import torch
from torch_geometric.nn import MessagePassing, GATConv, GATv2Conv, MLP
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention
from torch_geometric.nn.aggr import Set2Set
import torch.nn.functional as F
from torch_geometric.utils import degree
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.BatchNorm1d(emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr):
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_attr))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        row, col = edge_index

        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)
        out += F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

        return out

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GNN_2D(torch.nn.Module):

    def __init__(self, num_tasks, num_layer=5, emb_dim=300, gnn_type='gin',
                 residual=False, drop_ratio=0.0, JK="last", graph_pooling="mean"):
        '''
            num_tasks (int): number of labels to be predicted
        '''

        super(GNN_2D, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.residual = residual
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim=emb_dim)
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            elif gnn_type == 'gat':
                self.convs.append(GATConv(emb_dim, emb_dim, heads=4,
                                  concat=False, edge_dim=emb_dim))
            elif gnn_type == 'gatv2':
                self.convs.append(GATv2Conv(emb_dim, emb_dim, heads=4,
                                  concat=False, edge_dim=emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        # if gnn_type == 'gin':
        #     self.conv = GINConv(emb_dim)
        # elif gnn_type == 'gcn':
        #     self.conv = GCNConv(emb_dim)
        # elif gnn_type == 'gat':
        #     self.conv = GATConv(emb_dim, emb_dim, heads=8,
        #                         concat=False, edge_dim=emb_dim)
        # else:
        #     raise ValueError('Undefined GNN type called {}'.format(gnn_type))
        # self.batch_norm = torch.nn.BatchNorm1d(emb_dim)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*emb_dim, num_tasks)
            # self.mlp = MLP([2*emb_dim, emb_dim, num_tasks])
        else:
            self.graph_pred_linear = torch.nn.Linear(emb_dim, num_tasks)
            # self.mlp = MLP([emb_dim, emb_dim, num_tasks])

    def forward(self, data):

        # computing node embedding
        h_list = [self.atom_encoder(data.x)]
        edge_attr = self.bond_encoder(data.edge_attr)

        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], data.edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = self.conv(h_list[layer], data.edge_index, edge_attr)
            # h = self.batch_norm(h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            
            if self.residual:
                h += h_list[layer]
            
            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "last":
            h_node = h_list[-1]
        elif self.JK == "sum":
            h_node = 0
            for layer in range(self.num_layer + 1):
                h_node += h_list[layer]

        h_graph = self.pool(h_node, data.batch)
        out = self.graph_pred_linear(h_graph)
        # out = self.mlp(h_graph)

        return out.view(-1)
