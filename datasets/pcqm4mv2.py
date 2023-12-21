import os
import os.path as osp
import shutil
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset

from .utils import HData, smi2hgraph, edge_order


class PCQM4Mv2(InMemoryDataset):
    def __init__(self, root='dataset', smiles2graph=smi2hgraph, transform=None, pre_transform=None):
        '''
            The molecular hypergraph dataset class for PCQM4Mv2 dataset
            <https://ogb.stanford.edu/docs/lsc/pcqm4mv2/>
                - root (str): the dataset folder will be located at root/pcqm4m-v2
                - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                    * The default smiles2hgraph requires rdkit to be installed
        '''

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'pcqm4m-v2')
        self.version = 1
        
        # Old url hosted at Stanford
        # md5sum: 65b742bafca5670be4497499db7d361b
        # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip'
        # New url hosted by DGL team at AWS--much faster to download
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'

        # check version and update if necessary
        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4Mv2 dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.folder)

        super(PCQM4Mv2, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print('Stop download.')
            exit(-1)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        smiles_list = data_df['smiles']
        homolumogap_list = data_df['homolumogap']

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smiles_list))):

            smiles = smiles_list[i]
            homolumogap = homolumogap_list[i]
            atom_fvs, n_idx, e_idx, bond_fvs = self.smiles2graph(smiles)

            x = torch.tensor(atom_fvs, dtype=torch.long)
            edge_index0 = torch.tensor(n_idx, dtype=torch.long)
            edge_index1 = torch.tensor(e_idx, dtype=torch.long)
            edge_attr = torch.tensor(bond_fvs, dtype=torch.long)
            y = torch.tensor([homolumogap], dtype=torch.float)
            n_e = len(edge_index1.unique())
            e_order = torch.tensor(edge_order(e_idx), dtype=torch.long)

            data = HData(x=x, y=y, n_e=n_e, smi=smiles,
                         edge_index0=edge_index0,
                         edge_index1=edge_index1,
                         edge_attr=edge_attr,
                         e_order=e_order)
            data.__num_nodes__ = len(x)

            data_list.append(data)

        # double-check prediction target
        split_dict = self.get_idx_split()
        assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['train']]))
        assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
        assert(all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-dev']]))
        assert(all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-challenge']]))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'split_dict.pt')))
        return split_dict


if __name__ == '__main__':
    dataset = PCQM4Mv2()
    print(dataset)
    print(dataset.data.edge_index0)
    print(dataset.data.edge_index1)
    print(dataset.data.edge_index0.shape)
    print(dataset.data.x.shape)
    print(dataset[100])
    print(dataset[100].y)
    print(dataset.get_idx_split())
