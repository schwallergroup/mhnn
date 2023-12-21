import shutil
import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric.data import (
    InMemoryDataset,
    download_url,
)
from datasets.utils import smi2hgraph, HData, edge_order


class OCELOTv1(InMemoryDataset):
    r""" The molecular hypergraph dataset class for the OCELOT chromophore v1 dataset
    from the `"Chem. Sci., 2023, 14, 203-213." <https://doi.org/10.1039/D2SC04676H>`_ paper,
    consisting of about 25,251 molecules with 15 regression targets.
    
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    raw_url = 'https://data.materialsdatafacility.org/mdf_open/ocelot_chromophore_v1_v1.1/ocelot_chromophore_v1.csv'

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.ids = self.data.smi

    def mean(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return y[:, target].mean().item()

    def std(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return y[:, target].std().item()

    @property
    def raw_file_names(self):
        return ['ocelot_chromophore_v1.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        print('Downloading dataset...')
        try:
            download_url(self.raw_url, self.raw_dir)
        except:
            shutil.copy('datasets/raw/ocelot_chromophore_v1.csv', self.raw_dir)

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        smiles = df['smiles'].values.tolist()
        target = df.iloc[:, 1:-1].values
        target = torch.tensor(target, dtype=torch.float)

        data_list = []
        for i, smi in enumerate(tqdm(smiles)):

            atom_fvs, n_idx, e_idx, bond_fvs = smi2hgraph(smi)

            x = torch.tensor(atom_fvs, dtype=torch.long)
            edge_index0 = torch.tensor(n_idx, dtype=torch.long)
            edge_index1 = torch.tensor(e_idx, dtype=torch.long)
            edge_attr = torch.tensor(bond_fvs, dtype=torch.long)
            y = target[i].unsqueeze(0)
            n_e = len(edge_index1.unique())
            e_order = torch.tensor(edge_order(e_idx), dtype=torch.long)

            data = HData(x=x, y=y, n_e=n_e, smi=smi,
                         edge_index0=edge_index0,
                         edge_index1=edge_index1,
                         edge_attr=edge_attr,
                         e_order=e_order)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
    