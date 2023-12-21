import os
import os.path as osp
import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_gz,
)
from rdkit import Chem
from ogb.utils import smiles2graph

from datasets.utils import smi2hgraph, HData, edge_order


class OPVHGraph(InMemoryDataset):
    r""" The molecular hypergraph dataset class for the OPV dataset
    from the `"J. Chem. Phys., 2019, 150, 234111." <https://doi.org/10.1063/1.5099132>`_ paper,
    consisting of about 90,823 molecules with 8 regression targets.

    Args:
        root (string): Root directory where the dataset should be saved.
        polymer (bool): whether it's polymeric tasks
        partition (string): which dataset split
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

    raw_url0 = ('https://cscdata.nrel.gov/api/datasets/ad5d2c9a-af0a-4d72-b943-1e'
                '433d5750d6/download/cf0c78ad-6356-495b-b233-3fa5e1cd4ee7')
    raw_url1 = ('https://cscdata.nrel.gov/api/datasets/ad5d2c9a-af0a-4d72-b943-1e'
                '433d5750d6/download/1222cfcb-db92-4fc8-a310-bfab74d9217f')
    raw_url2 = ('https://cscdata.nrel.gov/api/datasets/ad5d2c9a-af0a-4d72-b943-1e'
                '433d5750d6/download/3085f235-be59-4b7f-93a6-1ea0505d9fde')

    def __init__(self, root, polymer=False, partition='train',
                 transform=None, pre_transform=None, pre_filter=None):
        assert polymer in [True, False]
        self.polymer = polymer
        assert partition in ['train', 'valid', 'test']
        self.partition = partition

        super().__init__(root, transform, pre_transform, pre_filter)

        if self.partition == 'train' and self.polymer is False:
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif self.partition == 'train' and self.polymer is True:
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif self.partition == 'valid':
            self.data, self.slices = torch.load(self.processed_paths[2])
        else:
            self.data, self.slices = torch.load(self.processed_paths[3])
        self.ids = self.data.smi

    def mean(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return y[:, target].mean().item()

    def std(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return y[:, target].std().item()

    @property
    def raw_file_names(self):
        return ['smiles_train.csv', 'smiles_valid.csv', 'smiles_test.csv']

    @property
    def processed_file_names(self):
        return ['train.pt', 'polymer_train.pt', 'valid.pt', 'test.pt']

    def download(self):
        print('Downloading OPV train dataset...')
        download_url(self.raw_url0, self.raw_dir)
        os.rename(osp.join(self.raw_dir, 'cf0c78ad-6356-495b-b233-3fa5e1cd4ee7'),
                  osp.join(self.raw_dir, 'smiles_train.csv.gz'))
        file_path = osp.join(self.raw_dir, 'smiles_train.csv.gz')
        extract_gz(file_path, self.raw_dir)
        os.unlink(file_path)

        print('Downloading OPV valid dataset...')
        download_url(self.raw_url1, self.raw_dir)
        os.rename(osp.join(self.raw_dir, '1222cfcb-db92-4fc8-a310-bfab74d9217f'),
                  osp.join(self.raw_dir, 'smiles_valid.csv.gz'))
        file_path = osp.join(self.raw_dir, 'smiles_valid.csv.gz')
        extract_gz(file_path, self.raw_dir)
        os.unlink(file_path)

        print('Downloading OPV test dataset...')
        download_url(self.raw_url2, self.raw_dir)
        os.rename(osp.join(self.raw_dir, '3085f235-be59-4b7f-93a6-1ea0505d9fde'),
                  osp.join(self.raw_dir, 'smiles_test.csv.gz'))
        file_path = osp.join(self.raw_dir, 'smiles_test.csv.gz')
        extract_gz(file_path, self.raw_dir)
        os.unlink(file_path)

    def compute_hgraph_data(self, df):
        # create hgraph data list from pd.DataFrame object
        smiles = df['smile'].values.tolist()
        target = df.iloc[:, 2:].values
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
        
        return data_list

    def process(self):

        # for training set
        df = pd.read_csv(self.raw_paths[0])
        data_list = self.compute_hgraph_data(df)
        torch.save(self.collate(data_list), self.processed_paths[0])

        # for polymer training set
        df = pd.read_csv(self.raw_paths[0])
        df = df.dropna(subset=['gap_extrapolated'])
        data_list = self.compute_hgraph_data(df)
        torch.save(self.collate(data_list), self.processed_paths[1])

        # for valid set
        df = pd.read_csv(self.raw_paths[1])
        data_list = self.compute_hgraph_data(df)
        torch.save(self.collate(data_list), self.processed_paths[2])

        # for test set
        df = pd.read_csv(self.raw_paths[2])
        data_list = self.compute_hgraph_data(df)
        torch.save(self.collate(data_list), self.processed_paths[3])


class OPVGraph(InMemoryDataset):
    r"""The molecular graph dataset class for the OPV dataset
    from the `"J. Chem. Phys., 2019, 150, 234111." <https://doi.org/10.1063/1.5099132>`_ paper,
    consisting of about 90,823 molecules with 8 regression targets.

    Args:
        root (string): Root directory where the dataset should be saved.
        polymer (bool): whether it's polymeric tasks
        partition (string): which dataset split
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

    raw_url0 = ('https://cscdata.nrel.gov/api/datasets/ad5d2c9a-af0a-4d72-b943-1e'
                '433d5750d6/download/cf0c78ad-6356-495b-b233-3fa5e1cd4ee7')
    raw_url1 = ('https://cscdata.nrel.gov/api/datasets/ad5d2c9a-af0a-4d72-b943-1e'
                '433d5750d6/download/1222cfcb-db92-4fc8-a310-bfab74d9217f')
    raw_url2 = ('https://cscdata.nrel.gov/api/datasets/ad5d2c9a-af0a-4d72-b943-1e'
                '433d5750d6/download/3085f235-be59-4b7f-93a6-1ea0505d9fde')

    def __init__(self, root, polymer=False, partition='train', 
                 transform=None, pre_transform=None, pre_filter=None):
        assert polymer in [True, False]
        self.polymer = polymer
        assert partition in ['train', 'valid', 'test']
        self.partition = partition
        super().__init__(root, transform, pre_transform, pre_filter)

        if self.partition == 'train' and self.polymer is False:
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif self.partition == 'train' and self.polymer is True:
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif self.partition == 'valid':
            self.data, self.slices = torch.load(self.processed_paths[2])
        else:
            self.data, self.slices = torch.load(self.processed_paths[3])

    def mean(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return y[:, target].mean().item()

    def std(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return y[:, target].std().item()

    @property
    def raw_file_names(self):
        return ['smiles_train.csv', 'smiles_valid.csv', 'smiles_test.csv']

    @property
    def processed_file_names(self):
        return ['train.pt', 'polymer_train.pt', 'valid.pt', 'test.pt']

    def download(self):
        print('Downloading OPV train dataset...')
        download_url(self.raw_url0, self.raw_dir)
        os.rename(osp.join(self.raw_dir, 'cf0c78ad-6356-495b-b233-3fa5e1cd4ee7'),
                  osp.join(self.raw_dir, 'smiles_train.csv.gz'))
        file_path = osp.join(self.raw_dir, 'smiles_train.csv.gz')
        extract_gz(file_path, self.raw_dir)
        os.unlink(file_path)

        print('Downloading OPV valid dataset...')
        download_url(self.raw_url1, self.raw_dir)
        os.rename(osp.join(self.raw_dir, '1222cfcb-db92-4fc8-a310-bfab74d9217f'),
                  osp.join(self.raw_dir, 'smiles_valid.csv.gz'))
        file_path = osp.join(self.raw_dir, 'smiles_valid.csv.gz')
        extract_gz(file_path, self.raw_dir)
        os.unlink(file_path)

        print('Downloading OPV test dataset...')
        download_url(self.raw_url2, self.raw_dir)
        os.rename(osp.join(self.raw_dir, '3085f235-be59-4b7f-93a6-1ea0505d9fde'),
                  osp.join(self.raw_dir, 'smiles_test.csv.gz'))
        file_path = osp.join(self.raw_dir, 'smiles_test.csv.gz')
        extract_gz(file_path, self.raw_dir)
        os.unlink(file_path)

    def compute_graph_data(self, df):
        # create graph data list from pd.DataFrame object
        smiles = df['smile'].values.tolist()
        target = df.iloc[:, 2:].values.tolist()

        # Convert SMILES into graph data
        data_list = []
        for i, smi in enumerate(tqdm(smiles)):

            # get graph data from SMILES
            graph = smiles2graph(smi)
            x = torch.tensor(graph['node_feat'], dtype=torch.long)
            edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
            edge_attr = torch.tensor(graph['edge_feat'], dtype=torch.long)
            y = torch.tensor([target[i]], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        
        return data_list

    def process(self):
        
        # for training set
        df = pd.read_csv(self.raw_paths[0])
        data_list = self.compute_graph_data(df)
        torch.save(self.collate(data_list), self.processed_paths[0])

        # for polymer training set
        df = pd.read_csv(self.raw_paths[0])
        df = df.dropna(subset=['gap_extrapolated'])
        data_list = self.compute_graph_data(df)
        torch.save(self.collate(data_list), self.processed_paths[1])

        # for valid set
        df = pd.read_csv(self.raw_paths[1])
        data_list = self.compute_graph_data(df)
        torch.save(self.collate(data_list), self.processed_paths[2])

        # for test set
        df = pd.read_csv(self.raw_paths[2])
        data_list = self.compute_graph_data(df)
        torch.save(self.collate(data_list), self.processed_paths[3])


class OPVGraph3D(InMemoryDataset):
    r""" The 3D graph dataset class for the OPV dataset
    from the `"J. Chem. Phys., 2019, 150, 234111." <https://doi.org/10.1063/1.5099132>`_ paper,
    consisting of about 90,823 molecules with 8 regression targets.

    Args:
        root (string): Root directory where the dataset should be saved.
        polymer (bool): whether it's polymeric tasks
        partition (string): which dataset split
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

    raw_url0 = ('https://cscdata.nrel.gov/api/datasets/ad5d2c9a-af0a-4d72-b943-1e'
                '433d5750d6/download/b69cf9a5-e7e0-405b-88cb-40df8007242e')
    raw_url1 = ('https://cscdata.nrel.gov/api/datasets/ad5d2c9a-af0a-4d72-b943-1e'
                '433d5750d6/download/1c8e7379-3071-4360-ba8e-0c6481c33d2c')
    raw_url2 = ('https://cscdata.nrel.gov/api/datasets/ad5d2c9a-af0a-4d72-b943-1e'
                '433d5750d6/download/4ef40592-0080-4f00-9bb7-34b25f94962a')

    def __init__(self, root, polymer=False, partition='train', 
                 transform=None, pre_transform=None, pre_filter=None):
        assert polymer in [True, False]
        self.polymer = polymer
        assert partition in ['train', 'valid', 'test']
        self.partition = partition
        super().__init__(root, transform, pre_transform, pre_filter)

        if self.partition == 'train' and self.polymer is False:
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif self.partition == 'train' and self.polymer is True:
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif self.partition == 'valid':
            self.data, self.slices = torch.load(self.processed_paths[2])
        else:
            self.data, self.slices = torch.load(self.processed_paths[3])

    def mean(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return y[:, target].mean().item()

    def std(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return y[:, target].std().item()

    @property
    def raw_file_names(self):
        return ['train.sdf', 'train.csv', 'polymer_train.sdf', 'polymer_train.csv',
                'valid.sdf', 'valid.csv', 'test.sdf', 'test.csv']

    @property
    def processed_file_names(self):
        return ['train.pt', 'polymer_train.pt', 'valid.pt', 'test.pt', ]

    def extract_data(self, df, fname_sdf, fname_csv):
        """ extract data from original CSV file to SDF file (3D coordinates)
            and CSV file (targets)

        Args:
            df (pd.DataFrame): df from original CSV file
            fname_sdf (str): output path of SDF file
            fname_csv (str): output path of CSV file
        """
        # convert molecule structures to SDF file
        mol_txt = df['mol'].values.tolist()
        with open(fname_sdf, 'w') as f:
            for _txt in mol_txt:
                f.write(_txt)
                f.write('$$$$\n')
        # extract targets from df to CSV file
        df_target = df[
            ['smile', 'gap', 'homo', 'lumo',
             'spectral_overlap', 'homo_extrapolated',
             'lumo_extrapolated', 'gap_extrapolated',
             'optical_lumo_extrapolated']
        ]
        df_target.to_csv(fname_csv)

    def download(self):
        print('Downloading OPV train dataset...')
        download_url(self.raw_url0, self.raw_dir)
        os.rename(osp.join(self.raw_dir, 'b69cf9a5-e7e0-405b-88cb-40df8007242e'),
                  osp.join(self.raw_dir, 'mol_train.csv.gz'))
        file_path = osp.join(self.raw_dir, 'mol_train.csv.gz')
        extract_gz(file_path, self.raw_dir)
        os.unlink(file_path)
        df_train = pd.read_csv(osp.join(self.raw_dir, 'mol_train.csv'))
        self.extract_data(df_train, self.raw_paths[0], self.raw_paths[1])
        df_train = df_train.dropna(subset=['gap_extrapolated'])
        self.extract_data(df_train, self.raw_paths[2], self.raw_paths[3])

        print('Downloading OPV valid dataset...')
        download_url(self.raw_url1, self.raw_dir)
        os.rename(osp.join(self.raw_dir, '1c8e7379-3071-4360-ba8e-0c6481c33d2c'),
                  osp.join(self.raw_dir, 'mol_valid.csv.gz'))
        file_path = osp.join(self.raw_dir, 'mol_valid.csv.gz')
        extract_gz(file_path, self.raw_dir)
        os.unlink(file_path)
        df_valid = pd.read_csv(osp.join(self.raw_dir, 'mol_valid.csv'))
        self.extract_data(df_valid, self.raw_paths[4], self.raw_paths[5])

        print('Downloading OPV test dataset...')
        download_url(self.raw_url2, self.raw_dir)
        os.rename(osp.join(self.raw_dir, '4ef40592-0080-4f00-9bb7-34b25f94962a'),
                  osp.join(self.raw_dir, 'mol_test.csv.gz'))
        file_path = osp.join(self.raw_dir, 'mol_test.csv.gz')
        extract_gz(file_path, self.raw_dir)
        os.unlink(file_path)
        df_test = pd.read_csv(osp.join(self.raw_dir, 'mol_test.csv'))
        self.extract_data(df_test, self.raw_paths[6], self.raw_paths[7])
    
    def compute_graph_data(self, sdf_path, csv_path):
        # create graph data list from SDF and CSV files
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        df = pd.read_csv(csv_path)
        target = df.iloc[:, 2:].values.tolist()

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)
            atomic_number = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            z = torch.tensor(atomic_number, dtype=torch.long)
            y = torch.tensor([target[i]], dtype=torch.float)
            data = Data(z=z, pos=pos, y=y, idx=i)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        
        return data_list

    def process(self):
        # for training set
        data_list = self.compute_graph_data(self.raw_paths[0], self.raw_paths[1])
        torch.save(self.collate(data_list), self.processed_paths[0])

        # for polymer training set
        data_list = self.compute_graph_data(self.raw_paths[2], self.raw_paths[3])
        torch.save(self.collate(data_list), self.processed_paths[1])

        # for valid set
        data_list = self.compute_graph_data(self.raw_paths[4], self.raw_paths[5])
        torch.save(self.collate(data_list), self.processed_paths[2])

        # for test set
        data_list = self.compute_graph_data(self.raw_paths[6], self.raw_paths[7])
        torch.save(self.collate(data_list), self.processed_paths[3])
