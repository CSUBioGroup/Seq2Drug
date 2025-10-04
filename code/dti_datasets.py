import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def load_data(
    batch_size, 
    data_args=None,
    split='train',
    loop=True
):

    data_dtis = get_input_data(data_args, split)
    dataset = DTIDataset(data_dtis)

    if split != 'test':
        sampler = DistributedSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
        )
    
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )

    if loop:
        return infinite_loader(data_loader)
    else:
        return iter(data_loader)


def get_input_data(data_args, split='train'):

    print('#'*30, '\nLoading dataset {} from {}...'.format(data_args.dataset, data_args.data_dir))

    if split == 'train':
        print('### Loading form the TRAIN set...')
        path = f'{data_args.data_dir}/train_interactions.csv'
    elif split == 'valid':
        print('### Loading form the VALID set...')
        path = f'{data_args.data_dir}/val_interactions.csv'
    elif split == 'test':
        print('### Loading form the TEST set...')
        path = f'{data_args.data_dir}/test_interactions.csv'
    else:
        assert False, "invalid split for dataset"

    dti_data = pd.read_csv(path)[['SMILES','target_sequence']]
    dti_lst = {'SMILES': list(dti_data['SMILES']), 'target_sequence': list(dti_data['target_sequence'])}

    return dti_lst


class DTIDataset(Dataset):
    def __init__(self, dti_datasets):
        super().__init__()
        self.dti_datasets = dti_datasets

    def __len__(self):
        return len(self.dti_datasets['SMILES'])

    def __getitem__(self, idx):

        smiles = self.dti_datasets['SMILES'][idx]
        target_sequence = self.dti_datasets['target_sequence'][idx]

        out_kwargs = {}
        out_kwargs['SMILES'] = smiles
        out_kwargs['target_sequence'] = target_sequence.upper()

        return out_kwargs


def infinite_loader(data_loader):
    while True:
        yield from data_loader
