import os
import json
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.entries = sorted(os.listdir(dataset_dir))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry_path = os.path.join(self.dataset_dir, self.entries[idx])
        with open(entry_path, 'r') as f:
            entry = json.load(f)
        return entry


def get_dataloader(dataset_dir, batch_size=8, shuffle=True):
    dataset = CustomDataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
