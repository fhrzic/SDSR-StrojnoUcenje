import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple, Sequence
import pandas as pd
import os
import sys

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(current_file_path)
print(main_dir)
sys.path.append(main_dir)

class XYPairDataset(Dataset):
    """Dataset for scalar x -> scalar y pairs."""
    def __init__(self, dataset_name: str = None, 
                 subset: str = "train"):
        """
        Method which implements dataset

        Args:
            * dataset_name, str, path to the dataset.
            * subset, str, it can be eiteher train or valid.
        """
        # Load data
        _data = pd.read_csv(os.path.join(main_dir, dataset_name), header=None)
        # Number of data samples
    
        _n_train = int(0.8 * len(_data))
        _n_val = len(_data) - _n_train
        # Get splits
        _train_ds, _val_ds = random_split(_data, 
                                          [_n_train, _n_val], 
                                          generator=torch.Generator().manual_seed(42))

        # Set dataset
        if subset == "train":
            self.data = _train_ds
        if subset == "valid":
            self.data = _val_ds
        print(self.data)
        # Count
        self.count = len(self.data)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        # Obtain x and y for given idx
        _x, _y = self.data[idx]

        # Change to torch tensor
        _x =  torch.tensor(_x)
        _y = torch.tensor (_y) 
        return _x, _y
    
def ordinary_dataloader(dataset: Dataset = None,
                        batch_size: int = 8)->DataLoader:
    """
    Function which builds dataset.

    Args:
        * dataset, Dataset, dataset class
        * batch_size, int, number of items in batch
    
    Returns:
        * torch.Dataloader
    """
    return DataLoader(dataset, batch_size = batch_size, shuffle=True)
