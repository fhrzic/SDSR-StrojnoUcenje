"""
Main class for MNIST dataset and simple visualisation
"""

# Učitavanja knjižica
import math
from typing import List, Optional

import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class mnist_dataset(Dataset):
    """
    Klasa za MNIST skup podataka s implementiranim __len__, __getitem__
    i metodom za vizualizaciju. Može koristiti train, valid ili test podskup.

    Dodatno:
      * use_augmentation (bool) – ako je True i subset == 'train',
        primjenjuje se jednostavna augmentacija na slike.
      * verbose (bool) – ako je True, ispisuje postavke dataseta pri inicijalizaciji.
    """

    def __init__(self,
                 root: str = "./data",
                 subset: str = "train",
                 valid_ratio: float = 0.1,
                 transform=None,
                 download: bool = True,
                 random_seed: int = 42,
                 use_augmentation: bool = False,
                 augment_transform: Optional[transforms.Compose] = None,
                 verbose: bool = True):
        """
        Inicijalizacijska metoda u kojoj dohvaćamo MNIST podatke i prema
        argumentu `subset` odabiremo train, valid ili test podskup.

        Args:
            * root, str, lokacija gdje se sprema / učitava MNIST
            * subset, str, koji podskup se koristi: 'train', 'valid' ili 'test'
            * valid_ratio, float, udio validacijskog skupa iz train dijela (0.1 = 10%)
            * transform, torchvision.transforms, osnovne transformacije za slike
            * download, bool, ako je True, preuzet će skup ako nije lokalno dostupan
            * random_seed, int, sjeme za reproducibilnu podjelu na train/valid
            * use_augmentation, bool, ako je True i subset='train', koristi augmentaciju
            * augment_transform, torchvision.transforms.Compose, augmentacijske
              transformacije koje se primjenjuju u __getitem__ (nakon osnovnog transforma)
            * verbose, bool, ako je True, ispisuje postavke dataseta pri inicijalizaciji
        """
        self.subset = subset.lower()
        assert self.subset in ["train", "valid", "test"], \
            f"subset mora biti 'train', 'valid' ili 'test', dobiveno: {subset}"

        self.use_augmentation = bool(use_augmentation)
        self.verbose = bool(verbose)

        # Zadane osnovne transformacije ako nisu zadane izvana
        # (ToTensor + Normalize, kao i prije)
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.base_transform = transform  # spremamo radi ispisa

        # Zadana augmentacija ako nije zadana izvana (rotacije + pomak + scale)
        if augment_transform is None:
            augment_transform = transforms.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            )
        self.augment_transform = augment_transform

        if self.subset in ["train", "valid"]:
            # Učitaj cijeli train dio
            _full_train = datasets.MNIST(
                root=root,
                train=True,
                download=download,
                transform=self.base_transform,
            )

            _num_train = len(_full_train)
            _num_valid = int(valid_ratio * _num_train)
            _num_train_final = _num_train - _num_valid

            # Reproducibilna podjela
            _generator = torch.Generator().manual_seed(random_seed)
            _train_dataset, _valid_dataset = random_split(
                _full_train,
                [_num_train_final, _num_valid],
                generator=_generator
            )

            if self.subset == "train":
                self.dataset = _train_dataset
            else:
                self.dataset = _valid_dataset
        else:
            # Test podskup – bez augmentacije
            self.dataset = datasets.MNIST(
                root=root,
                train=False,
                download=download,
                transform=self.base_transform,
            )

        # =============== ISPIS POSTAVKI ===============
        if self.verbose:
            print("==== Postavke MNIST dataseta ====")
            print(f"  subset           = {self.subset}")
            print(f"  root             = {root}")
            if self.subset in ["train", "valid"]:
                print(f"  valid_ratio      = {valid_ratio}")
                print(f"  random_seed      = {random_seed}")
            print(f"  use_augmentation = {self.use_augmentation}")
            base_t_name = type(self.base_transform).__name__ \
                if self.base_transform is not None else "None"
            print(f"  base_transform   = {base_t_name}")
            if self.use_augmentation and self.augment_transform is not None:
                aug_t_name = type(self.augment_transform).__name__
                print(f"  augment_transform= {aug_t_name}")
            print("==================================\n")
        # =============================================

    def __len__(self) -> int:
        """
        Metoda koja vraća ukupnu duljinu dataseta.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Metoda koja vraća jedan podatak temeljem njegovog indexa iz self.dataset.

        Augmentacija:
          * ako je subset == 'train' i use_augmentation == True,
            primjenjuje se self.augment_transform na sliku.
        """
        _img, _label = self.dataset[idx]

        # Augmentacija samo za train podskup
        if self.subset == "train" and self.use_augmentation and self.augment_transform is not None:
            # augment_transform radi nad Tensorom (RandomAffine podržava Tensor)
            _img = self.augment_transform(_img)

        return _img, _label

    def visualise(self,
                  n_samples: int = 16,
                  n_cols: int = 4,
                  cmap: str = "gray"):
        """
        Metoda koja vizualizira nekoliko uzoraka iz trenutnog podskupa.
        """
        _n_samples = min(n_samples, len(self.dataset))
        _n_cols = max(1, n_cols)
        _n_rows = math.ceil(_n_samples / _n_cols)

        _indices = torch.randint(
            low=0,
            high=len(self.dataset),
            size=(_n_samples,)
        )

        _fig, _axes = plt.subplots(
            _n_rows,
            _n_cols,
            figsize=(1.5 * _n_cols, 1.5 * _n_rows)
        )

        _axes = _axes.flatten() if _n_rows * _n_cols > 1 else [_axes]

        for _ax_idx, (_idx, _ax) in enumerate(zip(_indices, _axes)):
            _img, _label = self.dataset[int(_idx)]

            if isinstance(_img, torch.Tensor):
                _img_np = _img.squeeze().cpu().numpy()
            else:
                _img_np = _img

            _ax.imshow(_img_np, cmap=cmap)
            _ax.set_title(f"{int(_label)}", fontsize=8)
            _ax.axis("off")

        for _ax in _axes[_n_samples:]:
            _ax.axis("off")

        _fig.suptitle(f"Primjeri iz MNIST ({self.subset}) podskupa", fontsize=10)
        _fig.tight_layout()
        plt.show()


def ordinary_dataloader(dataset: Dataset = None,
                        batch_size: int = 1) -> DataLoader:
    """
    Funkcija koja kreira običan dataloader za dani dataset batch_sizom.

    Args:
        * dataset, Dataset, instanca klase dataset
        * batch_size, int, veličina batcha

    Returns:
        * PyTorch DataLoader
    """
    _loader = DataLoader(dataset=dataset,
                         batch_size=batch_size,
                         shuffle=True)
    return _loader
