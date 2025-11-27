"""
Main class for MNIST dataset and simple visualisation
"""

# Učitavanja knjižica
import math
from typing import List

import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class mnist_dataset(Dataset):
    """
    Klasa za MNIST skup podataka s implementiranim __len__, __getitem__
    i metodom za vizualizaciju. Može koristiti train, valid ili test podskup.

    """

    def __init__(self,
                 root: str = "./data",
                 subset: str = "train",
                 valid_ratio: float = 0.1,
                 transform=None,
                 download: bool = True,
                 random_seed: int = 42):
        """
        # NUŽNA METODA #

        Inicijalizacijska metoda u kojoj dohvaćamo MNIST podatke i prema
        argumentu `subset` odabiremo train, valid ili test podskup.

        Args:
            * root, str, lokacija gdje se sprema / učitava MNIST
            * subset, str, koji podskup se koristi: 'train', 'valid' ili 'test'
            * valid_ratio, float, udio validacijskog skupa iz train dijela (npr. 0.1 = 10%)
            * transform, torchvision.transforms, transformacije za slike
            * download, bool, ako je True, preuzet će skup ako nije lokalno dostupan
            * random_seed, int, sjeme za reproducibilnu podjelu na train/valid

        Returns:
            * metoda ne vraća ništa, ali populira self.dataset varijablu koja
              sadrži odabrani podskup podataka.
        """
        self.subset = subset.lower()
        assert self.subset in ["train", "valid", "test"], \
            f"subset mora biti 'train', 'valid' ili 'test', dobiveno: {subset}"

        # Zadane transformacije ako nisu zadane izvana
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        if self.subset in ["train", "valid"]:
            # Učitaj cijeli train dio
            _full_train = datasets.MNIST(
                root=root,
                train=True,
                download=download,
                transform=transform
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
            # Test podskup
            self.dataset = datasets.MNIST(
                root=root,
                train=False,
                download=download,
                transform=transform
            )

    def __len__(self) -> int:
        """
        # NUŽNA METODA #

        Metoda koja vraća ukupnu duljinu dataseta.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        # NUŽNA METODA #

        Metoda koja vraća jedan podatak temeljem njegovog indexa iz self.dataset.

        Args:
            * idx, int, index podataka koji se dohvaća iz skupa podataka.

        Returns:
            * slika, torch.Tensor, veličine (1, 28, 28) nakon transformacije
            * label, int, oznaka klase (0-9)
        """
        _img, _label = self.dataset[idx]
        return _img, _label

    def visualise(self,
                  n_samples: int = 16,
                  n_cols: int = 4,
                  cmap: str = "gray"):
        """
        Metoda koja vizualizira nekoliko uzoraka iz trenutnog podskupa
        (train/valid/test) u obliku mreže (grid).

        Args:
            * n_samples, int, ukupan broj uzoraka koje crtamo
            * n_cols, int, broj stupaca u mreži
            * cmap, str, colormap za prikaz slika ('gray' za MNIST)
        """
        _n_samples = min(n_samples, len(self.dataset))
        _n_cols = max(1, n_cols)
        _n_rows = math.ceil(_n_samples / _n_cols)

        # Nasumično odabrani indeksi iz trenutnog podskupa
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

            # MNIST slike su (1, 28, 28) -> pretvaramo u 2D array
            if isinstance(_img, torch.Tensor):
                _img_np = _img.squeeze().cpu().numpy()
            else:
                _img_np = _img

            _ax.imshow(_img_np, cmap=cmap)
            _ax.set_title(f"{int(_label)}", fontsize=8)
            _ax.axis("off")

        # Ugasimo viškove ako ih ima
        for _ax in _axes[_n_samples:]:
            _ax.axis("off")

        _fig.suptitle(f"Primjeri iz MNIST ({self.subset}) podskupa", fontsize=10)
        _fig.tight_layout()
        plt.show()


def ordinary_dataloader(dataset: Dataset = None,
                        batch_size: int = 1)->DataLoader:
    """
    Funkcija koja kreira običan dataloader za dani dataset batch_sizom koji definira
    koliko se podataka vraća u jednom pozivu. Dataloader se može shvatiti kao napredni
    iterator

    Args:
        * dataset, Dataset, instanca klase dataset koja ima implementirane metode getitem i len.
        * batch_size, int, količina podataka koji se vraćaju u jednoj iteraciji-pozivu dataloadera
    
    Returns:
        * običan dataloader kreiran u PyTorchu
    """
    _loader = DataLoader(dataset = dataset, 
                         batch_size = batch_size, 
                         shuffle = True)
    
    # Return
    return _loader