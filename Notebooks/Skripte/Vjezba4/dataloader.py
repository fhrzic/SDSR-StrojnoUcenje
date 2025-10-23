"""
Main class for dataloader and dataset building
"""

# Učitavanja knjižica
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List

def inspect_dataset(path_to_csv: str = None):
    """
    Metoda koja će ispisati kratki info o podacima koji se
    nalaze u skupu podataka.
    """
    # Provjera postoji li put
    assert os.path.exists(path_to_csv), f"Path does not exist: {path_to_csv}"
    
    # Ispis kolumni zajedno sa mini statistikom
    _df = pd.read_csv(path_to_csv)
    print("Label      Udio     Missing        Vrijednosti(5)")
    print("*****************************************************************")
    for _col in _df.columns:
        _count = _df[_col].count()        
        _total = len(_df)                
        _missing = _total - _count  
        _uniques = _df[_col].dropna().unique()[:5]      
        print(f"{_col}: {_count} / {_total} (missing: {_missing}) (Vales: {_uniques})")

class bikeRentalDatasetTwoOutputs(Dataset):
    """
    Klasa za skup podataka koji se može pronaći na:
    https://www.kaggle.com/datasets/archit9406/bike-sharing
    """
    def __init__(self, 
                 path_to_csv: str = None,
                 input_label: str = None,
                 target_label1: str = None,
                 target_label2: str = None,
                 normalizacija: bool = False):
        """
        # NUŽNA METODA #

        Inicijalizacijska metoda u kojoj dohvaćamo podatke. Kao ulaz
        prima dva argumenta nužna za ovaj skup podataka, a po potrebi
        može i više. Zaslužna je za dohvaćanje i u ovom slučaju učitavanje 
        te filtriranje podataka, a konačni rezultat jest varijabla koja sadrži
        učitane podatke.

        Args:
            * path_to_csv, str, put do csv datoteke koja se učitava
            * input_label, str, ime varijable koja se selektira kao
            ulazni podatak.
            * target_label1, str, ime prve varijable koja je izlazni podatak
            vrijednost koju nastojimo predvidjeti.
            * target_label2, str, ime druge varijable koja je izlazni podatak
            vrijednost koju nastojimo predvidjeti.
            * normalizacija, bool, ako je postavljeno na true, podaci će
            biti normalizirani metodom min-max mapirajući svaku značajku na 
            interval 0-1.

        Returns:
            * metoda ne vraća ništa, ali populira self.data varijablu koja
            sadrži podatke.
        """

        # Provjera postoji li put
        assert os.path.exists(path_to_csv), f"Path does not exist: {path_to_csv}"

                # Load dataset
        self.data = pd.read_csv(path_to_csv)

        # Check if columns exist
        assert input_label in self.data.columns, f"Input label '{input_label}' not found in CSV"
        assert target_label1 in self.data.columns, f"Input label '{target_label1}' not found in CSV"
        assert target_label2 in self.data.columns, f"Target label '{target_label2}' not found in CSV"

        # Extract inputs and target
        _x1 = self.data[[input_label]].values 
        _y1 = self.data[[target_label2]].values  
        _y2 = self.data[target_label2].values 

        # Normalizacija
        if normalizacija is True:
            _eps = 1e-8
            _x1_min = _x1.min(axis=0)
            _x1_max = _x1.max(axis=0)
            _y2_min = _y2.min()
            _y2_max = _y2.max()
            _y1_min = _y1.min()
            _y1_max = _y1.max()
            _x1 = (_x1 - _x1_min) / (_x1_max - _x1_min + _eps)
            _y1 = (_y1 - _y1_min) / (_y1_max - _y1_min + _eps)
            _y2 = (_y2 - _y2_min) / (_y2_max - _y2_min + _eps)


        # Now merge data
        self.data = [[_i, _j, _k] for _i, _j, _k in zip(_x1, _y1, _y2)]

    def __len__(self)->int:
        """
        # NUŽNA METODA "
        Metoda koja vraća ukupnu duljinu dataseta
        """
        return len(self.data)

    def __getitem__(self, idx)->List[torch.Tensor]:
        """
        #NUŽNA METODA#

        Metoda koja vraća jedan podatak temeljem njegovog indexa iz self.data

        Args:
            * idx, int, index podataka koji se dohvaća iz skupa podataka. Index bira
            Dataloader u PyTorchu u kojem, ako definiramo svojeg, možemo stvoriti vlastiti
            način dohvaćanja podataka.
        """
        # Convert to torch tensors
        _x_tensor = torch.tensor(self.data[idx][0], dtype=torch.float32)
        _y1_tensor = torch.tensor(self.data[idx][1], dtype=torch.float32)
        _y2_tensor = torch.tensor(self.data[idx][2], dtype=torch.float32)
        
        # Vratimo tensore
        return _x_tensor, _y1_tensor, _y2_tensor

class bikeRentalDatasetTwoInputs(Dataset):
    """
    Klasa za skup podataka koji se može pronaći na:
    https://www.kaggle.com/datasets/archit9406/bike-sharing
    """
    def __init__(self, 
                 path_to_csv: str = None,
                 input_label1: str = None,
                 input_label2: str = None,
                 target_label: str = None,
                 normalizacija: bool = False):
        """
        # NUŽNA METODA #

        Inicijalizacijska metoda u kojoj dohvaćamo podatke. Kao ulaz
        prima dva argumenta nužna za ovaj skup podataka, a po potrebi
        može i više. Zaslužna je za dohvaćanje i u ovom slučaju učitavanje 
        te filtriranje podataka, a konačni rezultat jest varijabla koja sadrži
        učitane podatke.

        Args:
            * path_to_csv, str, put do csv datoteke koja se učitava
            * input_label, str, ime varijable koja se selektira kao
            ulazni podatak.
            * target_label1, str, ime prve varijable koja je izlazni podatak
            vrijednost koju nastojimo predvidjeti.
            * target_label2, str, ime druge varijable koja je izlazni podatak
            vrijednost koju nastojimo predvidjeti.
            * normalizacija, bool, ako je postavljeno na true, podaci će
            biti normalizirani metodom min-max mapirajući svaku značajku na 
            interval 0-1.

        Returns:
            * metoda ne vraća ništa, ali populira self.data varijablu koja
            sadrži podatke.
        """

        # Provjera postoji li put
        assert os.path.exists(path_to_csv), f"Path does not exist: {path_to_csv}"

                # Load dataset
        self.data = pd.read_csv(path_to_csv)

        # Check if columns exist
        assert input_label1 in self.data.columns, f"Input label '{input_label1}' not found in CSV"
        assert input_label2 in self.data.columns, f"Input label '{input_label2}' not found in CSV"
        assert target_label in self.data.columns, f"Target label '{target_label}' not found in CSV"

        # Extract inputs and target
        _x1 = self.data[[input_label1]].values 
        _x2 = self.data[[input_label2]].values  
        _y = self.data[target_label].values 

        # Normalizacija
        if normalizacija is True:
            _eps = 1e-8
            _x1_min = _x1.min(axis=0)
            _x1_max = _x1.max(axis=0)
            _x2_min = _x2.min(axis=0)
            _x2_max = _x2.max(axis=0)
            _y_min = _y.min()
            _y_max = _y.max()
            _x1 = (_x1 - _x1_min) / (_x1_max - _x1_min + _eps)
            _x2 = (_x2 - _x2_min) / (_x2_max - _x2_min + _eps)
            _y = (_y - _y_min) / (_y_max - _y_min + _eps)

        # Now merge data
        self.data = [[_i, _j, _k] for _i, _j, _k in zip(_x1, _x2, _y)]

    def __len__(self)->int:
        """
        # NUŽNA METODA "
        Metoda koja vraća ukupnu duljinu dataseta
        """
        return len(self.data)

    def __getitem__(self, idx)->List[torch.Tensor]:
        """
        #NUŽNA METODA#

        Metoda koja vraća jedan podatak temeljem njegovog indexa iz self.data

        Args:
            * idx, int, index podataka koji se dohvaća iz skupa podataka. Index bira
            Dataloader u PyTorchu u kojem, ako definiramo svojeg, možemo stvoriti vlastiti
            način dohvaćanja podataka.
        """
        # Convert to torch tensors
        _x1_tensor = torch.tensor(self.data[idx][0], dtype=torch.float32)
        _x2_tensor = torch.tensor(self.data[idx][1], dtype=torch.float32)
        _y_tensor = torch.tensor(self.data[idx][2], dtype=torch.float32)
        
        # Vratimo tensore
        return _x1_tensor, _x2_tensor, _y_tensor


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
                         shuffle = False)
    
    # Return
    return _loader