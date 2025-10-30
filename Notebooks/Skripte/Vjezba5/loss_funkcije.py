"""
Skripta u kojoj se implementiraju funkcije likelihood, mse i negative loglikelihood 
"""

# Knjižnice
import numpy as np
import math

def normal_distribution(y:np.array = None, 
                        mu:float = None, 
                        sigma: float = None):
    """
    Methoda koja za dani y, mu i sigmu vraća vjerojatnost normalne distribucije

    Args:
        * y, np.array, niz brojeva za koje se računa vjerojatnost prema danoj Normalnoj distribuciji
        koja je definirana parametrima mu i sigma.
        * mu, float, srednja vrijednost normalne distribucije.
        * sigma, float, standardna devijacija normalne distribucije.
    """
    # Compute the coefficient (1 / (sigma * sqrt(2π)))
    coeff = 0
    
    # Compute exponent part: exp(-(y - mu)^2 / (2 * sigma^2))
    exponent = 0
    
    # Multiply them
    prob = 0
    return prob

def compute_sum_of_squares(y_train:np.array, 
                           y_pred:np.array)->float:
    """
    Metoda koja vraća sumu kvardata

    Args:
        * y_train, np.array, niz brojeva koje reprezentiraju oznake podataka
        * y_pred, np.array, niz brojeva koje reprezentiraju predikciju nekog
        modela

    Returns:
        * np.array, niz brojeva od kojeg svaki predstavlja sumu kvadrata odgovarajućih
        parova između označenih-pravih i predviđenih vrijednosti.

    """
    # Izračun sume kvadrata između predikcije i stvarne vrijednosti.
    sum_of_squares = 0
    return sum_of_squares




# Return the likelihood of all of the data under the model
def compute_likelihood(y_train: np.array = None, 
                       mu: np.array = None, 
                       sigma: float = None):
    """
    Methoda koja za dani y, mu i sigmu vraća vjerojatnost normalne distribucije

    Args:
        * y_train, np.array, niz brojeva za koje se računa vjerojatnost prema danoj Normalnoj distribuciji
        koja je definirana parametrima mu i sigma.
        * mu, np.array, niz predviđenih vrijednosti, odnosno sredina od strane modela za dani y_train.
        * sigma, float, standardna devijacija normalne distribucije.
    """

    # Izračunajte vjerojatnost za svaki dani uzorak
    _probs = 0

    # Umnožak svih izračunatih vjerojatnosti
    _likelihood = 0

    return _likelihood

def compute_negative_log_likelihood(y_train: np.array = None, 
                                    mu: np.array = None, 
                                    sigma: np.array= None)->float:
    """
    Methoda koja vraća izračunati negative loglikelihood za dani y_train i pripadajuće mu i sigma
    vrijednosti

    Args:
        * y_train, np.array, niz brojeva za koje se računa vjerojatnost prema danoj Normalnoj distribuciji
        koja je definirana parametrima mu i sigma.
        * mu, np.array, niz predviđenih vrijednosti, odnosno sredina od strane modela za dani y_train.
        * sigma, float, standardna devijacija normalne distribucije.
    """
    # Get probability of each data point
    probs = 0
    
    # Compute negative log likelihood
    nll = 0
    
    return nll