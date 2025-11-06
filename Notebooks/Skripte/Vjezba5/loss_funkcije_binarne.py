"""
Skripta u kojoj se implementiraju funkcije sigmoid, vjerojatnost prema Bernoullijevoj
distribuciji, likelihood i negativni log-likelihood.
"""

# Knjižnice
import numpy as np
import math
import torch


# Sigmoid funkcija koja preslikava [-∞, ∞] u [0, 1]
def sigmoid(model_out):
    """
    Izračunava logističku sigmoidnu funkciju:
        sigmoid(z) = 1 / (1 + exp(-z))

    Argumenti:
        model_out (float ili np.array): Ulazna vrijednost (logiti)

    Povratna vrijednost:
        sig_model_out (float ili np.array): Sigmoid izlaz u intervalu [0, 1]
    """
    sig_model_out = 1 / (1 + np.exp(-model_out))
    return sig_model_out


# Vraća vjerojatnost prema Bernoullijevoj distribuciji za opaženu klasu y
def bernoulli_distribution(y, lambda_param):
    """
    Izračunava P(y | λ) = (1 - λ)^(1 - y) * λ^y

    Argumenti:
        y (np.array ili float): Opažena klasa (0 ili 1)
        lambda_param (np.array ili float): Parametar λ u [0,1]

    Povratna vrijednost:
        prob (np.array ili float): Vjerojatnost da se opaža y za zadani λ
    """
    prob = np.power(lambda_param, y) * np.power(1 - lambda_param, 1 - y)
    return prob


# Vraća likelihood cijelog skupa podataka za zadani model
def compute_likelihood_b(y_train, lambda_param):
    """
    Računa likelihood skupa podataka:
        L = Π_i ( lambda_param[i]^y_i * (1 - lambda_param[i])^(1 - y_i) )

    Argumenti:
        y_train (np.array): Vektor opaženih oznaka (0 ili 1)
        lambda_param (np.array): Vektor vjerojatnosti λ u [0,1]

    Povratna vrijednost:
        likelihood (float): Ukupni likelihood opaženih podataka
    """
    # Izračunaj Bernoulli vjerojatnosti za svaki podatak
    sample_probs = bernoulli_distribution(y_train, lambda_param)

    # Likelihood je produkt pojedinačnih vjerojatnosti
    likelihood = np.prod(sample_probs)

    return likelihood


# Vraća negativni log-likelihood podataka pod modelom
def compute_negative_log_likelihood_b(y_train, lambda_param):
    """
    Računa negativni log-likelihood (NLL) Bernoullijevog modela:
        NLL = - Σ_i [ y_i * log(λ_i) + (1 - y_i) * log(1 - λ_i) ]

    Argumenti:
        y_train (np.array): Opažene oznake (0 ili 1)
        lambda_param (np.array): Predviđene vjerojatnosti λ u [0,1]

    Povratna vrijednost:
        nll (float): Vrijednost negativnog log-likelihooda
    """
    # Mali offset da se izbjegne log(0) (numerička stabilnost)
    eps = 1e-12
    lambda_param = np.clip(lambda_param, eps, 1 - eps)

    nll = -np.sum(y_train * np.log(lambda_param) + (1 - y_train) * np.log(1 - lambda_param))
    return nll
