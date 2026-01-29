from Easy_FFT import FFT_1D
import pytest as pt
import numpy as np

pi = np.pi
sigma = 1
mu = 0

def gaussian(x):
    return (1 / (np.sqrt(2 * pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def ft_of_gaussian(k):
        return np.exp(-(k**2) * sigma**2 / 2) * np.exp(-1j * k * mu)

def test_centered(self):

    x,k,f_hat=FFT_1D(gaussian,100,0.01)

    f_hat_actual = ft_of_gaussian(k)

    assert np.allclose(f_hat,f_hat_actual)