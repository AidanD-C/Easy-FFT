from Easy_FFT import FFT_1D
import numpy as np


def test_centered_gaussian():

    pi = np.pi
    sigma = 1
    mu = 0

    def gaussian(x):
        return (1 / (np.sqrt(2 * pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def ft_of_gaussian(k):
        return np.exp(-(k**2) * sigma**2 / 2) * np.exp(-1j * k * mu)

    x, k, f_hat = FFT_1D(gaussian, 100, 0.01)

    f_hat_actual = ft_of_gaussian(k)

    assert np.allclose(f_hat, f_hat_actual)


def test_off_center_gaussian_pos():

    pi = np.pi
    sigma = 1
    mu = 1

    def gaussian(x):
        return (1 / (np.sqrt(2 * pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def ft_of_gaussian(k):
        return np.exp(-(k**2) * sigma**2 / 2) * np.exp(-1j * k * mu)

    x, k, f_hat = FFT_1D(gaussian, 100, 0.01)

    f_hat_actual = ft_of_gaussian(k)

    assert np.allclose(f_hat, f_hat_actual)


def test_off_center_gaussian_neg():

    pi = np.pi
    sigma = 1
    mu = -1

    def gaussian(x):
        return (1 / (np.sqrt(2 * pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def ft_of_gaussian(k):
        return np.exp(-(k**2) * sigma**2 / 2) * np.exp(-1j * k * mu)

    x, k, f_hat = FFT_1D(gaussian, 100, 0.01)

    f_hat_actual = ft_of_gaussian(k)

    assert np.allclose(f_hat, f_hat_actual)


def test_centered_gaussian_times_x():

    pi = np.pi
    sigma = 1
    mu = 0

    def gaussian_times_x(x):
        return x * (1 / (np.sqrt(2 * pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def ft_of_gaussian_times_x(k):
        return 1j * np.exp(-(k**2) * sigma**2 / 2) * np.exp(-1j * k * mu) * -(k * sigma**2 + 1j * mu)

    x, k, f_hat = FFT_1D(gaussian_times_x, 100, 0.01)

    f_hat_actual = ft_of_gaussian_times_x(k)

    assert np.allclose(f_hat, f_hat_actual)


def test_off_center_gaussian_times_x_pos():

    pi = np.pi
    sigma = 1
    mu = 5

    def gaussian_times_x(x):
        return x * (1 / (np.sqrt(2 * pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def ft_of_gaussian_times_x(k):
        return 1j * np.exp(-(k**2) * sigma**2 / 2) * np.exp(-1j * k * mu) * -(k * sigma**2 + 1j * mu)

    x, k, f_hat = FFT_1D(gaussian_times_x, 100, 0.01)

    f_hat_actual = ft_of_gaussian_times_x(k)

    assert np.allclose(f_hat, f_hat_actual)


def test_off_center_gaussian_times_x_neg():

    pi = np.pi
    sigma = 1
    mu = -1

    def gaussian_times_x(x):
        return x * (1 / (np.sqrt(2 * pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def ft_of_gaussian_times_x(k):
        return 1j * np.exp(-(k**2) * sigma**2 / 2) * np.exp(-1j * k * mu) * -(k * sigma**2 + 1j * mu)

    x, k, f_hat = FFT_1D(gaussian_times_x, 100, 0.01)

    f_hat_actual = ft_of_gaussian_times_x(k)

    assert np.allclose(f_hat, f_hat_actual)
