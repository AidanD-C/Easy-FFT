from Easy_FFT import IFFT_2D
import numpy as np


def test_centered_gaussian_real():

    dx = 0.01
    L = 100

    sigma_x_real = 1
    sigma_y_real = 1
    x_center_real = 0
    y_center_real = 0
    A_real = 1

    sigma_x_imag = 1
    sigma_y_imag = 1
    x_center_imag = 0
    y_center_imag = 0
    A_imag = 0

    def gaussian(x, y):
        coeff_real = A_real / (2 * np.pi * sigma_x_real * sigma_y_real)
        exponent_real = -(((x - x_center_real) ** 2) / (2 * sigma_x_real**2) + ((y - y_center_real) ** 2) / (2 * sigma_y_real**2))

        coeff_imag = A_imag / (2 * np.pi * sigma_x_imag * sigma_y_imag)
        exponent_imag = -(((x - x_center_imag) ** 2) / (2 * sigma_x_imag**2) + ((y - y_center_imag) ** 2) / (2 * sigma_y_imag**2))

        return coeff_real * np.exp(exponent_real) + 1j * coeff_imag * np.exp(exponent_imag)

    def ft_of_gaussian(k):
        kx = k[..., 0]
        ky = k[..., 1]

        real_part = A_real * np.exp(-0.5 * ((kx * sigma_x_real) ** 2 + (ky * sigma_y_real) ** 2)) * np.exp(1j * (kx * x_center_real + ky * y_center_real))

        imag_part = A_imag * np.exp(-0.5 * ((kx * sigma_x_imag) ** 2 + (ky * sigma_y_imag) ** 2)) * np.exp(1j * (kx * x_center_imag + ky * y_center_imag))

        return real_part + 1j * imag_part

    x, k, f_hat = FFT_2D(gaussian, L, dx)

    assert np.allclose(f_hat, ft_of_gaussian(k))
