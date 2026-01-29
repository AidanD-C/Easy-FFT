import numpy as np
from numpy.typing import NDArray
from typing import Callable
from decimal import Decimal


def a_mod_b_is_zero(a, b):
    return Decimal(str(a)) % Decimal(str(b)) == 0


def FFT_1D(function: Callable[[float], complex], L: float, dx: float) -> tuple[NDArray, NDArray, NDArray]:
    """
    Docstring TBD
    """

    if L <= 0:
        raise ValueError("L must be positive.")

    if dx <= 0:
        raise ValueError("dx must be positive.")

    if not a_mod_b_is_zero(L, dx):
        raise ValueError("L must be an integer multiple of dx.")

    N = int(L / dx)

    if N % 2 == 0:
        x = np.linspace(-L / 2, L / 2, N, endpoint=False, dtype=float)
    else:
        x = np.linspace(-L / 2, L / 2, N, endpoint=True, dtype=float)

    f = np.array([function(float(xi)) for xi in x], dtype=complex)

    f_hat = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(f), norm="backward")) * dx

    k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))

    return (x, k, f_hat)


def IFFT_1D(function: Callable[[float], complex], L: float, dk: float) -> tuple[NDArray, NDArray, NDArray]:
    """
    Docstring TBD
    """

    if L == 0:
        raise ValueError("L must be positive.")

    if dk <= 0:
        raise ValueError("dk must be positive.")

    if not a_mod_b_is_zero(L, dk):
        raise ValueError("L must be an integer multiple of dk.")

    N = int(L / dk)

    if N % 2 == 0:
        k = np.linspace(-L / 2, L / 2, N, endpoint=False)
    else:
        k = np.linspace(-L / 2, L / 2, N, endpoint=True)

    f_hat = np.array([function(float(ki)) for ki in k], dtype=complex)

    f = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(f_hat), norm="forward")) * dk / (2 * np.pi)

    x = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dk))

    return (x, k, f)


def FFT_2D(function: Callable[[NDArray, NDArray], NDArray], box_width: float, dx: float) -> tuple[NDArray, NDArray, NDArray]:
    """
    function must be vectorized meaning it can take numpy arrays as inputs
    """

    if box_width <= 0:
        raise ValueError("box_width must be positive.")

    if dx <= 0:
        raise ValueError("dx must be positive.")

    if not a_mod_b_is_zero(box_width, dx):
        raise ValueError("box_width must be an integer multiple of dx.")

    N = int(box_width / dx)

    if N % 2 == 0:
        x = np.linspace(-box_width / 2, box_width / 2, N, endpoint=False, dtype=float)
    else:
        x = np.linspace(-box_width / 2, box_width / 2, N, endpoint=True, dtype=float)

    X, Y = np.meshgrid(x, x)

    dA = dx**2

    f = function(X, Y)

    f_hat = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f), norm="backward")) * dA

    k1 = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    k2 = k1

    K1, K2 = np.meshgrid(k1, k2)
    k = np.stack((K1, K2), axis=-1)

    x = np.stack((X, Y), axis=-1)

    return (x, k, f_hat)


def IFFT_2D(function: Callable[[NDArray, NDArray], NDArray], box_width: float, dk: float) -> tuple[NDArray, NDArray, NDArray]:
    """
    function must be vectorized meaning it can take numpy arrays as inputs
    """

    if box_width <= 0:
        raise ValueError("box_width must be positive.")

    if dk <= 0:
        raise ValueError("dk must be positive.")

    if not a_mod_b_is_zero(box_width, dk):
        raise ValueError("box_width must be an integer multiple of dk.")

    N = int(box_width / dk)

    if N % 2 == 0:
        k1 = np.linspace(-box_width / 2, box_width / 2, N, endpoint=False, dtype=float)
    else:
        k1 = np.linspace(-box_width / 2, box_width / 2, N, endpoint=True, dtype=float)

    K1, K2 = np.meshgrid(k1, k1)

    dA = dk**2

    f_hat = function(K1, K2)

    f = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(f_hat), norm="forward")) * dA / ((2 * np.pi) ** 2)

    x_temp = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dk))
    y_temp = x_temp

    X, Y = np.meshgrid(x_temp, y_temp)
    x = np.stack((X, Y), axis=-1)

    k = np.stack((K1, K2), axis=-1)

    return (x, k, f)


def vectorized_FFT_ND(dimension: int, function: Callable[[NDArray], complex], box_width: float, dx: float) -> tuple[NDArray, NDArray, NDArray]:
    """
    function must be "vectorized" meaning it can take numpy arrays as inputs
    """
    pass


def vectorized_IFFT_ND(dimension: int, function: Callable[[NDArray], complex], box_width: float, dk: float) -> tuple[NDArray, NDArray, NDArray]:
    pass
