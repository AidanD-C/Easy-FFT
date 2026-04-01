import numpy as np
from numpy.typing import NDArray
from typing import Callable
from decimal import Decimal


def a_mod_b_is_zero(a: float, b: float) -> bool:
    """
    Returns True if a is an integer multiple of b and False otherwise.
    """
    return Decimal(str(a)) % Decimal(str(b)) == 0


def FFT_1D(function: Callable[[float], complex], L: float, dx: float) -> tuple[NDArray, NDArray, NDArray]:
    r"""
    Returns a tuple of arrays (x, k, f_hat). Here is a description of each array in this tuple:
    - x: Points in x-space at which function was sampled to perform the fourier transform. In latex: $x[n] = n(dx)-L/2$ (for L/dx even... odd is similar)
    - k: Frequencies in k-space at which the fourier transform of function was evaluated. In latex: $k[n]=\frac{2\pi}{L} \left(n-\frac{L}{2(dx)}\right)$ (for L/dx even... odd is similar)
    - f_hat: Fourier transform of function evaluated at the frequencies k. In latex: $\hat{f}[n]= \sum^{N-1}_{m=0}f[m]e^{- i k[n]x[m]}dx$ where N=L/dx is the length of x,k,f_hat.

    Arguments:
    - function: A callable function which takes a float and returns a complex number or a subclass of a complex number (float, int). This is the function which will be fourier transformed.
                function does not need to be vectorized. 
    - L: Sampling window length in x-space. function will be sampled in an even interval of length L centered at the origin.
    - dx: Sample spacing in x-space. Within the sampling window of length L, samples will be taken at intervals of lenght dx.

    Preconditions:
    - L and dx must be positive.
    - L must be an integer multiple of dx.
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

    f = np.array([function(xi) for xi in x], dtype=complex)

    f_hat = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(f), norm="backward")) * dx

    k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))

    return (x, k, f_hat)


def IFFT_1D(function: Callable[[float], complex], L: float, dk: float) -> tuple[NDArray, NDArray, NDArray]:
    r"""
    Returns a tuple of arrays (x, k, f). Here is a description of each array in this tuple:
    - x: Points in x-space at which the inverse fourier transform of function was evaluated. In latex: $x[n]=\frac{2\pi}{L} \left(n-\frac{L}{2(dk)}\right)$ (for L/dk even... odd is similar)
    - k: Frequencies in k-space at which function was sampled to perform the inverse fourier transform. In latex: $k[n] = n(dk)-L/2$ (for L/dk even... odd is similar)
    - f: Inverse fourier transform of function evaluated at the points x. In latex: $f[n]= \frac{1}{2\pi}\sum^{N-1}_{m=0}\hat{f}[m]e^{i k[m]x[n]}dk$ where N=L/dk is the length of x,k,f.

    Arguments:
    - function: A callable function which takes a float and returns a complex number or a subclass of a complex number (float, int). This is the function which will be inverse fourier transformed.
                function does not need to be vectorized. 
    - L: Sampling window length in k-space. function will be sampled in an even interval of length L centered at the origin.
    - dk: Sample spacing in k-space. Within the sampling window of length L, samples will be taken at intervals of lenght dk.

    Preconditions:
    - L and dk must be positive.
    - L must be an integer multiple of dk.
    """

    if L == 0:
        raise ValueError("L must be positive.")

    if dk <= 0:
        raise ValueError("dk must be positive.")

    if not a_mod_b_is_zero(L, dk):
        raise ValueError("L must be an integer multiple of dk.")

    N = int(L / dk)

    if N % 2 == 0:
        k = np.linspace(-L / 2, L / 2, N, endpoint=False, dtype=float)
    else:
        k = np.linspace(-L / 2, L / 2, N, endpoint=True, dtype=float)

    f_hat = np.array([function(ki) for ki in k], dtype=complex)

    f = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(f_hat), norm="forward")) * dk / (2 * np.pi)

    x = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dk))

    return (x, k, f)


def vectorized_FFT_2D(function: Callable[[NDArray, NDArray], NDArray], box_width: float, dx: int) -> tuple[NDArray, NDArray, NDArray]:
    r"""
    Returns a tuple of arrays (x, k, f_hat). Here is a description of each array in this tuple:
    - x: 2-dimensional array of points in x-space at which function was sampled to perform the fourier transform.
    - k: 2-dimensional array of frequencies in k-space at which the fourier transform of function was evaluated.
    - f_hat: 2-dimensional array of the fourier transform of function evaluated at the frequencies k.

    Arguments:
    - function: A callable function which takes two arrays of x and y values and returns an array of complex numbers or a subclass of a complex number (float, int). This is the function which will be fourier transformed.
                function must be vectorized meaning it must take two arrays and output an array.
    - box_width: Sampling window width in x-space. function will be sampled in square of side length box_width centered at the origin.
    - dx: Sample spacing in x-space. Within the sampling window, samples will be taken at intervals of lenght dx in both the x and y directions.

    Preconditions:
    - box_width and dx must be positive.
    - box_width must be an integer multiple of dx.
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

    K1, K2 = np.meshgrid(k1, k1)
    k = np.stack((K1, K2), axis=-1)

    x = np.stack((X, Y), axis=-1)

    return (x, k, f_hat)


def vectorized_IFFT_2D(function: Callable[[NDArray, NDArray], NDArray], box_width: float, dk: int) -> tuple[NDArray, NDArray, NDArray]:
    r"""
    Returns a tuple of arrays (x, k, f). Here is a description of each array in this tuple:
    - x: 2-dimensional array of points in x-space at which the inverse fourier transform of function was evaluated.
    - k: 2-dimensional array of frequencies in k-space at which function was sampled to perform the inverse fourier transform.
    - f: 2-dimensional array of the inverse fourier transform of function evaluated at the points x.

    Arguments:
    - function: A callable function which takes two arrays of k_1 and k_2 values and returns an array of complex numbers or a subclass of a complex number (float, int). This is the function which will be inverse fourier transformed.
                function must be vectorized meaning it must take two arrays and output an array.
    - box_width: Sampling window width in k-space. function will be sampled in square of side length box_width centered at the origin.
    - dk: Sample spacing in k-space. Within the sampling window, samples will be taken at intervals of lenght dk in both the k_1 and k_2 directions.

    Preconditions:
    - box_width and dk must be positive.
    - box_width must be an integer multiple of dk.
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

    X, Y = np.meshgrid(x_temp, x_temp)
    x = np.stack((X, Y), axis=-1)

    k = np.stack((K1, K2), axis=-1)

    return (x, k, f)


def vectorized_FFT_ND() -> tuple[NDArray, NDArray, NDArray]:
    """TBD"""
    pass


def vectorized_IFFT_ND() -> tuple[NDArray, NDArray, NDArray]:
    """TBD"""
    pass
