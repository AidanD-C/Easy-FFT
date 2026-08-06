import numpy as np
from numpy.typing import NDArray
from typing import Callable
from decimal import Decimal


def a_mod_b_is_zero(a: float, b: float) -> bool:
    """
    Returns True if a / b is mathematically an integer, False otherwise.

    Uses Python's Decimal arithmetic to avoid IEEE-754 floating-point
    rounding errors.

    Parameters:
    - a: The dividend.
    - b: The divisor.

    Preconditions:
    - a and b must be faithfully represented by their finite decimal expansion in python.
      For example, if a=1.0 and b=1/3, this function will return False because the only
      faithful decimal expansion of 1/3 has infinitely many digits (0.333...).
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

    Raises:
    - ValueError: If L or dx is non-positive, or L is not an integer multiple of dx.
    """

    if L <= 0:
        raise ValueError("L must be positive.")

    if dx <= 0:
        raise ValueError("dx must be positive.")

    if not a_mod_b_is_zero(L, dx):
        raise ValueError("L must be an integer multiple of dx.")

    N = round(L / dx)

    if N % 2 == 0:
        n = np.arange(N)
        x = (n - N / 2) * dx
    else:
        n = np.arange(N)
        x = (n - (N - 1) / 2) * dx

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

    Raises:
    - ValueError: If L or dk is non-positive, or L is not an integer multiple of dk.
    """

    if L <= 0:
        raise ValueError("L must be positive.")

    if dk <= 0:
        raise ValueError("dk must be positive.")

    if not a_mod_b_is_zero(L, dk):
        raise ValueError("L must be an integer multiple of dk.")

    N = round(L / dk)

    if N % 2 == 0:
        n = np.arange(N)
        k = (n - N / 2) * dk
    else:
        n = np.arange(N)
        k = (n - (N - 1) / 2) * dk

    f_hat = np.array([function(ki) for ki in k], dtype=complex)

    f = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(f_hat), norm="forward")) * dk / (2 * np.pi)

    x = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dk))

    return (x, k, f)


def vectorized_FFT_2D(function: Callable[[NDArray, NDArray], NDArray], box_width: float, dx: float) -> tuple[NDArray, NDArray, NDArray]:
    r"""
    Returns a tuple of arrays (x, k, f_hat). Here is a description of each array in this tuple:
    - x: 2-dimensional array of points in x-space at which function was sampled to perform the fourier transform.
    - k: 2-dimensional array of wave vectors in k-space at which the fourier transform of function was evaluated.
    - f_hat: 2-dimensional array of the fourier transform of function evaluated at the wave vectors k.

    NOTE!!
    The meshgrids x,k,f_hat are returned in reverse index order (indexing='ji'). This ordering is suitable for plotting in matplotlib.
    For example, to obtain the first sample along the x axis from the origin, one would write x[0,1].

    Arguments:
    - function: A callable function which takes two meshgrids of shape (N,N) of points and returns a meshgrid of complex numbers or a subclass of a complex number (float, int).
                This is the function which will be fourier transformed. function must be vectorized meaning it must take two arrays and output an array.
    - box_width: Sampling window width in x-space. function will be sampled in square of side length box_width centered at the origin.
    - dx: Sample spacing in x-space. Within the sampling window, samples will be taken at intervals of lenght dx in both the x and y directions.

    Preconditions:
    - box_width and dx must be positive.
    - box_width must be an integer multiple of dx.

    Raises:
    - ValueError: If box_width or dx is non-positive, or box_width is not an integer multiple of dx.
    """

    if box_width <= 0:
        raise ValueError("box_width must be positive.")

    if dx <= 0:
        raise ValueError("dx must be positive.")

    if not a_mod_b_is_zero(box_width, dx):
        raise ValueError("box_width must be an integer multiple of dx.")

    N = round(box_width / dx)

    if N % 2 == 0:
        n = np.arange(N)
        x_1d = (n - N / 2) * dx
    else:
        n = np.arange(N)
        x_1d = (n - (N - 1) / 2) * dx

    X, Y = np.meshgrid(x_1d, x_1d)

    dA = dx**2

    f = function(X, Y)

    f_hat = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f), norm="backward")) * dA

    k1 = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))

    K1, K2 = np.meshgrid(k1, k1)
    k = np.stack((K1, K2), axis=-1)

    x = np.stack((X, Y), axis=-1)

    return (x, k, f_hat)


def vectorized_IFFT_2D(function: Callable[[NDArray, NDArray], NDArray], box_width: float, dk: float) -> tuple[NDArray, NDArray, NDArray]:
    r"""
    Returns a tuple of arrays (x, k, f). Here is a description of each array in this tuple:
    - x: 2-dimensional array of points in x-space at which the inverse fourier transform of function was evaluated.
    - k: 2-dimensional array of wave vectors in k-space at which function was sampled to perform the inverse fourier transform.
    - f: 2-dimensional array of the inverse fourier transform of function evaluated at the points x.

    NOTE!!
    The meshgrids x,k,f are returned in reverse index order (indexing='ji'). This ordering is suitable for plotting in matplotlib.
    For example, to obtain the first sample along the x axis from the origin, one would write x[0,1].

    Arguments:
    - function: A callable function which takes two meshgrids of shape (N,N) of points and returns a meshgrid of complex numbers or a subclass of a complex number (float, int).
                This is the function which will be inverse fourier transformed. function must be vectorized meaning it must take two arrays and output an array.
    - box_width: Sampling window width in k-space. function will be sampled in a square of side length box_width centered at the origin.
    - dk: Sample spacing in k-space. Within the sampling window, samples will be taken at intervals of lenght dk in both the k_1 and k_2 directions.

    Preconditions:
    - box_width and dk must be positive.
    - box_width must be an integer multiple of dk.

    Raises:
    - ValueError: If box_width or dk is non-positive, or box_width is not an integer multiple of dk.
    """

    if box_width <= 0:
        raise ValueError("box_width must be positive.")

    if dk <= 0:
        raise ValueError("dk must be positive.")

    if not a_mod_b_is_zero(box_width, dk):
        raise ValueError("box_width must be an integer multiple of dk.")

    N = round(box_width / dk)

    if N % 2 == 0:
        n = np.arange(N)
        k1_1d = (n - N / 2) * dk
    else:
        n = np.arange(N)
        k1_1d = (n - (N - 1) / 2) * dk

    K1, K2 = np.meshgrid(k1_1d, k1_1d)

    dA_k = dk**2

    f_hat = function(K1, K2)

    f = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(f_hat), norm="forward")) * dA_k / ((2 * np.pi) ** 2)

    x_1d = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dk))

    X, Y = np.meshgrid(x_1d, x_1d)
    x = np.stack((X, Y), axis=-1)

    k = np.stack((K1, K2), axis=-1)

    return (x, k, f)


def vectorized_FFT_ND(function: Callable[..., NDArray], ndim: int, box_width: float, dx: float) -> tuple[NDArray, NDArray, NDArray]:
    """
    Returns a tuple of arrays (x, k, f_hat). Here is a description of each array in this tuple:
    - x: Array with shape (N,...,(ndim times),...,N,3) of points in x-space at which function was sampled to perform the fourier transform.
    - k: Array with shape (N,...,(ndim times),...,N,3) of wave vectors in k-space at which the fourier transform of function was evaluated.
    - f_hat: Array with shape (N,...,(ndim times),...,N) of the fourier transform of function evaluated at the wave vectors k.

    NOTE!!
    The meshgrids x,k,f_hat are returned in reverse index order (indexing='ji'). This ordering is suitable for plotting in matplotlib.
    For example, to obtain the first sample along the x axis from the origin, one would write x[0,...,0,1].

    Arguments:
    - function: A callable function which takes ndim meshgrids of shape (N,...,(ndim times),...,N) of points and returns a meshgrid of complex numbers or a subclass of a complex number (float, int).
                This is the function which will be fourier transformed. function must be vectorized meaning it must take ndim arrays and output an array.
    - box_width: Sampling window width in x-space. function will be sampled in a hypercube of side length box_width centered at the origin.
    - dx: Sample spacing in x-space. Within the sampling window, samples will be taken at intervals of length dx along each axis (x,y,z,...).

    Preconditions:
    - box_width and dx must be positive.
    - box_width must be an integer multiple of dx.

    Raises:
    - ValueError: If ndim < 1, box_width or dx is non-positive, or box_width is not an integer multiple of dx.
    """

    if ndim < 1:
        raise ValueError("ndim must be at least 1.")

    if box_width <= 0:
        raise ValueError("box_width must be positive.")

    if dx <= 0:
        raise ValueError("dx must be positive.")

    if not a_mod_b_is_zero(box_width, dx):
        raise ValueError("box_width must be an integer multiple of dx.")

    N = round(box_width / dx)

    if N % 2 == 0:
        n = np.arange(N)
        x_1d = (n - N / 2) * dx
    else:
        n = np.arange(N)
        x_1d = (n - (N - 1) / 2) * dx

    grids = np.meshgrid(*([x_1d] * ndim))

    f = function(*grids)

    dV = dx**ndim

    f_hat = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(f), norm="backward")) * dV

    k_1d = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    k_grids = np.meshgrid(*([k_1d] * ndim))

    x = np.stack(grids, axis=-1)
    k = np.stack(k_grids, axis=-1)

    return (x, k, f_hat)


def vectorized_IFFT_ND(function: Callable[..., NDArray], ndim: int, box_width: float, dk: float) -> tuple[NDArray, NDArray, NDArray]:
    """
    Returns a tuple of arrays (x, k, f). Here is a description of each array in this tuple:
    - x: ndim-dimensional array of points in x-space at which the inverse fourier transform of function was evaluated.
    - k: ndim-dimensional array of wave vectors in k-space at which function was sampled to perform the inverse fourier transform.
    - f: ndim-dimensional array of the inverse fourier transform of function evaluated at the points x.

    NOTE!!
    The meshgrids x,k,f are returned in reverse index order (indexing='ji'). This ordering is suitable for plotting in matplotlib.
    For example, to obtain the first sample along the x axis from the origin, one would write x[0,...,0,1].

    Arguments:
    - function: A callable function which takes ndim meshgrids of shape (N,)*ndim of points and returns a meshgrid of complex numbers or a subclass of a complex number (float, int).
                This is the function which will be inverse fourier transformed. function must be vectorized meaning it must take ndim arrays and output an array.
    - box_width: Sampling window width in k-space. function will be sampled in a hypercube of side length box_width centered at the origin.
    - dk: Sample spacing in k-space. Within the sampling window, samples will be taken at intervals of lenght dk along all axes (k1,k2,k3,...).

    Preconditions:
    - box_width and dk must be positive.
    - box_width must be an integer multiple of dk.

    Raises:
    - ValueError: If ndim < 1, box_width or dk is non-positive, or box_width is not an integer multiple of dk.
    """

    if ndim < 1:
        raise ValueError("ndim must be at least 1.")

    if box_width <= 0:
        raise ValueError("box_width must be positive.")

    if dk <= 0:
        raise ValueError("dk must be positive.")

    if not a_mod_b_is_zero(box_width, dk):
        raise ValueError("box_width must be an integer multiple of dk.")

    N = round(box_width / dk)

    if N % 2 == 0:
        n = np.arange(N)
        k1_1d = (n - N / 2) * dk
    else:
        n = np.arange(N)
        k1_1d = (n - (N - 1) / 2) * dk

    k_grids = np.meshgrid(*([k1_1d] * ndim))

    f_hat = function(*k_grids)

    dV_k = dk**ndim

    f = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(f_hat), norm="forward")) * dV_k / ((2 * np.pi) ** ndim)

    x_1d = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dk))
    x_grids = np.meshgrid(*([x_1d] * ndim))

    x = np.stack(x_grids, axis=-1)
    k = np.stack(k_grids, axis=-1)

    return (x, k, f)
