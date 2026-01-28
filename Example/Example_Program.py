import numpy as np
from Easy_FFT import FFT_1D

def func(x):
    return np.exp(-(x**2))

x, k, f_hat = FFT_1D(function=func, L=100, dx=0.01)

# Plotting:
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2)

f = func(x)

ax[0].plot(x, np.real(f), label=r"$Re(f)$")
ax[0].plot(x, np.imag(f), label=r"$Im(f)$")
ax[0].set_title("f (Before FT):")
ax[0].set_xlabel("x")
ax[0].set_ylabel("f(x)")
ax[0].set_xlim(-10, 10)

ax[1].plot(k, np.real(f_hat), label=r"$Re(\hat{f})$")
ax[1].plot(k, np.imag(f_hat), label=r"$Im(\hat{f})$")
ax[1].set_title(r"$\hat{f}\;(After FT):$")
ax[1].set_xlabel("k")
ax[1].set_ylabel(r"$\hat{f}(x)$")
ax[1].set_xlim(-10, 10)

ax[0].legend()
ax[1].legend()

fig.tight_layout(pad=3.0)

plt.show()