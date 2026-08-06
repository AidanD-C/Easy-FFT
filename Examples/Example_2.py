"""
To run this example, please move this file (Example_Program.py) into the same directory as Easy_FFT.py
"""

import Easy_FFT as EFT
import numpy as np

func = lambda X, Y: np.exp(-(X**2 + Y**2))

x, k, f_hat = EFT.vectorized_FFT_ND(func, 2, 30, dx=0.1)

# Plotting:
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

f = func(x[..., 0], x[..., 1])

extents_x = (np.min(x[..., 1]), np.max(x[..., 1]), np.min(x[..., 0]), np.max(x[..., 0]))
extents_k = (np.min(k[..., 1]), np.max(k[..., 1]), np.min(k[..., 0]), np.max(k[..., 0]))

im0 = ax[0].imshow(f, origin="lower", extent=extents_x)
ax[0].set_title(r"$f$ (Before FT):")
ax[0].set_xlabel(r"$x$")
ax[0].set_ylabel(r"$y$")
ax[0].set_xlim(-10, 10)
ax[0].set_ylim(-10, 10)
divider0 = make_axes_locatable(ax[0])
cax0 = divider0.append_axes("right", size="10%", pad=0.05)
cbar = fig.colorbar(im0, cax=cax0)

im1 = ax[1].imshow(np.real(f_hat), origin="lower", extent=extents_k)
ax[1].set_title(r"$Re(\hat{f})$ (After FT):")
ax[1].set_xlabel(r"$k1$")
ax[1].set_ylabel(r"$k2$")
ax[1].set_xlim(-10, 10)
ax[1].set_ylim(-10, 10)
divider1 = make_axes_locatable(ax[1])
cax1 = divider1.append_axes("right", size="10%", pad=0.05)
cbar = fig.colorbar(im1, cax=cax1)

im2 = ax[2].imshow(np.imag(f_hat), origin="lower", extent=extents_k)
ax[2].set_title(r"$Im(\hat{f})$ (After FT):")
ax[2].set_xlabel(r"$k1$")
ax[2].set_ylabel(r"$k2$")
ax[2].set_xlim(-10, 10)
ax[2].set_ylim(-10, 10)
divider2 = make_axes_locatable(ax[2])
cax2 = divider2.append_axes("right", size="10%", pad=0.05)
cbar = fig.colorbar(im2, cax=cax2)

fig.tight_layout(pad=3.0)

plt.savefig("Example_2_output.png", dpi=100)
