# Easy-FFT
### About:
Easy-FFT is a collection of python functions which make it easier for physicists to perform Fourier transforms in Python. Easy-FFT contains functions to perform Fourier transforms in 1D, 2D and hopefully will be capable of N dimensional Fourier transforms soon.

### Workflow:
Here's an example of a basic program which uses Easy-FFT:
```python
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
```
Which outputs:
<p align="center">
  <img src="Example/Example Output.png" alt="Image 1" width="100%">
</p>

### Installation:
To begin the installation, we need to clone this repository onto your computer. First, open your terminal and navigate to a folder where you want to put this repository by entering the following in your terminal: 
```bash
cd <path-to-your-folder>
```
Once in the desired folder, enter the following command into your terminal to clone this repository into that folder:
```bash
git clone https://github.com/AidanD-C/Easy-FFT
```
Next, you might want to set up a python virtual environment in the Easy-FFT folder for dependency control, but this is optional. After that, you will want to download all the necessary dependencies for this repository. These dependencies can be found in the requirements.txt folder. After executing the command above, these next two commands can be used to download the dependencies in requirements.txt using pip:
```bash
cd Easy-FFT

# activate your virtual environment here if you chose to use one.

pip install -r requirements.txt
```

Now that the dependencies are installed, you're all ready to go. You can open the Easy-FFT folder in your IDE of choice and start doing some Fourier transforms.