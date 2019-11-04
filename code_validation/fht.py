import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    psnr = 10.0 * np.log10(1.0 / mse)
    return psnr

# -------------------- FFT Based Fast Hartley Transform ------------- #
def npFHT2D(x):
    X = np.fft.fft2(x) / np.sqrt(x.shape[0] * x.shape[1])
    return np.real(X) - np.imag(X)

# ------------------- Discreate Hartley Transform ------------------- #
def cas(x):
    return np.cos(x) + np.sin(x)
    # return np.sqrt(2) * np.cos(x - np.pi / 4)

def _DHT2D(x, axis=-1):
    N = x.shape[axis]
    n = np.arange(N)
    k = n[:,None]
    E = cas(2 * np.pi * k * n / N) / np.sqrt(N)
    if axis == 0:
        X = np.dot(E, x)
    else:
        X = np.dot(x, E)
    return X

def DHT2D(x):
    # apply 1D fourier transform to columns
    X = _DHT2D(x, axis=1)
    # apply 1D fourier transform to rows
    X = _DHT2D(X, axis=0)
    return X

# ------------------- Frequency shift ------------------- #
def freqshift(x, inplace=False):
    shift = [d // 2 for d in x.shape]
    x = np.concatenate((x[shift[0]:], x[:shift[0]]), axis=0)
    x = np.concatenate((x[:,shift[1]:], x[:,:shift[1]]), axis=1)
    return x

imagepath = os.path.expanduser("~/Desktop/mountains2.jpg")
m = np.asarray(Image.open(imagepath).convert('L').resize((240, 180)), dtype=np.float64)
m = m / 255.0
print(m.max(), m.min())
M = DHT2D(m) #/ np.sqrt(m.shape[0] * m.shape[1])
M = freqshift(M)
M_ = npFHT2D(m) #/ np.sqrt(m.shape[0] * m.shape[1])
M_ = np.fft.fftshift(M_)
print(np.mean((M - M_)**2))
print(M.max(), M.min())
print(M_.max(), M_.min())
M = freqshift(M)
m_ = DHT2D(M) #* np.sqrt(m.shape[0] * m.shape[1])
print(m_.max(), m_.min())

print(psnr(m, m_))

plt.figure(figsize=(6,6))
plt.imshow(m_, cmap='gray')
plt.show()