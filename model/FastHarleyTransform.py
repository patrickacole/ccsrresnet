import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Function

def cas(z):
    return torch.cos(z) + torch.sin(z)
    # return np.sqrt(2) * torch.cos(x - np.pi / 4)

def freqshift(x):
    """
    Assumes x is 4D - (N, 3, H, W)
    """
    shift = [d // 2 for d in x.shape[-2:]]
    x = torch.cat((x[:,:,shift[0]:], x[:,:,:shift[0]]), dim=-2)
    x = torch.cat((x[:,:,:,shift[1]:], x[:,:,:,:shift[1]]), dim=-1)
    return x

class FHT2D(nn.Module):
    def __init__(self, image_size):
        super(FHT2D, self).__init__()
        M, N = image_size
        device = torch.device(("cpu","cuda")[torch.cuda.is_available()])

        # set up column matrices
        n = torch.arange(N, dtype=torch.float32)
        k = n[:,None]
        E_col =  cas(2.0 * np.pi * k * n / N).to(device)

        # set up row matrices
        m = torch.arange(M, dtype=torch.float32)
        k = m[:,None]
        E_row =  cas(2.0 * np.pi * k * m / M).to(device)

        self.transform_matrices = [E_col, E_row]
        self.fht2d = FastHartleyTransform2D().apply

    def forward(self, x, inverse=False):
        return self.fht2d(x, inverse, self.transform_matrices)

class FastHartleyTransform2D(Function):
    @staticmethod
    def dft2d(x, transform_matrices=None):
        if not transform_matrices:
            M, N = x.shape[-2:]
            device = torch.device(("cpu","cuda")[torch.cuda.is_available()])

            # set up column matrices
            n = torch.arange(N, dtype=torch.float32)
            k = n[:,None]
            E_col =  cas(2.0 * np.pi * k * n / N).to(device)

            # set up row matrices
            m = torch.arange(M, dtype=torch.float32)
            k = m[:,None]
            E_row =  cas(2.0 * np.pi * k * m / M).to(device)
        else:
            E_col, E_row = transform_matrices

        X = torch.matmul(torch.matmul(x, E_col).permute(0, 1, 3, 2), E_row.t()).permute(0, 1, 3, 2)
        return X

    @staticmethod
    def forward(ctx, x, inverse=False, transform_matrices=None):
        if len(x.shape) != 4:
            raise ValueError("Input must be 4 Dimensional (batch_size, channels, height, width")

        if inverse:
            x = freqshift(x)
        X = FastHartleyTransform2D.dft2d(x, transform_matrices)

        inverse = torch.tensor([int(inverse)],dtype=torch.int8)
        if transform_matrices:
            E_col, E_row = transform_matrices
            ctx.save_for_backward(E_col, E_row, inverse)
        else:
            ctx.save_for_backward(inverse)

        if inverse[0] == 0:
            X = X / (X.size(2) * X.size(3))
            X = freqshift(X)
        return X

    @staticmethod
    def backward(ctx, grad_output):
        if len(ctx.saved_tensors) > 1:
            E_col, E_row, inverse = ctx.saved_tensors
            transform_matrices = [E_col, E_row]
        else:
            transform_matrices = None
            inverse, = ctx.saved_tensors

        inverse[0] = 1 + -inverse[0]
        if inverse[0]:
            grad_output = freqshift(grad_output)
        X = FastHartleyTransform2D.dft2d(grad_output, transform_matrices)

        if inverse[0] == 0:
            X = X / (X.size(2) * X.size(3))
            X = freqshift(X)
        return X, None, None


if __name__=="__main__":
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    fht2d = FHT2D((480, 720))
    imagepath = os.path.expanduser("~/Desktop/mountains2.jpg")
    m1_ = np.asarray(Image.open(imagepath).resize((720, 480)), dtype=np.float32)
    m1_ = m1_ / 255.0
    m1 = np.transpose(m1_, (2, 0, 1))[None,:]
    m = torch.from_numpy(m1)
    m_numpy = m1.copy()
    print("Orig Spatial\t max: ", m.max(), " min: ", m.min())
    M = fht2d(m)
    M_numpy = np.fft.fftshift(np.fft.fft2(m_numpy)) / (720 * 480)
    M_numpy = np.real(M_numpy) - np.imag(M_numpy)
    print("Fourier\t\t max: ", M.max(), " min: ", M.min())
    print("FourierN\t max: ", M_numpy.max(), " min: ", M_numpy.min())
    print("Fourier Diff: \t", np.sum((np.asarray(M.data) - M_numpy)**2))
    m = fht2d(M, inverse=True)
    m_numpy = np.fft.fft2(np.fft.fftshift(M_numpy))
    m_numpy = np.real(m_numpy) - np.imag(m_numpy)
    print("New Spatial\t max: ", m.max(), " min: ", m.min())
    print("New SpatialN\t max: ", m_numpy.max(), " min: ", m_numpy.min())
    img = np.transpose(np.asarray(m[0]), (1, 2, 0))
    def psnr(img1, img2):
        img1 = np.clip(img1, 0, 1)
        img2 = np.clip(img2, 0, 1)
        mse = np.mean((img1 - img2) ** 2)
        psnr = 10.0 * np.log10(1.0 / mse)
        return psnr
    print("Hartley transformed image with original image psnr ", psnr(img, m1_))

    # plot image
    plt.figure(figsize=(6,6))
    plt.imshow(np.clip(img, 0, 1))
    plt.show()
