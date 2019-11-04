import torch
import numpy as np

def cas(z):
    return torch.cos(z) + torch.sin(z)

class FHT2D():
    def __init__(self, image_size):
        M, N = image_size
        device = torch.device(("cpu","cuda")[torch.cuda.is_available()])

        # set up column matrices
        n = torch.arange(N, dtype=torch.float32)
        k = n[:,None]
        self.E_col =  cas(2.0 * np.pi * k * n / N).to(device)

        # set up row matrices
        m = torch.arange(M, dtype=torch.float32)
        k = m[:,None]
        self.E_row =  cas(2.0 * np.pi * k * m / M).to(device)

        self.M = M
        self.N = N

    def __call__(self, x, inverse=False):
        H = torch.matmul(torch.matmul(x, self.E_col).permute(0, 1, 3, 2),
                         self.E_row.t()).permute(0, 1, 3, 2)
        if not inverse:
            H = H / (self.M * self.N)
        return H


if __name__=="__main__":
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    fht2d = FHT2D((180, 240))
    imagepath = os.path.expanduser("~/Desktop/mountains2.jpg")
    m1_ = np.asarray(Image.open(imagepath).resize((240, 180)), dtype=np.float32)
    m1_ = m1_ / 255.0
    m1 = np.transpose(m1_, (2, 0, 1))[None,:]
    m = torch.from_numpy(m1)
    print(m.shape)
    M = fht2d(m)
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
