import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Function

class FHT2D(nn.Module):
    def __init__(self, image_size):
        super(FHT2D, self).__init__()
        M, N = image_size
        device = torch.device(("cpu","cuda")[torch.cuda.is_available()])

        # set up column matrices
        n = torch.arange(N, dtype=torch.float32)
        k = n[:,None]
        E_real_1 =  torch.cos(2.0 * np.pi * k * n / N).to(device)
        E_imag_1 = -torch.sin(2.0 * np.pi * k * n / N).to(device)

        # set up row matrices
        m = torch.arange(M, dtype=torch.float32)
        k = m[:,None]
        E_real_2 =  torch.cos(2.0 * np.pi * k * m / M).to(device)
        E_imag_2 = -torch.sin(2.0 * np.pi * k * m / M).to(device)

        self.transform_matrices = {"col" : (E_real_1, E_imag_1),
                                   "row" : (E_real_2, E_imag_2)}

        # self.fht2d = FastHartleyTransform2D().apply

    def dft2d(self, x, transform_matrices=None):
        if not transform_matrices:
            M, N = x.shape[-2], x.shape[-1]

            # set up column matrices
            n = torch.arange(N, dtype=torch.float32)
            k = n[:,None]
            E_real_1 =  torch.cos(2.0 * np.pi * k * n / N)
            E_imag_1 = -torch.sin(2.0 * np.pi * k * n / N)

            # set up row matrices
            m = torch.arange(M, dtype=torch.float32)
            k = m[:,None]
            E_real_2 =  torch.cos(2.0 * np.pi * k * m / M)
            E_imag_2 = -torch.sin(2.0 * np.pi * k * m / M)
        else:
            E_real_1, E_imag_1 = transform_matrices["col"]
            E_real_2, E_imag_2 = transform_matrices["row"]

        # calculate the real signal
        X_real  = torch.matmul(torch.matmul(x, E_real_1).permute(0, 1, 3, 2),
                               E_real_2.t()).permute(0, 1, 3, 2)
        X_real -= torch.matmul(torch.matmul(x, E_imag_1).permute(0, 1, 3, 2),
                               E_imag_2.t()).permute(0, 1, 3, 2)

        # calculate the imaginary signal
        X_imag  = torch.matmul(torch.matmul(x, E_imag_1).permute(0, 1, 3, 2),
                               E_real_2.t()).permute(0, 1, 3, 2)
        X_imag += torch.matmul(torch.matmul(x, E_real_1).permute(0, 1, 3, 2),
                               E_imag_2.t()).permute(0, 1, 3, 2)

        return X_real, X_imag

    def forward(self, x):
        # return self.fht2d(x, self.transform_matrices)
        X_real, X_imag = self.dft2d(x, self.transform_matrices)
        X = X_real - X_imag
        # normalize
        X_reshape = X.reshape(-1, x.size(1), x.size(2) * x.size(3))
        return X / X_reshape.max(dim=2)[0][:,:,None,None]

# class FastHartleyTransform2D(Function):
#     @staticmethod
#     def dft2d(x, transform_matrices=None):
#         if not transform_matrices:
#             M, N = x.shape[-2], x.shape[-1]

#             # set up column matrices
#             n = torch.arange(N, dtype=torch.float32)
#             k = n[:,None]
#             E_real_1 =  torch.cos(2.0 * np.pi * k * n / N)
#             E_imag_1 = -torch.sin(2.0 * np.pi * k * n / N)

#             # set up row matrices
#             m = torch.arange(M, dtype=torch.float32)
#             k = m[:,None]
#             E_real_2 =  torch.cos(2.0 * np.pi * k * m / M)
#             E_imag_2 = -torch.sin(2.0 * np.pi * k * m / M)
#         else:
#             E_real_1, E_imag_1 = transform_matrices["col"]
#             E_real_2, E_imag_2 = transform_matrices["row"]

#         # calculate the real signal
#         X_real  = torch.matmul(torch.matmul(x, E_real_1).permute(0, 1, 3, 2),
#                                E_real_2.t()).permute(0, 1, 3, 2)
#         X_real -= torch.matmul(torch.matmul(x, E_imag_1).permute(0, 1, 3, 2),
#                                E_imag_2.t()).permute(0, 1, 3, 2)

#         # calculate the imaginary signal
#         X_imag  = torch.matmul(torch.matmul(x, E_imag_1).permute(0, 1, 3, 2),
#                                E_real_2.t()).permute(0, 1, 3, 2)
#         X_imag += torch.matmul(torch.matmul(x, E_real_1).permute(0, 1, 3, 2),
#                                E_imag_2.t()).permute(0, 1, 3, 2)

#         return X_real, X_imag

#     @staticmethod
#     def forward(ctx, x, transform_matrices=None):
#         if len(x.shape) != 4:
#             raise ValueError("Input must be 4 Dimensional (batch_size, channels, height, width")

#         E_real_1, E_imag_1 = transform_matrices["col"]
#         E_real_2, E_imag_2 = transform_matrices["row"]
#         ctx.save_for_backward(E_real_1, E_imag_1, E_real_2, E_imag_2)
#         X_real, X_imag = FastHartleyTransform2D.dft2d(x, transform_matrices)
#         X = X_real - X_imag

#         # normalize
#         X_reshape = X.reshape(-1, x.size(1), x.size(2) * x.size(3))
#         return X / X_reshape.max(dim=2)[0][:,:,None,None]
#         # return X / np.sqrt(X.size(2) * X.size(3))

#     @staticmethod
#     def backward(ctx, grad_output):
#         E_real_1, E_imag_1, E_real_2, E_imag_2 = ctx.saved_tensors
#         transform_matrices = {"col" : (E_real_1, E_imag_1),
#                               "row" : (E_real_2, E_imag_2)}
#         X_real, X_imag = FastHartleyTransform2D.dft2d(grad_output, transform_matrices)
#         X = X_real - X_imag

#         # normalize
#         X_reshape = X.reshape(-1, grad_output.size(1), grad_output.size(2) * grad_output.size(3))
#         return X / X_reshape.max(dim=2)[0][:,:,None,None], None


if __name__=="__main__":
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    fht2d = FHT2D((480, 720))
    imagepath = os.path.expanduser("~/Desktop/mountains2.jpg")
    m1 = np.asarray(Image.open(imagepath).resize((720, 480)), dtype=np.float32)
    m1 = m1 / 255.0
    m1 = np.transpose(m1, (2, 0, 1))[None,:]
    m = torch.from_numpy(m1)
    print("Orig Spatial\t max: ", m.max(), " min: ", m.min())
    M = fht2d(m)
    print("Fourier\t\t max: ", M.max(), " min: ", M.min())
    m = fht2d(M)
    print("New Spatial\t max: ", m.max(), " min: ", m.min())
    img = np.transpose(np.asarray(m[0]), (1, 2, 0))
    img = (img - img.min()) / (img.max() - img.min())
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.show()
