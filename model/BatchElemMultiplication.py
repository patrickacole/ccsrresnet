import torch
import torch.nn as nn

class BatchElemMultiplication(nn.Module):
    def __init__(self, in_shape, out_channels):
        super(BatchElemMultiplication, self).__init__()
        nc, h, w = in_shape
        if out_channels % nc != 0:
            raise ValueError("Please make sure that output channels is divisible by the output channels")
        num_tensors = out_channels // nc
        self.weight = nn.Parameter(data=torch.Tensor(1, num_tensors, nc, h, w), requires_grad=True)
        # self.weight.data.uniform_(-1, 1)
        # self.weight.data.fill_(1.0)
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, x):
        x_expanded = x.unsqueeze(1)
        # force weights to be hermitian
        # y = ((self.weight + self.weight.permute(0, 1, 2, 4, 3)) / 2) * x_expanded
        y = self.weight * x_expanded
        return y.view(y.size(0), y.size(1) * y.size(2), y.size(3), y.size(4))

if __name__=="__main__":
    x = torch.Tensor(4, 1, 256, 256)
    bewm = BatchElemMultiplication((1, 256, 256), 5)
    y = bewm(x)
    print(y.shape)
    for param in bewm.parameters():
        print(param.data.max(), param.data.min())