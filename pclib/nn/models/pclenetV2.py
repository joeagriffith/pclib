import torch
import torch.nn as nn
from pclib.nn.layers import PCConvV2, PCLinear
from functools import reduce

class PCLeNetV2(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_shape: int
    num_classes: int

    def __init__(self, input_shape, num_classes, nu=1.0, mu=1.0, eta=0.1, steps=5, relu_errs=True):
        super(PCLeNetV2, self).__init__()

        self.in_shape = input_shape
        self.num_classes = num_classes


        shape = [
            input_shape,
            (6, input_shape[1]//2, input_shape[2]//2),
            (16, ((input_shape[1]//2)-4)//2, ((input_shape[2]//2)-4)//2),
            (120, ((input_shape[1]//2)-4)//2-4, ((input_shape[2]//2)-4)//2-4),
        ]

        self.steps = steps
        self.layers = nn.ModuleList([
            PCConvV2(shape[0], shape[1], (5,5), nu=nu, mu=mu, eta=eta, maxpool=2, padding=2, relu_errs=relu_errs),
            PCConvV2(shape[1], shape[2], (5,5), nu=nu, mu=mu, eta=eta, maxpool=2, relu_errs=relu_errs),
            PCConvV2(shape[2], shape[3], (5,5), nu=nu, mu=mu, eta=eta, relu_errs=relu_errs),

            PCLinear(reduce(lambda x, y: x * y, shape[3]), 84, nu, mu, eta, relu_errs=relu_errs),
            PCLinear(84, num_classes, nu, mu, eta, relu_errs=relu_errs),
        ])



    def step(self, x, R, E, y=None):
        if y is not None:
            assert y.shape == R[-1].shape
            R[-1] = y
        for i, layer in enumerate(self.layers):
            print(f"Layer {i}, R shape: {R[i].shape}, E shape: {E[i].shape}")
            if E[i].dim() == 2 and x.dim() == 4:
                x = x.view(x.shape[0], -1)
            td_err = E[i+1] if i+1 < len(E) else None
            R[i], E[i] = layer(x, R[i], td_err)
            R[i] = torch.tanh(R[i])
            x = R[i]
        return R, E

    def init_vars(self, batch_size: int):
        R = []
        E = []
        for layer in self.layers:
            r, e = layer.init_vars(batch_size)
            # r, e = layer.init_vars(batch_size, bias=True)
            R.append(r)
            E.append(e)
        return R, E

    def to(self, device):
        self.device = device
        for layer in self.layers:
            layer.to(device)
        return self


    def forward(self, x, y=None, steps=None, full_data=False):
        if steps is None:
            steps = self.steps

        R, E = self.init_vars(x.shape[0])

        for _ in range(steps):
            R, E = self.step(x, R, E, y)

        if full_data:
            return R[-1], R, E
        else:
            return R[-1]