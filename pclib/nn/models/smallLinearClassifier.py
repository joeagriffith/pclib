from pclib.nn.layers import PCLinearUniweighted
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallLinearClassifier(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int

    def __init__(self, input_size, output_size, hidden_sizes = [] , nu=1.0, mu=1.0, eta=0.1, relu_errs=True, steps=5, bias=False, device=torch.device('cpu'), dtype=None):
        factory_kwargs = {'bias': bias, 'device': device, 'dtype': dtype}
        super(SmallLinearClassifier, self).__init__()

        self.in_features = input_size
        self.out_features = output_size


        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(PCLinearUniweighted(prev_size, size, nu=nu, mu=mu, eta=eta, relu_errs=relu_errs, **factory_kwargs))
            prev_size = size
        layers.append(PCLinearUniweighted(prev_size, output_size, nu=nu, mu=mu, eta=eta, relu_errs=relu_errs, **factory_kwargs))

        self.layers = nn.ModuleList(layers)
        self.steps = steps
        self.device = device

    def step(self, x, R, E):
        for i, layer in enumerate(self.layers):
            td_err = E[i+1] if i+1 < len(E) else None
            R[i], E[i] = layer(x, R[i], td_err)
            R[i] = F.tanh(R[i])
            x = R[i]
        return R, E

    def init_vars(self, batch_size: int):
        R = []
        E = []
        for layer in self.layers:
            r, e = layer.init_vars(batch_size, bias=True)
            R.append(r)
            E.append(e)
        return R, E

    def forward(self, x, y=None, steps=None, full_data=False):
        assert len(x.shape) == 2, f"Input must be 2D, got {len(x.shape)}D"
        if steps is None:
            steps = self.steps

        R, E = self.init_vars(x.shape[0])
        if y is not None:
            assert y.shape == R[-1].shape
            R[-1] = y

        for _ in range(steps):
            R, E = self.step(x, R, E)
            if y is not None:
                R[-1] = y

        if full_data:
            return R, E
        else:
            return R[-1]




