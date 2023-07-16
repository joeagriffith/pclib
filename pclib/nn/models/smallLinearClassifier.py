from pclib.nn.layers import PCLinearUni as PCLinear
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallLinearClassifier(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    num_classes: int

    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=5, bias=False, device=torch.device('cpu'), dtype=None):
        factory_kwargs = {'bias': bias, 'device': device, 'dtype': dtype}
        super(SmallLinearClassifier, self).__init__()

        self.in_features = input_size
        self.num_classes = num_classes

        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(PCLinear(prev_size, size, **factory_kwargs))
            prev_size = size
        layers.append(PCLinear(prev_size, num_classes, **factory_kwargs))

        self.layers = nn.ModuleList(layers)
        self.steps = steps
        self.device = device

    def step(self, x, state, y=None):
        if y is not None:
            assert y.shape == state[-1][0].shape
            state[-1][0] = y
        for i, layer in enumerate(self.layers):
            state[i] = layer(x, state[i])
            x = state[i][0]
        return state

    def init_vars(self, batch_size: int):
        state = []
        for layer in self.layers:
            state_i = layer.init_vars(batch_size)
            state.append(state_i)
        return state

    def to(self, device):
        self.device = device
        for layer in self.layers:
            layer.to(device)
        return self

    def forward(self, x, state=None, y=None, steps=None):
        assert len(x.shape) == 2, f"Input must be 2D, got {len(x.shape)}D"
        if steps is None:
            steps = self.steps

        if state is None:
            state = self.init_vars(x.shape[0])

        for _ in range(steps):
            state = self.step(x, state, y)
            
        out = state[-1][0]
            
        return out, state
            




