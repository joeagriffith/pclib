# from pclib.nn.layers import PrecisionWeightedV2 as Linear
from pclib.nn.layers import Linear
import torch
import torch.nn as nn
import torch.nn.functional as F

class PCLinearClassifier(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    num_classes: int

    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=5, bias=True, symmetric=True, actv_fn=F.relu, actv_mode='Wf(r)', gamma=0.1, beta=1.0, device=torch.device('cpu'), dtype=None):
        factory_kwargs = {'bias': bias, 'symmetric': symmetric, 'device': device, 'dtype': dtype}
        super(PCLinearClassifier, self).__init__()

        self.in_features = input_size
        self.num_classes = num_classes
        self.bias = bias
        self.symmetric = symmetric
        self.gamma = gamma
        self.beta = beta

        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(Linear(prev_size, size, actv_fn=actv_fn, actv_mode=actv_mode, gamma=gamma, beta=beta, **factory_kwargs))
            prev_size = size
        layers.append(Linear(prev_size, num_classes, actv_fn=actv_fn, actv_mode=actv_mode, gamma=gamma, beta=beta, **factory_kwargs))

        self.layers = nn.ModuleList(layers)
        self.steps = steps
        self.device = device

    def step(self, x, state, y=None):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                td_error = state[i+1]['e']
            else:
                td_error = None
            state[i] = layer(x, state[i], td_error)
            x = state[i]['x']
        
        if y is not None:
            state[-1]['x'] = y
        return state

    def init_state(self, batch_size: int, mode='zeros'):
        state = []
        for layer in self.layers:
            state_i = layer.init_state(batch_size, mode)
            state.append(state_i)
        return state

    def to(self, device):
        self.device = device
        for layer in self.layers:
            layer.to(device)
        return self

    def forward(self, x, state=None, y=None, steps=None):
        assert len(x.shape) == 2, f"Input must be 2D, got {len(x.shape)}D"

        if y is not None:
            y_new = torch.ones_like(y) * 0.03
            y_new = y_new + (y * 0.94)

        if steps is None:
            steps = self.steps

        if state is None:
            state = self.init_state(x.shape[0])

        for _ in range(steps):
            state = self.step(x, state, y)
            
        out = state[-1]['x']
            
        return out, state

    def generate(self, y, steps=None, step_size=0.01):
        if steps is None:
            steps = self.steps

        state = self.init_state(y.shape[0])
        x = torch.randn((y.shape[0], self.in_features), device=self.device)
        for _ in range(steps):
            state = self.step(x, state, y)
            x -= step_size * state[0]['e']
            
        return x, state
            