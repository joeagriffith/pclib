from pclib.nn.layers import Linear
import torch
import torch.nn as nn
import torch.nn.functional as F

class PCLinearClassifier(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    num_classes: int

    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=5, bias=True, symmetric=True, gamma=0.1, device=torch.device('cpu'), dtype=None):
        factory_kwargs = {'bias': bias, 'symmetric': symmetric, 'device': device, 'dtype': dtype}
        super(PCLinearClassifier, self).__init__()

        self.in_features = input_size
        self.num_classes = num_classes
        self.gamma = gamma

        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(Linear(prev_size, size, gamma=gamma, **factory_kwargs))
            prev_size = size
        layers.append(Linear(prev_size, num_classes, gamma=gamma, **factory_kwargs))

        self.layers = nn.ModuleList(layers)
        self.steps = steps
        self.device = device

    def step(self, x, state, y=None):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                td_error = state[i+1][1]
            else:
                td_error = None
            state[i] = layer(x, state[i], td_error)
            x = state[i][0]
        if y is not None:
            y_norm = y / torch.norm(y, dim=1, keepdim=True)
            y_scaled = y_norm * torch.norm(state[-1][0], dim=1, keepdim=True)
            state[-1][0] = y_scaled
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
        if steps is None:
            steps = self.steps

        if state is None:
            state = self.init_state(x.shape[0])

        for _ in range(steps):
            state = self.step(x, state, y)

            
        out = state[-1][0]
            
        return out, state
            