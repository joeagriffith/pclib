import torch
import torch.nn as nn
import torch.nn.functional as F

class StochasticHebbian(nn.Module):
    __constants__ = ['in_features, out_features']
    in_features: int
    out_features: int
    def __init__(self, in_features, out_features, hidden_features = [], bias=True, actv_fn=F.relu, noise_coeff=1.0, steps=30, device=torch.device('cpu'), dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.actv_fn = actv_fn
        self.noise_coeff = noise_coeff
        self.steps = steps
        self.bias = bias
        self.device = device
        self.dtype = dtype

        self.layers = nn.ModuleList()
        for out_features in hidden_features + [out_features]:
            self.layers.append(nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype))
            in_features = out_features

    def to(self, device):
        self.device = device
        for layer in self.layers:
            layer.to(device)
        return self
    
    def calc_temperature(self, step_i, steps):
        return (1.0 - step_i / steps) * self.noise_coeff
    
    def forward(self, x, steps=None):
        if steps is None:
            steps = self.steps
        state = [x] + [None] * (len(self.layers))

        for step_i in range(steps):
            temp = self.calc_temperature(step_i, steps)
            for i, layer in enumerate(self.layers):
                x = state[i]
                state[i+1] = layer(x)
                noise = torch.randn_like(state[i+1], device=self.device, dtype=self.dtype) * temp
                state[i+1] = state[i+1] + noise

        return F.softmax(state[-1], dim=1), state
    
    # Multiplier is a tensor of shape (batch_size)
    def update_parameters(self, state, multiplier):
        for i, layer in enumerate(self.layers):
            grad = -(state[i+1].T @ (state[i] * multiplier.unsqueeze(1))) / state[i].shape[0]
            layer.weight.grad = grad
            if self.bias:
                layer.bias.grad = -(state[i+1] * multiplier.unsqueeze(1)).mean(0)