import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Optional

# Whittington & Bogacz 2017
class Conv2d(nn.Module):
    __constants__ = ['shape', 'prev_shape']
    shape: int
    prev_shape: Optional[int]

    def __init__(self,
                 shape: (int, int, int), # (channels, height, width)
                 prev_shape: (int, int, int) = None,
                 kernel_size: int = 3,
                 padding = 'same',
                 stride: int = 1,
                 bias: bool = True,
                 maxpool=1,
                 actv_fn: callable = F.relu,
                 d_actv_fn: callable = None,
                 gamma: float = 0.1,
                 beta: float = 1.0,
                 device=torch.device('cpu'),
                 dtype=None
                 ) -> None:
        
        assert stride == 1, "Stride > 1 not yet supported."

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Conv2d, self).__init__()

        self.shape = shape
        self.prev_shape = prev_shape
        self.actv_fn = actv_fn
        self.gamma = gamma
        self.beta = beta
        self.device = device


        # Default derivative of activation function
        if d_actv_fn is not None:
            self.d_actv_fn: callable = d_actv_fn
        elif actv_fn == F.relu:
            self.d_actv_fn: callable = lambda x: torch.sign(torch.relu(x))
        elif actv_fn == F.sigmoid:
            self.d_actv_fn: callable = lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))
        elif actv_fn == F.tanh:
            self.d_actv_fn: callable = lambda x: 1 - torch.tanh(x).square()
        
        # Initialise weights if not input layer
        if prev_shape is not None:
            self.conv_td = nn.Sequential(
                nn.Conv2d(shape[0], prev_shape[0], kernel_size, padding=padding, stride=stride, bias=bias, **factory_kwargs),
                nn.MaxPool2d(kernel_size=maxpool),
            )
            self.conv_bu = nn.Sequential(
                nn.Upsample(scale_factor=maxpool),
                nn.ConvTranspose2d(prev_shape[0], shape[0], kernel_size, padding=padding, stride=stride, bias=False, **factory_kwargs),
            )

    def init_state(self, batch_size):
        return {
            'x': torch.zeros((batch_size, self.shape[0], self.shape[1], self.shape[2]), device=self.device),
            'e': torch.zeros((batch_size, self.shape[0], self.shape[1], self.shape[2]), device=self.device),
        }

    def to(self, *args, **kwargs):
        self.device = args[0]
        return super().to(*args, **kwargs)

    # Returns a prediction of the state in the previous layer
    def predict(self, state):
        return self.conv_td(self.actv_fn(state['x']))
    
    # propogates error from layer below, return an update for x
    def propagate(self, e_below):
        return self.conv_bu(e_below)
    
    def update_x(self, state, e_below=None):
        # If not input layer, propagate error from layer below
        if e_below is not None:
            if e_below.dim() == 2:
                e_below = e_below.unsqueeze(-1).unsqueeze(-1)
            update = self.propagate(e_below)
            state['x'] += self.gamma * (-state['e'] + update * self.d_actv_fn(state['x']))
        # This update will be zero if top layer
        state['x'] += self.gamma * (self.beta * -state['e'])
        
    # Recalculates prediction-error between state and top-down prediction of it
    # With simulated annealing
    def update_e(self, state, pred=None, temp=None):
        with torch.no_grad():
            if pred is not None:
                state['pred'] = pred
            else:
                state['pred'] = state['x']
            state['e'] = state['x'] - state['pred']

            if temp is not None:
                eps = torch.randn_like(state['e'], device=self.device) * 0.034 * temp
                state['e'] += eps
        
    def update_grad(self, state, e_below):
        if state['e'].norm() > 0:
            state['e'].square().sum().backward()
