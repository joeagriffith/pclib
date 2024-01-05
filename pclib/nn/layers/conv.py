import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.nn.grad import conv2d_input, conv2d_weight
from typing import Optional, Tuple
from pclib.utils.functional import reTanh, identity

# Whittington & Bogacz 2017
class Conv2d(nn.Module):
    __constants__ = ['prev_shape', 'shape']
    shape: Tuple[int]
    prev_shape: Optional[Tuple[int]]

    def __init__(self,
                 prev_shape: Optional[Tuple[int]],
                 shape: (int, int, int), # (channels, height, width)
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding = 1,
                 maxpool=1,
                 has_bias: bool = True,
                 symmetric: bool = True,
                 actv_fn: callable = F.tanh,
                 d_actv_fn: callable = None,
                 gamma: float = 0.1,
                 device=torch.device('cpu'),
                 dtype=None
                 ) -> None:
        
        # assert stride == 1, "Stride != 1 not yet supported."

        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.prev_shape = prev_shape
        self.shape = shape
        self.actv_fn = actv_fn
        self.gamma = gamma
        self.device = device

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = has_bias
        self.maxpool = maxpool


        # Default derivative of activation function
        if d_actv_fn is not None:
            self.d_actv_fn: callable = d_actv_fn
        elif actv_fn == F.relu:
            self.d_actv_fn: callable = lambda x: torch.sign(torch.relu(x))
        elif actv_fn == F.leaky_relu:
            self.d_actv_fn: callable = lambda x: torch.sign(torch.relu(x)) + torch.sign(torch.minimum(x, torch.zeros_like(x))) * 0.01
        elif actv_fn == reTanh:
            self.d_actv_fn: callable = lambda x: torch.sign(torch.relu(x)) * (1 - torch.tanh(x).square())
        elif actv_fn == F.sigmoid:
            self.d_actv_fn: callable = lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))
        elif actv_fn == F.tanh:
            self.d_actv_fn: callable = lambda x: 1 - torch.tanh(x).square()
        elif actv_fn == identity:
            self.d_actv_fn: callable = lambda x: torch.ones_like(x)
        
        self.init_weights()
        self.norm = nn.LayerNorm(self.shape)

    def __str__(self):
        base_str = super().__str__()

        custom_info = "\n  (params): \n" + \
            f"    prev_shape: {self.prev_shape}\n" + \
            f"    shape: {self.shape}\n" + \
            f"    actv_fn: {self.actv_fn}\n" + \
            f"    gamma: {self.gamma}\n" + \
            f"    kernel_size: {self.kernel_size}\n" + \
            f"    stride: {self.stride}\n" + \
            f"    padding: {self.padding}\n" + \
            f"    has_bias: {self.has_bias}\n" + \
            f"    maxpool: {self.maxpool}\n"
        
        string = base_str[:base_str.find('\n')] + custom_info + base_str[base_str.find('\n'):]
        
        return string
        
    def init_weights(self):
        # Initialise weights if not input layer
        if self.prev_shape is not None:
            self.conv = nn.Sequential(
                nn.Conv2d(self.prev_shape[0], self.shape[0], self.kernel_size, padding=self.padding, stride=self.stride, bias=False, **self.factory_kwargs),
                nn.MaxPool2d(kernel_size=self.maxpool),
            )
            if self.has_bias:
                self.bias = Parameter(torch.zeros(self.prev_shape, device=self.device, requires_grad=True))
            # self.conv_bu = nn.Sequential(
            #     nn.Upsample(scale_factor=maxpool),
            #     nn.ConvTranspose2d(prev_shape[0], shape[0], kernel_size, padding=padding, stride=stride, bias=False, **factory_kwargs),
            # )

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
        x = F.interpolate(state['x'].detach(), scale_factor=self.maxpool, mode='nearest')
        prev_shape = (x.shape[0], self.prev_shape[0], self.prev_shape[1], self.prev_shape[2])
        pred = conv2d_input(prev_shape, self.conv[0].weight, x, stride=self.stride, padding=self.padding, dilation=1, groups=1)
        if self.has_bias:
            pred += self.bias
        return pred
    
    # propagates error from layer below, return an update for x
    def propagate(self, e_below):
        return self.conv(e_below)
    
    # Recalculates prediction-error between state and top-down prediction of it
    # With simulated annealing
    def update_e(self, state, pred, temp=None):
        if pred.dim() == 2:
            pred = pred.unsqueeze(-1).unsqueeze(-1)
        state['e'] = state['x'].detach() - self.actv_fn(pred)

        if temp is not None:
            eps = torch.randn_like(state['e'].detach(), device=self.device) * 0.034 * temp
            state['e'] += eps

    def update_x(self, state, e_below=None, pred_below=None, temp=None):
        # If not input layer, propagate error from layer below
        state['x'] = state['x'].detach()

        if e_below is not None:
            if e_below.dim() == 2:
                e_below = e_below.unsqueeze(-1).unsqueeze(-1)
            d_pred = self.d_actv_fn(pred_below)
            update = self.propagate(e_below * d_pred)
            state['x'] += self.gamma * update

        state['x'] += self.gamma * -state['e']

        state['x'] += self.gamma**2 * 0.1 * -state['x']
        
        if temp is not None:
            eps = torch.randn_like(state['x'], device=self.device) * 0.034 * temp
            state['x'] += eps


    def update_grad(self, state, e_below=None):
        """
        
        """
        if e_below is not None:
            b_size = e_below.shape[0]
            x = F.interpolate(self.actv_fn(state['x']), scale_factor=self.maxpool, mode='nearest')
            self.conv[0].weight.grad = 2*-conv2d_weight(e_below, self.conv[0].weight.shape, x, stride=self.stride, padding=self.padding, dilation=1, groups=1) / b_size
            # I dont trust this
            # Might need separate bias for prediction. this is for error prop
            if self.has_bias:
                self.bias.grad = 2*-e_below.mean(0)