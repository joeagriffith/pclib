import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.nn.grad import conv2d_input, conv2d_weight
from typing import Optional, Tuple, List
from pclib.utils.functional import reTanh, identity, trec

# Whittington & Bogacz 2017
class Conv2d(nn.Module):
    """
    | Convolutional layer with optional bias, assymetric weights not yet supported.
    | Layer has similar functionality to FC layer, but propagates errors via a convolution.
    | The predictions are calculated using pytorch's conv2d_input function.
    | This layer also defines predictions as: Wf(x) + Optional(bias).

    Parameters
    ----------
        prev_shape : Optional[Tuple[int]] 
            Shape of the previous layer, None if this is input layer.
        shape : (int, int, int)
            Shape of the current layer.
        kernel_size : int
            Size of the convolutional kernel.
        stride : int
            Stride of the convolutional kernel.
        padding : int
            Padding of the convolutional kernel.
        has_bi : bool 
            Whether the layer has a bias.
        symmetric : bool
            Whether the layer has symmetric weights.
        actv_fn : callable
            Activation function of the layer.
        d_actv_fn : Optional[callable]
            Derivative of the activation function of the layer (if None, it will be inferred from actv_fn).
        gamma : float
            Step size for x updates.
        device : Optional[torch.device]
            Device to run the layer on.
        dtype : Optional[torch.dtype]
            Data type of the layer.
    """


    __constants__ = ['prev_shape', 'shape']
    shape: Tuple[int]
    prev_shape: Optional[Tuple[int]]

    def __init__(self,
                 prev_shape: Optional[Tuple[int]],
                 shape: (int, int, int), # (channels, height, width)
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 maxpool: int = 1,
                 has_bias: bool = True,
                 symmetric: bool = True,
                 actv_fn: callable = F.tanh,
                 d_actv_fn: callable = None,
                 gamma: float = 0.1,
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = None
                 ) -> None:
        
        # assert stride == 1, "Stride != 1 not yet supported."
        assert symmetric, "Asymmetric convolution not yet supported."

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
        self.symmetric = symmetric
        self.maxpool = maxpool

        # Automatically set d_actv_fn if not provided
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
        elif actv_fn == F.gelu:
            self.d_actv_fn: callable = lambda x: torch.sigmoid(1.702 * x) * (1. + torch.exp(-1.702 * x) * (1.702 * x + 1.)) + 0.5
        elif actv_fn == F.softplus:
            self.d_actv_fn: callable = lambda x: torch.sigmoid(x)
        elif actv_fn == F.softsign:
            self.d_actv_fn: callable = lambda x: 1 / (1 + torch.abs(x)).square()
        elif actv_fn == F.elu:
            self.d_actv_fn: callable = lambda x: torch.sign(torch.relu(x)) + torch.sign(torch.minimum(x, torch.zeros_like(x))) * 0.01 + 1
        elif actv_fn == F.leaky_relu:
            self.d_actv_fn: callable = lambda x: torch.where(x > 0, torch.ones_like(x), 0.01 * torch.ones_like(x))
        elif actv_fn == trec:
            self.d_actv_fn: callable = lambda x: (x > 1.0).float()
        
        self.init_weights()

    def __str__(self):
        """
        | Returns a string representation of the layer.

        Returns
        -------
            str
        """
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
            f"    symmetric: {self.symmetric}\n" + \
            f"    maxpool: {self.maxpool}\n"
        
        string = base_str[:base_str.find('\n')] + custom_info + base_str[base_str.find('\n'):]
        
        return string
        
    def init_weights(self):
        """
        | Initialises the weights of the layer.
        | Includes optional maxpooling.
        """

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

    def init_state(self, batch_size:int):
        """
        | Builds a new state dictionary for the layer.

        Parameters
        ----------
            batch_size : int
                Batch size of the state dictionary.

        Returns
        -------
            dict
                A dictionary containing 2 equally shaped tensors, 'x' and 'e'.
        """
        return {
            'x': torch.zeros((batch_size, self.shape[0], self.shape[1], self.shape[2]), device=self.device),
            'e': torch.zeros((batch_size, self.shape[0], self.shape[1], self.shape[2]), device=self.device),
        }

    def to(self, *args, **kwargs):
        self.device = args[0]
        return super().to(*args, **kwargs)

    def predict(self, state:dict):
        """
        | Calculates the prediction of state['x'] the layer below.

        Parameters
        ----------
            state : dict
                The state dictionary for this layer.
        
        Returns
        -------
            torch.Tensor
                The layers prediction of 'x' in the layer below.
        """

        x = state['x'].detach() if self.symmetric else state['x']
        x = F.interpolate(x, scale_factor=self.maxpool, mode='nearest')
        prev_shape = (x.shape[0], self.prev_shape[0], self.prev_shape[1], self.prev_shape[2])
        actv = conv2d_input(prev_shape, self.conv[0].weight, self.actv_fn(x), stride=self.stride, padding=self.padding, dilation=1, groups=1)
        if self.has_bias:
            actv += self.bias
        return actv
    
    def propagate(self, e_below:torch.Tensor):
        """
        | Propagates error from layer below, returning an update for state['x'].

        Parameters
        ----------
            e_below : torch.Tensor
                The error from the layer below.

        Returns
        -------
            torch.Tensor
                The update for state['x'].
        """
        return self.conv(e_below)
    
    def update_e(self, state:dict, pred:torch.Tensor, temp:float=None):
        """
        | Updates the prediction error (state['e']) between state['x'] and pred.
        | Uses simulated annealing if temp is not None.

        Parameters
        ----------
            state : dict
                The state dictionary for this layer, containing 'x' and 'e'.
            pred : torch.Tensor
                The prediction of 'x' from the layer above.
            temp : Optional[float]
                The temperature for simulated annealing.
        """

        assert pred is not None, "Prediction must be provided to update_e()."

        if self.symmetric:
            state['x'] = state['x'].detach()

        if pred.dim() == 2:
            pred = pred.unsqueeze(-1).unsqueeze(-1)
        state['e'] = state['x'] - pred

        if temp is not None:
            eps = torch.randn_like(state['e'], device=self.device) * 0.034 * temp
            state['e'] += eps

    def update_x(self, state:dict, e_below:torch.Tensor=None, temp:float=None):
        """
        | Updates state['x'] using the error signal from the layer below and of current layer.
        | Formula: new_x = x + gamma * (-e + propagate(e_below) * d_actv_fn(x) - 0.1 * x + noise)

        Args:
            | state (dict): The state dictionary for this layer.
            | e_below (Optional[torch.Tensor]): The error from the layer below.
        """
        dx = torch.zeros_like(state['x'], device=self.device)
        if e_below is not None:
            e_below = e_below.detach()
            if e_below.dim() == 2:
                e_below = e_below.unsqueeze(-1).unsqueeze(-1)
            dx += self.propagate(e_below) * self.d_actv_fn(state['x'].detach())

        dx += -state['e'].detach()

        dx += 0.1 * -state['x'].detach()
        
        if temp is not None:
            dx += torch.randn_like(state['x'], device=self.device) * 0.034 * temp

        state['x'] = state['x'].detach() + self.gamma * dx