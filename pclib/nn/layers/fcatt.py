import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from typing import Optional
from pclib.utils.functional import reTanh, identity, trec
from pclib.nn.layers import FC

class FCAtt(FC):
    """
    | Fully connected layer with optional bias and optionally symmetric weights.
    | The layer stores its state in a dictionary with keys 'x' and 'e'.
    | Layer is defined such that 'x' and 'e' are the same shape, and 'x' precedes 'e' in the architecture.
    | The Layer defines predictions as: Wf(x) + Optional(bias).

    Parameters
    ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
        precision : float
            Coefficient for bottom-up prediction.
        has_bias : bool
            Whether to include a bias term.
        symmetric : bool
            Whether to reuse top-down prediction weights, for bottom-up error propagation.
        actv_fn : callable
            Activation function to use.
        d_actv_fn : Optional[callable]
            Derivative of activation function to use (if None, will be inferred from actv_fn).
        gamma : float
            step size for x updates.
        x_decay : float
            Decay rate for x.
        device : torch.device
            Device to use for computation.
        dtype : torch.dtype
            Data type to use for computation.
    """
    __constants__ = ['in_features', 'out_features']
    in_features: Optional[int]
    out_features: int
    weight: Optional[Tensor]
    weight_td: Optional[Tensor]
    bias: Optional[Tensor]

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 precision: float = 1.0,
                 has_bias: bool = True,
                 symmetric: bool = True,
                 actv_fn: callable = F.relu,
                 d_actv_fn: callable = None,
                 gamma: float = 0.1,
                 x_decay: float = 0.0,
                 dropout: float = 0.0,
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = None
                 ) -> None:

        assert symmetric == False, "FCAM layer does not support symmetric weights."
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            in_features,
            out_features,
            precision,
            has_bias,
            symmetric,
            actv_fn,
            d_actv_fn,
            gamma,
            x_decay,
            dropout,
            device,
            dtype
        )

    # Declare weights if not input layer
    def init_params(self):
        """
        | Creates and initialises weight tensors and bias tensor based on args from self.__init__().
        """
        if self.in_features is not None:
            self.weight = Parameter(torch.empty((self.out_features, self.in_features), **self.factory_kwargs))
            bound = 4 * math.sqrt(6 / (self.in_features + self.out_features)) if self.actv_fn == F.sigmoid else math.sqrt(6 / (self.in_features + self.out_features))
            self.weight.data.uniform_(-bound, bound)

            self.att_weight = Parameter(torch.empty((self.in_features, self.out_features), **self.factory_kwargs))
            bound = 4 * math.sqrt(6 / (self.in_features + self.out_features)) if self.actv_fn == F.sigmoid else math.sqrt(6 / (self.in_features + self.out_features))
            self.att_weight.data.uniform_(-bound, bound)

            if self.has_bias:
                #  Bias is used in prediction of layer below, so it has shape (in_features)
                self.bias = Parameter(torch.empty(self.in_features, **self.factory_kwargs))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight.T)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                self.register_parameter('bias', None)

            if not self.symmetric:
                self.weight_td = Parameter(torch.empty((self.in_features, self.out_features), **self.factory_kwargs))
                nn.init.kaiming_uniform_(self.weight_td, a=math.sqrt(5))
            else:
                self.register_parameter('weight_td', None)

        else:
            self.register_parameter('weight', None)
            self.register_parameter('weight_td', None)
            self.register_parameter('bias', None)
        
        self.dropout = nn.Dropout(p=self.dropout)

    def update_x(self, state:dict, e_below:torch.Tensor = None, gamma:torch.Tensor = None):
        """
        | Updates state['x'] inplace, using the error signal from the layer below and error of the current layer.
        | Formula: new_x = x + gamma * (-e + propagate(e_below * att) * d_actv_fn(x) - 0.1 * x + noise).
        | Where: att = actv_fn(x) @ att_weight.

        Parameters
        ----------
            state : dict
                Dictionary containing 'x' and 'e' tensors for this layer.
            e_below : Optional[torch.Tensor]
                state['e'] from the layer below. None if input layer.
            temp : Optional[float]
                Temperature for simulated annealing.
            gamma : Optional[torch.Tensor]
                Step size for x updates. If None, uses self.gamma. Shape: (batch_size).
        """
        if gamma is None:
            gamma = torch.ones(state['x'].shape[0]).to(self.device) * self.gamma

        # If not input layer, propagate error from layer below
        dx = torch.zeros_like(state['x'], device=self.device)
        if e_below is not None:
            e_below = e_below.detach()
            if e_below.dim() == 4:
                e_below = e_below.flatten(1)
            e_below = F.linear(self.actv_fn(state['x'].detach()), self.att_weight, None) * e_below
            dx += self.precision * self.propagate(e_below) * self.d_actv_fn(state['x'].detach())

        dx += -state['e'].detach()

        if self.x_decay > 0:
            dx += -self.x_decay * state['x'].detach() * self.d_actv_fn(state['x'].detach())

        state['x'] = state['x'].detach() + gamma.unsqueeze(-1) * dx