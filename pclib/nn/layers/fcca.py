import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from typing import Optional
from pclib.utils.functional import reTanh, identity, trec
from pclib.nn.layers import FC

class CrossAttention(nn.Module):

    def __init__(self, e_size, x_size, d_embed: int):
        super().__init__()

        self.scores = nn.Parameter(torch.randn(e_size, x_size))

        self.out_proj = nn.Linear(e_size, x_size, bias=False)
    
    def forward(self, e, x):

        scores = self.scores * x.unsqueeze(1)
        scores = scores.mean(dim=-1)

        values = scores * e
        output = self.out_proj(values)

        return output

#  Implements Cross Attention for the error propagation. 
class FCCA(FC):
    """
    | Fully connected layer with optional bias and optionally symmetric weights.
    | The layer stores its state in a dictionary with keys 'x' and 'e'.
    | Layer is defined such that 'x' and 'e' are the same shape, and 'x' precedes 'e' in the architecture.
    | The Layer implements predictions as: Wf(x) + Optional(b).

    Args:
        | in_features (int): Number of input features.
        | out_features (int): Number of output features.
        | has_bias (bool): Whether to include a bias term.
        | symmetric (bool): Whether to reuse top-down prediction weights, for bottom-up error propagation.
        | actv_fn (callable): Activation function to use.
        | d_actv_fn (callable): Derivative of activation function to use (if None, will be inferred from actv_fn).
        | gamma (float): step size for x updates.
        | device (torch.device): Device to use for computation.
        | dtype (torch.dtype): Data type to use for computation.

    Attributes:
        | weight (torch.Tensor): Weights for bottom-up error propagation
        | weight_td (torch.Tensor): Weights for top-down predictions. (if symmetric=False)
        | bias (torch.Tensor): Bias term (if has_bias=True).
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
                 has_bias: bool = True,
                 symmetric: bool = True,
                 actv_fn: callable = F.relu,
                 d_actv_fn: callable = None,
                 gamma: float = 0.1,
                 device=torch.device('cpu'),
                 dtype=None
                 ) -> None:

        if symmetric:
            raise NotImplementedError("Symmetric weights not possible with FCCA")

        super().__init__(
            in_features,
            out_features,
            has_bias,
            symmetric,
            actv_fn,
            d_actv_fn,
            gamma,
            device,
            dtype
        )

    # Declare weights if not input layer
    def init_params(self):
        """
        | Creates and initialises weight tensors and bias tensor based on init args.
        """
        if self.in_features is not None:
            self.ca_propagate = CrossAttention(self.in_features, self.out_features, 32)
            self.weight_td = Parameter(torch.empty((self.in_features, self.out_features), **self.factory_kwargs))
            nn.init.kaiming_uniform_(self.weight_td, a=math.sqrt(5))
            if self.has_bias:
                #  Bias is used in prediction of layer below, so it has shape (in_features)
                self.bias = Parameter(torch.empty(self.in_features, **self.factory_kwargs))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_td)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                self.register_parameter('bias', None)
            self.register_parameter('weight', None)

        else:
            self.register_parameter('weight', None)
            self.register_parameter('weight_td', None)
            self.register_parameter('bias', None)
    
    def propagate(self, e_below):
        """
        | Propagates error from layer below, returning an update signal for state['x'].

        Args:
            | e_below (torch.Tensor): Error signal from layer below.

        Returns:
            | update (torch.Tensor): Update signal for state['x'].
        """
        if e_below.dim() == 4:
            e_below = e_below.flatten(1)
        return F.linear(e_below, self.weight_td.t())
    
    def update_x(self, state, e_below=None, temp=None):
        """
        | Updates state['x'] inplace, using the error signal from the layer below and error of the current layer.
        | Formula: new_x = x + gamma * (-e + propagate(e_below) * d_actv_fn(x)).

        Args:
            | state (dict): Dictionary containing 'x' and 'e' tensors for this layer.
            | e_below (Optional[torch.Tensor]): Error of layer below. None if input layer.
        """
        # If not input layer, propagate error from layer below
        state['x'] = state['x'].detach()
        state['e'] = state['e'].detach()

        dx = torch.zeros_like(state['x'], device=self.device)
        if e_below is not None:
            e_below = e_below.detach()
            if e_below.dim() == 4:
                e_below = e_below.flatten(1)
            # dx += self.propagate(e_below) * self.d_actv_fn(state['x'])
            dx += self.ca_propagate(e_below, state['x']) * self.d_actv_fn(state['x'])

        dx += -state['e']

        dx += 0.1 * -state['x']

        if temp is not None:
            dx += torch.randn_like(state['x'], device=self.device) * temp * 0.034

        state['x'] = state['x'] + self.gamma * dx