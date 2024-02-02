import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from typing import Optional
from pclib.utils.functional import reTanh, identity, trec

class FC(nn.Module):
    """
    | Fully connected layer with optional bias and optionally symmetric weights.
    | The layer stores its state in a dictionary with keys 'x' and 'e'.
    | Layer is defined such that 'x' and 'e' are the same shape, and 'x' precedes 'e' in the architecture.
    | The Layer defines predictions as: Wf(x) + Optional(bias).

    Args:
        | in_features: Number of input features.
        | out_features: Number of output features.
        | has_bias: Whether to include a bias term.
        | symmetric: Whether to reuse top-down prediction weights, for bottom-up error propagation.
        | actv_fn: Activation function to use.
        | d_actv_fn: Derivative of activation function to use (if None, will be inferred from actv_fn).
        | gamma: step size for x updates.
        | device: Device to use for computation.
        | dtype: Data type to use for computation.

    Attributes:
        | in_features: Number of input features.
        | out_features: Number of output features; size of x and e vectors.
        | weight: Weights for bottom-up error propagation
        | weight_td: Weights for top-down predictions. (if symmetric=False)
        | bias: Bias term (if has_bias=True).
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

        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias
        self.symmetric = symmetric
        self.actv_fn = actv_fn
        self.gamma = gamma
        self.device = device

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

        self.init_params()

    def __str__(self):
        base_str = super().__str__()

        custom_info = "\n  (params): \n" + \
            f"    in_features: {self.in_features}\n" + \
            f"    out_features: {self.out_features}\n" + \
            f"    has_bias: {self.has_bias}\n" + \
            f"    symmetric: {self.symmetric}\n" + \
            f"    actv_fn: {self.actv_fn.__name__}\n" + \
            f"    gamma: {self.gamma}"
        
        string = base_str[:base_str.find('\n')] + custom_info + base_str[base_str.find('\n'):]
        
        return string
        
    # Declare weights if not input layer
    def init_params(self):
        """
        | Creates and initialises weight tensors and bias tensor based on args from self.__init__().
        """
        if self.in_features is not None:
            self.weight = Parameter(torch.empty((self.out_features, self.in_features), **self.factory_kwargs))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

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
            
    def init_state(self, batch_size):
        """
        | Builds a new state dictionary for the layer.

        Args:
            | batch_size (int): Batch size of the state.

        Returns:
            | state (dict): Dictionary containing 'x' and 'e' tensors of shape (batch_size, out_features).

        """
        return {
            'x': torch.zeros((batch_size, self.out_features), device=self.device),
            'e': torch.zeros((batch_size, self.out_features), device=self.device),
        }

    def to(self, *args, **kwargs):
        self.device = args[0]
        return super().to(*args, **kwargs)

    def predict(self, state):
        """
        | Calculates a prediction of state['x'] in the layer below.

        Args:
            | state (dict): Dictionary containing 'x' and 'e' tensors for this layer.
        
        Returns:
            | pred (torch.Tensor): Prediction of state['x'] in the layer below.
        """
        weight_td = self.weight.T if self.symmetric else self.weight_td
        x = state['x'].detach() if self.symmetric else state['x']
        return F.linear(self.actv_fn(x), weight_td, self.bias)
    
    
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
        return F.linear(e_below, self.weight, None)
        
    def update_e(self, state, pred, temp=None):
        """
        | Updates prediction-error (state['e']) inplace between state['x'] and the top-down prediction of it.
        | Uses simulated annealing if temp is not None.
        | Does nothing if pred is None. This is useful so the output layer doesn't need specific handling.

        Args:
            | state (dict): Dictionary containing 'x' and 'e' tensors for this layer.
            | pred (torch.Tensor): Top-down prediction of state['x'].
            | temp (Optional[float]): Temperature for simulated annealing.
        """
        assert pred is not None, "Prediction must be provided to update_e()."

        if self.symmetric:
            state['x'] = state['x'].detach()
        
        if pred.dim() == 4:
            pred = pred.flatten(1)
        state['e'] = state['x'] - pred

        if temp is not None:
            eps = torch.randn_like(state['e'], device=self.device) * 0.034 * temp
            state['e'] += eps
    
    def update_x(self, state, e_below=None, temp=None):
        """
        | Updates state['x'] inplace, using the error signal from the layer below and error of the current layer.
        | Formula: new_x = x + gamma * (-e + propagate(e_below) * d_actv_fn(x) - 0.1 * x + noise).

        Args:
            | state (dict): Dictionary containing 'x' and 'e' tensors for this layer.
            | e_below (Optional[torch.Tensor]): Error of layer below. None if input layer.
        """
        if not self.symmetric:
            state['x'] = state['x'].detach()
            state['e'] = state['e'].detach()

        # If not input layer, propagate error from layer below
        dx = torch.zeros_like(state['x'], device=self.device)
        if e_below is not None:
            e_below = e_below.detach()
            if e_below.dim() == 4:
                e_below = e_below.flatten(1)
            dx += self.propagate(e_below) * self.d_actv_fn(state['x'])

        dx += -state['e']

        dx += 0.1 * -state['x']

        if temp is not None:
            dx += torch.randn_like(state['x'], device=self.device) * temp * 0.034

        state['x'] = state['x'] + self.gamma * dx