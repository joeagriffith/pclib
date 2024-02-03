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

    Parameters
    ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
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
                 has_bias: bool = True,
                 symmetric: bool = True,
                 actv_fn: callable = F.relu,
                 d_actv_fn: callable = None,
                 gamma: float = 0.1,
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = None
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
        """
        | Returns a string representation of the layer.

        Returns
        -------
            str
        """
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
            
    def init_state(self, batch_size:int):
        """
        | Builds a new state dictionary for the layer, containing torch.tensors for 'x' and 'e'.

        Parameters
        ----------
            batch_size : int
                Batch size of the state.

        Returns
        -------
            dict
                Dictionary containing 'x' and 'e' tensors of shape (batch_size, out_features).
        """
        return {
            'x': torch.zeros((batch_size, self.out_features), device=self.device),
            'e': torch.zeros((batch_size, self.out_features), device=self.device),
        }

    def to(self, *args, **kwargs):
        self.device = args[0]
        return super().to(*args, **kwargs)

    def predict(self, state:dict):
        """
        | Calculates a prediction of state['x'] in the layer below.

        Parameters
        ----------
            state : dict
                Dictionary containing 'x' and 'e' tensors for this layer.
        
        Returns
        -------
            torch.Tensor
                Prediction of state['x'] in the layer below.
        """
        x = state['x'].detach() if self.symmetric else state['x']
        weight_td = self.weight.T if self.symmetric else self.weight_td
        return F.linear(self.actv_fn(x), weight_td, self.bias)
    
    
    def propagate(self, e_below:torch.Tensor):
        """
        | Propagates error from layer below, returning an update signal for state['x'].

        Parameters
        ----------
            e_below : torch.Tensor
                Error signal from layer below.

        Returns
        -------
            torch.Tensor
                Update signal for state['x'].
        """
        if e_below.dim() == 4:
            e_below = e_below.flatten(1)
        return F.linear(e_below, self.weight, None)
        
    def update_e(self, state:dict, pred:torch.Tensor, temp:float = None):
        """
        | Updates prediction-error (state['e']) inplace between state['x'] and the top-down prediction of it.
        | Uses simulated annealing if temp is not None.
        | Does nothing if pred is None. This is useful so the output layer doesn't need specific handling.

        Parameters
        ----------
            state : dict
                Dictionary containing 'x' and 'e' tensors for this layer.
            pred : torch.Tensor
                Top-down prediction of state['x'].
            temp : Optional[float]
                Temperature for simulated annealing.
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
    
    def update_x(self, state:dict, e_below:torch.Tensor = None, temp:float = None):
        """
        | Updates state['x'] inplace, using the error signal from the layer below and error of the current layer.
        | Formula: new_x = x + gamma * (-e + propagate(e_below) * d_actv_fn(x) - 0.1 * x + noise).

        Parameters
        ----------
            state : dict
                Dictionary containing 'x' and 'e' tensors for this layer.
            e_below : Optional[torch.Tensor]
                state['e'] from the layer below. None if input layer.
        """
        # If not input layer, propagate error from layer below
        dx = torch.zeros_like(state['x'], device=self.device)
        if e_below is not None:
            e_below = e_below.detach()
            if e_below.dim() == 4:
                e_below = e_below.flatten(1)
            dx += self.propagate(e_below) * self.d_actv_fn(state['x'].detach())

        dx += -state['e'].detach()

        dx += 0.1 * -state['x'].detach()

        if temp is not None:
            dx += torch.randn_like(state['x'], device=self.device) * temp * 0.034

        state['x'] = state['x'].detach() + self.gamma * dx

    def assert_grads(self, state, e_below):
        """
        | Assert grads are correct.
        """

        true_weight_grad = -(self.actv_fn(state['x']).T @ e_below) / state['x'].size(0)
        assert torch.allclose(self.weight.grad, true_weight_grad), f"Back Weight grad:\n{self.weight.grad}\nCalc weight grad:\n{true_weight_grad}"

        if self.memory_vector is not None:
            true_memory_grad = -state['e'].mean(0)
        assert torch.allclose(self.memory_vector.grad, true_memory_grad)
        
