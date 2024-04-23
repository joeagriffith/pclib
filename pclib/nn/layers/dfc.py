import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from typing import Optional
from pclib.utils.functional import reTanh, identity, trec, shrinkage, d_shrinkage
from pclib.nn.layers import FC

class DFC(FC):
    """
    | Fully connected layer with optional bias and optionally symmetric weights.
    | Predictions are made through MLPs with optional dropout.
    | Updates to value nodes are made via backpropagation
    | The layer stores its state in a dictionary with keys 'x' and 'e'.
    | Layer is defined such that 'x' and 'e' are the same shape, and 'x' precedes 'e' in the architecture.

    Parameters
    ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
        precision : float
            Coefficient for bottom-up error propagation.
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
        dropout : float
            Dropout rate for predictions.
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
        assert self.symmetric == True, "DFC must have symmetric=True."
        self.d_actv_fn = None
        self.init_params()

    # Declare weights if not input layer
    def init_params(self):
        """
        | Creates and initialises weight tensors and bias tensor based on args from self.__init__().
        """
        if self.in_features is not None:
            self.mlp = nn.Sequential(
                # nn.ReLU(),
                nn.Linear(self.out_features, self.in_features, bias=self.has_bias),

                # nn.Linear(self.out_features, self.out_features, bias=self.has_bias),
                # nn.ReLU(),
                # nn.Dropout(p=self.dropout),
                # nn.Linear(self.out_features, self.out_features, bias=self.has_bias),
                # nn.ReLU(),
                # nn.Dropout(p=self.dropout),
                # nn.Linear(self.out_features, self.in_features, bias=self.has_bias)
            )

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
        state = {
            # 'x': torch.zeros((batch_size, self.out_features), device=self.device, requires_grad=True),
            # 'x': torch.zeros((batch_size, self.out_features), device=self.device, requires_grad=True),
            'x': (torch.ones((batch_size, self.out_features), device=self.device)) / self.out_features,
            # 'e': torch.zeros((batch_size, self.out_features), device=self.device),
        }
        state['x'].requires_grad = True
        return state

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
        x = state['x'].flatten(1)
        # print(f'x grad_fn: {x.grad_fn}')
        # print(f'x requires grad: {x.requires_grad}')
        pred = self.mlp(self.actv_fn(x))
        # print(f'pred grad_fn: {pred.grad_fn}')
        return pred

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
        raise NotImplementedError("propagate() is not used in DFC, backpropagate VFE instead.")
    

    def update_x(self, state:dict, gamma:torch.Tensor = None):
        raise NotImplementedError("update_x() is not used in DFC, backpropagate VFE instead.")
    

    def update_e(self, state:dict, pred:torch.Tensor):
        raise NotImplementedError("update_e() is not used in DFC, backpropagate VFE instead.")
    
    def prediction_errors(self, state:dict, pred:torch.Tensor):
        assert pred is not None, "Prediction must be provided to update_e()."
        
        if pred.dim() == 4:
            pred = pred.flatten(1)
        return state['x'] - pred


    def assert_grads(self, state, e_below):
        """
        | Assert grads are correct.
        """
        raise NotImplementedError("Not Implemented.")