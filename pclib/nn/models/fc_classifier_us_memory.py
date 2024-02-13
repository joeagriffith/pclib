import torch
import torch.nn as nn
import torch.nn.functional as F

from pclib.nn.layers import FC
from pclib.nn.models import FCClassifierUs
from typing import List

# Based on Whittington and Bogacz 2017, but with targets predicting inputs
class FCClassifierUsMem(FCClassifierUs):
    """
    | An Unsupervised version of FCClassifier.
    | Learns a feature extractor (self.layers) via unsupervised learning.
    | Also learns an mlp classifier (self.classifier) which takes the output of self.layers as input, via supervised learning.

    Parameters
    ----------
        in_features : int
            Number of input features
        num_classes : int
            Number of classes
        hidden_sizes : list
            List of hidden layer sizes
        steps : int
            Number of steps to run inference for
        bias : bool
            Whether to include bias in layers
        symmetric : bool
            Whether to use same weights for top-down prediction and bottom-up error prop.
        actv_fn : callable
            Activation function to use
        d_actv_fn : Optional[callable]
            Derivative of activation function to use
        gamma : float
            step size for x updates
        temp_k : float
            Temperature constant for inference
        device : torch.device
            Device to run on
        dtype : torch.dtype
            Data type to use
    """
    def __init__(
            self, 
            in_features:int, 
            num_classes:int, 
            hidden_sizes:List[int] = [], 
            steps:int = 20, 
            bias:bool = True, 
            symmetric:bool = True, 
            actv_fn:callable = F.tanh, 
            d_actv_fn:callable = None, 
            gamma:float = 0.1, 
            temp_k:float = 1.0,
            device:torch.device = torch.device('cpu'), 
            dtype:torch.dtype = None
        ):
        super().__init__(in_features, num_classes, hidden_sizes, steps, bias, symmetric, actv_fn, d_actv_fn, gamma, temp_k, device, dtype)

    def init_layers(self):
        """
        | Initialises self.layers based on input parameters.
        | Initialises self.classifier as a simple MLP, to classify the output of self.layers.
        """
        layers = []
        in_features = None
        self.sizes = [self.in_features] + self.hidden_sizes
        for out_features in self.sizes:
            layers.append(FC(in_features, out_features, device=self.device, **self.factory_kwargs))
            in_features = out_features
        self.layers = nn.ModuleList(layers)

        self.memory_vector = nn.Parameter(torch.randn(self.sizes[-1], device=self.device, dtype=self.factory_kwargs['dtype']) * 0.01)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.sizes[-1], 200, bias=True, device=self.device, dtype=self.factory_kwargs['dtype']),
            nn.ReLU(),
            nn.Linear(200, self.num_classes, bias=False, device=self.device, dtype=self.factory_kwargs['dtype']),
        )

    def to(self, device):
        self.device = device
        for layer in self.layers:
            layer.to(device)
        for layer in self.classifier:
            layer.to(device)
        self.memory_vector = nn.Parameter(self.memory_vector.to(device))
        return self
    
    def step(self, state:List[dict], obs:torch.Tensor = None, y:torch.Tensor = None, temp:float = None, gamma:float = None):
        """
        | Performs one step of inference. Updates Xs first, then Es.
        | Both are updated from bottom to top.

        Parameters
        ----------
            state : List[dict]
                List of layer state dicts, each containing 'x' and 'e; (and 'eps' for FCPW)
            obs : Optional[torch.Tensor]
                Input data
            y : Optional[torch.Tensor]
                Target data
            temp : Optional[float]
                Temperature to use for update
        """
        for i, layer in enumerate(self.layers):
            if i > 0 or obs is None: # Don't update bottom x if obs is given
                e_below = state[i-1]['e'] if i > 0 else None
                layer.update_x(state[i], e_below, temp=temp, gamma=gamma)
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                pred = self.layers[i+1].predict(state[i+1])
                layer.update_e(state[i], pred, temp=temp)
            else:
                pred = self.memory_vector
                layer.update_e(state[i], pred, temp=temp)

