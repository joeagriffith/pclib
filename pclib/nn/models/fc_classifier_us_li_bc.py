
import torch
import torch.nn as nn
import torch.nn.functional as F

from pclib.nn.layers import FCLIBC, FC
from pclib.nn.models import FCClassifierUsLi

# Based on Whittington and Bogacz 2017, but with targets predicting inputs
class FCClassifierUsLiBc(FCClassifierUsLi):
    """
    | Inherits most functionality from FCClassifierSS, so is self-supervised.
    | However, it uses FCLI layer instead of FC layers to implement lateral inhibition.

    Args:
        | in_features (int): Number of input features
        | num_classes (int): Number of classes
        | hidden_sizes (list): List of hidden layer sizes
        | steps (int): Number of steps to run inference for
        | bias (bool): Whether to include bias in layers
        | symmetric (bool): Whether to use same weights for top-down prediction and bottom-up error prop.
        | precision_weighted (bool): Whether to use precision weighted layers (FCPW instead of FC)
        | actv_fn (torch.nn.functional): Activation function to use
        | d_actv_fn (torch.nn.functional): Derivative of activation function to use
        | gamma (float): step size for x updates
        | device (torch.device): Device to run on
        | dtype (torch.dtype): Data type to use

    Attributes:
        | layers (torch.nn.ModuleList): List of layers
        | in_features (int): Number of input features
        | num_classes (int): Number of classes
        | hidden_sizes (list): List of hidden layer sizes
        | steps (int): Number of steps to run inference for
        | bias (bool): Whether to include bias in layers
        | symmetric (bool): Whether to use same weights for top-down prediction and bottom-up error prop.
        | precision_weighted (bool): Whether to use precision weighted layers (FCPW instead of FC) (NOT IMPLEMENTED)
        | classifier (torch.nn.Sequential): Classifier to use
    """

    def __init__(self, in_features, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.tanh, d_actv_fn=None, gamma=0.1, device=torch.device('cpu'), dtype=None):
        super().__init__(in_features, num_classes, hidden_sizes, steps, bias, symmetric, precision_weighted, actv_fn, d_actv_fn, gamma, device, dtype)

    def init_layers(self):
        """
        | Initialises self.layers based on input parameters.
        | Uses FCLI layers instead of FC layers.
        | Initialises self.classifier as a simple MLP, to classify the output of self.layers.
        """
        layers = []
        layers.append(FC(None, self.in_features, device=self.device, **self.factory_kwargs))
        in_features = self.in_features
        for out_features in self.hidden_sizes:
            if self.precision_weighted:
                raise NotImplementedError("Precision weighted not implemented for FCClassifierSSLI")
            else:
                layers.append(FCLIBC(in_features, out_features, device=self.device, **self.factory_kwargs))
            in_features = out_features
        self.layers = nn.ModuleList(layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_sizes[-1], 256, bias=True, device=self.device, dtype=self.factory_kwargs['dtype']),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128, device=self.device, dtype=self.factory_kwargs['dtype']),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.num_classes, bias=True, device=self.device, dtype=self.factory_kwargs['dtype']),
        )

    def step(self, state, obs=None, y=None, temp=None):
        """
        | Performs one step of inference. Updates Es first, then Xs, then pins.
        | Es are updated top-down, Xs are updated bottom-up, due to the way the updates flow.

        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e; (and 'eps' for FCPW)
            | obs (Optional[torch.Tensor]): Input data
            | y (Optional[torch.Tensor]): Target data
            | temp (Optional[float]): Temperature to use for update

        """
        for i, layer in enumerate(self.layers):
            e_below = state[i-1]['e'] if i > 0 else None
            pred = self.layers[i+1].predict(state[i+1]) if i < len(self.layers) - 1 else None
            if i > 0 or obs is None: # Don't update bottom x if obs is given
                if i < len(self.layers) - 1 or y is None: # Don't update top x if y is given
                    with torch.no_grad():
                        layer.update_x(state[i], e_below, pred, temp=temp)
            if i < len(self.layers) - 1:
                layer.update_e(state[i], pred, temp=temp)