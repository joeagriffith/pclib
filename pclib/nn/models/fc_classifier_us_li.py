
import torch
import torch.nn as nn
import torch.nn.functional as F

from pclib.nn.layers import FCLI, FC
from pclib.nn.models import FCClassifierUs

# Based on Whittington and Bogacz 2017, but with targets predicting inputs
class FCClassifierUsLi(FCClassifierUs):
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
                layers.append(FCLI(in_features, out_features, device=self.device, **self.factory_kwargs))
            in_features = out_features
        self.layers = nn.ModuleList(layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_sizes[-1], 200, bias=True, device=self.device, dtype=self.factory_kwargs['dtype']),
            nn.ReLU(),
            nn.Linear(200, self.num_classes, bias=True, device=self.device, dtype=self.factory_kwargs['dtype']),
        )