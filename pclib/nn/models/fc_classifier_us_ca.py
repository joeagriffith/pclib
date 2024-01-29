import torch
import torch.nn as nn
import torch.nn.functional as F

from pclib.nn.layers import FCCA
from pclib.nn.models import FCClassifierUs

# Based on Whittington and Bogacz 2017, but with targets predicting inputs
class FCClassifierUsCa(FCClassifierUs):
    """
    | A Self-Supervised version of FCClassifier.
    | It learns a feature extractor (self.layers) only from observations.
    | It separately learns a classifier (self.classifier) which takes the output of self.layers as input.

    Args:
        | in_features (int): Number of input features
        | num_classes (int): Number of classes
        | hidden_sizes (list): List of hidden layer sizes
        | steps (int): Number of steps to run inference for
        | bias (bool): Whether to include bias in layers
        | symmetric (bool): Whether to use same weights for top-down prediction and bottom-up error prop.
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
        | classifier (torch.nn.Sequential): Classifier to use
    """
    def __init__(self, in_features, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, actv_fn=F.tanh, d_actv_fn=None, gamma=0.1, device=torch.device('cpu'), dtype=None):
        super().__init__(in_features, num_classes, hidden_sizes, steps, bias, symmetric, actv_fn, d_actv_fn, gamma, device, dtype)

    def init_layers(self):
        """
        | Initialises self.layers based on input parameters.
        | Initialises self.classifier as a simple MLP, to classify the output of self.layers.
        """
        layers = []
        in_features = None
        self.sizes = [self.in_features] + self.hidden_sizes
        for out_features in self.sizes:
            layers.append(FCCA(in_features, out_features, device=self.device, **self.factory_kwargs))
            in_features = out_features
        self.layers = nn.ModuleList(layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.sizes[-1], 200, bias=True, device=self.device, dtype=self.factory_kwargs['dtype']),
            nn.ReLU(),
            nn.Linear(200, self.num_classes, bias=False, device=self.device, dtype=self.factory_kwargs['dtype']),
        )
    
    # def vfe(self, state, batch_reduction='mean', unit_reduction='sum'):
    #     """
    #     | Calculates the Variational Free Energy (VFE) of the model.
    #     | This is the sum of the squared prediction errors of each layer.
    #     | how batches and units are reduced is controlled by batch_reduction and unit_reduction.

    #     Args:
    #         | state (list): List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)
    #         | batch_reduction (str): How to reduce over batches ['sum', 'mean', None]
    #         | unit_reduction (str): How to reduce over units ['sum', 'mean']

    #     Returns:
    #         | vfe (torch.Tensor): VFE of the model (scalar)
    #     """
    #     # Reduce units for each layer
    #     if unit_reduction == 'sum':
    #         vfe = [state_i['e'].square().sum(dim=[i for i in range(1, state_i['e'].dim())]) +  state_i['x'].abs().sum(dim=[i for i in range(1, state_i['x'].dim())]) for state_i in state]
    #     elif unit_reduction =='mean':
    #         vfe = [state_i['e'].square().mean(dim=[i for i in range(1, state_i['e'].dim())]) + state_i['x'].abs().mean(dim=[i for i in range(1, state_i['x'].dim())]) for state_i in state]
    #     # Reduce layers
    #     vfe = sum(vfe)
    #     # Reduce batches
    #     if batch_reduction == 'sum':
    #         vfe = vfe.sum()
    #     elif batch_reduction == 'mean':
    #         vfe = vfe.mean()

    #     return vfe
