import torch
import torch.nn as nn
import torch.nn.functional as F

from pclib.nn.layers import FCBCDIM
from pclib.nn.models import FCClassifierUs

# Based on Whittington and Bogacz 2017, but with targets predicting inputs
class FCClassifierUsBcDim(FCClassifierUs):
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
        | Initialises self.classifier as a simple MLP, to classify the output of self.layers.
        """
        layers = []
        in_features = None
        for out_features in [self.in_features] + self.hidden_sizes:
            if self.precision_weighted:
                raise NotImplementedError
                # layers.append(FCPW(in_features, out_features, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs))
            else:
                layers.append(FCBCDIM(in_features, out_features, device=self.device, **self.factory_kwargs))
            in_features = out_features
        self.layers = nn.ModuleList(layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_sizes[-1], 200, bias=True, device=self.device, dtype=self.factory_kwargs['dtype']),
            nn.ReLU(),
            nn.Linear(200, self.num_classes, bias=False, device=self.device, dtype=self.factory_kwargs['dtype']),
        )

    def forward(self, obs=None, steps=None):
        """
        | Performs inference for the model.
        
        Args:
            | obs (Optional[torch.Tensor]): Input data
            | y (Optional[torch.Tensor]): Target data
            | steps (Optional[int]): Number of steps to run inference for. Uses self.steps if not provided.

        Returns:
            | out (torch.Tensor): Output of the model
            | state (list): List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)
        """
        if steps is None:
            steps = self.steps

        for layer in self.layers:
            if layer.weight_td is not None:
                layer.weight_td.data = F.relu(layer.weight_td.data)

        state = self.init_state(obs)

        for i in range(steps):
            temp = self.calc_temp(i, steps)
            self.step(state, obs, temp)
            
        out = self.get_output(state)
            
        return out, state


    # def step(self, state, obs=None, y=None, temp=None):
    #     """
    #     | Performs one step of inference. Updates Es first, then Xs, then pins.
    #     | Es are updated top-down, Xs are updated bottom-up, due to the way the updates flow.

    #     Args:
    #         | state (list): List of layer state dicts, each containing 'x' and 'e; (and 'eps' for FCPW)
    #         | obs (Optional[torch.Tensor]): Input data
    #         | y (Optional[torch.Tensor]): Target data
    #         | temp (Optional[float]): Temperature to use for update

    #     """
    #     for i, layer in enumerate(self.layers):
    #         e_below = state[i-1]['e'] if i > 0 else None
    #         pred = self.layers[i+1].predict(state[i+1]) if i < len(self.layers) - 1 else None
    #         if i > 0 or obs is None: # Don't update bottom x if obs is given
    #             if i < len(self.layers) - 1 or y is None: # Don't update top x if y is given
    #                 with torch.no_grad():
    #                     layer.update_x(state[i], e_below, temp=temp)
    #         if i < len(self.layers) - 1:
    #             layer.update_e(state[i], pred, temp=temp)