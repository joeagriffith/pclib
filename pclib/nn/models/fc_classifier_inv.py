import torch
import torch.nn as nn
import torch.nn.functional as F
from pclib.nn.layers import FC, FCPW
from pclib.nn.models import FCClassifier

# Based on Whittington and Bogacz 2017
class FCClassifierInv(FCClassifier):
    """
    | Inherits most functionality from FCClassifier, except it predicts targets from inputs (inputs at top, targets at bottom).
    | Based on Whittington and Bogacz 2017.
    | This model is the most effective classifier, acheiving >98% val accuracy on MNIST.

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
        | precision_weighted (bool): Whether to use precision weighted layers (FCPW instead of FC)
    """

    def __init__(self, in_features, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.tanh, d_actv_fn=None, gamma=0.1, device=torch.device('cpu'), dtype=None):
        super().__init__(in_features, num_classes, hidden_sizes, steps, bias, symmetric, precision_weighted, actv_fn, d_actv_fn, gamma, device, dtype)

    def init_layers(self):
        """
        | Almost identical to FCClassifier, except it swaps the position of in_features and num_classes.
        """
        layers = []
        in_features = None
        for out_features in [self.num_classes] + self.hidden_sizes + [self.in_features]:
            if self.precision_weighted:
                layers.append(FCPW(in_features, out_features, device=self.device, **self.factory_kwargs))
            else:
                layers.append(FC(in_features, out_features, device=self.device, **self.factory_kwargs))
            in_features = out_features
        self.layers = nn.ModuleList(layers)

    def pin(self, state, obs=None, y=None):
        """
        | Pins the input and/or target to the state if provided.
        | Overrides FCClassifier.pin() to pin targets at bottom, and inputs at top.
        """
        if obs is not None:
            state[-1]['x'] = obs.clone()
        if y is not None:
            state[0]['x'] = y.clone()

    def _init_xs(self, state, obs=None, y=None):
        """
        | Initialises xs dependant using either obs or y if provided.    
        | Similar to FCClassifier._init_xs(), but generates predictions from obs, or propagates error from target.

        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)
            | obs (Optional[torch.Tensor]): Input data
            | y (Optional[torch.Tensor]): Target data
        """
        if obs is not None:
            for i, layer in reversed (list(enumerate(self.layers))):
                if i == len(self.layers) - 1: # last layer
                    state[i]['x'] = obs.clone()
                if i > 0:
                    state[i-1]['x'] = layer.predict(state[i])
            if y is not None:
                state[0]['x'] = y.clone()
        elif y is not None:
            for i, layer in enumerate(self.layers):
                if i == 0:
                    state[0]['x'] = y.clone()
                else:
                    state[i]['x'] = layer.propagate(state[i-1]['x'])

    def get_output(self, state):
        """
        | Gets the output of the model.
        | Output is pulled from bottom layer, instead of top layer.

        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)
        
        Returns:
            | out (torch.Tensor): Output of the model
        """
        return state[0]['x']