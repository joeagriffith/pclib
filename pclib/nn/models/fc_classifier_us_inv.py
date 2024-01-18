import torch
import torch.nn as nn
import torch.nn.functional as F

from pclib.nn.layers import FC
from pclib.nn.models import FCClassifierUs

class FCClassifierUsInv(FCClassifierUs):
    """
    | Inherits most functionality from FCClassifierInv, except it is self-supervised.
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
        | Doesn't include num_classes layer in self.layers.
        | Initialises self.classifier as a simple MLP, to classify the output of self.layers.
        """
        layers = []
        in_features = None
        for out_features in self.hidden_sizes + [self.in_features]:
            layers.append(FC(in_features, out_features, device=self.device, **self.factory_kwargs))
            in_features = out_features
        self.layers = nn.ModuleList(layers)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_sizes[-1], 200, bias=True, device=self.device, dtype=self.factory_kwargs['dtype']),
            nn.ReLU(),
            nn.Linear(200, self.num_classes, bias=False, device=self.device, dtype=self.factory_kwargs['dtype']),
        )

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
        | Takes the output from the feature extractor (bottom layer) and passes it through the classifier.

        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e_l' and 'e_u'

        Returns:
            | out (torch.Tensor): Output of the classifier
        """
        x = state[0]['x']
        out = self.classifier(x.detach())
        return out
    

    def reconstruct(self, obs, steps=None):
        """
        | Initialises the state of the model using the observation.
        | Runs inference without pinning the observation.
        | In theory should reconstruct the observation.

        Args:
            | obs (torch.Tensor): Input data
            | steps (Optional[int]): Number of steps to run inference for. Uses self.steps if not provided.

        Returns:
            | out (torch.Tensor): Reconstructed observation
            | state (list): List of layer state dicts, each containing 'x' and 'e'
        """
        if steps is None:
            steps = self.steps
        
        state = self.init_state(obs)

        for i in range(steps):
            temp = self.calc_temp(i, steps)
            self.step(state, temp=temp)
        
        out = state[-1]['x']

        return out, state