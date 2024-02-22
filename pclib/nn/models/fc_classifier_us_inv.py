import torch
import torch.nn as nn
import torch.nn.functional as F

from pclib.nn.layers import FC
from pclib.nn.models import FCClassifierUs
from typing import List

class FCClassifierUsInv(FCClassifierUs):
    """
    | Inherits most functionality from FCClassifierUs, except it predictions from inputs to targets (inputs at top, targets at bottom).

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
        x_decay : float
            Decay rate for x
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
            d_actv_fn:bool = None, 
            gamma:float = 0.1, 
            x_decay:float = 0.0,
            temp_k:float = 1.0,
            device:torch.device = torch.device('cpu'), 
            dtype:torch.dtype = None
        ):
        super().__init__(
            in_features, 
            num_classes, 
            hidden_sizes, 
            steps, 
            bias, 
            symmetric, 
            actv_fn, 
            d_actv_fn, 
            x_decay,
            gamma, 
            temp_k,
            device, 
            dtype
        )

    def init_layers(self):
        """
        | Initialises self.layers based on input parameters.
        | Doesn't include num_classes layer in self.layers.
        | Initialises self.classifier as a simple MLP, to classify the output of self.layers.
        """
        layers = []
        in_features = None
        self.sizes = [self.in_features] + self.hidden_sizes
        self.sizes = self.sizes[::-1]
        for out_features in self.sizes:
            layers.append(FC(in_features, out_features, device=self.device, **self.factory_kwargs))
            in_features = out_features
        self.layers = nn.ModuleList(layers)

        self.classifier = nn.Sequential(
            nn.Linear(self.sizes[0], 200, bias=True, device=self.device, dtype=self.factory_kwargs['dtype']),
            nn.ReLU(),
            nn.Linear(200, self.num_classes, bias=False, device=self.device, dtype=self.factory_kwargs['dtype']),
        )
            
    def _init_xs(self, state:List[dict], obs:torch.Tensor = None, y:torch.Tensor = None):
        """
        | Initialises xs dependant using either obs or y if provided.    
        | Similar to FCClassifier._init_xs(), but generates predictions from obs, or propagates error from target.

        Parameters
        ----------
            state : List[dict]
                List of layer state dicts, each containing 'x' and 'e'
            obs : Optional[torch.Tensor]
                Input data
            y : Optional[torch.Tensor]
                Target data
        """
        assert y is None, "y should not be provided for unsupervised models"
        if obs is not None:
            for i, layer in reversed (list(enumerate(self.layers))):
                if i == len(self.layers) - 1: # last layer
                    state[i]['x'] = obs.clone()
                if i > 0:
                    state[i-1]['x'] = layer.predict(state[i])
    

    def get_output(self, state:List[dict]):
        """
        | Takes the output from the feature extractor (bottom layer) and passes it through the classifier.

        Parameters
        ----------
            state : List[dict]
                List of layer state dicts, each containing 'x' and 'e'

        Returns
        -------
            torch.Tensor
                Output of the classifier
        """
        x = state[0]['x']
        out = self.classifier(x.detach())
        return out


    def reconstruct(self, obs:torch.Tensor, steps:int = None):
        """
        | Initialises the state of the model using the observation.
        | Runs inference without pinning the observation.
        | In theory should reconstruct the observation.

        Parameters
        ----------
            obs : torch.Tensor
                Input data
            steps : Optional[int]
                Number of steps to run inference for. Uses self.steps if not provided.

        Returns
        -------
            torch.Tensor
                Reconstructed observation
            List[dict]
                List of state dicts for each layer, each containing 'x' and 'e'
        """
        if steps is None:
            steps = self.steps
        
        state = self.init_state(obs)

        prev_vfe = None
        gamma = self.gamma
        for i in range(steps):
            temp = self.calc_temp(i, steps)
            self.step(state, temp=temp, gamma=gamma)
            vfe = self.vfe(state)
            if prev_vfe is not None and vfe < prev_vfe:
                gamma = gamma * 0.9
            prev_vfe = vfe
        
        out = state[-1]['x']

        return out, state