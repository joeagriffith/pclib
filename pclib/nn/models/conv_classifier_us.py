from pclib.nn.layers import Conv2d, FC
from pclib.nn.models import ConvClassifier
from pclib.utils.functional import format_y
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.grad import conv2d_input, conv2d_weight
from typing import List, Optional

# Based on Whittington and Bogacz 2017
class ConvClassifierUs(ConvClassifier):
    """
    | Similar to the ConvClassifer, except it learns an unsupervised feature extractor, and a separate backprop trained classifier.
    | This network is not currently customisable, but requires altering the init_layers() code to change the architecture.

    Parameters
    ----------
        steps : int
            Number of steps to run the network for.
        bias : bool
            Whether to include bias terms in the network.
        symmetric : bool
            Whether to use symmetric weights. 
        actv_fn : callable
            Activation function to use in the network.
        d_actv_fn : Optional[callable]
            Derivative of the activation function to use in the network.
        gamma : float
            step size for x updates
        device : torch.device
            Device to run the network on.
        dtype : torch.dtype
            Data type to use for network parameters.
    """
    def __init__(
            self, 
            steps:int = 20, 
            bias:bool = True, 
            symmetric:bool = True, 
            actv_fn:callable = F.relu, 
            d_actv_fn:callable = None, 
            gamma:float = 0.1, 
            device:torch.device = torch.device('cpu'), 
            dtype:torch.dtype = None
        ):
        super().__init__(
            steps=steps,
            bias=bias,
            symmetric=symmetric,
            actv_fn=actv_fn,
            d_actv_fn=d_actv_fn,
            gamma=gamma,
            device=device,
            dtype=dtype,
        )

    def init_layers(self):
        """
        | Initialises the layers of the network.
        """
        layers = []
        layers.append(Conv2d(None,          (1, 32, 32),                  **self.factory_kwargs))
        layers.append(Conv2d((1, 32, 32),   (32, 16, 16),  5, 2, 2, **self.factory_kwargs))
        layers.append(Conv2d((32, 16, 16),  (64, 8, 8),    3, 2, 1, **self.factory_kwargs))
        layers.append(Conv2d((64, 8, 8),    (128, 4, 4),    3, 2, 1, **self.factory_kwargs))
        layers.append(Conv2d((128, 4, 4),    (256, 2, 2),    3, 2, 1, **self.factory_kwargs))
        layers.append(Conv2d((256, 2, 2),    (256, 1, 1),    3, 2, 1, **self.factory_kwargs))
        self.layers = nn.ModuleList(layers)

        self.classifier = nn.Sequential(
            nn.Linear(self.layers[-1].shape[0], 200, bias=True, device=self.device, dtype=self.factory_kwargs['dtype']),
            nn.ReLU(),
            nn.Linear(200, self.num_classes, bias=False, device=self.device, dtype=self.factory_kwargs['dtype']),
        )


    def to(self, device):
        self.device = device
        for layer in self.layers:
            layer.to(device)
        for layer in self.classifier:
            layer.to(device)
        return self

    def get_output(self, state:List[dict]):
        """
        | Returns the output of the network.

        Parameters
        ----------
            state : List[dict]
                List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)

        Returns
        -------
            torch.Tensor
                Output of the network
        """
        x = state[-1]['x']
        out = self.classifier(x.detach().flatten(1))
        return out
        

    def forward(self, obs:torch.Tensor = None, steps:int = None, back_on_step:bool = False):
        """
        | Performs inference for the network.

        Parameters
        ----------
            obs : Optional[torch.Tensor]
                Input data
            steps : Optional[int]
                Number of steps to run inference for
            back_on_step : bool
                Whether to backpropagate on each step. Default False.
        
        Returns
        -------
            torch.Tensor
                Output of the network
            List[dict]
                List of layer state dicts, each containing 'x' and 'e'
        """
        if steps is None:
            steps = self.steps

        state = self.init_state(obs)

        for i in range(steps):
            temp = self.calc_temp(i, steps)
            self.step(state, obs, temp)
            if back_on_step:
                self.vfe(state).backward()
            
        out = self.get_output(state)
            
        return out, state

    def classify(self, obs:torch.Tensor, steps:int = None):
        """
        | Classifies the input obs.

        Parameters
        ----------
            obs : torch.Tensor
                Input data
            steps : Optional[int]
                Number of steps to run inference for
        
        Returns
        -------
            torch.Tensor
                Predicted class
        """
        return self.forward(obs, steps)[0].argmax(dim=1)


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

        Returns:
            torch.Tensor
                Reconstructed observation
            List[dict]
                List of states for each layer, each is a dict containing 'x' and 'e'.
        """
        if steps is None:
            steps = self.steps
        
        state = self.init_state(obs)

        for i in range(steps):
            temp = self.calc_temp(i, steps)
            self.step(state, temp=temp)
        
        out = state[0]['x']

        return out, state

    
    def generate(self, y, steps=None):
        """
        | Not implemented as one cannot generate an input without a target, and this model does not pin targets.
        """
        raise(NotImplementedError)

