from pclib.nn.layers import Conv2d, FC
from pclib.nn.models import ConvClassifierUs
from pclib.utils.functional import format_y
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.grad import conv2d_input, conv2d_weight
from typing import List, Optional

class ConvAM(ConvClassifierUs):
    """
    | A model of associative memory, inspired by 'ASSOCIATIVE MEMORIES VIA PREDICTIVE CODING' 2021, Salvatori et al.
    | Stores data by learning to predict it, minimising VFE.

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
        temp_k : float
            Temperature constant for inference
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
            temp_k:float = 1.0,
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
            temp_k=temp_k,
            device=device,
            dtype=dtype,
        )

    def init_layers(self):
        """
        | Initialises the layers of the network.
        """
        layers = []
        layers.append(Conv2d(None,           (3, 32, 32),                  **self.factory_kwargs))
        layers.append(Conv2d((3, 32, 32),    (64, 32, 32),  3, 1, 1, **self.factory_kwargs))
        layers.append(Conv2d((64, 32, 32),   (64, 16, 16),  3, 2, 1, **self.factory_kwargs))
        layers.append(Conv2d((64, 16, 16),   (64, 16, 16),  3, 1, 1, **self.factory_kwargs))
        layers.append(Conv2d((64, 16, 16),   (128, 8, 8),    3, 2, 1, **self.factory_kwargs))
        layers.append(Conv2d((128, 8, 8),    (128, 8, 8),    3, 1, 1, **self.factory_kwargs))
        layers.append(Conv2d((128, 8, 8),    (256, 4, 4),    3, 2, 1, **self.factory_kwargs))
        layers.append(Conv2d((256, 4, 4),    (256, 4, 4),    3, 1, 1, **self.factory_kwargs))
        layers.append(Conv2d((256, 4, 4),    (512, 2, 2),    3, 2, 1, **self.factory_kwargs))
        layers.append(Conv2d((512, 2, 2),    (512, 2, 2),    3, 1, 1, **self.factory_kwargs))
        layers.append(Conv2d((512, 2, 2),    (512, 1, 1),    3, 2, 1, **self.factory_kwargs))
        layers.append(Conv2d((512, 1, 1),    (512, 1, 1),    3, 1, 1, **self.factory_kwargs))
        self.layers = nn.ModuleList(layers)

        self.memory_vector = nn.Parameter(torch.empty(256, 1, 1, device=self.device, dtype=self.dtype))
        nn.init.normal_(self.memory_vector, mean=0.0, std=0.01)

    def to(self, device):
        self.device = device
        for layer in self.layers:
            layer.to(device)
        self.memory_vector = nn.Parameter(self.memory_vector.to(device))
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
        out = self.layers[1].predict(state[1])
        return out
        

    def forward(self, obs:torch.Tensor = None, steps:int = None, learn_on_step:bool = False):
        """
        | Performs inference for the network.

        Parameters
        ----------
            obs : Optional[torch.Tensor]
                Input data
            steps : Optional[int]
                Number of steps to run inference for
            learn_on_step : bool
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

        prev_vfe = None
        gamma = self.gamma
        for i in range(steps):
            temp = self.calc_temp(i, steps)
            self.step(state, obs, temp, gamma=gamma)
            vfe = self.vfe(state)
            if learn_on_step:
                vfe.backward()
            if prev_vfe is not None and vfe < prev_vfe:
                gamma = gamma * 0.9
            prev_vfe = vfe
            
        out = self.get_output(state)
            
        return out, state

    def reconstruct(self, obs:torch.Tensor, steps:int = None):
        raise(NotImplementedError, "Use forward (call) instead.")

    
    def generate(self, y, steps=None):
        raise(NotImplementedError, "Use forward (call) instead.")

