import torch
import torch.nn as nn
import torch.nn.functional as F

from pclib.nn.layers import FC
from pclib.nn.models import FCClassifierUs
from typing import List

class FCAM(FCClassifierUs):
    """
    | A model of associative memory, based on 'ASSOCIATIVE MEMORIES VIA PREDICTIVE CODING' 2021, Salvatori et al.
    | Stores data by learning to predict it, minimising VFE.

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
        x_decay : float
            Decay rate for x
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
            x_decay:float = 0.0,
            temp_k:float = 1.0,
            device:torch.device = torch.device('cpu'), 
            dtype:torch.dtype = None
        ):
        super().__init__(in_features, num_classes, hidden_sizes, steps, bias, symmetric, actv_fn, d_actv_fn, gamma, x_decay, temp_k, device, dtype)

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

        self.memory_vector = nn.Parameter(torch.empty(self.hidden_sizes[-1], device=self.device, dtype=self.dtype))
        fan_in = self.hidden_sizes[-1]
        bound = 1 / fan_in if fan_in > 0 else 0
        nn.init.uniform_(self.memory_vector, -bound, bound)

    def to(self, device):
        self.device = device
        for layer in self.layers:
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
                if i < len(self.layers) - 1 or y is None: # Don't update top x if y is given
                    e_below = state[i-1]['e'] if i > 0 else None
                    layer.update_x(state[i], e_below, temp=temp, gamma=gamma)
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                pred = self.layers[i+1].predict(state[i+1])
                layer.update_e(state[i], pred, temp=temp)
            elif i == len(self.layers) - 1:
                layer.update_e(state[i], self.memory_vector, temp=temp)

    def forward(self, obs:torch.Tensor = None, steps:int = None):
        """
        | Performs inference phase of the model. 
        | Uses self.classifier to get output.

        Parameters
        ----------
            obs : Optional[torch.Tensor]
                Input data
            steps : Optional[int]
                Number of steps to run inference for. Uses self.steps if not provided.

        Returns
        -------
            torch.Tensor
                Output of the model
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
            self.step(state, obs, temp=temp, gamma=gamma)
            vfe = self.vfe(state)
            if prev_vfe is not None and vfe < prev_vfe:
                gamma = gamma * 0.9
            prev_vfe = vfe
            
        out = self.get_output(state)
            
        return out, state



    def get_output(self, state:List[dict]):
        """
        Takes the output of the last layer from the feature extractor and passes it through the classifier.
        Returns the output of the classifier.

        Parameters
        ----------
            state : List[dict]
                List of layer state dicts, each containing 'x' and 'e'

        Returns
        -------
            torch.Tensor
                Output of the classifier
        """
        out = self.layers[1].predict(state[1])
        return out

    def assert_grads(model, state:List[dict]):
        """
        | Uses assertions to compare current gradients of each layer to manually calculated gradients.
        | Only useful if using autograd=True in training, otherwise comparing manual grads to themselves.

        Parameters
        ----------
            state : List[dict]
                List of state dicts for each layer, each containing 'x' and 'e'
        """
        for i, layer in enumerate(model.layers):
            if i > 0:
                layer.assert_grads(state[i], state[i-1]['e'])                
            

    def reconstruct(self, obs:torch.Tensor, steps:int = None):
        raise(NotImplementedError, "Use .forward() instead.")

    
    def generate(self, y:torch.Tensor, steps:int = None):
        """
        | Not implemented as one cannot generate an input without a target, and this model does not pin targets.
        """
        raise(NotImplementedError)