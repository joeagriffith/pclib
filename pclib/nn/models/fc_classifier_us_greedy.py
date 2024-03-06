import torch
import torch.nn as nn
import torch.nn.functional as F

from pclib.nn.layers import FC
from pclib.nn.models import FCClassifierUs
from typing import List

# Based on Whittington and Bogacz 2017, but with targets predicting inputs
class FCClassifierUsGreedy(FCClassifierUs):
    """
    | An Unsupervised version of FCClassifier.
    | Learns a feature extractor (self.layers) via unsupervised learning.
    | Also learns an mlp classifier (self.classifier) which takes the output of self.layers as input, via supervised learning.

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
            d_actv_fn:callable = None, 
            gamma:float = 0.1, 
            x_decay:float = 0.0,
            temp_k:float = 1.0,
            device:torch.device = torch.device('cpu'), 
            dtype:torch.dtype = None
        ):
        super().__init__(in_features, num_classes, hidden_sizes, steps, bias, symmetric, actv_fn, d_actv_fn, gamma, x_decay, temp_k, device, dtype)

    
    def vfe(self, state:List[dict], batch_reduction:str = 'mean', unit_reduction:str = 'sum', learn_layer:int = None):
        """
        | Calculates the Variational Free Energy (VFE) of the model.
        | This is the sum of the squared prediction errors of each layer.
        | how batches and units are reduced is controlled by batch_reduction and unit_reduction.

        Parameters
        ----------
            state : List[dict] 
                List of layer state dicts, each containing 'x' and 'e'
            batch_reduction : str 
                How to reduce over batches ['sum', 'mean', None], default='mean'
            unit_reduction : str
                How to reduce over units ['sum', 'mean'], default='sum'

        Returns
        -------
            torch.Tensor
                VFE of the model (scalar)
        """
        # Reduce units for each layer
        if unit_reduction == 'sum':
            if learn_layer is not None:
                vfe = [0.5 * state[learn_layer-1]['e'].square().sum(dim=[i for i in range(1, state[learn_layer-1]['e'].dim())])]
            else:
                vfe = [0.5 * state_i['e'].square().sum(dim=[i for i in range(1, state_i['e'].dim())]) for state_i in state]
        elif unit_reduction =='mean':
            if learn_layer is not None:
                vfe = [0.5 * state[learn_layer-1]['e'].square().mean(dim=[i for i in range(1, state[learn_layer-1]['e'].dim())])]
            else:
                vfe = [0.5 * state_i['e'].square().mean(dim=[i for i in range(1, state_i['e'].dim())]) for state_i in state]
        # Reduce layers
        vfe = sum(vfe)
        # Reduce batches
        if batch_reduction == 'sum':
            vfe = vfe.sum()
        elif batch_reduction == 'mean':
            vfe = vfe.mean()

        return vfe


    def _init_es(self, state:List[dict], learn_layer:int = None):
        """
        | Calculates the initial errors for each layer.
        | Assumes that Xs have already been initialised.

        Parameters
        ----------
            state : List[dict]
                List of state dicts for each layer, each containing 'x' and 'e' tensors.
        """
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                if learn_layer is not None and i >= learn_layer:
                    break
                if i < len(self.layers) - 1:
                    pred = self.layers[i+1].predict(state[i+1])
                    layer.update_e(state[i], pred, temp=1.0)

    def init_state(self, obs:torch.Tensor = None, y:torch.Tensor = None, learn_layer:int = None):
        """
        | Initialises the state of the model. Xs are calculated using _init_xs().

        Parameters
        ----------
            obs : Optional[torch.Tensor]
                Input data
            y : Optional[torch.Tensor]
                Target data
            learn_layer : Optional[int]
                If provided, only initialises errors for layers i where i < learn_layer

        Returns
        -------
            List[dict]
                List of state dicts for each layer, each containing 'x' and 'e'
        """
        if obs is not None:
            b_size = obs.shape[0]
        elif y is not None:
            b_size = y.shape[0]
        else:
            raise ValueError("Either obs or y must be provided to init_state.")
        state = []
        for layer in self.layers:
            state.append(layer.init_state(b_size))
            
        self._init_xs(state, obs, y)
        self._init_es(state, learn_layer=learn_layer)

        return state

    def step(self, state:List[dict], pin_obs:bool = False, temp:float = None, gamma:float = None, learn_layer:int = None):
        """
        | Performs one step of inference. Updates Xs first, then Es.
        | Both are updated from bottom to top.

        Parameters
        ----------
            state : List[dict]
                List of layer state dicts, each containing 'x' and 'e; (and 'eps' for FCPW)
            obs : Optional[torch.Tensor]
                Input data
            pin_obs : bool
                Whether to pin the observation or not
            temp : Optional[float]
                Temperature to use for update
            gamma : Optional[float]
                Step size for update
            learn_layer : Optional[int]
                If provided, only updates Xs for layers i where i <= learn_layer, and Es for layers i where i < learn_layer.
        """
        for i, layer in enumerate(self.layers):
            if learn_layer is not None and i > learn_layer:
                break
            if i > 0 or not pin_obs: # Don't update bottom x if pinned
                e_below = state[i-1]['e'] if i > 0 else None
                layer.update_x(state[i], e_below, temp=temp, gamma=gamma)
        for i, layer in enumerate(self.layers):
            if learn_layer is not None and i >= learn_layer:
                break
            if i < len(self.layers) - 1:
                pred = self.layers[i+1].predict(state[i+1])
                layer.update_e(state[i], pred, temp=temp)


    def forward(self, obs:torch.Tensor = None, pin_obs:bool = False, steps:int = None, learn_layer:int = None):
        """
        | Performs inference phase of the model. 
        | Uses self.classifier to get output.

        Parameters
        ----------
            obs : Optional[torch.Tensor]
                Input data
            pin_obs : bool
                Whether to pin the observation or not
            steps : Optional[int]
                Number of steps to run inference for. Uses self.steps if not provided.
            learn_layer : Optional[int]
                If provided, only layers i where i <= learn_layer are updated.

        Returns
        -------
            torch.Tensor
                Output of the model
            List[dict]
                List of layer state dicts, each containing 'x' and 'e'
        """
        if steps is None:
            steps = self.steps

        state = self.init_state(obs, learn_layer=learn_layer)

        prev_vfe = None
        gamma = self.gamma
        for i in range(steps):
            temp = self.calc_temp(i, steps)
            self.step(state, pin_obs=pin_obs, temp=temp, gamma=gamma, learn_layer=learn_layer)
            vfe = self.vfe(state, learn_layer=learn_layer)
            if prev_vfe is not None and vfe < prev_vfe:
                gamma = gamma * 0.9
            prev_vfe = vfe
            
        out = self.get_output(state)
            
        return out, state