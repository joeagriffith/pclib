import torch
import torch.nn as nn
import torch.nn.functional as F

# from pclib.nn.layers import FCAtt as FC
from pclib.nn.layers import DFC
from pclib.utils.functional import format_y, identity
from pclib.nn.models import FCPCN
from typing import List, Optional

from torchviz import make_dot
import time


class FCDPCN(FCPCN):
    """
    | A PC model which uses MLPs for prediction, and backpropagation for value node updates.
    | Predictions flow from targets to inputs (top-down).
    | Heavily customisable, however, the default settings usually give best results.

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
            Decay constant for x
        dropout : float
            Dropout rate for predictions
        inverted : bool
            Whether to switch observation and target in forward pass (WhittingtonBogacz2017)
        has_top : bool
            Whether to include a recurrent layer for top prediction
        device : torch.device
            Device to run on
        dtype : torch.dtype
            Data type to use
    """
    __constants__ = ['in_features', 'num_classes']
    in_features: int
    num_classes: int

    def __init__(
            self, 
            sizes:List[int] = [], 
            precisions:List[float] = [],
            steps:int = 20, 
            bias:bool = True, 
            symmetric:bool = True, 
            actv_fn:callable = F.tanh, 
            d_actv_fn:callable = None, 
            gamma:float = 0.1, 
            x_decay: float = 0.0,
            dropout:float = 0.0,
            inverted:bool = False,
            has_top: bool = False,
            device:torch.device = torch.device('cpu'), 
            dtype:torch.dtype = None
        ):
        super().__init__(
            sizes,
            precisions,
            steps,
            bias,
            symmetric,
            actv_fn,
            d_actv_fn,
            gamma,
            x_decay,
            dropout,
            inverted,
            has_top,
            device,
            dtype
        )
    
    def init_layers(self):
        """
        | Initialises self.layers based on input parameters.
        """
        layers = []
        in_features = None
        for i, out_features in enumerate(self.sizes):
            layers.append(DFC(in_features, out_features, self.precisions[i], device=self.device, **self.factory_kwargs))
            in_features = out_features
        self.layers = nn.ModuleList(layers)

        if self.has_top:
            raise NotImplementedError("Top layer not implemented yet.")
            top_fac_kwargs = self.factory_kwargs.copy()
            # top_fac_kwargs['actv_fn'] = identity
            self.top = DFC(self.sizes[-1], self.sizes[-1], self.precisions[-1], device=self.device, **top_fac_kwargs)
            self.top.weight.data = (self.top.weight.data - torch.diag(self.top.weight.data.diag())) * 0.01

    def init_state(self, obs:torch.Tensor = None, y:torch.Tensor = None, b_size:int = None, pin_obs:bool = False, pin_target:bool=False, learn_layer:int = None):
        """
        | Initialises the state of the model. Xs are calculated using _init_xs().
        | Atleast one of the arguments obs/y/b_size must be provided.

        Parameters
        ----------
            obs : Optional[torch.Tensor]
                Input data
            y : Optional[torch.Tensor]
                Target data
            b_size : Optional[int]
                Batch size
            learn_layer : Optional[int]
                If provided, only initialises Xs for layers i where i <= learn_layer
                and only initialise Es for layers i where i < learn_layer

        Returns
        -------
            List[dict]
                List of state dicts for each layer, each containing 'x' and 'e'
        """
        if obs is not None:
            b_size = obs.shape[0]
        elif y is not None:
            b_size = y.shape[0]
        elif b_size is not None:
            pass
        else:
            raise ValueError("Either obs/y/b_size must be provided to init_state.")
        state = []
        for layer in self.layers:
            state.append(layer.init_state(b_size))
            
        if self.inverted:
            obs, y = y, obs
        if obs is not None:
            state[0]['x'] = obs.clone().requires_grad_(pin_obs)
        if y is not None:
            state[-1]['x'] = y.clone().requires_grad_(pin_target)

        optimiser = torch.optim.SGD([state[i]['x'] for i in range(len(state)) if state[i]['x'].requires_grad], lr=self.gamma, momentum=0.9)
    
        return state, optimiser


    def vfe(self, state:List[dict], batch_reduction:str = 'mean', learn_layer:int = None, normalise=False):
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
            learn_layer : Optional[int]
                If provided, only error from layer 'learn_layer-1' is included in the VFE calculation.
            normalise : bool
                Whether to normalise the VFE by the sum of the squared activations.

        Returns
        -------
            torch.Tensor
                VFE of the model
        """
        errors = []
        for i, layer in enumerate(self.layers):
            # Dont update e if >= learn_layer, unless top layer
            if learn_layer is not None:
                if learn_layer < len(self.layers) - 1 or not self.has_top:
                    if i >= learn_layer:
                        break
            if i < len(self.layers) - 1:
                pred = self.layers[i+1].predict(state[i+1])
            elif self.has_top:
                pred = self.top.predict(state[i])
            else:
                continue
            # errors.append(layer.squared_errors(state[i], pred))
            errors.append(layer.squared_errors(state[i], pred).sum())
        
        return sum(errors)

        # # Reduce units for each layer
        # vfe = [0.5 * e.sum(dim=[i for i in range(1, e.dim())]) for e in errors[:learn_layer]]
        
        # if normalise:
        #     vfe = [vfe_i / (vfe_i.detach() + 1e-6) for vfe_i in vfe]

        # # Reduce layers
        # vfe = sum(vfe)
        # # Reduce batches
        # if batch_reduction == 'sum':
        #     vfe = vfe.sum()
        # elif batch_reduction == 'mean':
        #     vfe = vfe.mean()

        # return vfe


    def step(self, state:List[dict], optimiser:torch.optim.Optimizer, gamma:torch.Tensor):
        """
        | Performs one step of inference. Updates Xs first, then Es.
        | Both are updated from bottom to top.

        Parameters
        ----------
            state : List[dict]
                List of layer state dicts, each containing 'x' and 'e; (and 'eps' for FCPW)
            gamma : Optional[torch.Tensor]
                Step size for x updates, size = (batch_size,)
            pin_obs : bool
                Whether to pin the observation
            pin_target : bool
                Whether to pin the target
            learn_layer : Optional[int]
                If provided, only layers i where i <= learn_layer are updated.
        """
        optimiser.zero_grad()
        vfe = (self.vfe(state, batch_reduction='none') * gamma.unsqueeze(1)).sum() 
        vfe.backward()
        # optimiser.step()

    def forward(self, obs:torch.Tensor = None, y:torch.Tensor = None, pin_obs:bool = False, pin_target:bool = False, steps:int = None, learn_layer:int = None):

        """
        | Performs inference phase of the model.
        
        Parameters
            obs : Optional[torch.Tensor]
                Input data
            y : Optional[torch.Tensor]
                Target data
            pin_obs : bool
                Whether to pin the observation
            pin_target : bool
                Whether to pin the target
            steps : Optional[int]
                Number of steps to run inference for. Uses self.steps if not provided.
            learn_layer : Optional[int]
                If provided, only layers i where i <= learn_layer are updated.

        Returns
        -------
            torch.Tensor
                Output of the model
            List[dict]
                List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)
            torch.utils.optim.Optimizer
                Optimiser used for inference
        """
        if steps is None:
            steps = self.steps

        state, optimiser = self.init_state(obs, y, learn_layer=learn_layer)

        prev_vfe = None
        gamma = torch.ones(state[0]['x'].shape[0]).to(self.device)
        for i in range(steps):
            self.step(state, optimiser, gamma)
            with torch.no_grad():
                gamma, prev_vfe = self.update_gamma(state, gamma, prev_vfe)
            
        out = self.get_output(state)
            
        return out, state, optimiser