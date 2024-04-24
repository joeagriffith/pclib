import torch
import torch.nn as nn
import torch.nn.functional as F

# from pclib.nn.layers import FCAtt as FC
from pclib.nn.layers import FC
from pclib.utils.functional import format_y, identity
from typing import List, Optional


class FCPCN(nn.Module):
    """
    | The standard PC model which stacks FC layers.
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
        momentum: float
            Momentum for value node updates
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
            momentum:float = 0.0,
            inverted:bool = False,
            has_top: bool = False,
            device:torch.device = torch.device('cpu'), 
            dtype:torch.dtype = None
        ):
        super().__init__()

        assert len(precisions) == len(sizes), "Precisions must be the same length as sizes"

        self.factory_kwargs = {'actv_fn': actv_fn, 'd_actv_fn': d_actv_fn, 'gamma': gamma, 'has_bias': bias, 'symmetric': symmetric, 'x_decay': x_decay, 'dropout': dropout, 'momentum': momentum, 'dtype': dtype}
        self.sizes = sizes
        self.precisions = precisions
        self.bias = bias
        self.symmetric = symmetric
        self.gamma = gamma
        self.x_decay = x_decay
        self.dropout = dropout
        self.momentum = momentum
        self.steps = steps
        self.inverted = inverted
        self.has_top = has_top
        self.device = device
        self.dtype = dtype

        self.init_layers()
        self.register_buffer('epochs_trained', torch.tensor(0, dtype=torch.long))
        self.register_buffer('max_vfe', torch.tensor(-float('inf'), dtype=torch.float32))

    def __str__(self):
        base_str = super().__str__()

        custom_info = "\n  (params): \n" + \
            f"    sizes: {self.sizes}\n" + \
            f"    precisions: {self.precisions}\n" + \
            f"    bias: {self.bias}\n" + \
            f"    symmetric: {self.symmetric}\n" + \
            f"    actv_fn: {self.factory_kwargs['actv_fn'].__name__}\n" + \
            f"    gamma: {self.gamma}\n" + \
            f"    x_decay: {self.x_decay}\n" + \
            f"    dropout: {self.dropout}\n" + \
            f"    momentum: {self.momentum}\n" + \
            f"    steps: {self.steps}\n" + \
            f"    inverted: {self.inverted}\n" + \
            f"    has_top: {self.has_top}\n" + \
            f"    device: {self.device}\n" + \
            f"    dtype: {self.factory_kwargs['dtype']}\n" + \
            f"    epochs_trained: {self.epochs_trained}\n" + \
            f"    max_vfe: {self.max_vfe}\n"
        
        string = base_str[:base_str.find('\n')] + custom_info + base_str[base_str.find('\n'):]
        
        return string
    
    @property
    def num_classes(self):
        """
        | Returns the number of classes.
        """
        if self.inverted:
            return self.sizes[0]
        else:
            return self.sizes[-1]


    def init_layers(self):
        """
        | Initialises self.layers based on input parameters.
        """
        layers = []
        in_features = None
        for i, out_features in enumerate(self.sizes):
            layers.append(FC(in_features, out_features, self.precisions[i], device=self.device, **self.factory_kwargs))
            in_features = out_features
        self.layers = nn.ModuleList(layers)

        if self.has_top:
            top_fac_kwargs = self.factory_kwargs.copy()
            # top_fac_kwargs['actv_fn'] = identity
            self.top = FC(self.sizes[-1], self.sizes[-1], self.precisions[-1], device=self.device, **top_fac_kwargs)
            self.top.weight.data = (self.top.weight.data - torch.diag(self.top.weight.data.diag())) * 0.01
    
    def inc_epochs(self, n:int=1):
        """
        | Increments the number of epochs trained by n.
        | Used by the trainer to keep track of the number of epochs trained.

        Parameters
        ----------
            n : int
                Number of epochs to increment by.
        """
        self.epochs_trained += n    

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
        # Reduce units for each layer
        if learn_layer is not None:
            vfe = [0.5 * state[learn_layer-1]['e'].square().sum(dim=[i for i in range(1, state[learn_layer-1]['e'].dim())])]
        else:
            vfe = [0.5 * state_i['e'].square().sum(dim=[i for i in range(1, state_i['e'].dim())]) for state_i in state]
        
        if normalise:
            vfe = [vfe_i / (vfe_i.detach() + 1e-6) for vfe_i in vfe]

        # Reduce layers
        vfe = sum(vfe)
        # Reduce batches
        if batch_reduction == 'sum':
            vfe = vfe.sum()
        elif batch_reduction == 'mean':
            vfe = vfe.mean()

        return vfe

    # def _init_xs(self, state:List[dict], obs:torch.Tensor = None, y:torch.Tensor = None, learn_layer:int = None):
    #     """
    #     | Initialises xs using either y or obs if provided.
    #     | If y is provided, then top down predictions are calculated and used as initial xs.
    #     | Else if obs is provided, then bottom up error propagations (pred=0) are calculated and used as initial xs.

    #     Parameters
    #     ----------
    #         state : List[dict]
    #             List of layer state dicts, each containing 'x' and 'e'
    #         obs : Optional[torch.Tensor]
    #             Input data
    #         y : Optional[torch.Tensor]
    #             Target data
    #         learn_layer : Optional[int]
    #             If provided, only initialises Xs for layers i where i <= learn_layer
    #     """
    #     if self.inverted:
    #         obs, y = y, obs

    #     if y is not None and learn_layer is None:
    #         for i, layer in reversed(list(enumerate(self.layers))):
    #             if i == len(self.layers) - 1: # last layer
    #                 state[i]['x'] = y.detach()
    #             if i > 0:
    #                 pred = layer.predict(state[i])
    #                 state[i-1]['x'] = pred.detach()
    #         if obs is not None:
    #             state[0]['x'] = obs.detach()
    #     elif obs is not None:
    #         for i, layer in enumerate(self.layers):
    #             if learn_layer is not None and i > learn_layer:
    #                 break
    #             if i == 0:
    #                 state[0]['x'] = obs.clone()
    #             else:
    #                 x_below = state[i-1]['x'].detach()
    #                 # Found that using actv_fn here gives better results
    #                 state[i]['x'] = layer.propagate(x_below)
    #                 # state[i]['x'] = torch.randn_like(state[i]['x']) * 0.1
    #     else:
    #         for i, layer in enumerate(self.layers):
    #             if learn_layer is not None and i > learn_layer:
    #                 break
    #             state[i]['x'] = torch.randn_like(state[i]['x']) * 0.01

    # def _init_es(self, state:List[dict], learn_layer:int = None):
    #     """
    #     | Calculates the initial errors for each layer.
    #     | Assumes that Xs have already been initialised.

    #     Parameters
    #     ----------
    #         state : List[dict]
    #             List of state dicts for each layer, each containing 'x' and 'e' tensors.
    #         learn_layer : Optional[int]
    #             If provided, only initialises errors for layers i where i < learn_layer
    #     """
    #     for i, layer in enumerate(self.layers):
    #         # Dont update e if >= learn_layer, unless top layer
    #         if learn_layer is not None:
    #             if learn_layer < len(self.layers) - 1 or not self.has_top:
    #                 if i >= learn_layer:
    #                     break
    #         if i < len(self.layers) - 1:
    #             pred = self.layers[i+1].predict(state[i+1])
    #         elif self.has_top:
    #             pred = self.top.predict(state[i])
    #         else:
    #             continue
    #         layer.update_e(state[i], pred)

    def init_state(self, obs:torch.Tensor = None, y:torch.Tensor = None, b_size:int = None, learn_layer:int = None):
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
            state[0]['x'] = obs.clone()
        if y is not None:
            state[-1]['x'] = y.clone()
    
        # # Alternative Initialisation
        # self._init_xs(state, obs, y, learn_layer=learn_layer)
        # self._init_es(state, learn_layer=learn_layer)
        # Update Es
        for i, layer in enumerate(self.layers):
            # Dont update e if >= learn_layer, unless top layer
            if i < len(self.layers) - 1:
                pred = self.layers[i+1].predict(state[i+1])
            else:
                continue
            layer.update_e(state[i], pred)

        return state

    def to(self, device):
        self.device = device
        for layer in self.layers:
            layer.to(device)
        if self.has_top:
            self.top.to(device)
        return self

    def get_output(self, state:List[dict]):
        """
        | Gets the output of the model.

        Parameters
        ----------
            state : List[dict] 
                List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)

        Returns
        -------
            torch.Tensor
                Output of the model
        """
        if self.inverted:
            return state[0]['x']
            # return torch.cat([state_i['x'] for state_i in state[:-1]], dim=1)
        else:
            return state[-1]['x']
            # return torch.cat([state_i['x'] for state_i in state[1:]], dim=1)


    def update_gamma(self, state, gamma, prev_vfe=None):
        """
        | Decays gamma based on the change in VFE.
        | If VFE increases, gamma is decayed, else it is increased.

        Parameters
        ----------
            state : List[dict]
                List of layer state dicts, each containing 'x' and 'e'
            gamma : torch.Tensor
                Current gamma
            prev_vfe : Optional[torch.Tensor]
                Previous VFE
        
        Returns
        -------
            torch.Tensor
                Updated gamma
            torch.Tensor
                Current VFE
        """
        vfe = self.vfe(state, batch_reduction=None)
        if prev_vfe is None:
            return gamma, vfe
        else:
            mult = torch.where(vfe < prev_vfe, 1.0, 0.9)
            return gamma * mult, vfe
    
    def step(self, state:List[dict], gamma:torch.Tensor=None, pin_obs:bool = False, pin_target:bool = False, learn_layer:int = None):
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
        if self.inverted:
            pin_obs, pin_target = pin_target, pin_obs

        # Update Xs
        for i, layer in enumerate(self.layers):
            if learn_layer is not None and i > learn_layer:
                break
            if i > 0 or not pin_obs: # Don't update bottom x if pin_obs is True
                if i < len(self.layers) - 1 or not pin_target: # Don't update top x if pin_target is True
                    e_below = state[i-1]['e'] if i > 0 else None
                    layer.update_x(state[i], e_below, gamma=gamma)
        
        # Update Es
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
            layer.update_e(state[i], pred)

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
        """
        if steps is None:
            steps = self.steps

        state = self.init_state(obs, y, learn_layer=learn_layer)

        prev_vfe = None
        gamma = torch.ones(state[0]['x'].shape[0]).to(self.device) * self.gamma
        for i in range(steps):
            self.step(state, gamma, pin_obs, pin_target, learn_layer)
            with torch.no_grad():
                gamma, prev_vfe = self.update_gamma(state, gamma, prev_vfe)
            
        out = self.get_output(state)
            
        return out, state

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
    
    def classify(self, obs:torch.Tensor, steps:int=None):
        """
        | Performs inference on the obs once with each possible target pinned.
        | Returns the target with the lowest VFE.
        
        Parameters
        ----------
            obs : torch.Tensor
                Input data
            steps : Optional[int]
                Number of steps to run inference for. Uses self.steps if not provided.

        Returns
        -------
            torch.Tensor
                Argmax(dim=1) output of the classifier
        """
        assert len(obs.shape) == 2, f"Input must be 2D, got {len(obs.shape)}D"
        if self.inverted:
            num_classes = self.sizes[0]
        else:
            num_classes = self.sizes[-1]
        if steps is None:
            steps = self.steps

        vfes = torch.zeros(obs.shape[0], num_classes, device=self.device)
        for target in range(num_classes):
            targets = torch.full((obs.shape[0],), target, device=self.device, dtype=torch.long)
            y = format_y(targets, num_classes)
            _, state = self.forward(obs, y, pin_obs=True, pin_target=True, steps=steps)
            vfes[:, target] = self.vfe(state, batch_reduction=None)
        
        return vfes.argmin(dim=1)

    def reconstruct(self, obs:torch.Tensor = None, y:torch.Tensor = None, steps:int = None, learn_layer:int = None, beta:float = 0.1):

        """
        | Performs inference phase of the model.
        
        Parameters
            obs : Optional[torch.Tensor]
                Input data
            y : Optional[torch.Tensor]
                Target data
            steps : Optional[int]
                Number of steps to run inference for. Uses self.steps if not provided.
            learn_layer : Optional[int]
                If provided, only layers i where i <= learn_layer are updated.

        Returns
        -------
            torch.Tensor
                Final prediction of input data
            List[dict]
                List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)
        """
        _, state = self.forward(obs, y, pin_obs=True, pin_target=y is not None, steps=steps, learn_layer=learn_layer)
        if len(self.layers) == 1:
            pred = self.top.predict(state[0])
        else:
            pred = self.layers[1].predict(state[1])
        
        out = (1-beta) * state[0]['x'] + beta * pred
        # self.step(state)
        # out = state[0]['x']
        return out, state