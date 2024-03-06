import torch
import torch.nn as nn
import torch.nn.functional as F

from pclib.nn.layers import FC
from pclib.utils.functional import format_y
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
        temp_k : float
            Temperature constant for inference
        inverted : bool
            Whether to switch observation and target in forward pass (WhittingtonBogacz2017)
        has_memory_vec : bool
            Whether to include a memory vector for predicting the top layer
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
            steps:int = 20, 
            bias:bool = True, 
            symmetric:bool = True, 
            actv_fn:callable = F.tanh, 
            d_actv_fn:callable = None, 
            gamma:float = 0.1, 
            x_decay: float = 0.0,
            temp_k:float = 1.0,
            inverted:bool = False,
            has_memory_vec:bool = False,
            device:torch.device = torch.device('cpu'), 
            dtype:torch.dtype = None
        ):
        super().__init__()

        self.factory_kwargs = {'actv_fn': actv_fn, 'd_actv_fn': d_actv_fn, 'gamma': gamma, 'has_bias': bias, 'symmetric': symmetric, 'x_decay': x_decay, 'dtype': dtype}
        self.sizes = sizes
        self.bias = bias
        self.symmetric = symmetric
        self.gamma = gamma
        self.x_decay = x_decay
        self.temp_k = temp_k
        self.steps = steps
        self.inverted = inverted
        self.has_memory_vec = has_memory_vec
        self.device = device
        self.dtype = dtype

        self.init_layers()
        self.register_buffer('epochs_trained', torch.tensor(0, dtype=torch.long))
        self.register_buffer('max_vfe', torch.tensor(-float('inf'), dtype=torch.float32))

    def __str__(self):
        base_str = super().__str__()

        custom_info = "\n  (params): \n" + \
            f"    sizes: {self.sizes}\n" + \
            f"    steps: {self.steps}\n" + \
            f"    bias: {self.bias}\n" + \
            f"    symmetric: {self.symmetric}\n" + \
            f"    actv_fn: {self.factory_kwargs['actv_fn'].__name__}\n" + \
            f"    gamma: {self.gamma}\n" + \
            f"    x_decay: {self.x_decay}\n" + \
            f"    temp_k: {self.temp_k}\n" + \
            f"    inverted: {self.inverted}\n" + \
            f"    has_memory_vec: {self.has_memory_vec}\n" + \
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
        for out_features in self.sizes:
            layers.append(FC(in_features, out_features, device=self.device, **self.factory_kwargs))
            in_features = out_features
        self.layers = nn.ModuleList(layers)

        if self.has_memory_vec:
            self.memory_vector = nn.Parameter(torch.empty(self.sizes[-1], device=self.device, dtype=self.dtype))
            fan_in = self.sizes[-1]
            bound = 1 / fan_in if fan_in > 0 else 0
            nn.init.uniform_(self.memory_vector, -bound, bound)
    
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
            learn_layer : Optional[int]
                If provided, only error from layer 'learn_layer-1' is included in the VFE calculation.

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

    def _init_xs(self, state:List[dict], obs:torch.Tensor = None, y:torch.Tensor = None, learn_layer:int = None):
        """
        | Initialises xs using either y or obs if provided.
        | If y is provided, then top down predictions are calculated and used as initial xs.
        | Else if obs is provided, then bottom up error propagations (pred=0) are calculated and used as initial xs.

        Parameters
        ----------
            state : List[dict]
                List of layer state dicts, each containing 'x' and 'e'
            obs : Optional[torch.Tensor]
                Input data
            y : Optional[torch.Tensor]
                Target data
            learn_layer : Optional[int]
                If provided, only initialises Xs for layers i where i <= learn_layer
        """
        if self.inverted:
            obs, y = y, obs

        with torch.no_grad():
            if y is not None and learn_layer is None:
                for i, layer in reversed(list(enumerate(self.layers))):
                    if i == len(self.layers) - 1: # last layer
                        state[i]['x'] = y.detach()
                    if i > 0:
                        pred = layer.predict(state[i])
                        state[i-1]['x'] = pred.detach()
                if obs is not None:
                    state[0]['x'] = obs.detach()
            elif obs is not None:
                for i, layer in enumerate(self.layers):
                    if learn_layer is not None and i > learn_layer:
                        break
                    if i == 0:
                        state[0]['x'] = obs.clone()
                    else:
                        x_below = state[i-1]['x'].detach()
                        # Found that using actv_fn here gives better results
                        state[i]['x'] = layer.actv_fn(layer.propagate(x_below))
            else:
                for i, layer in enumerate(self.layers):
                    if learn_layer is not None and i > learn_layer:
                        break
                    state[i]['x'] = torch.randn_like(state[i]['x'])

    def _init_es(self, state:List[dict], learn_layer:int = None):
        """
        | Calculates the initial errors for each layer.
        | Assumes that Xs have already been initialised.

        Parameters
        ----------
            state : List[dict]
                List of state dicts for each layer, each containing 'x' and 'e' tensors.
            learn_layer : Optional[int]
                If provided, only initialises errors for layers i where i < learn_layer
        """
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                if learn_layer is not None and i >= learn_layer:
                    break
                if i < len(self.layers) - 1:
                    pred = self.layers[i+1].predict(state[i+1])
                    layer.update_e(state[i], pred, temp=1.0)
                if i == len(self.layers) - 1 and self.has_memory_vec:
                    layer.update_e(state[i], self.memory_vector, temp=1.0)

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
            
        self._init_xs(state, obs, y, learn_layer=learn_layer)
        self._init_es(state, learn_layer=learn_layer)

        return state

    def to(self, device):
        self.device = device
        for layer in self.layers:
            layer.to(device)
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
        else:
            return state[-1]['x']

    def calc_temp(self, step_i:int, steps:int):
        """
        | Calculates the temperature for the current step.
        | Uses a geometric sequence to decay the temperature.
        | temp = self.temp_k * \alpha^{step_i}
        | where \alpha = (\frac{0.001}{1})^{\frac{1}{steps}}

        Parameters
        ----------
            step_i : int
                Current step
            steps : int
                Total number of steps

        Returns
        -------
            float
                Temperature for the current step
        """
        alpha = (0.001/1)**(1/steps)
        return self.temp_k * alpha**step_i

    def step(self, state:List[dict], gamma:torch.Tensor=None, pin_obs:bool = False, pin_target:bool = False, temp:float = None, learn_layer:int = None):
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
            temp : Optional[float]
                Temperature to use for update
            learn_layer : Optional[int]
                If provided, only layers i where i <= learn_layer are updated.
        """
        if self.inverted:
            pin_obs, pin_target = pin_target, pin_obs

        for i, layer in enumerate(self.layers):
            if learn_layer is not None and i > learn_layer:
                break
            if i > 0 or not pin_obs: # Don't update bottom x if pin_obs is True
                if i < len(self.layers) - 1 or not pin_target: # Don't update top x if pin_target is True
                    e_below = state[i-1]['e'] if i > 0 else None
                    layer.update_x(state[i], e_below, temp=temp, gamma=gamma)
        for i, layer in enumerate(self.layers):
            if learn_layer is not None and i >= learn_layer:
                break
            if i < len(self.layers) - 1:
                pred = self.layers[i+1].predict(state[i+1])
                layer.update_e(state[i], pred, temp=temp)
            elif i == len(self.layers) - 1 and self.has_memory_vec:
                layer.update_e(state[i], self.memory_vector, temp=temp)

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
            temp = self.calc_temp(i, steps)
            self.step(state, gamma, pin_obs, pin_target, temp, learn_layer)
            vfe = self.vfe(state, batch_reduction=None, learn_layer=learn_layer).detach()
            if prev_vfe is not None:
                mult = torch.ones_like(gamma)
                mult[vfe > prev_vfe] = 0.9
                gamma = gamma * mult
            prev_vfe = vfe
            
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






