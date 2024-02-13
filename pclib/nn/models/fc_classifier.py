import torch
import torch.nn as nn
import torch.nn.functional as F

from pclib.nn.layers import FC
from pclib.utils.functional import format_y
from typing import List, Optional


class FCClassifier(nn.Module):
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
        temp_k : float
            Temperature constant for inference
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
            in_features:int, 
            num_classes:int, 
            hidden_sizes:List[int] = [], 
            steps:int = 20, 
            bias:bool = True, 
            symmetric:bool = True, 
            actv_fn:callable = F.tanh, 
            d_actv_fn:callable = None, 
            gamma:float = 0.1, 
            temp_k:float = 1.0,
            device:torch.device = torch.device('cpu'), 
            dtype:torch.dtype = None
        ):
        super().__init__()

        self.factory_kwargs = {'actv_fn': actv_fn, 'd_actv_fn': d_actv_fn, 'gamma': gamma, 'has_bias': bias, 'symmetric': symmetric, 'dtype': dtype}
        self.in_features = in_features
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.bias = bias
        self.symmetric = symmetric
        self.gamma = gamma
        self.temp_k = temp_k
        self.steps = steps
        self.device = device
        self.dtype = dtype

        self.init_layers()
        self.register_buffer('epochs_trained', torch.tensor(0, dtype=torch.long))
        self.register_buffer('max_vfe', torch.tensor(-float('inf'), dtype=torch.float32))

    def __str__(self):
        base_str = super().__str__()

        custom_info = "\n  (params): \n" + \
            f"    in_features: {self.in_features}\n" + \
            f"    num_classes: {self.num_classes}\n" + \
            f"    hidden_sizes: {self.hidden_sizes}\n" + \
            f"    steps: {self.steps}\n" + \
            f"    bias: {self.bias}\n" + \
            f"    symmetric: {self.symmetric}\n" + \
            f"    actv_fn: {self.factory_kwargs['actv_fn'].__name__}\n" + \
            f"    gamma: {self.gamma}\n" + \
            f"    device: {self.device}\n" + \
            f"    dtype: {self.factory_kwargs['dtype']}\n" + \
            f"    epochs_trained: {self.epochs_trained}\n" + \
            f"    max_vfe: {self.max_vfe}\n"
        
        string = base_str[:base_str.find('\n')] + custom_info + base_str[base_str.find('\n'):]
        
        return string


    def init_layers(self):
        """
        | Initialises self.layers based on input parameters.
        """
        layers = []
        in_features = None
        for out_features in [self.in_features] + self.hidden_sizes + [self.num_classes]:
            layers.append(FC(in_features, out_features, device=self.device, **self.factory_kwargs))
            in_features = out_features
        self.layers = nn.ModuleList(layers)
    
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

    def vfe(self, state:List[dict], batch_reduction:str = 'mean', unit_reduction:str = 'sum'):
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
            vfe = [-0.5 * state_i['e'].square().sum(dim=[i for i in range(1, state_i['e'].dim())]) for state_i in state]
        elif unit_reduction =='mean':
            vfe = [-0.5 * state_i['e'].square().mean(dim=[i for i in range(1, state_i['e'].dim())]) for state_i in state]
        # Reduce layers
        vfe = sum(vfe)
        # Reduce batches
        if batch_reduction == 'sum':
            vfe = vfe.sum()
        elif batch_reduction == 'mean':
            vfe = vfe.mean()

        return vfe

    def _init_xs(self, state:List[dict], obs:torch.Tensor = None, y:torch.Tensor = None):
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
        """
        with torch.no_grad():
            if y is not None:
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
                    if i == 0:
                        state[0]['x'] = obs.clone()
                    else:
                        x_below = state[i-1]['x'].detach()
                        # Found that using actv_fn here gives better results
                        state[i]['x'] = layer.actv_fn(layer.propagate(x_below))

    def _init_es(self, state:List[dict]):
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
                if i < len(self.layers) - 1:
                    pred = self.layers[i+1].predict(state[i+1])
                    layer.update_e(state[i], pred, temp=1.0)

    def init_state(self, obs:torch.Tensor = None, y:torch.Tensor = None):
        """
        | Initialises the state of the model. Xs are calculated using _init_xs().

        Parameters
        ----------
            obs : Optional[torch.Tensor]
                Input data
            y : Optional[torch.Tensor]
                Target data

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
        self._init_es(state)

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

    def forward(self, obs:torch.Tensor = None, y:torch.Tensor = None, steps:int = None, learn_on_step:bool = False):
        """
        | Performs inference phase of the model.
        
        Parameters
            obs : Optional[torch.Tensor]
                Input data
            y : Optional[torch.Tensor]
                Target data
            steps : Optional[int]
                Number of steps to run inference for. Uses self.steps if not provided.

        Returns
        -------
            torch.Tensor
                Output of the model
            List[dict]
                List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)
        """
        if steps is None:
            steps = self.steps

        state = self.init_state(obs, y)

        prev_vfe = None
        gamma = self.gamma
        for i in range(steps):
            temp = self.calc_temp(i, steps)
            self.step(state, obs, y, temp, gamma)
            vfe = self.vfe(state)
            if learn_on_step:
                vfe.backward()
            if prev_vfe is not None and vfe < prev_vfe:
                gamma = gamma * 0.9
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
    
    def generate(self, y, steps=None):
        """
        | Performs inference with target pinned and input free to relax.
        | In theory, should generate an input from a target.

        Parameters
        ----------
            y : torch.Tensor    
                Target data
            steps : Optional[int]
                Number of steps to run inference for. Uses self.steps if not provided.
        
        Returns
        -------
            torch.Tensor
                Generated input
        """
        y = format_y(y, self.num_classes)
        _, state = self.forward(y=y, steps=steps)
        return state[0]['x']

    def reconstruct(self, obs:torch.Tensor, y:torch.Tensor = None, steps:int = None):
        """
        | Initialises the state of the model using the observation (and target if supplied).
        | Runs inference without pinning the observation.
        | In theory should reconstruct the observation.

        Parameters
        ----------
            obs : torch.Tensor
                Input data
            y : Optional[torch.Tensor]
                Target data
            steps : Optional[int]
                Number of steps to run inference for. Uses self.steps if not provided.

        Returns
        -------
            torch.Tensor
                Reconstructed observation
        """
        if steps is None:
            steps = self.steps
        
        state = self.init_state(obs, y)

        for i in range(steps):
            temp = self.calc_temp(i, steps)
            self.step(state, y=y, temp=temp)
        
        out = state[0]['x']

        return out
    
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
    
        if steps is None:
            steps = self.steps

        vfes = torch.zeros(obs.shape[0], self.num_classes, device=self.device)
        for target in range(self.num_classes):
            targets = torch.full((obs.shape[0],), target, device=self.device, dtype=torch.long)
            y = format_y(targets, self.num_classes)
            _, state = self.forward(obs, y, steps)
            vfes[:, target] = self.vfe(state, batch_reduction=None)
        
        return vfes.argmin(dim=1)






