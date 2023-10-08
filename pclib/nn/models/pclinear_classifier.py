# from pclib.nn.layers import PrecisionWeightedV2 as Linear
from pclib.nn.layers import Linear
from pclib.utils.functional import vfe, format_y
import torch
import torch.nn as nn
import torch.nn.functional as F

class PCLinearClassifier(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    num_classes: int

    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=5, bias=True, symmetric=True, actv_fn=F.relu, gamma=0.1, beta=1.0, device=torch.device('cpu'), dtype=None):
        factory_kwargs = {'bias': bias, 'symmetric': symmetric, 'device': device, 'dtype': dtype}
        super(PCLinearClassifier, self).__init__()

        self.in_features = input_size
        self.num_classes = num_classes
        self.bias = bias
        self.symmetric = symmetric
        self.gamma = gamma
        self.beta = beta

        layers = []
        prev_size = None
        for size in [num_classes] + hidden_sizes + [input_size]:
            layers.append(Linear(size, prev_size, actv_fn=actv_fn, gamma=gamma, beta=beta, **factory_kwargs))
            prev_size = size

        self.layers = nn.ModuleList(layers)
        self.steps = steps
        self.device = device

    def step(self, state, y=None):

        # Update Xs Top-down, ignore top layer
        for i, layer in reversed(list(enumerate(self.layers))):
            if i == len(self.layers) - 1: # don't update last layer
                continue
            e_below = state[i-1]['e'] if i > 0 else None
            state[i] = layer.update_x(state[i], e_below)
        if y is not None:
            state[0]['x'] = y.clone()

        # Update Es Top-down, collecting predictions as we descend.
        for i, layer in reversed(list(enumerate(self.layers))):
            if i < len(self.layers) - 1: # don't update last layer
                state[i] = layer.update_e(state[i], pred)
            if i > 0: # Bottom layer can't predict
                pred = layer.predict(state[i])

        return state

    # Initialises xs in state using 1 sweep of top-down predictions
    def _init_xs(self, obs, state, y=None):
        for i, layer in reversed(list(enumerate(self.layers))):
            if i == len(self.layers) - 1: # last layer
                state[i]['x'] = obs.clone()
            if i > 0:
                state[i-1]['x'] = layer.predict(state[i])
        if y is not None:
            state[0]['x'] = y.clone()
        return state

    def _init_es(self, state):
        # Doesn't have to be done top-down, done for consistency
        for i, layer in reversed(list(enumerate(self.layers))):
            if i > 0:
                state[i-1]['e'] = state[i-1]['x'] - layer.predict(state[i])
        return state

    def init_state(self, obs, y=None):
        state = []
        for layer in self.layers:
            state_i = layer.init_state(obs.shape[0])
            state.append(state_i)
        
        state = self._init_xs(obs, state, y)

        return state

    def to(self, device):
        self.device = device
        for layer in self.layers:
            layer.to(device)
        return self

    def get_output(self, state):
        return state[0]['x']

    def forward(self, obs, state=None, y=None, steps=None):
        assert len(obs.shape) == 2, f"Input must be 2D, got {len(obs.shape)}D"

        if steps is None:
            steps = self.steps

        if state is None:
            state = self.init_state(obs, y)

        for _ in range(steps):
            state = self.step(state, y)
            
        out = state[0]['x']
            
        return out, state
    
    def classify(self, obs, state=None, steps=None):
        assert len(obs.shape) == 2, f"Input must be 2D, got {len(obs.shape)}D"

        if steps is None:
            steps = self.steps

        vfes = torch.zeros(obs.shape[0], self.num_classes, device=self.device)
        for target in range(self.num_classes):
            targets = torch.full((obs.shape[0],), target, device=self.device, dtype=torch.long)
            y = format_y(targets, self.num_classes)
            _, state = self.forward(obs, None, y, steps)
            vfes[:, target] = vfe(state, batch_reduction=None)
        
        return vfes.argmin(dim=1)

