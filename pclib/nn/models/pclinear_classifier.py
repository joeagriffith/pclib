from pclib.nn.layers import PrecisionWeighted as PrecisionWeighted
from pclib.nn.layers import Linear
from pclib.utils.functional import vfe, format_y
import torch
import torch.nn as nn
import torch.nn.functional as F


# Based on Whittington and Bogacz 2017
class PCLinearClassifier(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    num_classes: int

    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.relu, d_actv_fn=None, gamma=0.1, beta=1.0, device=torch.device('cpu'), dtype=None):
        factory_kwargs = {'bias': bias, 'symmetric': symmetric, 'device': device, 'dtype': dtype}
        super(PCLinearClassifier, self).__init__()

        self.in_features = input_size
        self.num_classes = num_classes
        self.bias = bias
        self.symmetric = symmetric
        self.precision_weighted = precision_weighted
        self.gamma = gamma
        self.beta = beta

        layers = []
        prev_size = None
        for size in [num_classes] + hidden_sizes + [input_size]:
            if precision_weighted:
                layers.append(PrecisionWeighted(size, prev_size, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs))
            else:
                layers.append(Linear(size, prev_size, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs))
            prev_size = size

        self.layers = nn.ModuleList(layers)
        self.steps = steps
        self.device = device

    def step(self, state, obs=None, y=None, temp=None):

        # Update Es, Top-down so we can collect predictions as we descend
        for i, layer in reversed(list(enumerate(self.layers))):
            if i < len(self.layers) - 1: # don't update top e (no prediction)
                layer.update_e(state[i], pred, temp=temp)
            if i > 0: # Bottom layer can't predict
                pred = layer.predict(state[i])

        # Update Xs
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                e_below = state[i-1]['e'] if i > 0 else None
                layer.update_x(state[i], e_below)
        
        # Pin input and output Xs if provided
        if obs is not None:
            state[-1]['x'] = obs.clone()
        if y is not None:
            state[0]['x'] = y.clone()


    # Initialises xs in state using 1 sweep of top-down predictions
    def _init_xs(self, state, obs=None, y=None):
        if obs is not None:
            for i, layer in reversed(list(enumerate(self.layers))):
                if i == len(self.layers) - 1: # last layer
                    state[i]['x'] = obs.clone()
                if i > 0:
                    state[i-1]['x'] = layer.predict(state[i])
            if y is not None:
                state[0]['x'] = y.clone()
        elif y is not None:
            for i, layer in enumerate(self.layers):
                if i == 0:
                    state[0]['x'] = y.clone()
                elif i < len(self.layers) - 1:
                    state[i]['x'] = layer.propagate(state[i-1]['x'])

    def init_state(self, obs=None, y=None):
        with torch.no_grad():
            if obs is not None:
                b_size = obs.shape[0]
            elif y is not None:
                b_size = y.shape[0]
            state = []
            for layer in self.layers:
                state.append(layer.init_state(b_size))
            
            self._init_xs(state, obs, y)
            return state

    def to(self, device):
        self.device = device
        for layer in self.layers:
            layer.to(device)
        return self

    def get_output(self, state):
        return state[0]['x']

    def calc_temp(self, step_i, steps):
        return 1 - (step_i / steps)

    def forward(self, obs=None, y=None, steps=None):
        if steps is None:
            steps = self.steps

        state = self.init_state(obs, y)

        for i in range(steps):
            temp = self.calc_temp(i, steps)
            self.step(state, obs, y, temp)
            
        out = self.get_output(state)
            
        return out, state
    
    def classify(self, obs, state=None, steps=None):
        assert len(obs.shape) == 2, f"Input must be 2D, got {len(obs.shape)}D"

        if steps is None:
            steps = self.steps

        vfes = torch.zeros(obs.shape[0], self.num_classes, device=self.device)
        for target in range(self.num_classes):
            targets = torch.full((obs.shape[0],), target, device=self.device, dtype=torch.long)
            y = format_y(targets, self.num_classes)
            _, state = self.forward(obs, y, steps)
            vfes[:, target] = vfe(state, batch_reduction=None)
        
        return vfes.argmin(dim=1)

