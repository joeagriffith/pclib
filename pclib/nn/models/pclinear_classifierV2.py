# from pclib.nn.layers import PrecisionWeightedV2 as Linear
from pclib.nn.layers import LinearV2 as Linear
import torch
import torch.nn as nn
import torch.nn.functional as F

class PCLinearClassifierV2(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    num_classes: int

    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=5, bias=True, symmetric=True, actv_fn=F.relu, actv_mode='Wf(r)', gamma=0.1, beta=1.0, device=torch.device('cpu'), dtype=None):
        factory_kwargs = {'bias': bias, 'symmetric': symmetric, 'device': device, 'dtype': dtype}
        super(PCLinearClassifierV2, self).__init__()

        self.in_features = input_size
        self.num_classes = num_classes
        self.bias = bias
        self.symmetric = symmetric
        self.gamma = gamma
        self.beta = beta

        layers = []
        size = num_classes
        for next_size in hidden_sizes + [input_size]:
            layers.append(Linear(size, next_size, actv_fn=actv_fn, actv_mode=actv_mode, gamma=gamma, beta=beta, **factory_kwargs))
            size = next_size
        layers.append(Linear(size, None, actv_fn=actv_fn, actv_mode=actv_mode, gamma=gamma, beta=beta, **factory_kwargs))

        self.layers = nn.ModuleList(layers)
        self.steps = steps
        self.device = device

    def step(self, obs, state, y=None):

        # Update Xs Top-down, ignore top layer
        for i, layer in reversed(list(enumerate(self.layers))):
            if i == len(self.layers) - 1: # last layer
                continue
            bu_error = self.layers[i-1].forward_error(state[i-1]) if i > 0 else None
            state[i] = layer.update_x(state[i], bu_error)
        if y is not None:
            state[0]['x'] = y.clone()
        state[-1]['x'] = obs.clone() # TODO: redundant, here for "compatibility"

        # Update Es Top-down, ignore top layer
        for i, layer in reversed(list(enumerate(self.layers))):
            if i == len(self.layers) - 1: # last layer
                continue
            f_x_lp1 = self.layers[i+1].actv_fn(state[i+1]['x'])
            state[i] = layer.update_e(state[i], f_x_lp1)

        return state

    def init_state(self, batch_size: int, mode='zeros'):
        state = []
        for layer in self.layers:
            state_i = layer.init_state(batch_size, mode)
            state.append(state_i)
        return state

    def to(self, device):
        self.device = device
        for layer in self.layers:
            layer.to(device)
        return self

    def get_output(self, state):
        return state[0]['x']
    
    # Initialises xs in state using 1 sweep of top-down predictions
    def init_xs(self, obs, state, y=None):
        for i, layer in reversed(list(enumerate(self.layers))):
            if i == len(self.layers) - 1: # last layer
                state[i]['x'] = obs.clone()
            else:
                state[i]['x'] = layer.predict(f_x_lp1)
            f_x_lp1 = layer.actv_fn(state[i]['x'])

        if y is not None:
            state[0]['x'] = y.clone()
        return state

    def init_es(self, state):
        for i, layer in enumerate(self.layers):
            f_x_lp1 = None
            if i < len(self.layers) - 1:
                f_x_lp1 = layer.actv_fn(state[i+1]['x'])
            state[i] = layer.update_e(state[i], f_x_lp1)
        return state


    def forward(self, obs, state=None, y=None, steps=None):
        assert len(obs.shape) == 2, f"Input must be 2D, got {len(obs.shape)}D"

        if steps is None:
            steps = self.steps

        if state is None:
            state = self.init_state(obs.shape[0])
        state[-1]['x'] = obs.clone()

        if y is not None:
            temp = torch.ones_like(y) * 0.03
            y = temp + (y * 0.94)
            state[0]['x'] = y.clone()
        
        state = self.init_xs(obs, state, y)
        state = self.init_es(state)

        for _ in range(steps):
            state = self.step(obs, state, y)
            
        out = state[0]['x']
            
        return out, state