from pclib.nn.layers import Conv2d, ConvTranspose2d, FC
from pclib.utils.functional import format_y
import torch
import torch.nn as nn
import torch.nn.functional as F


# Based on Whittington and Bogacz 2017
class ConvClassifier(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    num_classes: int

    def __init__(self, steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.relu, d_actv_fn=None, gamma=0.1, beta=1.0, device=torch.device('cpu'), dtype=None):
        self.factory_kwargs = {'actv_fn': actv_fn, 'd_actv_fn': d_actv_fn, 'gamma': gamma, 'beta': beta, 'has_bias': bias, 'symmetric': symmetric, 'dtype': dtype}
        super().__init__()

        self.num_classes = 10
        self.precision_weighted = precision_weighted
        self.steps = steps
        self.device = device

    def init_layers(self):
        layers = []
        layers.append(ConvTranspose2d((1, 28, 28), None, **self.factory_kwargs))
        layers.append(ConvTranspose2d((32, 24, 24), 1, 5, padding=0, **self.factory_kwargs))
        layers.append(ConvTranspose2d((64, 10, 10), 32, 5, padding=0, upsample=2, **self.factory_kwargs))
        layers.append(ConvTranspose2d((64, 3, 3), 64, 5, padding=0, upsample=2, **self.factory_kwargs))
        layers.append(FC(128, 64*3*3, **self.factory_kwargs))
        layers.append(FC(10, 128, **self.factory_kwargs))
        self.layers = nn.ModuleList(layers)

    def vfe(self, state, batch_reduction='mean', layer_reduction='sum'):
        if layer_reduction == 'sum':
            vfe = sum([state_i['e'].square().sum(dim=[i for i in range(1, state_i['e'].dim())]) for state_i in state])
        elif layer_reduction =='mean':
            vfe = sum([state_i['e'].square().mean(dim=[i for i in range(1, state_i['e'].dim())]) for state_i in state])
        if batch_reduction == 'sum':
            vfe = vfe.sum()
        elif batch_reduction == 'mean':
            vfe = vfe.mean()

        return vfe

    def step(self, state, obs=None, y=None, temp=None):
        # Update Es, Top-down so we can collect predictions as we descend
        for i, layer in reversed(list(enumerate(self.layers))):
            if i < len(self.layers) - 1: # don't update top e (no prediction)
                layer.update_e(state[i], pred, temp=temp)
            if i > 0: # Bottom layer can't predict
                pred = layer.predict(state[i])
                if isinstance(layer, FC) and isinstance(self.layers[i-1], [Conv2d, ConvTranspose2d]):
                    shape = self.layers[i-1].shape
                    pred = pred.reshape(pred.shape[0], shape[0], shape[1], shape[2])
                elif isinstance(layer, [Conv2d, ConvTranspose2d]) and isinstance(self.layers[i-1], FC):
                    pred = pred.flatten(1)

        self.vfe(state).backward(retain_graph=True)

        for i, layer in enumerate(self.layers):
            e_below = state[i-1]['e'] if i > 0 else None
            if isinstance(layer, FC) and isinstance(self.layers[i-1], ConvTranspose2d):
                e_below = e_below.flatten(1)
            
            layer.update_x(state[i], e_below)
        self.zero_grad()

        self.pin(state, obs, y)

    def pin(self, state, obs=None, y=None):
        # Pin input and output Xs if provided
        if obs is not None:
            state[0]['x'] = obs.clone()
            if isinstance(self.layers[0], ConvTranspose2d):
                state[0]['x'].requires_grad = True
        if y is not None:
            state[-1]['x'] = y.clone()
            if isinstance(self.layers[-1], ConvTranspose2d):
                state[-1]['x'].requires_grad = True



    # Initialises xs in state using 1 sweep of top-down predictions
    def _init_xs(self, state, obs=None, y=None):
        if y is not None:
            for i, layer in reversed(list(enumerate(self.layers))):
                if i == len(self.layers) - 1: # last layer
                    state[i]['x'] = y.clone()
                if i > 0:
                    pred = layer.predict(state[i])
                    if isinstance(layer, FC) and isinstance(self.layers[i-1], ConvTranspose2d):
                        shape = self.layers[i-1].shape
                        pred = pred.reshape(pred.shape[0], shape[0], shape[1], shape[2])
                    state[i-1]['x'] = pred.detach()
            if obs is not None:
                state[0]['x'] = obs.clone()
        elif obs is not None:
            for i, layer in enumerate(self.layers):
                if i == 0:
                    state[0]['x'] = obs.clone()
                else:
                    state[i]['x'] = 0.01 * torch.randn_like(state[i]['x']).to(self.device)

        # elif obs is not None:
            # raise(NotImplementedError, "Initialising xs from obs not implemented, because propagate dont work.")
            # for i, layer in enumerate(self.layers):
            #     if i == 0:
            #         state[0]['x'] = obs.clone()
            #     else:
            #         x_below = state[i-1]['x']
            #         if isinstance(layer, FC) and isinstance(self.layers[i-1], ConvTranspose2d):
            #             x_below = x_below.flatten(1)
            #         state[i]['x'] = layer.propagate(x_below)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, [Conv2d, ConvTranspose2d]):
                state[i]['x'].requires_grad = True

    def init_state(self, obs=None, y=None):
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
        return state[-1]['x']

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

    def generate(self, y, steps=None):
        y = format_y(y, self.num_classes)
        _, state = self.forward(y=y, steps=steps)
        return state[0]['x']
    
    def classify(self, obs, state=None, steps=None):
        if steps is None:
            steps = self.steps

        vfes = torch.zeros(obs.shape[0], self.num_classes, device=self.device)
        for target in range(self.num_classes):
            targets = torch.full((obs.shape[0],), target, device=self.device, dtype=torch.long)
            y = format_y(targets, self.num_classes)
            _, state = self.forward(obs, y, steps)
            vfes[:, target] = vfe(state, batch_reduction=None)
        
        return vfes.argmin(dim=1)


# Based on Whittington and Bogacz 2017
class ConvClassifierInv(ConvClassifier):
    def __init__(self, steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.relu, d_actv_fn=None, gamma=0.1, beta=1.0, device=torch.device('cpu'), dtype=None):
        super().__init__()

    def init_layers(self):
        # (1, 28, 28) -> (32, 24, 24) -> (64, 20, 20), (128, 16, 16), (256, 12, 12)
        layers = []
        layers.append(FC(10, None, **self.factory_kwargs))
        layers.append(FC(3*3*64, 10, **self.factory_kwargs))
        layers.append(Conv2d((32, 10, 10), 64, padding=0, maxpool=2, **self.factory_kwargs))
        layers.append(Conv2d((32, 24, 24), 32, padding=0, maxpool=2, **self.factory_kwargs))
        layers.append(Conv2d((1, 28, 28), 32, padding=0, **self.factory_kwargs))
        self.layers = nn.ModuleList(layers)

    def pin(self, state, obs=None, y=None): 
        # Pin input and output Xs if provided
        if obs is not None:
            state[-1]['x'] = obs.clone()
            if isinstance(self.layers[-1], Conv2d):
                state[-1]['x'].requires_grad = True
        if y is not None:
            state[0]['x'] = y.clone()
            if isinstance(self.layers[0], Conv2d):
                state[0]['x'].requires_grad = True

    # Initialises xs in state using 1 sweep of top-down predictions
    def _init_xs(self, state, obs=None, y=None):
        if obs is not None:
            for i, layer in reversed(list(enumerate(self.layers))):
                if i == len(self.layers) - 1: # last layer
                    state[i]['x'] = obs.clone()
                if i > 0:
                    # print(f"i: {i}")
                    pred = layer.predict(state[i])
                    if isinstance(layer, Conv2d) and isinstance(self.layers[i-1], FC):
                        pred = pred.flatten(1)
                    # print(f"pred shape: {pred.shape}")
                    state[i-1]['x'] = pred.detach()
            if y is not None:
                state[0]['x'] = y.clone()
        elif y is not None:
            raise NotImplementedError
        #     for i, layer in enumerate(self.layers):
        #         if i == 0:
        #             state[0]['x'] = y.clone()
        #         else:
        #             x_below = state[i-1]['x']
        #             if state[i]['x'].dim() == 4 and state[i-1]['x'].dim() == 2:
        #                 x_below = x_below.unsqueeze(-1).unsqueeze(-1)
        #             state[i]['x'] = layer.propagate(x_below)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, [Conv2d, ConvTranspose2d]):
                state[i]['x'].requires_grad = True

    def get_output(self, state):
        return state[0]['x']
    
    def generate(self, y, steps=None):
        raise NotImplementedError

    def classify(self, obs, state=None, steps=None):
        if steps is None:
            steps = self.steps

        vfes = torch.zeros(obs.shape[0], self.num_classes, device=self.device)
        for target in range(self.num_classes):
            targets = torch.full((obs.shape[0],), target, device=self.device, dtype=torch.long)
            y = format_y(targets, self.num_classes)
            _, state = self.forward(obs, y, steps)
            vfes[:, target] = vfe(state, batch_reduction=None)
        
        return vfes.argmin(dim=1)
