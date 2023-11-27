from pclib.nn.layers import FCLI, FC, FCPW
from pclib.nn.models import FCClassifier
from pclib.utils.functional import vfe, format_y
import torch
import torch.nn as nn
import torch.nn.functional as F


# Based on Whittington and Bogacz 2017, but with targets predicting inputs
class FCClassifierSSLI(FCClassifier):

    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, lat_inhib=False, precision_weighted=False, actv_fn=F.relu, d_actv_fn=None, gamma=0.1, beta=1.0, device=torch.device('cpu'), dtype=None):
        factory_kwargs = {'has_bias': bias, 'symmetric': symmetric, 'device': device, 'dtype': dtype}
        super().__init__(input_size, num_classes, hidden_sizes, steps, bias, symmetric, precision_weighted, actv_fn, d_actv_fn, gamma, beta, device, dtype)

        layers = []
        layers.append(FC(None, input_size, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs))
        in_features = input_size
        for out_features in hidden_sizes:
            if precision_weighted:
                raise NotImplementedError("Precision weighted not implemented for FCClassifierSSLI")
            else:
                layers.append(FCLI(in_features, out_features, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs))
            in_features = out_features
        self.layers = nn.ModuleList(layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 200, bias=True, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(200, num_classes, bias=True, device=device, dtype=dtype),
        )

    def to(self, device):
        self.device = device
        for layer in self.layers:
            layer.to(device)
        for layer in self.classifier:
            layer.to(device)
        return self

    def get_output(self, state):
        # x = torch.cat([state[i]['x'] for i in range(1,len(state))], dim=1)
        x = state[-1]['x']
        out = self.classifier(x.detach())
        return out

    def forward(self, obs=None, steps=None):
        if steps is None:
            steps = self.steps

        state = self.init_state(obs)

        for i in range(steps):
            temp = self.calc_temp(i, steps)
            self.step(state, obs, temp=temp)
            
        out = self.get_output(state)
            
        return out, state

    def classify(self, obs, steps=None):
        return self.forward(obs, steps)[0].argmax(dim=1)

    def reconstruct(self, obs, steps=None):
        if steps is None:
            steps = self.steps
        
        state = self.init_state(obs)

        for i in range(steps):
            temp = self.calc_temp(i, steps)
            self.step(state, temp=temp)
        
        out = state[0]['x']

        return out, state

    
    def generate(self, y, steps=None):
        raise(NotImplementedError)
