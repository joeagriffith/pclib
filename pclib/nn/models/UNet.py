from pclib.nn.layers import Conv2d, ConvTranspose2d, FC, Conv2dSkip
from pclib.utils.functional import vfe, format_y
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, shape, prev_channels=None, upsample=1, actv_fn=F.relu, d_actv_fn=None, gamma=0.1, beta=1.0, device=torch.device('cpu'), dtype=None):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ConvTranspose2d(shape, prev_channels, kernel_size=3, padding=1, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, device=device, dtype=dtype),
            ConvTranspose2d(shape, shape[0], kernel_size=3, padding=1, upsample=upsample, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, device=device, dtype=dtype),
        )


class UNet(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    num_classes: int

    def __init__(self, steps=100, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.relu, d_actv_fn=None, gamma=0.1, beta=1.0, device=torch.device('cpu'), dtype=None):
        factory_kwargs = {'bias': bias, 'device': device, 'dtype': dtype}
        super(UNet, self).__init__()

        self.bias = bias
        self.symmetric = symmetric
        self.precision_weighted = precision_weighted
        self.gamma = gamma
        self.beta = beta

        # self.encoder = nn.ModuleList([
        #     DoubleConv((64, 1920, 1280), 3, upsample=2, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
        #     DoubleConv((128, 960, 640), 64, upsample=2, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
        #     DoubleConv((256, 480, 320), 128, upsample=2, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
        #     DoubleConv((512, 240, 160), 256, upsample=2, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
        # ])

        self.encoder = nn.ModuleList([
            ConvTranspose2d((3, 160, 256), None, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),

            ConvTranspose2d((64, 160, 256), 3, kernel_size=3, padding=1, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
            ConvTranspose2d((64, 160, 256), 64, kernel_size=3, padding=1, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),

            ConvTranspose2d((128, 80, 128), 64, kernel_size=3, padding=1, upsample=2, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
            ConvTranspose2d((128, 80, 128), 128, kernel_size=3, padding=1, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
            
            ConvTranspose2d((256, 40, 64), 128, kernel_size=3, padding=1, upsample=2, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
            ConvTranspose2d((256, 40, 64), 256, kernel_size=3, padding=1, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),

            ConvTranspose2d((512, 20, 32), 256, kernel_size=3, padding=1, upsample=2, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
            ConvTranspose2d((512, 20, 32), 512, kernel_size=3, padding=1, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
        ])

        self.bottleneck = nn.ModuleList([
            ConvTranspose2d((1024, 10, 16), 512, kernel_size=3, padding=1, upsample=2, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
            ConvTranspose2d((1024, 10, 16), 1024, kernel_size=3, padding=1, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
        ])

        self.decoder = nn.ModuleList([
            Conv2dSkip((1024, 20, 32), 1024, kernel_size=2, stride=2, padding=0, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
            # Conv2d((1024, 20, 32), 1024, kernel_size=2, stride=2, padding=0, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
            ConvTranspose2d((512, 20, 32), 1024, kernel_size=3, padding=1, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
            ConvTranspose2d((512, 20, 32), 512, kernel_size=3, padding=1, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),

            Conv2dSkip((512, 40, 64), 512, kernel_size=2, stride=2, padding=0, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
            # Conv2d((512, 40, 64), 512, kernel_size=2, stride=2, padding=0, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
            ConvTranspose2d((256, 40, 64), 512, kernel_size=3, padding=1, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
            ConvTranspose2d((256, 40, 64), 256, kernel_size=3, padding=1, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),

            Conv2dSkip((256, 80, 128), 256, kernel_size=2, stride=2, padding=0, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
            # Conv2d((256, 80, 128), 256, kernel_size=2, stride=2, padding=0, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
            ConvTranspose2d((128, 80, 128), 256, kernel_size=3, padding=1, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
            ConvTranspose2d((128, 80, 128), 128, kernel_size=3, padding=1, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),

            Conv2dSkip((128, 160, 256), 128, kernel_size=2, stride=2, padding=0, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
            # Conv2d((128, 160, 256), 128, kernel_size=2, stride=2, padding=0, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
            ConvTranspose2d((64, 160, 256), 128, kernel_size=3, padding=1, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
            ConvTranspose2d((64, 160, 256), 64, kernel_size=3, padding=1, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),

        ])

        self.final = nn.ModuleList([
            ConvTranspose2d((1, 160, 256), 64, kernel_size=3, padding=1, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs),
        ])

        # Key is layer index receiving skip, value is layer index sending skip
        self.skips = {
            20: 2,
            17: 4,
            14: 6,
            11: 8,
        }
        
        self.steps = steps
        self.device = device

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
        for i, module in reversed(list(enumerate(self.encoder + self.bottleneck + self.decoder + self.final))):
            if i < len(self.encoder + self.bottleneck + self.decoder + self.final) - 1: # don't update top e (no prediction)
                if i in self.skips:
                    skip = state[self.skips[i]]['x']
                    module.update_e(state[i], skip, pred, temp=temp)
                else:
                    module.update_e(state[i], pred, temp=temp)
                # module.update_e(state[i], pred, temp=temp)
            if i > 0: # Bottom layer can't predict
                if i in self.skips:
                    skip = state[self.skips[i]]['x']
                    pred = module.predict(state[i], skip)
                else:
                    pred = module.predict(state[i])

        # print("vfe before: ", self.vfe(state))
        self.vfe(state).backward(retain_graph=True)
        # for i, layer in enumerate(self.layers):
        #     if state[i]['x'].requires_grad:
        #         has_grad = state[i]['x'].grad is not None
        #         print(f"layer {i} has grad: ", has_grad)

        for i, module in enumerate(self.encoder + self.bottleneck + self.decoder + self.final):
            e_below = state[i-1]['e'] if i > 0 else None
            
            # print(f"layer {i} of type {type(layer)} requires_grad {state[i]['x'].requires_grad}, has_grad {state[i]['x'].grad is not None}")
            module.update_x(state[i], e_below)
        self.zero_grad()

        # with torch.no_grad():
            # Update Xs
            # for i, layer in enumerate(self.layers):
            #     e_below = state[i-1]['e'] if i > 0 else None
            #     layer.update_x(state[i], e_below)
            
        # Pin input and output Xs if provided
        if obs is not None:
            state[0]['x'] = obs.clone()
            state[0]['x'].requires_grad = True
        if y is not None:
            state[-1]['x'] = y.clone()
            state[-1]['x'].requires_grad = True



    # Initialises xs in state using 1 sweep of top-down predictions
    def _init_xs(self, state, obs=None, y=None):
        for i, module in enumerate(self.encoder + self.bottleneck + self.decoder + self.final):
            if i == 0:
                state[0]['x'] = obs.clone()
            else:
                # Could propagate xs as errs (if build conv propagate)
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
        for i, _ in enumerate(self.encoder + self.bottleneck + self.decoder + self.final):
            state[i]['x'].requires_grad = True

    def init_state(self, obs=None, y=None):
        if obs is not None:
            b_size = obs.shape[0]
        elif y is not None:
            b_size = y.shape[0]
        state = []
        for module in self.encoder + self.bottleneck + self.decoder + self.final:
            state.append(module.init_state(b_size))
        
        self._init_xs(state, obs, y)
        return state

    def to(self, device):
        self.device = device
        for module in self.encoder + self.bottleneck + self.decoder + self.final:
            module.to(device)
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

