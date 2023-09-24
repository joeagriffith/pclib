import torch
import matplotlib.pyplot as plt
from pclib.utils.functional import vfe

def track_vfe(model, x, y=None, steps=100, init_mode='rand'):
    assert len(x.shape) == 2, f"Invalid shape {x.shape}, input and targets must be pre-processed."
    state = model.init_state(x.shape[0], mode=init_mode)
    vfes = []
    for step_i in range(steps):
        with torch.no_grad():
            state = model.step(x, state, y)
            vfes.append(vfe(state).item())
        
    plt.plot(vfes)

def accuracy(model, dataset, batch_size=1024, steps=100, return_all=False, plot=True):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
    correct = [0 for _ in range(steps)]
    for x, y in dataloader:
        x = x.flatten(start_dim=1)
        state = model.init_state(x.shape[0], mode='rand')

        for step_i in range(steps):
            with torch.no_grad():
                state = model.step(x, state)
            
            pred = state[-1][0].argmax(dim=1)
            correct[step_i] += (pred == y).sum().item()

    acc = [c/len(dataset) for c in correct]
    if plot:
        plt.plot(acc)
        print(f"Max accuracy: {max(acc)}")
    if return_all:
        return acc
    else:
        return max(acc)