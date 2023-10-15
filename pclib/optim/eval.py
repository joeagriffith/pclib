import torch
import matplotlib.pyplot as plt
from pclib.utils.functional import vfe

def topk_accuracy(output, target, k=1):
    """Computes the precision for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(k, 1)
        correct = pred.eq(target.unsqueeze(1).expand_as(pred)).sum(dim=0)
        accuracy = correct * (100 / batch_size)
        return accuracy

def track_vfe(model, x, y=None, steps=100, plot_Es=False):
    # assert len(x.shape) == 2, f"Invalid shape {x.shape}, input and targets must be pre-processed."
    state = model.init_state(x, y)
    vfes = []
    E = [[] for _ in range(len(model.layers))]
    for step_i in range(steps):
        with torch.no_grad():
            temp = model.calc_temp(step_i, steps)
            model.step(state, x, y, temp)
            vfes.append(vfe(state).item())
            for i in range(len(model.layers)):
                E[i].append(state[i]['e'].square().sum(dim=1).mean().item())
        
    plt.plot(vfes, label='VFE')

    if plot_Es:
        for i in range(len(model.layers)):
            plt.plot(E[i], label=f'layer {i} E')
    plt.legend()
    plt.show()
        


def accuracy(model, dataset, batch_size=1024, steps=100):
    with torch.no_grad():
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
        correct = 0
        for x, y in dataloader:
            if type(model.layers[-1].shape) == int:
                x = x.flatten(start_dim=1)
            pred = model.classify(x, steps)
            correct += (pred == y).sum().item()
        acc = correct/len(dataset)
    return acc