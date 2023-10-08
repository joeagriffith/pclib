import torch
import matplotlib.pyplot as plt
from pclib.utils.functional import vfe

def topk_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        device = "cuda" if output.is_cuda else "cpu"
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = torch.zeros(len(topk), dtype=float, device=device)
        for i, k in enumerate(topk):
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res[i] = correct_k.mul_(100.0 / batch_size)
        return res

def evaluate_pc(model, data_loader, criterion, device, flatten=False):
    with torch.no_grad():
        model.eval()

        loss = 0.0
        acc = torch.zeros(2, device=device)
        errs = torch.zeros(len(model.layers), device=device)

        for images, y in data_loader:
            x = images.to(device)
            if flatten:
                x = torch.flatten(x, start_dim=1)
            target = y.to(device)
            out, R, E = model(x, full_data=True)

            loss += criterion(out, target).item()

            acc += topk_accuracy(out, target, (1,3))

            for i, e in enumerate(E):
                errs[i] += e.square().mean()
        
        loss /= len(data_loader)
        acc /= len(data_loader) 
        errs /= len(data_loader)

        return loss, acc, errs


def track_vfe(model, x, y=None, steps=100, init_mode='rand', plot_Es=False):
    assert len(x.shape) == 2, f"Invalid shape {x.shape}, input and targets must be pre-processed."
    state = model.init_state(x.shape[0], mode=init_mode)
    vfes = []
    E = [[] for _ in range(len(model.layers))]
    for step_i in range(steps):
        with torch.no_grad():
            state = model.step(x, state, y)
            vfes.append(vfe(state).item())
            for i in range(len(model.layers)):
                E[i].append(state[i]['e'].square().sum(dim=1).mean().item())
        
    plt.plot(vfes, label='VFE')

    if plot_Es:
        for i in range(len(model.layers)):
            plt.plot(E[i], label=f'layer {i} E')
    plt.legend()
    plt.show()
        


def accuracy(model, dataset, batch_size=1024, steps=100, return_all=False, plot=True):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
    correct = [0 for _ in range(steps)]
    for x, y in dataloader:
        x = x.flatten(start_dim=1)
        state = model.init_state(x.shape[0], mode='rand')

        for step_i in range(steps):
            with torch.no_grad():
                state = model.step(x, state)
            
            pred = model.get_output(state).argmax(dim=1)
            correct[step_i] += (pred == y).sum().item()

    acc = [c/len(dataset) for c in correct]
    if plot:
        plt.plot(acc)
        print(f"Max accuracy: {max(acc)}")
    if return_all:
        return acc
    else:
        return max(acc)