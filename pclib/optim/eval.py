import torch
import matplotlib.pyplot as plt

def topk_accuracy(output:torch.Tensor, target:torch.Tensor, k:int = 1):
    """
    | Computes the top-k accuracy for the specified values of k

    Parameters
    ----------
        output : torch.Tensor
            Output of the classifier
        target : torch.Tensor
            Target labels
        k : int
            Number of top predictions to consider
    
    Returns
    -------
        torch.Tensor
            Accuracy for each value of k
    """
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(k, 1)
        correct = pred.eq(target.unsqueeze(1).expand_as(pred)).sum(dim=0)
        accuracy = correct * (100 / batch_size)
        return accuracy

def track_vfe(model:torch.nn.Module, x:torch.Tensor = None, y:torch.Tensor = None, steps:int = 100, plot_Es:bool = False, plot_Xs:bool = False, flatten:bool = True):
    """
    | Tracks the VFE of the model over the specified number of steps.

    Parameters
    ----------
        model : torch.nn.Module
            Model to perform inference with
        x : torch.Tensor
            Input data
        y : torch.Tensor
            Target data, provided to top of model. If not provided, model is assumed to be unsupervised.
        steps : int
            Number of steps to track VFE for
        plot_Es : bool
            Whether to plot the E values of each layer
        plot_Xs : bool
            Whether to plot the X values of each layer
        flatten : bool
            Whether to flatten the input data
    
    Returns
    -------
        list
            List of VFE values
        list
            List of E values for each layer
        list
            List of X values for each layer
    """
    if flatten:
        x = x.flatten(start_dim=1)
    state = model.init_state(x, y)
    vfes = []
    E = [[] for _ in range(len(model.layers))]
    X = [[] for _ in range(len(model.layers))]
    for step_i in range(steps):
        temp = model.calc_temp(step_i, steps)
        model.step(state, x, y, temp)
        vfes.append(model.vfe(state).item())
        for i in range(len(model.layers)):
            E[i].append(state[i]['e'].square().sum(dim=1).mean().item())
            X[i].append(state[i]['x'].square().sum(dim=1).mean().item())
        
    plt.plot(vfes, label='VFE')

    if plot_Es:
        for i in range(len(model.layers)):
            plt.plot(E[i], label=f'layer {i} E')
    plt.legend()
    plt.show()

    if plot_Xs:
        for i in range(len(model.layers)):
            plt.plot(X[i], label=f'layer {i} X')
    plt.legend()
    plt.show()

    return vfes, E, X
        
def accuracy(model:torch.nn.Module, dataset:torch.utils.data.Dataset, batch_size:int = 1024, steps:int = 100, flatten:bool = True):
    """
    | Computes the accuracy of the model on the specified dataset.

    Parameters
    ----------
        model : torch.nn.Module
            Model to perform inference with
        dataset : torch.utils.data.Dataset
            Dataset to evaluate on
        batch_size : int
            Batch size to use
        steps : int
            Number of steps to run inference for
        flatten : bool
            Whether to flatten the input data
        
    Returns
    -------
        float
            Accuracy of the model on the dataset
    """
    with torch.no_grad():
        dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
        correct = 0
        for x, y in dataloader:
            if flatten:
                x = x.flatten(start_dim=1)
            pred = model.classify(x, steps)
            correct += (pred == y).sum().item()
        acc = correct/len(dataset)
    return acc