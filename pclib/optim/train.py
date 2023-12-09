import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchviz import make_dot

from pclib.optim.eval import topk_accuracy
from pclib.nn.layers import FCPW, FCLI
from pclib.utils.functional import format_y, calc_corr

def get_optimiser(parameters, lr, weight_decay, optimiser='AdamW'):
    """
    | Builds an optimiser from the specified arguments

    Args:
        | parameters (list): list of PyTorch parameters to optimiser. Usually model.parameters()
        | lr (float): learning rate
        | weight_decay (float): weight decay
        | optimiser (str): optimiser to use. ['AdamW', 'Adam', 'SGD', 'RMSprop']
    
    Returns:
        | optimiser (torch.optim): optimiser
    """
    assert optimiser in ['AdamW', 'Adam', 'SGD', 'RMSprop'], f"Invalid optimiser {optimiser}"
    if optimiser == 'AdamW':
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif optimiser == 'Adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimiser == 'SGD':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimiser == 'RMSprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9)

def init_stats(model, minimal=False):
    """
    | Initialises a dictionary to store statistics
    
    Args:
        | model (nn.Module): model to track statistics for
        | minimal (bool): if True, only track minimal statistics (train_vfe, train_corr, val_vfe, val_acc)
    
    Returns:
        | stats (dict): dictionary to store statistics
    """
    if not minimal:
        stats = {
            "X_norms": [[] for _ in range(len(model.layers))],
            "E_mags": [[] for _ in range(len(model.layers))],
            "WeightTD_means": [[] for _ in range(len(model.layers)-1)],
            "WeightTD_stds": [[] for _ in range(len(model.layers)-1)],
            "Bias_means": [[] for _ in range(len(model.layers)-1)],
            "Bias_stds": [[] for _ in range(len(model.layers)-1)],
            "WeightBU_means": [[] for _ in range(len(model.layers)-1)],
            "WeightBU_stds": [[] for _ in range(len(model.layers)-1)],
            "WeightVar_means": [[] for _ in range(len(model.layers)-1)],
            "WeightVar_stds": [[] for _ in range(len(model.layers)-1)],
            "train_vfe": [],
            "train_corr": [],
            "val_vfe": [],
            "val_acc": [],
        }
    else:
        stats = {
            "train_vfe": [],
            "train_corr": [],
            "val_vfe": [],
            "val_acc": [],
        }
    return stats

def neg_pass(model, x, targets, neg_coeff):
    """
    | Calculates incorrect ys and performs inference on them.
    | Then multiply vfe by -neg_coeff and backpropagate to increase vfe for negative data.

    Args:
        | model (nn.Module): model to train
        | x (torch.Tensor): input data
        | targets (torch.Tensor): targets
        | neg_coeff (float): coefficient to multiply vfe by. 1.0 for balanced positive and negative passes. Must be positive.
    """
    assert neg_coeff > 0, f"neg_coeff must be positive, got {neg_coeff}"
    false_targets = (targets + torch.randint_like(targets, low=1, high=model.num_classes)) % model.num_classes
    false_y = format_y(false_targets, model.num_classes)

    # Forward pass
    _, neg_state = model(x, y=false_y)
    loss = -neg_coeff * model.vfe(neg_state)
    loss.backward()


def val_pass(model, val_loader, flatten=True):
    """
    | Performs a validation pass on the model

    Args:
        | model (nn.Module): model to validate
        | val_loader (DataLoader): validation data
        | flatten (bool): if True, flatten input data

    Returns:
        | val_vfe (float): average VFE for the validation data
        | val_acc (float): accuracy for the validation data
    """
    with torch.no_grad():
        model.eval()
        val_correct = 0
        val_vfe = 0
        for images, target in val_loader:
            if flatten:
                x = images.flatten(start_dim=1)
            else:
                x = images
            # x = images.flatten(start_dim=1)

            # Forward pass
            out, val_state = model(x)
            val_vfe += model.vfe(val_state, batch_reduction='sum').item()
            val_correct += (out.argmax(dim=1) == target).sum().item()

        val_acc = val_correct / len(val_loader.dataset)
        val_vfe /= len(val_loader.dataset)
    return val_vfe, val_acc

def train(
    model, 
    train_data,
    val_data,
    num_epochs,
    lr = 3e-4,
    c_lr = 1e-3,
    batch_size=1,
    reg_coeff = 1e-2,
    flatten=True,
    neg_coeff=None,
    step=0, 
    stats=None,
    minimal_stats=False,
    assert_grads=False,
    grad_mode='auto',
    optim='AdamW',
):
    """
    | Trains a model with the specified parameters
    | Can train any model, supervised or unsupervised, standard, symmetric or inverted.

    Args:
        | model (nn.Module): model to train
        | train_data (Dataset or DataLoader): training data
        | val_data (Dataset or DataLoader): validation data
        | num_epochs (int): number of epochs to train for
        | lr (float): learning rate
        | c_lr (float): learning rate for classifier. Ignored if model has no classifier.
        | batch_size (int): batch size
        | reg_coeff (float): weight decay. Also used for optimising classifier.
        | flatten (bool): if True, flatten input data
        | neg_coeff (float): coefficient to multiply vfe by during negative pass. 1.0 for balanced positive and negative passes. Must be positive.
        | step (int): step number, used for logging
        | stats (dict): dictionary to store statistics
        | minimal_stats (bool): if True, only track minimal statistics (train_vfe, train_corr, val_vfe, val_acc)
        | assert_grads (bool): if True, assert that gradients are close to manual gradients. Must be false if grad_mode is 'manual'.
        | grad_mode (str): gradient mode. ['auto', 'manual']
        | optim (str): optimiser to use. ['AdamW', 'Adam', 'SGD', 'RMSprop']

    Returns:
        | step (int): step number
        | stats (dict): dictionary of statistics
    """


    assert grad_mode in ['auto', 'manual'], f"Invalid grad_mode {grad_mode}, must be 'auto' or 'manual'"
    if grad_mode == 'manual':
        assert(assert_grads == False), "assert_grads must be False when grad_mode is 'manual'"


    optimiser = get_optimiser(model.parameters(), lr, reg_coeff, optim)
    if hasattr(model, 'classifier'):
        c_optimiser = get_optimiser(model.classifier.parameters(), c_lr, reg_coeff, optim)
        loss_fn = F.cross_entropy
    else:
        c_optimiser = None

    train_loader = train_data if isinstance(train_data, DataLoader) else DataLoader(train_data, batch_size, shuffle=True)
    val_loader = val_data if isinstance(val_data, DataLoader) else DataLoader(val_data, batch_size, shuffle=False)

    # Track epochs, indexed: [layer][epoch]
    if stats is None:
        stats = init_stats(model, minimal_stats)

    for epoch in range(num_epochs):

        train_data.apply_transform()

        # A second set of statistics for each epoch
        # Later aggregated into stats
        epoch_stats = init_stats(model, minimal_stats)
        
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)    
        if epoch > 0:
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(
                train_VFE = stats['train_vfe'][-1],
                val_VFE = stats['val_vfe'][-1],
                val_acc = stats['val_acc'][-1],
            )

        for batch_i, (images, targets) in loop:
            if flatten:
                x = images.flatten(start_dim=1)
            else:
                x = images
            y = format_y(targets, model.num_classes)
            b_size = x.shape[0]
            step += b_size

            if grad_mode == 'manual':
                with torch.no_grad():
                    # Forward pass and gradient calculation
                    try:
                        out, state = model(x, y=y)
                    # catch typeerror if model is not supervised
                    except TypeError:
                        out, state = model(x)
                model.zero_grad()
                for i, layer in enumerate(model.layers):
                    if i == 0:
                        continue
                    e_below = state[i-1]['e']
                    layer.update_grad(state[i], e_below)
            else:
                # Forward pass and gradient calculation
                try:
                    out, state = model(x, y=y)
                # catch typeerror if model is not supervised
                except TypeError:
                    out, state = model(x)

                model.zero_grad()
                vfe = model.vfe(state)
                # Plots computation graph for vfe, for debugging
                if epoch == 0 and batch_i == 9:
                    make_dot(vfe).render("vfe", format="png")
                vfe.backward()

            for i, layer in enumerate(model.layers):
                if isinstance(layer, FCLI):
                    layer.weight_lat.grad = -state[i]['x'].t() @ state[i]['x'] / b_size
                    # push lat weights towards identity matrix
                    (reg_coeff * (layer.weight_lat - torch.eye(layer.weight_lat.shape[0], device=layer.weight_lat.device)).norm()).backward()
                    # Apply grad update here so optimiser doesn't add weight decay
                    layer.weight_lat.grad -= layer.weight_lat.grad.diag().diag()
                    layer.weight_lat.data -= lr * layer.weight_lat.grad
                    layer.weight_lat.grad = None
                elif isinstance(layer, FCPW): # updating data directly as we don't want weight decay
                    layer.weight_var.data -= lr * layer.weight_var.grad
                    layer.weight_var.data = torch.clamp(layer.weight_var.data, min=0.01)
                    layer.weight_var.grad = None

            if assert_grads: model.assert_grads(state)

            # Parameter Update (Grad Descent)
            optimiser.step()
            if c_optimiser is not None and grad_mode=='auto':
                loss_fn(out, targets).backward()
                c_optimiser.step()

            # A negative phase pass, increases VFE for negative data
            if neg_coeff is not None and neg_coeff > 0: neg_pass(model, x, targets, neg_coeff)

            # Track batch statistics
            epoch_stats['train_vfe'].append(model.vfe(state).item())
            epoch_stats['train_corr'].append(calc_corr(state).item())
            if not minimal_stats:
                for i, layer in enumerate(model.layers):
                    epoch_stats['X_norms'][i].append(state[i]['x'].norm(dim=1).mean().item())
                    epoch_stats['E_mags'][i].append(state[i]['e'].square().mean().item())
                    if layer.in_features is not None:
                        epoch_stats['WeightTD_means'][i-1].append(layer.weight_td.mean().item())
                        epoch_stats['WeightTD_stds'][i-1].append(layer.weight_td.std().item())
                        if not model.layers[i].symmetric:
                            epoch_stats['WeightBU_means'][i-1].append(layer.weight_bu.mean().item())
                            epoch_stats['WeightBU_stds'][i-1].append(layer.weight_bu.std().item())
                        if model.layers[i].bias is not None:
                            epoch_stats['Bias_means'][i-1].append(layer.bias.mean().item())
                            epoch_stats['Bias_stds'][i-1].append(layer.bias.std().item())
                    if isinstance(layer, FCPW) and i < len(epoch_stats['WeightVar_means']):
                        epoch_stats['WeightVar_means'][i].append(layer.weight_var.mean().item())
                        epoch_stats['WeightVar_stds'][i].append(layer.weight_var.std().item())

        val_vfe, val_acc = val_pass(model, val_loader)

        # Track epoch statistics
        stats['train_vfe'].append(torch.tensor(epoch_stats['train_vfe']).mean().item())
        stats['train_corr'].append(torch.tensor(epoch_stats['train_corr']).mean().item())
        stats['val_acc'].append(val_acc)
        stats['val_vfe'].append(val_vfe)
        if not minimal_stats:
            for i, layer in enumerate(model.layers):
                stats['X_norms'][i].append(torch.tensor(epoch_stats['X_norms'][i]).mean().item())
                stats['E_mags'][i].append(torch.tensor(epoch_stats['E_mags'][i]).mean().item())
                if layer.in_features is not None:
                    stats['WeightTD_means'][i-1].append(torch.tensor(epoch_stats['WeightTD_means'][i-1]).mean().item())
                    stats['WeightTD_stds'][i-1].append(torch.tensor(epoch_stats['WeightTD_stds'][i-1]).mean().item())
                    if not layer.symmetric:
                        stats['WeightBU_means'][i-1].append(torch.tensor(epoch_stats['WeightBU_means'][i-1]).mean().item())
                        stats['WeightBU_stds'][i-1].append(torch.tensor(epoch_stats['WeightBU_stds'][i-1]).mean().item())
                    if layer.bias is not None:
                        stats['Bias_means'][i-1].append(torch.tensor(epoch_stats['Bias_means'][i-1]).mean().item())
                        stats['Bias_stds'][i-1].append(torch.tensor(epoch_stats['Bias_stds'][i-1]).mean().item())
                if isinstance(layer, FCPW) and i < len(stats['WeightVar_means']):
                    stats['WeightVar_means'][i].append(torch.tensor(epoch_stats['WeightVar_means'][i]).mean().item())
                    stats['WeightVar_stds'][i].append(torch.tensor(epoch_stats['WeightVar_stds'][i]).mean().item())
    return step, stats
