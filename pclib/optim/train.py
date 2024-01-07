import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter

from pclib.optim.eval import topk_accuracy
from pclib.nn.layers import FCPW, FCLI, Conv2dLi
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

def init_stats(model, minimal=False, loss=False):
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
    if loss:
        stats['train_loss'] = []
        stats['val_loss'] = []
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

def untr_pass(model, x, untr_coeff):
    """
    | Calculates incorrect ys and performs inference on them.
    | Then multiply vfe by -neg_coeff and backpropagate to increase vfe for negative data.

    Args:
        | model (nn.Module): model to train
        | x (torch.Tensor): input data
        | targets (torch.Tensor): targets
        | neg_coeff (float): coefficient to multiply vfe by. 1.0 for balanced positive and negative passes. Must be positive.
    """
    input = torch.randn_like(x)*2 + x
    input = F.normalize(input, dim=1) * x.norm(dim=1).mean()

    grad_before = model.layers[1].weight_td.grad.clone()
    # Forward pass
    _, neg_state = model(x)
    loss = -untr_coeff * model.vfe(neg_state)
    loss.backward()
    grad_after = model.layers[1].weight_td.grad.clone()
    if (grad_before == grad_after).all():
        raise RuntimeError("Untrained pass did not update gradients")

def val_pass(model, val_loader, flatten=True, allow_grads=False, return_loss=False):
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
    with torch.set_grad_enabled(allow_grads):
        model.eval()
        acc = torch.tensor(0.0).to(model.device)
        vfe = torch.tensor(0.0).to(model.device)
        loss = torch.tensor(0.0).to(model.device)
        for images, target in val_loader:
            if flatten:
                x = images.flatten(start_dim=1)
            else:
                x = images

            # Forward pass
            out, state = model(x)
            if return_loss:
                loss += F.cross_entropy(out, target, reduction='sum')
            vfe += model.vfe(state, batch_reduction='sum')
            acc += (out.argmax(dim=1) == target).sum()

        acc /= len(val_loader.dataset)
        vfe /= len(val_loader.dataset)
        loss /= len(val_loader.dataset)
    return {'vfe': vfe, 'acc': acc, 'loss': loss}

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
    untr_coeff=None,
    log_dir=None,
    minimal_stats=False,
    track_corr=False,
    assert_grads=False,
    val_grads=False,
    save_best=True,
    grad_mode='auto',
    optim='AdamW',
    scheduler=None,
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
    assert scheduler in [None, 'ReduceLROnPlateau'], f"Invalid scheduler '{scheduler}', or not yet implemented"

    optimiser = get_optimiser(model.parameters(), lr, reg_coeff, optim)
    if scheduler == 'ReduceLROnPlateau':
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, verbose=True, factor=0.1)


    if hasattr(model, 'classifier') and c_lr > 0:
        c_optimiser = get_optimiser(model.classifier.parameters(), c_lr, reg_coeff, optim)
        loss_fn = F.cross_entropy
        if scheduler == 'ReduceLROnPlateau':
            c_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(c_optimiser, patience=5, cooldown=10, verbose=True, factor=0.1)
    else:
        c_optimiser = None

    train_params = {
        'num_epochs': num_epochs,
        'lr': lr,
        'c_lr': c_lr,
        'batch_size': batch_size,
        'reg_coeff': reg_coeff,
        'flatten': flatten,
        'neg_coeff': neg_coeff,
        'untr_coeff': untr_coeff,
        'log_dir': log_dir,
        'grad_mode': grad_mode,
        'optim': optim,
    }

    if log_dir is not None:
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_text('model', str(model).replace('\n', '<br/>').replace(' ', '&nbsp;'))
        writer.add_text('modules', '\n'.join([str(module) for module in model.modules()]).replace('\n', '<br/>').replace(' ', '&nbsp;'))
        writer.add_text('train_params', str(train_params).replace(',', '<br/>').replace('{', '').replace('}', '').replace(' ', '&nbsp;').replace("'", ''), model.epochs_trained.item())
        writer.add_text('optimiser', str(optimiser).replace('\n', '<br/>').replace(' ', '&nbsp;'), model.epochs_trained.item())
        if c_optimiser is not None:
            writer.add_text('c_optimiser', str(c_optimiser).replace('\n', '<br/>').replace(' ', '&nbsp;'), model.epochs_trained.item())

    if save_best:
        if log_dir is not None:
            weight_dir = log_dir.replace('logs', 'weights')
            weight_dir = weight_dir + '.pt'
        else:
            raise ValueError("save_best=True requires log_dir to be specified")

    assert grad_mode in ['auto', 'manual'], f"Invalid grad_mode {grad_mode}, must be 'auto' or 'manual'"
    if grad_mode == 'manual':
        assert(assert_grads == False), "assert_grads must be False when grad_mode is 'manual'"



    train_loader = train_data if isinstance(train_data, DataLoader) else DataLoader(train_data, batch_size, shuffle=True)
    if val_data is not None:
        val_loader = val_data if isinstance(val_data, DataLoader) else DataLoader(val_data, batch_size, shuffle=False)

    stats = {}
    for epoch in range(num_epochs):

        # This applies the same transform to every image.
        # Might be better to apply a different transform to each image.
        # or atleast to each batch.
        train_data.apply_transform()

        # A second set of statistics for each epoch
        # Later aggregated into stats
        epoch_stats = init_stats(model, minimal_stats, c_optimiser is not None)
        
        model.train()
        loop = tqdm(train_loader, total=len(train_loader), leave=False)    

        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
        loop.set_postfix(stats)

        for batch_i, (images, targets) in enumerate(loop):
            if flatten:
                x = images.flatten(start_dim=1)
            else:
                x = images
            y = format_y(targets, model.num_classes)
            b_size = x.shape[0]

            if grad_mode == 'manual':
                # with torch.no_grad():
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
                # if epoch == 0 and batch_i == 0:
                #     make_dot(vfe).render("vfe", format="png")
                vfe.backward()

            for i, layer in enumerate(model.layers):
                if isinstance(layer, FCLI):
                    # Hebbian update to reduce correlations between neurons
                    layer.weight_lat.grad = state[i]['x'].t() @ state[i]['x'] / b_size
                    # zero diagonal of grad so self-connections are not updated, (stay at 1.0)
                    layer.weight_lat.grad -= layer.weight_lat.grad.diag().diag()
                    # Apply grad update here so optimiser doesn't add weight decay
                    if lr > 0:
                        with torch.no_grad():
                            layer.update_mov_avg(state[i])
                # elif isinstance(layer, FCPW): # updating data directly as we don't want weight decay
                #     layer.weight_var.data -= lr * layer.weight_var.grad
                #     layer.weight_var.data = torch.clamp(layer.weight_var.data, min=0.01)
                #     layer.weight_var.grad = None

            if assert_grads: model.assert_grads(state)

            # Peroforms a negative pass, check function for details
            if neg_coeff is not None and neg_coeff > 0: neg_pass(model, x, targets, neg_coeff)

            # Performs untraining pass, check function for details
            if untr_coeff is not None and untr_coeff > 0: untr_pass(model, x, untr_coeff)

            # Parameter Update (Grad Descent)
            optimiser.step()
            if c_optimiser is not None:# and grad_mode=='auto':
                train_loss = loss_fn(out, targets)
                train_loss.backward()
                c_optimiser.step()
                epoch_stats['train_loss'].append(train_loss.item())
            optimiser.step()


            # Track batch statistics
            epoch_stats['train_vfe'].append(model.vfe(state, batch_reduction='sum').item())
            if track_corr:
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
                    # if isinstance(layer, FCPW) and i < len(epoch_stats['WeightVar_means']):
                    #     epoch_stats['WeightVar_means'][i].append(layer.weight_var.mean().item())
                    #     epoch_stats['WeightVar_stds'][i].append(layer.weight_var.std().item())

        

        # Compiles statistics from each batch into a mean statistic for the epoch
        stats['train_vfe'] = torch.tensor(epoch_stats['train_vfe']).sum().item() / len(train_loader.dataset)
        if log_dir:
            writer.add_scalar('VFE/train', stats['train_vfe'], model.epochs_trained.item())
        if track_corr:
            stats['train_corr'] = torch.tensor(epoch_stats['train_corr']).mean().item()
            if log_dir:
                writer.add_scalar('Corr/train', stats['train_corr'], model.epochs_trained.item())
        if c_optimiser is not None:
            stats['train_loss'] = torch.tensor(epoch_stats['train_loss']).mean().item()
            if log_dir:
                writer.add_scalar('Loss/train', stats['train_loss'], model.epochs_trained.item())
        
        if scheduler is not None:
            sched.step(stats['train_vfe'])
        if c_optimiser is not None and scheduler is not None:
            c_sched.step(stats['train_loss'])

        # Collects statistics for validation data if it exists
        if val_data is not None:
            val_results = val_pass(model, val_loader, flatten, val_grads, c_optimiser is not None)
            stats['val_vfe'] = val_results['vfe'].item()
            stats['val_acc'] = val_results['acc'].item()
            if c_optimiser is not None:
                stats['val_loss'] = val_results['loss'].item()

            if log_dir:
                writer.add_scalar('Accuracy/val', stats['val_acc'], model.epochs_trained.item())
                writer.add_scalar('VFE/val', stats['val_vfe'], model.epochs_trained.item())
                if c_optimiser is not None:
                    writer.add_scalar('Loss/val', stats['val_loss'], model.epochs_trained.item())
        
        # Saves model if it has the lowest validation VFE (or training VFE if no validation data) compared to previous training
        if save_best:
            current_vfe = stats['val_vfe'] if val_data is not None else stats['train_vfe']
            if current_vfe < model.min_vfe:
                torch.save(model.state_dict(), weight_dir)
                model.min_vfe = torch.tensor(current_vfe)

        if log_dir:
            if not minimal_stats:
                for i, layer in enumerate(model.layers):
                    writer.add_scalar(f'X_norms/layer_{i}', torch.tensor(epoch_stats['X_norms'][i]).mean().item(), model.epochs_trained.item())
                    writer.add_scalar(f'E_mags/layer_{i}', torch.tensor(epoch_stats['E_mags'][i]).mean().item(), model.epochs_trained.item())
                    if layer.in_features is not None:
                        writer.add_scalar(f'WeightTD_means/layer_{i}', torch.tensor(epoch_stats['WeightTD_means'][i-1]).mean().item(), model.epochs_trained.item())
                        writer.add_scalar(f'WeightTD_stds/layer_{i}', torch.tensor(epoch_stats['WeightTD_stds'][i-1]).mean().item(), model.epochs_trained.item())
                        if not layer.symmetric:
                            writer.add_scalar(f'WeightBU_means/layer_{i}', torch.tensor(epoch_stats['WeightBU_means'][i-1]).mean().item(), model.epochs_trained.item())
                            writer.add_scalar(f'WeightBU_stds/layer_{i}', torch.tensor(epoch_stats['WeightBU_stds'][i-1]).mean().item(), model.epochs_trained.item())
                        if layer.bias is not None:
                            writer.add_scalar(f'Bias_means/layer_{i}', torch.tensor(epoch_stats['Bias_means'][i-1]).mean().item(), model.epochs_trained.item())
                            writer.add_scalar(f'Bias_stds/layer_{i}', torch.tensor(epoch_stats['Bias_stds'][i-1]).mean().item(), model.epochs_trained.item())
                    # if isinstance(layer, FCPW):
                    #     writer.add_scalar(f'WeightVar_means/layer_{i}', torch.tensor(epoch_stats['WeightVar_means'][i]).mean().item(), model.epochs_trained.item())
                    #     writer.add_scalar(f'WeightVar_stds/layer_{i}', torch.tensor(epoch_stats['WeightVar_stds'][i]).mean().item(), model.epochs_trained.item())
        
        model.inc_epochs()