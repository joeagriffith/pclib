import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from pclib.optim.eval import topk_accuracy
from pclib.utils.functional import format_y, calc_corr, calc_sparsity

def get_optimiser(parameters:list, lr:float, weight_decay:float, maximize:bool, optimiser:str = 'AdamW', no_momentum:bool = False):
    """
    | Builds an optimiser from the specified arguments

    Parameters
    ----------
        parameters : list
            list of PyTorch parameters to optimiser. Usually model.parameters()
        lr : float
            learning rate
        weight_decay : float
            weight decay
        optimiser : str
            optimiser to use. ['AdamW', 'Adam', 'SGD', 'RMSprop']
    
    Returns
    -------
        torch.optim.Optimiser
    """
    assert optimiser in ['AdamW', 'Adam', 'SGD', 'RMSprop'], f"Invalid optimiser {optimiser}"
    if optimiser == 'AdamW':
        if no_momentum:
            return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, maximize=maximize, betas=(0.0, 0.0))
        else:
            return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, maximize=maximize)
    elif optimiser == 'Adam':
        if no_momentum:
            return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay, maximize=maximize, betas=(0.0, 0.0))
        else:
            return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay, maximize=maximize)
    elif optimiser == 'SGD':
        if no_momentum:
            return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, maximize=maximize, momentum=0.0)
        else:
            return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, maximize=maximize, momentum=0.8)
    elif optimiser == 'RMSprop':
        if no_momentum:
            return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay, maximize=maximize, momentum=0.0)
        else:
            return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay, maximize=maximize, momentum=0.8)

def init_stats(model:torch.nn.Module, minimal:bool = False, loss:bool = False):
    """
    | Initialises a dictionary to store statistics
    
    Parameters
    ----------
        model : torch.nn.Module
            model to track statistics for
        minimal : bool
            if True, only track minimal statistics (train_vfe, train_corr, val_vfe, val_acc)
    
    Returns
    -------
        dict
            dictionary to store statistics
    """
    if not minimal:
        stats = {
            "X_norms": [[] for _ in range(len(model.layers))],
            "E_mags": [[] for _ in range(len(model.layers))],
            "train_vfe": [],
            "train_corr": [],
            "train_sparsity": [],
            "val_vfe": [],
            "val_acc": [],
            "train_sparsity": [],
            "train_corr": [],
        }
    else:
        stats = {
            "train_vfe": [],
            "val_vfe": [],
            "val_acc": [],
        }
    if loss:
        stats['train_loss'] = []
        stats['val_loss'] = []
    return stats

def neg_pass(model:torch.nn.Module, x:torch.Tensor, labels:torch.Tensor, neg_coeff:float=1.0):
    """
    | Calculates incorrect y
    | Then multiply vfe by -neg_coeff and backpropagate to increase vfe for negative data.

    Parameters
    ----------
        model : torch.nn.Module
            model to train
        x : torch.Tensor
            input data
        labels : torch.Tensor
            target labels
        neg_coeff : float
            coefficient to multiply vfe by. 1.0 for balanced positive and negative passes. Must be positive.
    """
    assert neg_coeff > 0, f"neg_coeff must be positive, got {neg_coeff}"
    false_targets = (labels + torch.randint_like(labels, low=1, high=model.num_classes)) % model.num_classes
    false_y = format_y(false_targets, model.num_classes)

    # Forward pass
    _, neg_state = model(x, y=false_y)
    loss = -neg_coeff * model.vfe(neg_state)
    loss.backward()

def untr_pass(model:torch.nn.Module, x:torch.Tensor, y:torch.Tensor=None, untr_coeff:float=0.1):
    """
    | Performs an inference pass with the model, always supplying x, and y if it is not None.
    | However, the gradients calculated with respect to 'vfe * -untr_coeff', so that the model is untrained.
    | This function is inspired by work in the Hopfield Network literature, which aims to reduce the occurence of spurious minima.

    Parameters
    ----------
        model : torch.nn.Module
            model to train
        x : torch.Tensor
            input data
        untr_coeff : float
            coefficient to multiply vfe by. 1.0 for balanced positive and negative passes. Must be positive.
    """
    assert untr_coeff > 0, f"untr_coeff must be positive, got {untr_coeff}"
    # Forward pass
    x = torch.rand_like(x)
    try:
        _, neg_state = model(x, y=y)
    # catch typeerror if model is not supervised
    except TypeError:
        _, neg_state = model(x)
    loss = -untr_coeff * model.vfe(neg_state)
    loss.backward()

def val_pass(model:torch.nn.Module, val_loader:torch.utils.data.DataLoader, flatten:bool=True, return_loss:bool=False, learn_layer:int=None):
    """
    | Performs a validation pass on the model

    Parameters
    ----------
        model : torch.nn.Module
            model to validate
        val_loader : torch.utils.data.DataLoader
            validation data
        flatten : bool
            if True, flatten input data
        learn_layer : int
            if not None, only performs inference using first learn_layer layers, and calculates vfe only from layer learn_layer. Only works for unsupervised models.

    Returns
    -------
        dict
            dictionary of validation results
    """
    with torch.no_grad():
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
            if learn_layer is not None:
                out, state = model(x, learn_layer=learn_layer)
            else:
                out, state = model(x)
            if return_loss:
                loss += F.cross_entropy(out, target, reduction='sum')
            if learn_layer is not None:
                vfe += model.vfe(state, batch_reduction='sum', learn_layer=learn_layer)
            else:
                vfe += model.vfe(state, batch_reduction='sum')
            acc += (out.argmax(dim=1) == target).sum()

        acc /= len(val_loader.dataset)
        vfe /= len(val_loader.dataset)
        loss /= len(val_loader.dataset)
    return {'vfe': vfe, 'acc': acc, 'loss': loss}

def train(
    model:torch.nn.Module, 
    train_data:torch.utils.data.Dataset,
    val_data:torch.utils.data.Dataset,
    num_epochs:int,
    lr:float = 3e-4,
    c_lr:float = 1e-3,
    learn_on_step:bool = False,
    batch_size:int = 1,
    reg_coeff:float = 1e-2,
    flatten:bool = True,
    neg_coeff:float = None,
    untr_coeff:float = None,
    log_dir:str = None,
    model_dir:str = None,
    minimal_stats:bool = False,
    assert_grads:bool = False,
    optim:str = 'AdamW',
    scheduler:str = None,
    no_momentum:bool = False,
    norm_grads:bool = False,
    norm_weights:bool = False,
    learn_layer:int = None
):
    """
    | Trains a model with the specified parameters
    | Can train any model, supervised or unsupervised, standard, symmetric or inverted.

    Parameters
    ----------
        model : torch.nn.Module
            model to train
        train_data : Dataset or DataLoader
            training data
        val_data : Dataset or DataLoader
            validation data
        num_epochs : int
            number of epochs to train for
        lr : float
            learning rate for PC layers.
        c_lr : float
            learning rate for classifier. Ignored if model has no classifier.
        learn_on_step : bool
            if True, backpropagate on every model.step(), if False, backpropagate after model.forward().
        batch_size : int
            batch size
        reg_coeff : float
            weight decay. Also used for optimising classifier.
        flatten : bool
            if True, flatten input data
        neg_coeff : float
            coefficient to multiply vfe by during negative pass. 1.0 for balanced positive and negative passes. Must be positive.
        untr_coeff : float
            coefficient to multiply vfe by during untraining pass. 1.0 for equal training and untraining (suggest ~0.1). Must be positive.
        minimal_stats : bool
            if True, only track minimal statistics (train_vfe, val_vfe, val_acc)
        assert_grads : bool
            if True, assert that gradients are close to manual gradients. Must be false if grad_mode is 'manual'.
        grad_mode : str
            gradient mode. ['auto', 'manual']
        optim : str
            optimiser to use. ['AdamW', 'Adam', 'SGD', 'RMSprop']
        scheduler : str
            scheduler to use. [None, 'ReduceLROnPlateau']
        no momentum : bool
            if True, momentum is set to 0.0 for optimiser. Only works for AdamW, Adam, SGD, RMSprop
        norm_grads : bool
            if True, normalise gradients to have norm 1.0 along last dimension.
        norm_weights: bool
            if True, layer weights are constrained to have unit columns
        learn_layer : int
            if not None, only learn the specified layer. Must be in range(model.num_layers). Only works for unsupervised models. Remember, layer 0 does not have weights, so start from 1 for greedy layer-wise learning.
    """
    assert scheduler in [None, 'ReduceLROnPlateau'], f"Invalid scheduler '{scheduler}', or not yet implemented"
    if learn_on_step:
        assert lr > 0, "lr must be positive when learn_on_step=True"

    optimiser = get_optimiser(model.parameters(), lr, reg_coeff, True, optim, no_momentum)
    if scheduler == 'ReduceLROnPlateau':
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, verbose=True, factor=0.1, mode='max')


    if hasattr(model, 'classifier') and c_lr > 0:
        c_optimiser = get_optimiser(model.classifier.parameters(), c_lr, reg_coeff, False, optim,)
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

    train_loader = train_data if isinstance(train_data, DataLoader) else DataLoader(train_data, batch_size, shuffle=True)
    if val_data is not None:
        val_loader = val_data if isinstance(val_data, DataLoader) else DataLoader(val_data, batch_size, shuffle=False)

    stats = {}
    for epoch in range(num_epochs):

        # This applies the same transform to dataset in batches.
        # Items in same batch with have same augmentation, but process is much faster.
        if hasattr(train_data, 'apply_transform'):
            train_data.apply_transform(batch_size=batch_size)

        # A second set of statistics for each epoch
        # Later aggregated into stats
        epoch_stats = init_stats(model, minimal_stats, c_optimiser is not None)
        
        model.train()
        loop = tqdm(train_loader, total=len(train_loader), leave=False)    

        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
        loop.set_postfix(stats)

        for images, targets in loop:
            if flatten:
                x = images.flatten(start_dim=1)
            else:
                x = images
            y = format_y(targets, model.num_classes)

            model.zero_grad()
            # Forward pass and gradient calculation
            try:
                out, state = model(x, y=y, learn_on_step=learn_on_step)
            # catch typeerror if model is not supervised
            except TypeError:
                if learn_layer is None:
                    out, state = model(x, learn_on_step=learn_on_step)
                else:
                    out, state = model(x, learn_on_step=learn_on_step, learn_layer=learn_layer)
            if lr > 0 and not learn_on_step:
                if learn_layer is None:
                    vfe = model.vfe(state)
                else:
                    vfe = model.vfe(state, learn_layer=learn_layer)
                if norm_grads:
                    vfe /= vfe.abs().item()
                vfe.backward()

            # if norm_grads:
            #     for p in model.parameters():
            #         # p.grad = F.normalize(p.grad, p=2, dim=tuple(range(p.grad.dim())))
            #         p.grad = F.normalize(p.grad, p=2, dim=-1)

            if assert_grads: model.assert_grads(state)

            # Peroforms a negative pass, check function for details
            if neg_coeff is not None and neg_coeff > 0: neg_pass(model, x, targets, neg_coeff)

            # Performs untraining pass, check function for details
            if untr_coeff is not None and untr_coeff > 0: untr_pass(model, x, untr_coeff)

            # Parameter Update (Grad Descent)
            if lr > 0:
                optimiser.step()
            if c_optimiser is not None:# and grad_mode=='auto':
                train_loss = loss_fn(out, targets)
                train_loss.backward()
                c_optimiser.step()
                epoch_stats['train_loss'].append(train_loss.item())
            
            if norm_weights:
                for i, layer in enumerate(model.layers):
                    if i > 0:
                        layer.weight.data = F.normalize(layer.weight.data, dim=1)

            # Track batch statistics
            epoch_stats['train_vfe'].append(model.vfe(state, batch_reduction='sum').item())
            
            if not minimal_stats:
                epoch_stats['train_corr'].append(calc_corr(state).item())
                epoch_stats['train_sparsity'].append(calc_sparsity(state).item())
                for i, layer in enumerate(model.layers):
                    epoch_stats['X_norms'][i].append(state[i]['x'].norm(dim=1).mean().item())
                    epoch_stats['E_mags'][i].append(state[i]['e'].square().mean().item())


        # Collects statistics for validation data if it exists
        if val_data is not None:
            val_results = val_pass(model, val_loader, flatten, c_optimiser is not None, learn_layer=learn_layer)
            stats['val_vfe'] = val_results['vfe'].item()
            stats['val_acc'] = val_results['acc'].item()
            if c_optimiser is not None:
                stats['val_loss'] = val_results['loss'].item()

            if log_dir:
                writer.add_scalar('Accuracy/val', stats['val_acc'], model.epochs_trained.item())
                writer.add_scalar('VFE/val', stats['val_vfe'], model.epochs_trained.item())
                if c_optimiser is not None:
                    writer.add_scalar('Loss/val', stats['val_loss'], model.epochs_trained.item())

        stats['train_vfe'] = torch.tensor(epoch_stats['train_vfe']).sum().item() / len(train_loader.dataset)
        if c_optimiser is not None:
            stats['train_loss'] = torch.tensor(epoch_stats['train_loss']).mean().item()
        if log_dir:
            writer.add_scalar('VFE/train', stats['train_vfe'], model.epochs_trained.item())
            if c_optimiser is not None:
                writer.add_scalar('Loss/train', stats['train_loss'], model.epochs_trained.item())
            if not minimal_stats:
                stats['train_corr'] = torch.tensor(epoch_stats['train_corr']).mean().item()
                writer.add_scalar('Corr/train', stats['train_corr'], model.epochs_trained.item())
                stats['train_sparsity'] = torch.tensor(epoch_stats['train_sparsity']).mean().item()
                writer.add_scalar('Sparsity/train', stats['train_sparsity'], model.epochs_trained.item())
                for i, layer in enumerate(model.layers):
                    writer.add_scalar(f'X_norms/layer_{i}', torch.tensor(epoch_stats['X_norms'][i]).mean().item(), model.epochs_trained.item())
                    writer.add_scalar(f'E_mags/layer_{i}', torch.tensor(epoch_stats['E_mags'][i]).mean().item(), model.epochs_trained.item())
        
        if scheduler is not None:
            sched.step(stats['train_vfe'])
        if c_optimiser is not None and scheduler is not None:
            c_sched.step(stats['train_loss'])
        
        # Saves model if it has the lowest validation VFE (or training VFE if no validation data) compared to previous training
        if model_dir is not None:
            current_vfe = stats['val_vfe'] if val_data is not None else stats['train_vfe']
            if current_vfe > model.max_vfe:
                torch.save(model.state_dict(), model_dir)
                model.max_vfe = torch.tensor(current_vfe)

        model.inc_epochs()