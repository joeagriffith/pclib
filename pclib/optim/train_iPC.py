import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List

from pclib.optim.eval import topk_accuracy
from pclib.utils.functional import format_y, calc_corr, calc_sparsity

def get_optimiser(parameters:list, lr:float, weight_decay:float, optimiser:str = 'AdamW', no_momentum:bool = False):
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
            return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, betas=(0.0, 0.0))
        else:
            return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif optimiser == 'Adam':
        if no_momentum:
            return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay, betas=(0.0, 0.0))
        else:
            return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimiser == 'SGD':
        if no_momentum:
            return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=0.0)
        else:
            return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif optimiser == 'RMSprop':
        if no_momentum:
            return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay, momentum=0.0)
        else:
            return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay, momentum=0.8)

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
            "Layer_VFE": [[] for _ in range(len(model.layers))],
            "grad_norms": [[] for _ in range(len(model.layers)-1)],
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

def neg_pass(model:torch.nn.Module, pos_state:List[dict], neg_state:List[dict], gamma:torch.Tensor, neg_coeff:float=1.0, norm_grads:bool=False):
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
    neg_state[-1]['x'] = pos_state[-1]['x']
    model.step(neg_state, gamma)
    neg_vfe = model.vfe(neg_state, normalise=norm_grads)
    loss = -neg_coeff * neg_vfe
    loss.backward()
    with torch.no_grad():
        neg_mse = F.mse_loss(neg_state[0]['x'], pos_state[0]['x'])

    return neg_vfe, neg_mse


def contrastive_divergence(model:torch.nn.Module, pos_state:List[dict], gamma:torch.Tensor, cd_coeff:float=1.0, norm_grads:bool=False, cd_steps:int=1):
    cd_state = [{k: v.clone() for k, v in state.items()} for state in pos_state]
    for _ in range(cd_steps):
        model.step(cd_state, gamma)

    assert cd_coeff > 0, f"cd_coeff must be positive, got {cd_coeff}"
    cd_vfe = model.vfe(cd_state, normalise=norm_grads)
    loss = -cd_coeff * cd_vfe
    loss.backward()
    with torch.no_grad():
        cd_mse = F.mse_loss(cd_state[0]['x'], pos_state[0]['x'])
    
    return cd_vfe, cd_mse

def val_pass(model:torch.nn.Module, val_loader:torch.utils.data.DataLoader, flatten:bool=True, learn_layer:int=None):
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
        vfe = torch.tensor(0.0).to(model.device)
        for images, target in val_loader:
            if flatten:
                x = images.flatten(start_dim=1)
            else:
                x = images

            # Forward pass
            _, state = model(x, pin_obs=True, learn_layer=learn_layer)
            vfe += model.vfe(state, batch_reduction='sum', learn_layer=learn_layer)

        vfe /= len(val_loader.dataset)
    return {'vfe': vfe}

def val_pass_classifier(classifier:torch.nn.Module, val_batches):
    """
    | Performs a validation pass on the model

    Parameters
    ----------
        classifier : Optional[torch.nn.Module]
            classifies model output. Ignored if classifier does not exist.
        val_batches : List[Tuple[torch.Tensor, torch.Tensor]]
            list of validation data batches

    Returns
    -------
        dict
            dictionary of validation results
    """
    with torch.no_grad():
        acc = torch.tensor(0.0).to(classifier.device)
        loss = torch.tensor(0.0).to(classifier.device)
        N = 0
        for i in range(len(val_batches)):
            out = classifier(val_batches[i][0])
            loss += F.cross_entropy(out, val_batches[i][1], reduction='sum')
            acc += (out.argmax(dim=1) == val_batches[i][1]).sum()
            N += len(val_batches[i][1])

        acc /= N/100
        loss /= N
    return {'acc': acc, 'loss': loss}


def train_iPC(
    model:torch.nn.Module, 
    supervised:bool,
    train_data:torch.utils.data.Dataset,
    val_data:torch.utils.data.Dataset,
    num_epochs:int,
    eval_every:int = 1,
    lr:float = 3e-4,
    batch_size:int = 1,
    reg_coeff:float = 1e-2,
    flatten:bool = True,
    neg_coeff:float = None,
    cd_coeff:float = None,
    cd_steps:int = 1,
    log_dir:str = None,
    model_dir:str = None,
    minimal_stats:bool = False,
    assert_grads:bool = False,
    optim:str = 'AdamW',
    scheduler:str = None,
    no_momentum:bool = False,
    norm_grads:bool = False,
    norm_weights:bool = False,
    learn_layer:int = None,
):
    """
    | Trains a model with the specified parameters
    | Can train any model, supervised or unsupervised, standard, symmetric or inverted.

    Parameters
    ----------
        model : torch.nn.Module
            model to train
        supervised : bool
            if True, model is supervised. If False, model is unsupervised. (whether output layer is pinned to target or not).
        train_data : Dataset or DataLoader
            training data
        val_data : Dataset or DataLoader
            validation data
        num_epochs : int
            number of epochs to train for
        eval_every : int
            evaluate model every eval_every epochs
        lr : float
            learning rate for PC layers.
        batch_size : int
            batch size
        reg_coeff : float
            weight decay. Also used for optimising classifier.
        flatten : bool
            if True, flatten input data
        neg_coeff : float
            coefficient to multiply vfe by during negative pass. 1.0 for balanced positive and negative passes. Must be positive.
        cd_coeff : float
            coefficient to multiply vfe by during contrastive divergence. 1.0 for balanced positive and negative passes. Must be positive.
        cd_steps : int
            number of steps to take in contrastive divergence
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
            if True, normalise gradients by normalising the VFE.
        norm_weights: bool
            if True, layer weights are constrained to have unit columns
        learn_layer : int
            if not None, only learn the specified layer. Must be in range(model.num_layers). Only works for unsupervised models. Remember, layer 0 does not have weights, so start from 1 for greedy layer-wise learning.
    """
    assert scheduler in [None, 'ReduceLROnPlateau'], f"Invalid scheduler '{scheduler}', or not yet implemented"

    optimiser = get_optimiser(model.parameters(), lr, reg_coeff, optim, no_momentum)
    if scheduler == 'ReduceLROnPlateau':
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, verbose=True, factor=0.1)

    train_params = {
        'supervised': supervised,
        'num_epochs': num_epochs,
        'lr': lr,
        'batch_size': batch_size,
        'reg_coeff': reg_coeff,
        'flatten': flatten,
        'neg_coeff': neg_coeff,
        'cd_coeff': cd_coeff,
        'cd_steps': cd_steps,
        'optim': optim,
        'scheduler': scheduler,
        'no_momentum': no_momentum,
        'norm_grads': norm_grads,
        'norm_weights': norm_weights,
        'learn_layer': learn_layer,
    }

    if log_dir is not None:
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_text('model', str(model).replace('\n', '<br/>').replace(' ', '&nbsp;'))
        writer.add_text('modules', '\n'.join([str(module) for module in model.modules()]).replace('\n', '<br/>').replace(' ', '&nbsp;'))
        writer.add_text('train_params', str(train_params).replace(',', '<br/>').replace('{', '').replace('}', '').replace(' ', '&nbsp;').replace("'", ''), model.epochs_trained.item())
        writer.add_text('optimiser', str(optimiser).replace('\n', '<br/>').replace(' ', '&nbsp;'), model.epochs_trained.item())

    train_loader = train_data if isinstance(train_data, DataLoader) else DataLoader(train_data, batch_size, shuffle=True)
    if val_data is not None:
        val_loader = val_data if isinstance(val_data, DataLoader) else DataLoader(val_data, batch_size, shuffle=False)

    if flatten:
        pos_states = [model.init_state(images.flatten(1)) for images, _ in train_loader]
        neg_states = [model.init_state(images.flatten(1)) for images, _ in train_loader]
    else:
        pos_states = [model.init_state(images) for images, _ in train_loader]
        neg_states = [model.init_state(images) for images, _ in train_loader]
    if supervised:
        for i, (_, targets) in enumerate(train_loader):
            pos_states[i][-1]['x'] = format_y(targets, model.num_classes)
            neg_states[i][-1]['x'] = format_y(targets, model.num_classes)
    
    gammas = [torch.ones(pos_states[i][-1]['x'].shape[0]).to(model.device) * model.gamma for i in range(len(pos_states))]

    stats = {}
    loop = tqdm(range(num_epochs), leave=False)
    for epoch in loop:

        # This applies the same transform to dataset in batches.
        # Items in same batch with have same augmentation, but process is much faster.
        # if hasattr(train_data, 'apply_transform'):
        #     train_data.apply_transform(batch_size=batch_size)

        # A second set of statistics for each epoch
        # Later aggregated into stats
        epoch_stats = init_stats(model, minimal_stats)
        
        model.train()

        if epoch > 0:
            loop.set_postfix(stats)

        for i in range(len(train_loader)):

            optimiser.zero_grad()
            # Forward pass and gradient calculation
            if supervised:
                model.step(pos_states[i], gammas[i], pin_obs=True, pin_target=True, learn_layer=learn_layer)
            else:
                model.step(pos_states[i], gammas[i], pin_obs=True, learn_layer=learn_layer)
            if lr > 0:
                vfe = model.vfe(pos_states[i], learn_layer=learn_layer, normalise=norm_grads)
                vfe.backward()

            if assert_grads: model.assert_grads(pos_states[i])

            # Peroforms a negative pass, check function for details
            if neg_coeff is not None and neg_coeff > 0: neg_pass(model, pos_states[i], neg_states[i], gammas[i], neg_coeff, norm_grads)

            # Performs untraining pass, check function for details
            if cd_coeff is not None and cd_coeff > 0: contrastive_divergence(model, pos_states[i], gammas[i], cd_coeff, norm_grads, cd_steps)

            # Parameter Update (Grad Descent)
            if lr > 0:
                optimiser.step()
            
            if norm_weights:
                for layer_idx, layer in enumerate(model.layers):
                    if layer_idx > 0:
                        if hasattr(layer, 'weight'):
                            layer.weight.data = F.normalize(layer.weight.data, dim=-1)
                        elif hasattr(layer, 'conv'):
                            layer.conv[0].weight.data = F.normalize(layer.conv[0].weight.data, dim=(0, 2, 3))
            
            if model.has_top:
                model.top.weight.data = model.top.weight.data - torch.diag(model.top.weight.data.diag())

            if lr > 0:
                # Track batch statistics
                epoch_stats['train_vfe'].append(model.vfe(pos_states[i], batch_reduction='sum').item())
                
                if not minimal_stats:
                    epoch_stats['train_corr'].append(calc_corr(pos_states[i]).item())
                    epoch_stats['train_sparsity'].append(calc_sparsity(pos_states[i]).item())
                    for l, layer in enumerate(model.layers):
                        epoch_stats['X_norms'][l].append(pos_states[i][l]['x'].norm(dim=1).mean().item())
                        epoch_stats['Layer_VFE'][l].append(0.5 * pos_states[i][l]['e'].square().sum(dim=[d for d in range(1, pos_states[i][l]['e'].dim())]).mean(0).item())
                        if l > 0:
                            if learn_layer is not None and l != learn_layer:
                                continue
                            epoch_stats['grad_norms'][l-1].append(layer.weight.grad.norm().item())

        # Collects statistics for validation data if it exists
        if val_data is not None and epoch % eval_every == 0:
            val_results = val_pass(model, val_loader, flatten, learn_layer=learn_layer)
            stats['val_vfe'] = val_results['vfe'].item()

            if log_dir:
                writer.add_scalar('VFE/val', stats['val_vfe'], model.epochs_trained.item())

        if lr > 0:
            stats['train_vfe'] = torch.tensor(epoch_stats['train_vfe']).sum().item() / len(train_loader.dataset)
            if log_dir:
                writer.add_scalar('VFE/train', stats['train_vfe'], model.epochs_trained.item())
                if not minimal_stats:
                    stats['train_corr'] = torch.tensor(epoch_stats['train_corr']).mean().item()
                    writer.add_scalar('Corr/train', stats['train_corr'], model.epochs_trained.item())
                    stats['train_sparsity'] = torch.tensor(epoch_stats['train_sparsity']).mean().item()
                    writer.add_scalar('Sparsity/train', stats['train_sparsity'], model.epochs_trained.item())
                    for l, layer in enumerate(model.layers):
                        writer.add_scalar(f'X_norms/layer_{l}', torch.tensor(epoch_stats['X_norms'][l]).mean().item(), model.epochs_trained.item())
                        writer.add_scalar(f'Layer_VFE/layer_{l}', torch.tensor(epoch_stats['Layer_VFE'][l]).mean().item(), model.epochs_trained.item())
                        if l > 0:
                            if learn_layer is not None and l != learn_layer:
                                continue
                            writer.add_scalar(f'grad_norms/layer_{l}', torch.tensor(epoch_stats['grad_norms'][l-1]).mean().item(), model.epochs_trained.item())
        
        if scheduler is not None and lr > 0:
            sched.step(stats['train_vfe'])
        
        if lr > 0:
            # Saves model if it has the lowest validation VFE (or training VFE if no validation data) compared to previous training
            if model_dir is not None:
                current_vfe = stats['val_vfe'] if val_data is not None else stats['train_vfe']
                if current_vfe > model.max_vfe:
                    torch.save(model.state_dict(), model_dir)
                    model.max_vfe = torch.tensor(current_vfe)

        model.inc_epochs()
    

def train_iPC_classifier(
    model:torch.nn.Module, 
    classifier:torch.nn.Module,
    train_data:torch.utils.data.Dataset,
    val_data:torch.utils.data.Dataset,
    num_epochs:int,
    lr:float = 3e-4,
    batch_size:int = 1,
    reg_coeff:float = 1e-2,
    flatten:bool = True,
    log_dir:str = None,
    optim:str = 'AdamW',
    scheduler:str = None,
):
    """
    | Trains a model with the specified parameters
    | Can train any model, supervised or unsupervised, standard, symmetric or inverted.

    Parameters
    ----------
        model : torch.nn.Module
            model to train
        classifier : torch.nn.Module
            classifier to train on model output. Ignored if classifier does not exist.
        pos_states : List[dict]
            list of positive states to train on
        val_data : Dataset or DataLoader
            validation data
        num_epochs : int
            number of epochs to train for
        lr : float
            learning rate for classifier.
        batch_size : int
            batch size
        reg_coeff : float
            weight decay. Also used for optimising classifier.
        flatten : bool
            if True, flatten input data
        log_dir : str
            directory to save tensorboard logs
        model_dir : str
            directory to save model
        optim : str
            optimiser to use. ['AdamW', 'Adam', 'SGD', 'RMSprop']
        scheduler : str
            scheduler to use. [None, 'ReduceLROnPlateau']
    """
    assert scheduler in [None, 'ReduceLROnPlateau'], f"Invalid scheduler '{scheduler}', or not yet implemented"

    optimiser = get_optimiser(classifier.parameters(), lr, reg_coeff, optim, False)
    if scheduler == 'ReduceLROnPlateau':
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, cooldown=10, verbose=True, factor=0.1)

    train_params = {
        'classifier': classifier is not None,
        'num_epochs': num_epochs,
        'lr': lr,
        'batch_size': batch_size,
        'reg_coeff': reg_coeff,
        'optim': optim,
        'scheduler': scheduler,
    }

    if log_dir is not None:
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_text('model', str(model).replace('\n', '<br/>').replace(' ', '&nbsp;'))
        writer.add_text('classifier', str(classifier).replace('\n', '<br/>').replace(' ', '&nbsp;'))
        writer.add_text('modules', '\n'.join([str(module) for module in model.modules()]).replace('\n', '<br/>').replace(' ', '&nbsp;'))
        writer.add_text('train_params', str(train_params).replace(',', '<br/>').replace('{', '').replace('}', '').replace(' ', '&nbsp;').replace("'", ''), model.epochs_trained.item())
        writer.add_text('optimiser', str(optimiser).replace('\n', '<br/>').replace(' ', '&nbsp;'), model.epochs_trained.item())

    train_loader = train_data if isinstance(train_data, DataLoader) else DataLoader(train_data, batch_size, shuffle=True)
    loop = tqdm(train_loader, leave=False)
    loop.set_description('Preprocessing training data...')
    model.eval()
    with torch.no_grad():
        if flatten:
            train_batches = [(model(x.flatten(1), pin_obs=True)[0], y) for x, y in loop]
        else:
            train_batches = [(model(x, pin_obs=True)[0], y) for x, y in loop]

        if val_data is not None:
            val_loader = val_data if isinstance(val_data, DataLoader) else DataLoader(val_data, batch_size, shuffle=False)
            loop = tqdm(val_loader, leave=False)
            loop.set_description('Preprocessing validation data...')
            if flatten:
                val_batches = [(model(x.flatten(1), pin_obs=True)[0], y) for x, y in loop]
            else:
                val_batches = [(model(x, pin_obs=True)[0], y) for x, y in loop]

    stats = {}
    loop = tqdm(range(num_epochs), leave=False)
    for epoch in loop:

        # A set of statistics for each epoch
        # Later aggregated into stats
        epoch_stats = {
            'train_loss': [],
            'train_acc': [],
        }
        
        model.train()

        if epoch > 0:
            loop.set_postfix(stats)

        for i in range(len(train_loader)):

            optimiser.zero_grad()

            out = classifier(train_batches[i][0])
            train_loss = F.cross_entropy(out, train_batches[i][1])
            train_loss.backward()
            optimiser.step()
            epoch_stats['train_loss'].append(train_loss.item())
            epoch_stats['train_acc'].append(topk_accuracy(out, train_batches[i][1], k=1).item())

            
        # Collects statistics for validation data if it exists
        if val_data is not None:
            val_results = val_pass_classifier(classifier, val_batches)
            stats['val_acc'] = val_results['acc'].item()
            stats['val_loss'] = val_results['loss'].item()
            if log_dir:
                writer.add_scalar('Accuracy/val', stats['val_acc'], epoch)
                writer.add_scalar('Loss/val', stats['val_loss'], epoch)

        stats['train_loss'] = torch.tensor(epoch_stats['train_loss']).mean().item()
        stats['train_acc'] = torch.tensor(epoch_stats['train_acc']).mean().item()
        if log_dir:
            writer.add_scalar('Loss/train', stats['train_loss'], epoch)
            writer.add_scalar('Accuracy/train', stats['train_acc'], epoch)
        
        if scheduler is not None:
            sched.step(stats['train_loss'])