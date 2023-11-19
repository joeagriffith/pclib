import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from pclib.optim.eval import topk_accuracy
from pclib.nn.layers import PrecisionWeighted
from pclib.utils.functional import vfe, format_y, calc_corr

def train(
    model, 
    train_data,
    val_data,
    num_epochs,
    pc_lr = 3e-4,
    c_lr = 3e-4,
    batch_size=1,
    reg_coeff = 1e-2,
    flatten=True,
    init_mode='rand',
    neg_coeff=None,
    step=0, 
    stats=None,
    device="cpu",
    pc_optim='AdamW',
    c_optim='AdamW',
):
    assert pc_optim in ['AdamW', 'Adam', 'SGD', 'RMSprop'], f"Invalid optimiser {pc_optim}"
    if pc_optim == 'AdamW':
        pc_optimiser = torch.optim.AdamW(model.layers.parameters(), lr=pc_lr, weight_decay=reg_coeff)
    elif pc_optim == 'Adam':
        pc_optimiser = torch.optim.Adam(model.layers.parameters(), lr=pc_lr, weight_decay=reg_coeff)
    elif pc_optim == 'SGD':
        pc_optimiser = torch.optim.SGD(model.layers.parameters(), lr=pc_lr, weight_decay=reg_coeff, momentum=0.9)
    elif pc_optim == 'RMSprop':
        pc_optimiser = torch.optim.RMSprop(model.layers.parameters(), lr=pc_lr, weight_decay=reg_coeff, momentum=0.9)
    
    assert c_optim in [None, 'AdamW', 'Adam', 'SGD', 'RMSprop'], f"Invalid optimiser {c_optim}"
    if c_optim == 'AdamW':
        c_optimiser = torch.optim.AdamW(model.classifier.parameters(), lr=c_lr, weight_decay=reg_coeff)
    elif c_optim == 'Adam':
        c_optimiser = torch.optim.Adam(model.classifier.parameters(), lr=c_lr, weight_decay=reg_coeff)
    elif c_optim == 'SGD':
        c_optimiser = torch.optim.SGD(model.classifier.parameters(), lr=c_lr, weight_decay=reg_coeff, momentum=0.9)
    elif c_optim == 'RMSprop':
        c_optimiser = torch.optim.RMSprop(model.classifier.parameters(), lr=c_lr, weight_decay=reg_coeff, momentum=0.9)

    loss_fn = F.cross_entropy

    train_loader = train_data if isinstance(train_data, DataLoader) else DataLoader(train_data, batch_size, shuffle=True)
    val_loader = val_data if isinstance(val_data, DataLoader) else DataLoader(val_data, batch_size, shuffle=False)

    # Track epochs, indexed: [layer][epoch]
    if stats is None:
        stats = {
            "R_norms": [[] for _ in range(len(model.layers))],
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
            "val_acc": [],
            "val_vfe": [],
        }
        

    for epoch in range(num_epochs):

        train_data.apply_transform()

        # Track batches, indexed: [layer][batch]
        epoch_stats = {
            "R_norms": [[] for _ in range(len(model.layers))],
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
        }
        
        model.train()
        loop = tqdm(train_loader, total=len(train_loader), leave=False)    
        if epoch > 0:
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(
                train_VFE = stats['train_vfe'][-1],
                val_acc = stats['val_acc'][-1],
                val_VFE = stats['val_vfe'][-1],
            )

        for images, targets in loop:
            if flatten:
                x = images.flatten(start_dim=1)
            else:
                x = images
            b_size = x.shape[0]
            step += b_size

            # Forward pass
            # with torch.no_grad():
            out, state = model(x)

            # Calculate grads for pc layers and classifier
            model.zero_grad()
            vfe(state).backward()
            # for i, layer in enumerate(model.layers):
            #     layer.weight_lat.grad = F.relu(state[i]['x'].t() @ state[i]['x']) / b_size
            if c_optim is not None:
                loss_fn(out, targets).backward()

            # Assert grads
            # for i, layer in enumerate(model.layers):
            #     if i > 0:
            #         layer.assert_grad(state[i], state[i-1]['e'])                

            # Track batch statistics
            epoch_stats['train_vfe'].append(model.vfe(state).item())
            epoch_stats['train_corr'].append(calc_corr(state).item())

            # A negative phase pass, increases VFE for negative data
            if neg_coeff is not None and neg_coeff > 0:
                # Forward pass
                raise(NotImplementedError) # MOVE vfe and corr trackers up from below
                
            # Parameter Update (Grad Descent)
            pc_optimiser.step()
            if c_optim is not None:
                c_optimiser.step()
            for layer in model.layers:
                # Isn't this already done by pc_optimiser.step()?
                if isinstance(layer, PrecisionWeighted):
                    layer.weight_var.data -= pc_lr * layer.weight_var.grad
                    layer.weight_var.data = torch.clamp(layer.weight_var.data, min=0.01)

            # Track batch statistics
            for i, layer in enumerate(model.layers):
                epoch_stats['R_norms'][i].append(state[i]['x'].norm(dim=1).mean().item())
                epoch_stats['E_mags'][i].append(state[i]['e'].square().mean().item())
                if layer.prev_shape is not None:
                    epoch_stats['WeightTD_means'][i-1].append(layer.weight_td.mean().item())
                    epoch_stats['WeightTD_stds'][i-1].append(layer.weight_td.std().item())
                    if not model.layers[i].symmetric:
                        epoch_stats['WeightBU_means'][i-1].append(layer.weight_bu.mean().item())
                        epoch_stats['WeightBU_stds'][i-1].append(layer.weight_bu.std().item())
                    if model.layers[i].bias is not None:
                        epoch_stats['Bias_means'][i-1].append(layer.bias.mean().item())
                        epoch_stats['Bias_stds'][i-1].append(layer.bias.std().item())
                if isinstance(layer, PrecisionWeighted) and i < len(epoch_stats['WeightVar_means']):
                    epoch_stats['WeightVar_means'][i].append(layer.weight_var.mean().item())
                    epoch_stats['WeightVar_stds'][i].append(layer.weight_var.std().item())


        # Validation pass
        with torch.no_grad():
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
                val_vfe += vfe(val_state, batch_reduction='sum').item()
                val_correct += (out.argmax(dim=1) == target).sum().item()

            val_acc = val_correct / len(val_data)
            val_vfe /= len(val_data)


        # Track epoch statistics
        for i, layer in enumerate(model.layers):
            stats['R_norms'][i].append(torch.tensor(epoch_stats['R_norms'][i]).mean().item())
            stats['E_mags'][i].append(torch.tensor(epoch_stats['E_mags'][i]).mean().item())
            if layer.prev_shape is not None:
                stats['WeightTD_means'][i-1].append(torch.tensor(epoch_stats['WeightTD_means'][i-1]).mean().item())
                stats['WeightTD_stds'][i-1].append(torch.tensor(epoch_stats['WeightTD_stds'][i-1]).mean().item())
                if not layer.symmetric:
                    stats['WeightBU_means'][i-1].append(torch.tensor(epoch_stats['WeightBU_means'][i-1]).mean().item())
                    stats['WeightBU_stds'][i-1].append(torch.tensor(epoch_stats['WeightBU_stds'][i-1]).mean().item())
                if layer.bias is not None:
                    stats['Bias_means'][i-1].append(torch.tensor(epoch_stats['Bias_means'][i-1]).mean().item())
                    stats['Bias_stds'][i-1].append(torch.tensor(epoch_stats['Bias_stds'][i-1]).mean().item())
            if isinstance(layer, PrecisionWeighted) and i < len(stats['WeightVar_means']):
                stats['WeightVar_means'][i].append(torch.tensor(epoch_stats['WeightVar_means'][i]).mean().item())
                stats['WeightVar_stds'][i].append(torch.tensor(epoch_stats['WeightVar_stds'][i]).mean().item())
        stats['train_vfe'].append(torch.tensor(epoch_stats['train_vfe']).mean().item())
        stats['train_corr'].append(torch.tensor(epoch_stats['train_corr']).mean().item())
        stats['val_acc'].append(val_acc)
        stats['val_vfe'].append(val_vfe)
    return step, stats
