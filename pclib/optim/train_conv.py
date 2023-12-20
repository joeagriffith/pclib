import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from pclib.optim.eval import topk_accuracy
from pclib.nn.layers import FCPW
from pclib.utils.functional import format_y

def train_conv(
    model, 
    train_data,
    val_data,
    num_epochs,
    lr = 3e-4,
    batch_size=1,
    reg_coeff = 1e-2,
    flatten=True,
    init_mode='rand',
    neg_coeff=None,
    step=0, 
    stats=None,
    device="cpu",
    optim='AdamW',
):
    assert optim in ['AdamW', 'Adam', 'SGD', 'RMSprop'], f"Invalid optimiser {optim}"
    if optim == 'AdamW':
        optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=reg_coeff)
    elif optim == 'Adam':
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg_coeff)
    elif optim == 'SGD':
        optimiser = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=reg_coeff, momentum=0.9)
    elif optim == 'RMSprop':
        optimiser = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=reg_coeff, momentum=0.9)

    train_loader = train_data if isinstance(train_data, DataLoader) else DataLoader(train_data, batch_size, shuffle=True)
    val_loader = val_data if isinstance(val_data, DataLoader) else DataLoader(val_data, batch_size, shuffle=False)

    # Track epochs, indexed: [layer][epoch]
    if stats is None:
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
            "val_acc": [],
            "val_vfe": [],
        }

    for epoch in range(num_epochs):
        train_data.apply_transform()

        # Track batches, indexed: [layer][batch]
        epoch_stats = {
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
            y = format_y(targets, model.num_classes)
            false_targets = (targets + torch.randint_like(targets, low=1, high=model.num_classes)) % model.num_classes
            false_y = format_y(false_targets, model.num_classes)
            b_size = x.shape[0]
            step += b_size

            # Forward pass
            # with torch.no_grad():
            out, state = model(x, y=y)

            # Calculate grads, different equations for each implementation, top_down is f(Wr) or Wf(r)
            # Grads calculated in last step
            # model.zero_grad()
            # model.vfe(state).backward()

            # Assert grads
            # for i, layer in enumerate(model.layers):
            #     if i > 0:
            #         layer.assert_grad(state[i], state[i-1]['e'])                

            # A negative phase pass, increases VFE for negative data
            if neg_coeff is not None and neg_coeff > 0:
                # Forward pass
                with torch.no_grad():
                    out, neg_state = model(x, y=false_y)

                # Calculate grads, different equations for each implementation, top_down is f(Wr) or Wf(r)
                # for i, layer in enumerate(model.layers):
                #     if i > 0:
                #         layer.update_grad(neg_state[i], -neg_coeff * neg_state[i-1]['e'])
                loss = -neg_coeff * model.vfe(neg_state)
                loss.backward()
                
            # Parameter Update (Grad Descent)
            optimiser.step()
            for layer in model.layers:
                if isinstance(layer, FCPW):
                    raise NotImplementedError
                    layer.weight_var.data -= lr * layer.weight_var.grad
                    layer.weight_var.data = torch.clamp(layer.weight_var.data, min=0.01)

            # Track batch statistics
            epoch_stats['train_vfe'].append(model.vfe(state).item())
            for i, layer in enumerate(model.layers):
                epoch_stats['X_norms'][i].append(state[i]['x'].norm(dim=1).mean().item())
                epoch_stats['E_mags'][i].append(state[i]['e'].square().mean().item())
                # if layer.in_features is not None:
                #     epoch_stats['WeightTD_means'][i-1].append(layer.conv_td.weight.mean().item())
                #     epoch_stats['WeightTD_stds'][i-1].append(layer.conv_td.weight.std().item())
                #     if not model.layers[i].symmetric:
                #         epoch_stats['WeightBU_means'][i-1].append(layer.conv_bu.weight.mean().item())
                #         epoch_stats['WeightBU_stds'][i-1].append(layer.conv_bu.weight.std().item())
                #     if model.layers[i].bias is not None:
                #         epoch_stats['Bias_means'][i-1].append(layer.conv.bias.mean().item())
                #         epoch_stats['Bias_stds'][i-1].append(layer.conv.bias.std().item())
                # if isinstance(layer, PrecisionWeighted) and i < len(epoch_stats['WeightVar_means']):
                #     epoch_stats['WeightVar_means'][i].append(layer.weight_var.mean().item())
                #     epoch_stats['WeightVar_stds'][i].append(layer.weight_var.std().item())


        # Validation pass
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

        val_acc = val_correct / len(val_data)
        val_vfe /= len(val_data)

        # Track epoch statistics
        for i, layer in enumerate(model.layers):
            stats['X_norms'][i].append(torch.tensor(epoch_stats['X_norms'][i]).mean().item())
            stats['E_mags'][i].append(torch.tensor(epoch_stats['E_mags'][i]).mean().item())
            # if layer.in_features is not None:
            #     stats['WeightTD_means'][i-1].append(torch.tensor(epoch_stats['WeightTD_means'][i-1]).mean().item())
            #     stats['WeightTD_stds'][i-1].append(torch.tensor(epoch_stats['WeightTD_stds'][i-1]).mean().item())
            #     if not layer.symmetric:
            #         stats['WeightBU_means'][i-1].append(torch.tensor(epoch_stats['WeightBU_means'][i-1]).mean().item())
            #         stats['WeightBU_stds'][i-1].append(torch.tensor(epoch_stats['WeightBU_stds'][i-1]).mean().item())
            #     if layer.bias is not None:
            #         stats['Bias_means'][i-1].append(torch.tensor(epoch_stats['Bias_means'][i-1]).mean().item())
            #         stats['Bias_stds'][i-1].append(torch.tensor(epoch_stats['Bias_stds'][i-1]).mean().item())
            # if isinstance(layer, PrecisionWeighted) and i < len(stats['WeightVar_means']):
            #     stats['WeightVar_means'][i].append(torch.tensor(epoch_stats['WeightVar_means'][i]).mean().item())
            #     stats['WeightVar_stds'][i].append(torch.tensor(epoch_stats['WeightVar_stds'][i]).mean().item())
        stats['train_vfe'].append(torch.tensor(epoch_stats['train_vfe']).mean().item())
        stats['val_acc'].append(val_acc)
        stats['val_vfe'].append(val_vfe)
    return step, stats