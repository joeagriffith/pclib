import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from pclib.optim.eval import topk_accuracy, evaluate_pc
from pclib.nn.layers import DualPopulation, PrecisionWeighted, PrecisionWeightedV2
from pclib.utils.functional import vfe

def train(
    model, 
    train_data,
    val_data,
    num_epochs,
    lr = 3e-4,
    batch_size=1,
    reg_coeff = 1e-2,
    init_mode='rand',
    step=0, 
    stats=None,
    device="cpu",
):

    train_loader = train_data if isinstance(train_data, DataLoader) else DataLoader(train_data, batch_size, shuffle=True)
    val_loader = val_data if isinstance(val_data, DataLoader) else DataLoader(val_data, batch_size, shuffle=False)

    # Track epochs, indexed: [layer][epoch]
    if stats is None:
        stats = {
            "R_norms": [[] for _ in range(len(model.layers))],
            "E_mags": [[] for _ in range(len(model.layers))],
            "WeightTD_means": [[] for _ in range(len(model.layers))],
            "WeightTD_stds": [[] for _ in range(len(model.layers))],
            "Bias_means": [[] for _ in range(len(model.layers))],
            "Bias_stds": [[] for _ in range(len(model.layers))],
            "WeightBU_means": [[] for _ in range(len(model.layers))],
            "WeightBU_stds": [[] for _ in range(len(model.layers))],
            "train_vfe": [],
            "val_acc": [],
            "val_vfe": [],
        }
        

    for epoch in range(num_epochs):

        # Track batches, indexed: [layer][batch]
        epoch_stats = {
            "R_norms": [[] for _ in range(len(model.layers))],
            "E_mags": [[] for _ in range(len(model.layers))],
            "WeightTD_means": [[] for _ in range(len(model.layers))],
            "WeightTD_stds": [[] for _ in range(len(model.layers))],
            "Bias_means": [[] for _ in range(len(model.layers))],
            "Bias_stds": [[] for _ in range(len(model.layers))],
            "WeightBU_means": [[] for _ in range(len(model.layers))],
            "WeightBU_stds": [[] for _ in range(len(model.layers))],
            "train_vfe": [],
        }
        
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)    
        if epoch > 0:
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(
                train_VFE = stats['train_vfe'][-1],
                val_acc = stats['val_acc'][-1],
                val_VFE = stats['val_vfe'][-1],
            )

        for batch_idx, (images, target) in loop:
            x = images.flatten(start_dim=1)
            y = F.one_hot(target, model.num_classes).float()
            b_size = x.shape[0]
            step += b_size

            # Initialise belief and error tensors
            state = model.init_state(b_size, mode=init_mode)

            with torch.no_grad():
                # Forward pass
                out, state = model(x, state, y)

                # Calculate grads
                # for i in range(len(model.layers)):
                for i, layer in enumerate(model.layers):
                    # layer.weight_td.grad = -(state[i][1].T @ state[i][0]) / b_size
                    # layer.weight_td.grad = -((state[i][1].T @ state[i][0]) * layer.d_actv_fn(layer.weight_td.data)) / b_size
                    pred = F.linear(state[i][0], layer.weight_td, layer.bias)
                    layer.weight_td.grad = -((state[i][1] * layer.d_actv_fn(pred)).T @ state[i][0]) / b_size
                    if layer.bias is not None:
                        if isinstance(layer, DualPopulation):
                            raise("Not Tested with new error update to bias")
                            target_bias = x.mean(axis=0) if i == 0 else state[i-1][0].mean(axis=0)
                            target_bias = torch.cat((target_bias, target_bias))
                            layer.bias.grad = layers.bias - target_bias.expand(layer.bias.shape)
                        else:
                            # layer.bias.grad = -state[i][1].mean(axis=0)
                            layer.bias.grad = -(state[i][1] * layer.d_actv_fn(pred)).mean(axis=0)
                        
                        if not model.layers[i].symmetric:
                            layer.weight_bu.grad = -(state[i][0].T @ state[i][1]) / b_size

                        if isinstance(model.layers[i], (PrecisionWeighted, PrecisionWeightedV2)):
                            layer.weight_var.grad = 0.1 * -((state[i][2].T @ state[i][1]) / b_size - torch.eye(layer.weight_var.shape[0], device=device))

                
            # Regularisation
            reg = 0
            for i in range(len(model.layers)):
                reg += model.layers[i].weight_td.square().sum()
                if model.layers[i].bias is not None:
                    reg += model.layers[i].bias.square().sum()
                if not model.layers[i].symmetric:
                    reg += model.layers[i].weight_bu.square().sum()
                # if isinstance(model.layers[i], (PrecisionWeighted, PrecisionWeightedV2):
                #     reg += model.layers[i].weight_var.square().sum()
            reg *= reg_coeff
            reg.backward()

            # Parameter Update
            for i in range(len(model.layers)):
                model.layers[i].weight_td.data -= lr * model.layers[i].weight_td.grad
                if model.layers[i].bias is not None:
                    assert model.layers[i].bias.grad is not None, f"layer {i} bias has no grad"
                    model.layers[i].bias.data -= lr * model.layers[i].bias.grad
                if not model.layers[i].symmetric:
                    model.layers[i].weight_bu.data -= lr * model.layers[i].weight_bu.grad
                if isinstance(model.layers[i], (PrecisionWeighted, PrecisionWeightedV2)):
                    model.layers[i].weight_var.data -= lr * model.layers[i].weight_var.grad

            # Track batch statistics
            for i in range(len(model.layers)):
                epoch_stats['R_norms'][i].append(state[i][0].norm(dim=1).mean().item())
                epoch_stats['E_mags'][i].append(state[i][1].square().mean().item())
                epoch_stats['WeightTD_means'][i].append(model.layers[i].weight_td.mean().item())
                epoch_stats['WeightTD_stds'][i].append(model.layers[i].weight_td.std().item())
                epoch_stats['train_vfe'].append(vfe(state).item())
                if not model.layers[i].symmetric:
                    epoch_stats['WeightBU_means'][i].append(model.layers[i].weight_bu.mean().item())
                    epoch_stats['WeightBU_stds'][i].append(model.layers[i].weight_bu.std().item())
                if model.layers[i].bias is not None:
                    epoch_stats['Bias_means'][i].append(model.layers[i].bias.mean().item())
                    epoch_stats['Bias_stds'][i].append(model.layers[i].bias.std().item())

        with torch.no_grad():
            val_correct = 0
            val_vfe = 0
            for batch_idx, (images, target) in enumerate(val_loader):
                x = images.flatten(start_dim=1)
                
                state = model.init_state(x.shape[0], mode=init_mode)

                # Forward pass
                out, state = model(x, state)
                val_vfe += vfe(state, batch_reduction='sum').item()
                val_correct += (out.argmax(dim=1) == target).sum().item()

            val_acc = val_correct / len(val_data)
            val_vfe /= len(val_data)


        # Track epoch statistics
        for i in range(len(model.layers)):
            stats['R_norms'][i].append(torch.tensor(epoch_stats['R_norms'][i]).mean().item())
            stats['E_mags'][i].append(torch.tensor(epoch_stats['E_mags'][i]).mean().item())
            stats['WeightTD_means'][i].append(torch.tensor(epoch_stats['WeightTD_means'][i]).mean().item())
            stats['WeightTD_stds'][i].append(torch.tensor(epoch_stats['WeightTD_stds'][i]).mean().item())
            stats['train_vfe'].append(torch.tensor(epoch_stats['train_vfe']).mean().item())
            if not model.layers[i].symmetric:
                stats['WeightBU_means'][i].append(torch.tensor(epoch_stats['WeightBU_means'][i]).mean().item())
                stats['WeightBU_stds'][i].append(torch.tensor(epoch_stats['WeightBU_stds'][i]).mean().item())
            if model.layers[i].bias is not None:
                stats['Bias_means'][i].append(torch.tensor(epoch_stats['Bias_means'][i]).mean().item())
                stats['Bias_stds'][i].append(torch.tensor(epoch_stats['Bias_stds'][i]).mean().item())
        stats['val_acc'].append(val_acc)
        stats['val_vfe'].append(val_vfe)
    return step, stats
