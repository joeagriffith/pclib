import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from pclib.optim.eval import topk_accuracy, evaluate_pc
from pclib.nn.layers import FCPW
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
            "X_norms": [[] for _ in range(len(model.layers))],
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
            "X_norms": [[] for _ in range(len(model.layers))],
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
            false_y = F.one_hot((target + torch.randint_like(target, low=1, high=model.num_classes)) % model.num_classes, model.num_classes).float()
            y = F.one_hot(target, model.num_classes).float()
            b_size = x.shape[0]
            step += b_size


            state = model.init_state(b_size, mode=init_mode)
            # Forward pass
            with torch.no_grad():
                out, state = model(x, state, y)

            # Calculate grads, different equations for each implementation, top_down is f(Wr) or Wf(r)
            for i, layer in enumerate(model.layers):
                
                if layer.actv_mode == 'Wf(x)':
                    layer.weight_td.grad = -(state[i]['e'].T @ layer.actv_fn(state[i]['x'])) / b_size
                    if layer.bias is not None:
                        layer.bias.grad = -state[i]['e'].mean(axis=0)
                    if not layer.symmetric:
                        layer.weight_bu.grad = -(layer.actv_fn(state[i]['x']).T @ state[i]['e']) / b_size

                elif layer.actv_mode == 'f(Wx)':
                    layer.weight_td.grad = -((state[i]['e'] * layer.d_actv_fn(state[i]['pred'])).T @ state[i]['x']) / b_size
                    if layer.bias is not None:
                        layer.bias.grad = -(state[i]['e'] * layer.d_actv_fn(state[i]['pred'])).mean(axis=0)
                    if not layer.symmetric:
                        layer.weight_bu.grad = -(state[i]['x'].T @ (state[i]['e'] * layer.d_actv_fn(state[i]['pred']))) / b_size

                if isinstance(layer, FCPW):
                    layer.weight_var.grad = 0.1 * -((state[i]['eps'].T @ state[i]['e']) / b_size - torch.eye(layer.weight_var.shape[0], device=device))

            neg_state = model.init_state(b_size, mode=init_mode)
            # Forward pass
            with torch.no_grad():
                out, neg_state = model(x, neg_state, false_y)

            # Calculate grads, different equations for each implementation, top_down is f(Wr) or Wf(r)
            for i, layer in enumerate(model.layers):
                
                if layer.actv_mode == 'Wf(x)':
                    layer.weight_td.grad -= 0.5 * -(neg_state[i]['e'].T @ layer.actv_fn(neg_state[i]['x'])) / b_size
                    if layer.bias is not None:
                        layer.bias.grad -= 0.5 * -neg_state[i]['e'].mean(axis=0)
                    if not layer.symmetric:
                        layer.weight_bu.grad -= 0.5 * -(layer.actv_fn(neg_state[i]['x']).T @ neg_state[i]['e']) / b_size

                elif layer.actv_mode == 'f(Wx)':
                    layer.weight_td.grad -= 0.5 * -((neg_state[i]['e'] * layer.d_actv_fn(neg_state[i]['pred'])).T @ neg_state[i]['x']) / b_size
                    if layer.bias is not None:
                        layer.bias.grad -= 0.5 * -(neg_state[i]['e'] * layer.d_actv_fn(neg_state[i]['pred'])).mean(axis=0)
                    if not layer.symmetric:
                        layer.weight_bu.grad -= 0.5 * -(neg_state[i]['x'].T @ (neg_state[i]['e'] * layer.d_actv_fn(neg_state[i]['pred']))) / b_size
                
            # Regularisation (L2)
            reg = 0
            for i, layer in enumerate(model.layers):
                reg += layer.weight_td.square().sum()
                if layer.bias is not None:
                    reg += layer.bias.square().sum()
                if not layer.symmetric:
                    reg += layer.weight_bu.square().sum()
                # # Can't regularise variance, it's not a parameter (kinda)
                # if isinstance(model.layers[i], (PrecisionWeighted, PrecisionWeightedV2):
                #     reg += model.layers[i].weight_var.square().sum()
            reg *= reg_coeff
            reg.backward()

            # Parameter Update (Grad Descent)
            with torch.no_grad():
                for i, layer in enumerate(model.layers):
                    layer.weight_td.data -= lr * layer.weight_td.grad
                    if layer.bias is not None:
                        assert layer.bias.grad is not None, f"layer {i} bias has no grad"
                        layer.bias.data -= lr * layer.bias.grad
                    if not layer.symmetric:
                        layer.weight_bu.data -= lr * layer.weight_bu.grad
                    if isinstance(layer, FCPW):
                        layer.weight_var.data -= lr * layer.weight_var.grad

            # Track batch statistics
            for i, layer in enumerate(model.layers):
                epoch_stats['X_norms'][i].append(state[i]['x'].norm(dim=1).mean().item())
                epoch_stats['E_mags'][i].append(state[i]['e'].square().mean().item())
                epoch_stats['WeightTD_means'][i].append(layer.weight_td.mean().item())
                epoch_stats['WeightTD_stds'][i].append(layer.weight_td.std().item())
                epoch_stats['train_vfe'].append(vfe(state).item())
                if not model.layers[i].symmetric:
                    epoch_stats['WeightBU_means'][i].append(layer.weight_bu.mean().item())
                    epoch_stats['WeightBU_stds'][i].append(layer.weight_bu.std().item())
                if model.layers[i].bias is not None:
                    epoch_stats['Bias_means'][i].append(layer.bias.mean().item())
                    epoch_stats['Bias_stds'][i].append(layer.bias.std().item())


        # Validation pass
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
        for i, layer in enumerate(model.layers):
            stats['X_norms'][i].append(torch.tensor(epoch_stats['X_norms'][i]).mean().item())
            stats['E_mags'][i].append(torch.tensor(epoch_stats['E_mags'][i]).mean().item())
            stats['WeightTD_means'][i].append(torch.tensor(epoch_stats['WeightTD_means'][i]).mean().item())
            stats['WeightTD_stds'][i].append(torch.tensor(epoch_stats['WeightTD_stds'][i]).mean().item())
            stats['train_vfe'].append(torch.tensor(epoch_stats['train_vfe']).mean().item())
            if not layer.symmetric:
                stats['WeightBU_means'][i].append(torch.tensor(epoch_stats['WeightBU_means'][i]).mean().item())
                stats['WeightBU_stds'][i].append(torch.tensor(epoch_stats['WeightBU_stds'][i]).mean().item())
            if layer.bias is not None:
                stats['Bias_means'][i].append(torch.tensor(epoch_stats['Bias_means'][i]).mean().item())
                stats['Bias_stds'][i].append(torch.tensor(epoch_stats['Bias_stds'][i]).mean().item())
        stats['val_acc'].append(val_acc)
        stats['val_vfe'].append(val_vfe)
    return step, stats
