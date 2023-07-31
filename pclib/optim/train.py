import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from pclib.optim.eval import topk_accuracy, evaluate_pc
from pclib.nn.layers import DualPELayer

def train(
    model, 
    train_data,
    val_data,
    num_epochs,
    lr = 3e-4,
    batch_size=1,
    reg_coeff = 1e-2,
    step=0, 
    stats=None,
    device="cpu",
):

    train_loader = train_data if isinstance(train_data, DataLoader) else DataLoader(train_data, batch_size, shuffle=True)
    val_loader = val_data if isinstance(val_data, DataLoader) else DataLoader(val_data, batch_size, shuffle=False)

    # Track epochs, indexed: [layer][epoch]
    if stats is None:
        epochs_R_mags = [[] for _ in range(len(model.layers))]
        epochs_E_mags = [[] for _ in range(len(model.layers))]
        epochs_WeightTD_means = [[] for _ in range(len(model.layers))]
        epochs_WeightTD_stds = [[] for _ in range(len(model.layers))]
        epochs_WeightBU_means = [[] for _ in range(len(model.layers))]
        epochs_WeightBU_stds = [[] for _ in range(len(model.layers))]
        

    for epoch in range(num_epochs):

        # Track batches, indexed: [layer][batch]
        batches_R_mags = [[] for _ in range(len(model.layers))]
        batches_E_mags = [[] for _ in range(len(model.layers))]
        batches_WeightTD_means = [[] for _ in range(len(model.layers))]
        batches_WeightTD_stds = [[] for _ in range(len(model.layers))]
        batches_WeightBU_means = [[] for _ in range(len(model.layers))]
        batches_WeightBU_stds = [[] for _ in range(len(model.layers))]
        
        #  Initialise variables and prepare data for new epoch
        model.train()

        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)    
        if epoch > 0:
            norm_e = sum([epochs_E_mags[i][-1] for i in range(len(model.layers))]) / len(model.layers)
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(
                norm_e = norm_e,
        #         train_err_mean = train_err.mean() / (batch_idx + 1),
        #         val_loss = val_loss, 
        #         val_acc = val_acc[0].item(),
        #         val_err_mean = val_err.mean()
            )

        for batch_idx, (images, target) in loop:
            x = images.flatten(start_dim=1).to(device)
            y = F.one_hot(target, model.num_classes).float().to(device)
            b_size = x.shape[0]
            step += b_size

            # Initialise belief and error tensors
            state = model.init_state(b_size)

            for step_i in range(model.steps):
                with torch.no_grad():
                    state = model.step(x, state, y)

                if step_i > 0:
                    with torch.no_grad():
                        for i in range(len(model.layers)):
                            model.layers[i].weight_td.grad = -(state[i][1].T @ state[i][0]) / b_size
                            if model.layers[i].bias is not None:
                                target_bias = x.mean(axis=0) if i == 0 else state[i-1][0].mean(axis=0)
                                if isinstance(model.layers[i], DualPELayer):
                                    target_bias = torch.cat((target_bias, target_bias))
                                model.layers[i].bias.grad = model.layers[i].bias - target_bias.expand(model.layers[i].bias.shape)
                            if not model.layers[i].symmetric:
                                model.layers[i].weight_bu.grad = -(state[i][0].T @ state[i][1]) / b_size
                
            # Regularisation
            reg = torch.zeros(1).to(device)
            for i in range(len(model.layers)):
                reg += model.layers[i].weight_td.square().mean()
                if not model.layers[i].symmetric:
                    reg += model.layers[i].weight_bu.square().mean()
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

            # Track batch statistics
            for i in range(len(model.layers)):
                batches_R_mags[i].append(state[i][0].norm(dim=1).mean().item())
                batches_E_mags[i].append(state[i][1].norm(dim=1).mean().item())
                batches_WeightTD_means[i].append(model.layers[i].weight_td.mean().item())
                batches_WeightTD_stds[i].append(model.layers[i].weight_td.std().item())
                if not model.layers[i].symmetric:
                    batches_WeightBU_means[i].append(model.layers[i].weight_bu.mean().item())
                    batches_WeightBU_stds[i].append(model.layers[i].weight_bu.std().item())

        # Track epoch statistics
        for i in range(len(model.layers)):
            epochs_R_mags[i].append(torch.tensor(batches_R_mags[i]).mean().item())
            epochs_E_mags[i].append(torch.tensor(batches_E_mags[i]).mean().item())
            epochs_WeightTD_means[i].append(torch.tensor(batches_WeightTD_means[i]).mean().item())
            epochs_WeightTD_stds[i].append(torch.tensor(batches_WeightTD_stds[i]).mean().item())
            if not model.layers[i].symmetric:
                epochs_WeightBU_means[i].append(torch.tensor(batches_WeightBU_means[i]).mean().item())
                epochs_WeightBU_stds[i].append(torch.tensor(batches_WeightBU_stds[i]).mean().item())

    stats = {
        "R_mags": epochs_R_mags,
        "E_mags": epochs_E_mags,
        "Weight_means": epochs_WeightTD_means,
        "Weight_stds": epochs_WeightTD_stds,
        "WeightBU_means": epochs_WeightBU_means,
        "WeightBU_stds": epochs_WeightBU_stds,
    }

    return step, stats
