import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from pclib.nn.layers import PCLinearUni, PCLinear

from pclib.optim.eval import topk_accuracy, evaluate_pc

def train_lin(
    model, 
    train_data,
    val_data,
    num_epochs,
    lr = 3e-4,
    batch_size=1,
    reg_coeff = 1e-2,
    step=0, 
    stats=None,
    track_td=False,
    device="cpu",
):

    train_loader = train_data if isinstance(train_data, DataLoader) else DataLoader(train_data, batch_size, shuffle=True)
    val_loader = val_data if isinstance(val_data, DataLoader) else DataLoader(val_data, batch_size, shuffle=False)

    # Track epochs, indexed: [layer][epoch]
    if stats is None:
        epochs_R_mags = [[] for _ in range(len(model.layers))]
        epochs_E_mags = [[] for _ in range(len(model.layers))]
        epochs_Weight_means = [[] for _ in range(len(model.layers))]
        epochs_Weight_stds = [[] for _ in range(len(model.layers))]
        epochs_WeightTD_means = [[] for _ in range(len(model.layers))]
        epochs_WeightTD_stds = [[] for _ in range(len(model.layers))]
        

    for epoch in range(num_epochs):

        # Track batches, indexed: [layer][batch]
        batches_R_mags = [[] for _ in range(len(model.layers))]
        batches_E_mags = [[] for _ in range(len(model.layers))]
        batches_Weight_means = [[] for _ in range(len(model.layers))]
        batches_Weight_stds = [[] for _ in range(len(model.layers))]
        if track_td:
            batches_WeightTD_means = [[] for _ in range(len(model.layers))]
            batches_WeightTD_stds = [[] for _ in range(len(model.layers))]
        
        #  Initialise variables and prepare data for new epoch
        model.train()

        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)    
        if epoch > 0:
            mean_e = sum([epochs_E_mags[i][-1] for i in range(len(model.layers))]) / len(model.layers)
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(
                mean_e = mean_e,
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
                            model.layers[i].weight.grad = -(state[i][0].T @ state[i][1]) / b_size
                            if model.layers[i].bias is not None:
                                target_bias = x.mean(axis=0) if i == 0 else state[i-1][0].mean(axis=0)
                                model.layers[i].bias.grad = model.layers[i].bias - target_bias
                            if isinstance(model.layers[i], PCLinear):
                                model.layers[i].weight_td.grad = -(state[i][1].T @ state[i][0]) / b_size
                
            # Regularisation
            reg = torch.zeros(1).to(device)
            for i in range(len(model.layers)):
                reg += model.layers[i].weight.square().mean()
                if isinstance(model.layers[i], PCLinear):
                    reg += model.layers[i].weight_td.square().mean()
            reg *= reg_coeff
            reg.backward()

            # Parameter Update
            for i in range(len(model.layers)):
                model.layers[i].weight.data -= lr * model.layers[i].weight.grad
                if model.layers[i].bias is not None:
                    assert model.layers[i].bias.grad is not None, f"layer {i} bias has no grad"
                    model.layers[i].bias.data -= lr * model.layers[i].bias.grad
                if isinstance(model.layers[i], PCLinear):
                    model.layers[i].weight_td.data -= lr * model.layers[i].weight_td.grad

            # Track batch statistics
            for i in range(len(model.layers)):
                batches_R_mags[i].append(state[i][0].norm(dim=1).mean().item())
                batches_E_mags[i].append(state[i][1].norm(dim=1).mean().item())
                batches_Weight_means[i].append(model.layers[i].weight.mean().item())
                batches_Weight_stds[i].append(model.layers[i].weight.std().item())
                if track_td:
                    batches_WeightTD_means[i].append(model.layers[i].weight_td.mean().item())
                    batches_WeightTD_stds[i].append(model.layers[i].weight_td.std().item())

        # Track epoch statistics
        for i in range(len(model.layers)):
            epochs_R_mags[i].append(torch.tensor(batches_R_mags[i]).mean().item())
            epochs_E_mags[i].append(torch.tensor(batches_E_mags[i]).mean().item())
            epochs_Weight_means[i].append(torch.tensor(batches_Weight_means[i]).mean().item())
            epochs_Weight_stds[i].append(torch.tensor(batches_Weight_stds[i]).mean().item())
            if track_td:
                epochs_WeightTD_means[i].append(torch.tensor(batches_WeightTD_means[i]).mean().item())
                epochs_WeightTD_stds[i].append(torch.tensor(batches_WeightTD_stds[i]).mean().item())

    stats = {
        "R_mags": epochs_R_mags,
        "E_mags": epochs_E_mags,
        "Weight_means": epochs_Weight_means,
        "Weight_stds": epochs_Weight_stds,
        "WeightTD_means": epochs_WeightTD_means,
        "WeightTD_stds": epochs_WeightTD_stds,
    }

    return step, stats

    
def train(
    model, 
    train_data,
    val_data,
    optimiser,
    criterion, 
    model_name, 
    num_epochs,
    learning_rate = 3e-4,
    weight_decay = 1e-2,
    flatten=False, 
    model_dir="out/weights",
    log_dir="out/logs", 
    step=0, 
    best="loss",
    save_model=True,
    batch_size=100,
    plot_err=True,
    e_weight_mode="same",
    augment=True,
    train_loss=False,
    train_on_loss=False,
    device="cpu",
):
    writer = SummaryWriter(f"{log_dir}/{model_name}")

    #  For determining best model. Either can be used.
    best_val_acc = 0.0
    best_val_loss = float("inf")

    train_loader = train_data if isinstance(train_data, DataLoader) else DataLoader(train_data, batch_size, shuffle=True)
    val_loader = val_data if isinstance(val_data, DataLoader) else DataLoader(val_data, batch_size, shuffle=False)

    if train_on_loss:
        train_loss = True

    # Weighting coefficients for layers in loss function
    assert e_weight_mode in ["same", "l0_only", "l0_heavy"], "train_pc received invalid 'e_weight_mode'"
    if e_weight_mode == "same":
        layer_loss_weights = [1.0 for _ in range(len(model.layers))]
    elif e_weight_mode == "l0_heavy":
        layer_loss_weights = [0.1 for _ in range(len(model.layers))]
    elif e_weight_mode == "L_0":
        layer_loss_weights = [0.0 for _ in range(len(model.layers))]
    layer_loss_weights[0] = 1.0

    # step_loss_weights = [1.0 for _ in range(model.steps)]
    # step_loss_weights[0] = 0.0
    step_loss_weights = [0.0 for _ in range(model.steps)]
    step_loss_weights[-1] = 1.0

    for epoch in range(num_epochs):
        
        #  Initialise variables and prepare data for new epoch
        model.train()

        # Re-calculate data augmentation if train_data is Dataset (not DataLoader)
        if not isinstance(train_data, DataLoader) and augment:
            train_data.apply_transform()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)    

        train_err = torch.zeros(len(model.layers), device=device)
        if train_loss:
            epoch_train_loss = 0.0

        n = 0
        for batch_idx, (images, target) in loop:
            n += images.shape[0]
            x = images.to(device)
            if flatten:
                x = torch.flatten(x, start_dim=1)
            target = target.to(device)

            y = F.one_hot(target, model.num_classes).to(torch.float32)

            # Initialise belief and error tensors
            R, E = model.init_vars(x.shape[0])
            # Set last belief to one-hot encoding of label
            optimiser.zero_grad()
            for step_i in range(model.steps):
                R, E = model.step(x, R, E, y)
                # Keep last belief as one-hot encoding of label

                e_loss = 0.0
                for i, e in enumerate(E):
                    e_loss += e.square().sum() * layer_loss_weights[i] * step_loss_weights[step_i]
                e_loss.backward()
                
                # r_loss = 0.0
                # for i, r in enumerate(R):
                #     r_loss += r.square().sum() * layer_loss_weights[i] * step_loss_weights[step_i]
                # if not train_on_loss:
                #     r_loss.backward()

                for i in range(len(model.layers)):
                    train_err[i] += E[i].detach().square().mean()
                    E[i] = E[i].detach()
                    R[i] = R[i].detach()


            if train_loss:
                out = model(x)

                batch_train_loss = criterion(out, y)
                epoch_train_loss += batch_train_loss.item()
                if train_on_loss:
                    batch_train_loss.backward()


            optimiser.step()

            #  TQDM bar update
            if epoch > 0 and batch_idx > 0:
                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(
                    train_err_mean = train_err.mean() / (batch_idx + 1),
                    val_loss = val_loss, 
                    val_acc = val_acc[0].item(),
                    val_err_mean = val_err.mean()
                )

        #  Convert running totals of training metrics to epoch means
        train_err /= len(train_loader)
        if train_loss:
            epoch_train_loss /= len(train_loader)
        
        #  Calculate validation metrics
        val_loss, val_acc, val_err = evaluate_pc(model, val_loader, criterion, device, flatten)
            
        #  Save model if selected metric is a high score
        if save_model:
            if best == "loss":
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f'{model_dir}/{model_name}.pth')
            elif best in ["acc", "accuracy"]:
                if best_val_acc < val_acc[0]:
                    best_val_acc = val_acc[0]
                    torch.save(model.state_dict(), f'{model_dir}/{model_name}.pth')
            else:
                raise Exception("Invalid value for 'best'")

        #  Log metrics for tensorboard
        step += n
        writer.add_scalar(f"Mean Training MSE", train_err.mean(), step)
        for i, e in enumerate(train_err):
            writer.add_scalar(f"Training layer {i} MSE", e.item(), step) 
        writer.add_scalar("Mean Validation MSE", val_err.mean(), step)
        for i, e in enumerate(val_err):
            writer.add_scalar(f"Validation layer {i} MSE", e.item(), step) 
        if train_loss:
            writer.add_scalar("Training Loss", epoch_train_loss, step)
        writer.add_scalar("Validation Loss", val_loss, step)
        writer.add_scalar("Validation Accuracy Top1", val_acc[0].item(), step)
        writer.add_scalar("Validation Accuracy Top3", val_acc[1].item(), step)

    return step