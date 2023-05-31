import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from pclib.optim.eval import topk_accuracy, evaluate_pc

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
    model_dir="out/models",
    log_dir="out/logs", 
    step=0, 
    best="loss",
    save_model=True,
    batch_size=100,
    plot_err=True,
    e_weight_mode="same",
    device="cpu"
):
    writer = SummaryWriter(f"{log_dir}/{model_name}")

    #  For determining best model. Either can be used.
    best_val_acc = 0.0
    best_val_loss = float("inf")

    train_loader = train_data if isinstance(train_data, DataLoader) else DataLoader(train_data, batch_size, shuffle=True)
    val_loader = val_data if isinstance(val_data, DataLoader) else DataLoader(val_data, batch_size, shuffle=False)

    # Weighting coefficients for layers in loss function
    assert e_weight_mode in ["same", "l0_only", "l0_heavy"], "train_pc received invalid 'e_weight_mode'"
    if e_weight_mode == "same":
        layer_loss_weights = [1.0 for _ in range(len(model.layers))]
    elif e_weight_mode == "l0_heavy":
        layer_loss_weights = [0.1 for _ in range(len(model.layers))]
    elif e_weight_mode == "L_0":
        layer_loss_weights = [0.0 for _ in range(len(model.layers))]
    layer_loss_weights[0] = 1.0

    step_loss_weights = [1.0 for _ in range(model.steps)]
    step_loss_weights[0] = 0.0

    for epoch in range(num_epochs):
        
        #  Initialise variables and prepare data for new epoch
        model.train()

        # Re-calculate data augmentation if train_data is Dataset (not DataLoader)
        if not isinstance(train_data, DataLoader):
            train_data.apply_transform()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)    

        train_err = torch.zeros(len(model.layers), device=device)

        n = 0
        for batch_idx, (images, y) in loop:
            n += images.shape[0]
            x = images.to(device)
            if flatten:
                x = torch.flatten(x, start_dim=1)
            y = y.to(device)

            # Initialise belief and error tensors
            R, E = model.init_vars(x.shape[0])
            # Set last belief to one-hot encoding of label
            R[-1] = F.one_hot(y, model.out_features).to(torch.float32)
            optimiser.zero_grad()
            for step_i in range(model.steps):
                R, E = model.step(x, R, E)
                # Keep last belief as one-hot encoding of label
                R[-1] = F.one_hot(y, model.out_features).to(torch.float32)

                e_loss = 0.0
                for i, e in enumerate(E):
                    e_loss += e.square().sum() * layer_loss_weights[i] * step_loss_weights[step_i]
                e_loss.backward()

                for i in range(len(model.layers)):
                    train_err[i] += E[i].detach().square().mean()
                    E[i] = E[i].detach()
                    R[i] = R[i].detach()

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
        writer.add_scalar("Validation Loss", val_loss, step)
        writer.add_scalar("Validation Accuracy Top1", val_acc[0].item(), step)
        writer.add_scalar("Validation Accuracy Top3", val_acc[1].item(), step)
        writer.add_scalar("Validation Accuracy Top5", val_acc[2].item(), step)

    return step