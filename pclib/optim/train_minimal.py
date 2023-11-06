import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from pclib.optim.eval import topk_accuracy
from pclib.nn.layers import PrecisionWeighted
from pclib.utils.functional import vfe, format_y

def train(
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
            "train_vfe": [],
            "val_acc": [],
            "val_vfe": [],
        }

    for epoch in range(num_epochs):

        train_data.apply_transform()

        # Track batches, indexed: [layer][batch]
        epoch_stats = {
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
            out, state = model(x, y=y)

            # Calculate grads, different equations for each implementation, top_down is f(Wr) or Wf(r)
            model.zero_grad()
            model.vfe(state).backward()

            # Assert grads
            # for i, layer in enumerate(model.layers):
            #     e_below = state[i-1]['e_u'] if i > 0 else None
            #     e_above = state[i+1]['e_l'] if i < len(model.layers)-1 else None
            #     layer.assert_grad(state[i], e_below, e_above)                

            if neg_coeff is not None and neg_coeff > 0:
                # Forward Pass
                out, state = model(x, y=false_y)
                loss = -neg_coeff * model.vfe(state)
                loss.backward()


            # Parameter Update (Grad Descent)
            optimiser.step()
            for layer in model.layers:
                if isinstance(layer, PrecisionWeighted):
                    layer.weight_var.data -= lr * layer.weight_var.grad
                    layer.weight_var.data = torch.clamp(layer.weight_var.data, min=0.01)

            epoch_stats['train_vfe'].append(model.vfe(state).item())

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
                val_vfe += model.vfe(val_state, batch_reduction='sum').item()
                val_correct += (out.argmax(dim=1) == target).sum().item()

            val_acc = val_correct / len(val_data)
            val_vfe /= len(val_data)

        # Track epoch statistics
        stats['train_vfe'].append(torch.tensor(epoch_stats['train_vfe']).mean().item())
        stats['val_acc'].append(val_acc)
        stats['val_vfe'].append(val_vfe)
    return step, stats
