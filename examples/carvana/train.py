from tqdm import tqdm
import torch
from examples.carvana.utils import (
    load_checkpoint,    
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
)

def train(
        model, 
        train_loader, 
        val_loader, 
        epochs, 
        lr=3e-4,
        batch_size=1,
        reg_coeff=1e-2,
        step=0,
        stats=None,
        device="cpu",
        optim='AdamW', 
        save_every=None,
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
    
    num_layers = len(model.encoder + model.bottleneck + model.decoder + model.final)

    if stats is None:
        stats = {
            "R_norms": [[] for _ in range(num_layers)],
            "E_mags": [[] for _ in range(num_layers)],
            "WeightTD_means": [[] for _ in range(num_layers-1)],
            "WeightTD_stds": [[] for _ in range(num_layers-1)],
            "Bias_means": [[] for _ in range(num_layers-1)],
            "Bias_stds": [[] for _ in range(num_layers-1)],
            "WeightBU_means": [[] for _ in range(num_layers-1)],
            "WeightBU_stds": [[] for _ in range(num_layers-1)],
            "WeightVar_means": [[] for _ in range(num_layers-1)],
            "WeightVar_stds": [[] for _ in range(num_layers-1)],
            "train_vfe": [],
            "val_acc": [],
            "val_vfe": [],
        }


    for epoch in range(epochs):
        # Track batches, indexed: [layer][batch]
        epoch_stats = {
            "R_norms": [[] for _ in range(num_layers)],
            "E_mags": [[] for _ in range(num_layers)],
            "WeightTD_means": [[] for _ in range(num_layers-1)],
            "WeightTD_stds": [[] for _ in range(num_layers-1)],
            "Bias_means": [[] for _ in range(num_layers-1)],
            "Bias_stds": [[] for _ in range(num_layers-1)],
            "WeightBU_means": [[] for _ in range(num_layers-1)],
            "WeightBU_stds": [[] for _ in range(num_layers-1)],
            "WeightVar_means": [[] for _ in range(num_layers-1)],
            "WeightVar_stds": [[] for _ in range(num_layers-1)],
            "train_vfe": [],
        }

        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        if epoch > 0:
            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(
                train_VFE = stats['train_vfe'][-1],
                val_acc = stats['val_acc'][-1],
                val_VFE = stats['val_vfe'][-1],
            )
        
        for batch_idx, (images, targets) in loop:
            images = images.to(device)
            targets = targets.float().unsqueeze(1).to(device)
        
            # Forward
            with torch.cuda.amp.autocast():
                out, state = model(images, y=targets)
                model.vfe(state).backward()

            optimiser.step()
            
            # Track batch statistics
            epoch_stats['train_vfe'].append(model.vfe(state).item())
            for i in range(num_layers):
                epoch_stats['R_norms'][i].append(state[i]['x'].norm(dim=1).mean().item())
                epoch_stats['E_mags'][i].append(state[i]['e'].square().mean().item())

        
        # save model
        if save_every is not None and epoch % save_every == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimiser": optimiser.state_dict(),
            }
            save_checkpoint(checkpoint)

        # check accuracy
        num_correct = 0
        num_pixels = 0
        dice_score = 0
        val_vfe = 0
        model.eval()

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device).unsqueeze(1)
                out, val_state = model(x, y=y)
                val_vfe += model.vfe(val_state).item()
                preds = torch.sigmoid(out)
                preds = (preds > 0.5).float()
                num_correct += (preds == y).sum()
                num_pixels += torch.numel(preds)
                dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
                )
            val_acc = num_correct/num_pixels*100
            dice_score = dice_score/len(val_loader)
            val_vfe = val_vfe/len(val_loader)


        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="examples/carvana/saved_images/", device=device
        )

        # Track epoch statistics
        for i, _ in range(num_layers):
            stats['R_norms'][i].append(torch.tensor(epoch_stats['R_norms'][i]).mean().item())
            stats['E_mags'][i].append(torch.tensor(epoch_stats['E_mags'][i]).mean().item())
        stats['train_vfe'].append(torch.tensor(epoch_stats['train_vfe']).mean().item())
        stats['val_acc'].append(val_acc)
        stats['val_vfe'].append(val_vfe)