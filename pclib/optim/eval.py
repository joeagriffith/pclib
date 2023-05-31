import torch

def topk_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        device = "cuda" if output.is_cuda else "cpu"
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = torch.zeros(len(topk), dtype=float, device=device)
        for i, k in enumerate(topk):
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res[i] = correct_k.mul_(100.0 / batch_size)
        return res

def evaluate_pc(model, data_loader, criterion, device, flatten=False):
    with torch.no_grad():
        model.eval()

        loss = 0.0
        acc = torch.zeros(3, device=device)
        errs = torch.zeros(len(model.layers), device=device)

        for images, y in data_loader:
            x = images.to(device)
            if flatten:
                x = torch.flatten(x, start_dim=1)
            target = y.to(device)
            R, E = model(x, full_data=True)

            loss += criterion(R[-1], target).item()

            acc += topk_accuracy(R[-1], target, (1,3,5))

            for i, e in enumerate(E):
                errs[i] += e.square().mean()
        
        loss /= len(data_loader)
        acc /= len(data_loader) 
        errs /= len(data_loader)

        return loss, acc, errs