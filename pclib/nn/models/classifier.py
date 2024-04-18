import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_size, n_classes):
        super().__init__()
        self.fc = nn.Linear(input_size, n_classes, bias=False)
        self.device = torch.device('cpu')
    
    def forward(self, x):
        return self.fc(x)
    
    def to(self, device):
        self.device = device
        return super().to(device)