import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

def cross_entropy_loss(output, target):
    loss = CrossEntropyLoss()
    return loss(output, target)

def nll_loss(output, target):
    return F.nll_loss(output, target)
