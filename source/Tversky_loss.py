import torch
from torch import nn


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-10):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Convert targets to one-hot encoding if they're not already
        if targets.dim() == 1:
            targets = torch.nn.functional.one_hot(
                targets, num_classes=inputs.shape[1])
        targets = targets.float()

        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha *
                                        FP + self.beta * FN + self.smooth)

        return 1 - Tversky
