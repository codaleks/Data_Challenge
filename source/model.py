import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from source.Tversky_loss import TverskyLoss

import pytorch_lightning as pl
from torch.nn.functional import softmax


class MLP(pl.LightningModule):
    def __init__(self, lr=0.01, batch_size=32, lambda_debias=0.05):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(768, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 28),
            nn.Softmax(dim=1)
        )
        self.lr = lr
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_debias = lambda_debias

    def forward(self, x):
        logits = self.MLP(x)
        return logits

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def fairness_loss(self, outputs, sensitive_attrs):
        """Simple fairness loss based on differences in prediction distributions."""
        group_0_mask = sensitive_attrs == 0
        group_1_mask = sensitive_attrs == 1
        preds_0 = softmax(outputs[group_0_mask], dim=1).mean(dim=0)
        preds_1 = softmax(outputs[group_1_mask], dim=1).mean(dim=0)
        fairness_loss = torch.norm(
            preds_0 - preds_1, p=2)  # L2 norm as an example
        return fairness_loss

    def training_step(self, batch, batch_idx):
        # Assuming batch includes sensitive attributes
        inputs, labels, sensitive_attrs = batch
        outputs = self(inputs)
        classification_loss = self.loss_fn(outputs, labels)
        fairness_loss = self.fairness_loss(outputs, sensitive_attrs)
        total_loss = classification_loss + self.lambda_debias * fairness_loss
        self.log('train_loss', total_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log('fairness_loss', fairness_loss,
                 on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, _ = batch  # Validation step unchanged; ignores sensitive attributes
        outputs = self(inputs)
        val_loss = self.loss_fn(outputs, labels)
        self.log('val_loss', val_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        return val_loss
