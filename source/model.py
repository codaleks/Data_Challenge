import torch
from torch import nn, optim
import torchmetrics
import pytorch_lightning as pl
from torch.nn.functional import softmax


class MLP(pl.LightningModule):
    def __init__(self, lr=0.01, batch_size=32, lambda_debias=0.01):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(768, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 28),
            nn.Softmax(dim=1)
        )
        self.lr = lr
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_debias = lambda_debias
        self.accuracy = torchmetrics.Accuracy('multiclass', num_classes=28)

    def forward(self, x):
        logits = self.MLP(x)
        return logits

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr)
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
        # classification_loss = self.loss_fn(outputs, labels)
        # fairness_loss = self.fairness_loss(outputs, sensitive_attrs)
        loss = self.loss_fn(outputs, labels)
        self.accuracy(outputs, labels)
        self.log('train_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        # self.log('fairness_loss', loss,
        #         on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', self.accuracy,
                 on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, _ = batch  # Validation step unchanged; ignores sensitive attributes
        outputs = self(inputs)
        val_loss = self.loss_fn(outputs, labels)
        self.accuracy(outputs, labels)
        self.log('val_loss', val_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log('val_accuracy', self.accuracy(outputs, labels),
                 on_step=False, on_epoch=True, prog_bar=True)
        return val_loss
