from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from source.Tversky_loss import TverskyLoss
import pytorch_lightning as pl


class MLP(pl.LightningModule):
    def __init__(self, lr=0.01, batch_size=32):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(768, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 28),
            nn.Softmax(dim=1)
        )
        self.lr = lr
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        logits = self.MLP(x)
        return logits

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=1e-8)
        scheduler = {
            'scheduler': StepLR(optimizer, step_size=30, gamma=0.1),
            'interval': 'epoch',  # or 'step' to update the lr each step
            'frequency': 5,
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        self.log('train_loss', loss)  # Automatically logs training loss
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        outputs = self(inputs)
        val_loss = self.loss_fn(outputs, labels)
        self.log('val_loss', val_loss)
        return val_loss
