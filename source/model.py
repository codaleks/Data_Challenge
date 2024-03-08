from torch import nn, optim
import pytorch_lightning as pl


class MLP(pl.LightningModule):
    def __init__(self, lr=0.01, batch_size=32):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(284, 64),
            nn.ReLU(),
            nn.Linear(64, 28),
            nn.Softmax()  # Specify the dimension for Softmax
        )
        self.lr = lr
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        logits = self.MLP(x)
        return logits

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        self.log('train_loss', loss)  # Automatically logs training loss
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        self.log('test_loss', loss)
