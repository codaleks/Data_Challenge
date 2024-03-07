from torch import nn, optim
import pytorch_lightning as pl


class MLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(284, 250),
            nn.ReLU(),
            nn.Linear(250, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 28),
            nn.Softmax(dim=1)  # Specify the dimension for Softmax
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.MLP(x)

    def configure_optimizers(self, lr=0.01):
        optimizer = optim.Adam(self.parameters(), lr=lr)
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
