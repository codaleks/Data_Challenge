from torch.utils.data import DataLoader, TensorDataset
from preprocessing import preprocessing
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch import nn, optim
import torch
import pytorch_lightning as pl


class MLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(332, 250),
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
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        self.log('train_loss', loss)  # Automatically logs training loss
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        self.log('test_loss', loss)

    ####################
    # DATA RELATED HOOKS
    ####################
    def setup(self):
        
        

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=128)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=128)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=128)
    

# Create a DataLoader
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the Lightning module
model = MLP()

# Initialize a Trainer
trainer = pl.Trainer(
    max_epochs=4000,
    gpus=1 if torch.cuda.is_available() else 0,
    callbacks=[TQDMProgressBar(refresh_rate=20)])

# Train the model
trainer.fit(model, train_loader)
