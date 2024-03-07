import torch
import pytorch_lightning as pl
from source.model import MLP
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from source.evaluator import gap_eval_scores
from source.dataset import prepare_data
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import MLFlowLogger
import mlflow

torch.set_float32_matmul_precision('high')


def main():
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    # Prepare the data
    train_dataset, valid_dataset, test_dataset = prepare_data(
        data_path="A:\MSBGD\Data_Challenge\data\preprocessed_data.pickle")
    batch_size = 64
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=23, persistent_workers=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=23, persistent_workers=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the Lightning module
    model = MLP()
    mlf_logger = MLFlowLogger(
        experiment_name="lightning_logs", tracking_uri="http://127.0.0.1:8080")
    # Initialize a Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        devices=1,
        callbacks=[TQDMProgressBar(refresh_rate=20)])

    # Train the model
    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    main()
