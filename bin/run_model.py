import torch
import pytorch_lightning as pl
from source.model import MLP
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from source.evaluator import gap_eval_scores
from source.dataset import prepare_data
from torch.utils.data import DataLoader
from mlflow import MlflowClient
import mlflow

torch.set_float32_matmul_precision('high')


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient(
    ).list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")


def main():
    datapath = "/home/aleksander/Documents/Data_Challenge/data/preprocessed_data.pickle"
    # Prepare the data
    train_dataset, valid_dataset, test_dataset = prepare_data(
        data_path=datapath)
    batch_size = 128
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, persistent_workers=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=16, persistent_workers=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the Lightning module
    model = MLP(lr=0.001, batch_size=batch_size)

# Train the model.
    trainer = pl.Trainer(
        max_epochs=30,
        devices=1,
        callbacks=[TQDMProgressBar(refresh_rate=20)])

    mlflow.autolog()

    with mlflow.start_run() as run:
        trainer.fit(model, train_loader, valid_loader)

    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


if __name__ == "__main__":
    main()
