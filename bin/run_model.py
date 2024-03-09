import torch
import pytorch_lightning as pl
import mlflow
import pandas as pd

from pytorch_lightning.callbacks.progress import TQDMProgressBar
from source.model import MLP
from source.evaluator import gap_eval_scores
from source.dataset import prepare_data
from torch.utils.data import DataLoader
from mlflow import MlflowClient

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


def train(batch_size=128, lr=0.001, max_epochs=30, num_workers=16, data_path="../data/data-challenge-student.pickle"):
    datapath = data_path
    # Prepare the data
    train_dataset, valid_dataset = prepare_data(
        data_path=datapath)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)

    # Initialize the Lightning module
    model = MLP(lr=lr, batch_size=batch_size)

# Train the model.
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[TQDMProgressBar(refresh_rate=20)])

    mlflow.autolog()
    with mlflow.start_run() as run:
        trainer.fit(model, train_loader, valid_loader)

    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


def main(batch_size=128, lr=0.001, max_epochs=30, num_workers=16, datapath="../data/data-challenge-student.pickle"):
    train_dataset, valid_dataset = prepare_data(data_path=datapath)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)

    # Initialize the Lightning module
    model = MLP(lr=lr, batch_size=batch_size)

# Train the model.
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[TQDMProgressBar(refresh_rate=20)])

    mlflow.autolog()
    with mlflow.start_run() as run:
        trainer.fit(model, train_loader, valid_loader)

    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

    # submission
    with open(datapath, 'rb') as handle:
        dat = pd.read_pickle(handle)
    X_test = dat['X_test']
    X_test = (X_test - X_test.mean()) / X_test.std()

    inputs = X_test
    inputs = torch.tensor(inputs.values).float()

    pred = MLP(inputs)
    pred = pred.argmax(1)

    results = pd.DataFrame(pred, columns=['score'])
    results.to_csv("Data_Challenge_MDI_341.csv", header=None, index=None)


if __name__ == "__main__":
    main(batch_size=128, lr=0.001, max_epochs=30, num_workers=16,
         data_path="data/preprocessed_data.pickle")
