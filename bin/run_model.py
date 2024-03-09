from mlflow import MlflowClient
from torch.utils.data import DataLoader
from source.dataset import create_datasets
from source.model import MLP
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from source.evaluator import gap_eval_scores
import torch
import pytorch_lightning as pl
import mlflow
import pandas as pd


torch.set_float32_matmul_precision('high')

###########################################
#                Variables                #
###########################################

datapath = "../data/data-challenge-student.pickle"
batch_size = 128
lr = 1e-3
max_epochs = 30
num_workers = 16


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient(
    ).list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")


def train(batch_size=batch_size, lr=lr, max_epochs=max_epochs, num_workers=num_workers, datapath=datapath):
    # Prepare the data
    train_dataset, valid_dataset, scaler = create_datasets(
        datapath=datapath, test_size=0.2)

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

    return model, valid_loader, scaler


def test_model(model, test_loader):
    # test over the whole test set
    model.eval()
    all_preds = []
    all_labels = []
    all_bias = []
    for batch in test_loader:
        inputs, labels, bias = batch
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.append(predicted)
        all_labels.append(labels)
        all_bias.append(bias)
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    all_bias = torch.cat(all_bias).cpu().numpy()

    eval_scores, _ = gap_eval_scores(
        all_preds, all_labels, all_bias, metrics=['TPR'])
    final_score = (eval_scores['macro_fscore'] + (1-eval_scores['TPR_GAP']))/2
    return print(f"Final score: {final_score}")


def submission(model, scaler, datapath=datapath):
    with open(datapath, 'rb') as handle:
        dat = pd.read_pickle(handle)
    X_test = dat['X_test']
    X_test = scaler.transform(X_test)

    inputs = X_test
    inputs = torch.tensor(inputs.values).float()

    pred = MLP(inputs)
    pred = pred.argmax(1)

    results = pd.DataFrame(pred, columns=['score'])
    results.to_csv("Data_Challenge_MDI_341.csv", header=None, index=None)


def main(batch_size=batch_size, lr=lr, max_epochs=max_epochs, num_workers=num_workers, datapath=datapath):
    model, valid_loader, scaler = train(
        batch_size, lr, max_epochs, num_workers, datapath)

    test_model(model, valid_loader)
    # submission()


if __name__ == "__main__":
    main(batch_size=batch_size, lr=lr, max_epochs=max_epochs, num_workers=num_workers,
         data_path="data/preprocessed_data.pickle")
