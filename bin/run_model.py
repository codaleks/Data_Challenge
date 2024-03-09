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

    eval_scores, confusion_matrices_eval = gap_eval_scores(
        all_preds, all_labels, all_bias, metrics=['TPR'])
    final_score = (eval_scores['macro_fscore'] + (1-eval_scores['TPR_GAP']))/2
    return print(f"Final score: {final_score}")


def main(batch_size=128, lr=0.001, max_epochs=30, num_workers=16, data_path="/home/aleksander/Documents/Data_Challenge/data/preprocessed_data.pickle"):
    datapath = data_path
    # Prepare the data
    train_dataset, valid_dataset, test_dataset = prepare_data(
        data_path=datapath)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False)

    # Initialize the Lightning module
    model = MLP(lr=lr, batch_size=batch_size)

# Train the model.
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=1,
        callbacks=[TQDMProgressBar(refresh_rate=20)])

    mlflow.autolog()

    with mlflow.start_run() as run:
        trainer.fit(model, train_loader, valid_loader)

    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

    # Test the model
    test_model(model, test_loader)


if __name__ == "__main__":
    main(batch_size=128, lr=0.001, max_epochs=30, num_workers=16,
         data_path="data/preprocessed_data.pickle")
