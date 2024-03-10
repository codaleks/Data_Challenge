import torch
import pytorch_lightning as pl
import random
import pandas as pd
import mlflow

from torch import from_numpy
from torch.utils.data import DataLoader
from source.dataset import create_datasets
from source.model import MLP
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from source.evaluator import gap_eval_scores
from pytorch_model_summary import summary

torch.set_float32_matmul_precision('high')

###########################################
#                Variables                #
###########################################

datapath = "data/data-challenge-student.pickle"
batch_size = 256
lr = 0.00001
max_epochs = 60
num_workers = 16

###########################################


def train(batch_size=batch_size, lr=lr, max_epochs=max_epochs, num_workers=num_workers, datapath=datapath):
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.pytorch.autolog()
    # Prepare the data
    train_dataset, valid_dataset, scaler = create_datasets(
        datapath=datapath, test_size=0.2)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
    # Initialize the Lightning module
    model = MLP(lr=lr, batch_size=batch_size)
    print(summary(model, torch.zeros((1, 768)),
          show_input=False, show_hierarchical=True))

# Train the model.
    with mlflow.start_run():
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[TQDMProgressBar(refresh_rate=20)])

        trainer.fit(model, train_loader, valid_loader)

    return model, valid_loader, scaler


def test_model(model, val_loader):
    # test over the whole test set
    model.eval()
    all_preds = []
    all_labels = []
    all_bias = []
    for batch in val_loader:
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
    inputs = from_numpy(inputs).float()

    pred = model(inputs)
    pred = pred.argmax(1)

    results = pd.DataFrame(pred, columns=['score'])
    # random name for the submission file*
    random_number = random.randint(1, 100)
    results.to_csv(f"outputs/Data_Challenge_MDI_341_{random_number}.csv",
                   header=None, index=None)


def main(batch_size=batch_size, lr=lr, max_epochs=max_epochs, num_workers=num_workers, datapath=datapath):

    # Scaler useful for the submission
    model, val_loader, scaler = train(
        batch_size, lr, max_epochs, num_workers, datapath)

    test_model(model, val_loader)
    submission(model, scaler, datapath)


if __name__ == "__main__":
    main(batch_size=batch_size, lr=lr, max_epochs=max_epochs, num_workers=num_workers,
         datapath=datapath)
