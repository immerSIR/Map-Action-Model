import torch
from tqdm import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import mlflow.pytorch
from typing import Annotated, Optional, Tuple, List, Dict
from zenml import step, pipeline
from zenml.integrations.mlflow.experiment_trackers import (
    MLFlowExperimentTracker,
)


@step(enable_cache=False, experiment_tracker="mlflow_tracker")
def train_model(model: torch.nn.Module, train_dataloader:DataLoader, epochs:int) -> Tuple[
    Annotated[torch.nn.Module, "model"],
    Annotated[Dict, "results"]
]:
    """
    Train a PyTorch model.

    Args:
        model (nn.Module): The neural network model.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        optimizer (Optimizer): The optimizer for model training.
        loss_fn (nn.Module): The loss function.
        epochs (int, optional): Number of training epochs. Defaults to 20.

    Returns:
        Tuple[nn.Module, dict]: Trained model and dictionary containing training results.
        
    """
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    results = {
        "train_loss": [],
        "train_acc": [],
    }

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss, train_acc = 0, 0

        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.cuda(), y.cuda()
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss.cpu().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        train_loss = train_loss / len(train_dataloader)
        train_acc = train_acc / len(train_dataloader)

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} "
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        
    
    with mlflow.start_run() as run:
        mlflow.pytorch.log_model(model, "model")
            
    return model, results
