import zenml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Annotated, Optional, Tuple, List, Dict
from zenml import step
from zenml.pipelines import pipeline
from zenml.integrations.mlflow.experiment_trackers import (
    MLFlowExperimentTracker,
)


@step(enable_cache=False, experiment_tracker="mlflow_tracker")
def test_step(model: nn.Module, test_dataloader: DataLoader, loss_fn: nn.Module) -> Tuple[
    Annotated[float, "test_loss"],
    Annotated[float, "test_acc"]
]:
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.cuda(), y.cuda()
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.cpu().item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    test_loss = test_loss / len(test_dataloader)
    test_acc = test_acc / len(test_dataloader)
    
    
    with mlflow.start_run() as run:
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_acc", test_acc)

        # Log the PyTorch model as an artifact
        mlflow.pytorch.log_model(model, "model")
    return test_loss, test_acc