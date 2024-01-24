import zenml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Tuple, List, Dict
from zenml.steps import step, Output, BaseStepConfig
from zenml.pipelines import pipeline
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow

@enable_mlflow 
@step(enable_cache=False)
def test_step(model: nn.Module, test_dataloader: DataLoader, loss_fn: nn.Module) -> Output(
    test_loss = float,
    test_acc = float
):
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