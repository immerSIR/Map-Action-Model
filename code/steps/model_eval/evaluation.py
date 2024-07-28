import os
import torch
import torch.nn as nn
from tqdm import tqdm
import mlflow
from torch.utils.data import DataLoader
from typing import Annotated, Tuple, Dict
from zenml import step
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker


@step(experiment_tracker="mlflow_tracker")
def test_step(
    model: nn.Module,
    test_dataloader: DataLoader,
    loss_fn: nn.Module,
    results: Dict,
    epochs: int
) -> Tuple[
    Annotated[float, "test_loss"],
    Annotated[float, "test_acc"],
    Annotated[Dict, "results"]
]:
    """
    Perform testing step for a PyTorch model.
    Args:
    model (nn.Module): The PyTorch model to evaluate.
    test_dataloader (DataLoader): DataLoader for the test dataset.
    loss_fn (nn.Module): Loss function for the evaluation.
    results (Dict): Dictionary to store results.
    epochs (int): Number of epochs.
    Returns:
    Tuple[float, float, Dict]: Tuple containing test loss, test accuracy, and results.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Initialize test_loss and test_acc lists if they don't exist
    if "test_loss" not in results:
        results["test_loss"] = []
    if "test_acc" not in results:
        results["test_acc"] = []

    for epoch in range(epochs):
        epoch_test_loss, epoch_test_acc = 0, 0
        with torch.inference_mode():
            for X, y in tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Test]"):
                X, y = X.to(device), y.to(device)
                test_pred_logits = model(X)
                loss = loss_fn(test_pred_logits, y)
                epoch_test_loss += loss.item()
                test_pred_labels = test_pred_logits.argmax(dim=1)
                epoch_test_acc += (test_pred_labels == y).sum().item() / len(y)

        epoch_test_loss /= len(test_dataloader)
        epoch_test_acc /= len(test_dataloader)

        results["test_loss"].append(epoch_test_loss)
        results["test_acc"].append(epoch_test_acc)

        mlflow.log_metric("test_loss", epoch_test_loss, step=epoch)
        mlflow.log_metric("test_acc", epoch_test_acc, step=epoch)

        print(
            f"Epoch {epoch+1}/{epochs} - Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.4f}")

    # Log the PyTorch model as an artifact
    mlflow.pytorch.log_model(model, "model")

    # Create 'save' directory if it doesn't exist
    save_dir = 'save'
    os.makedirs(save_dir, exist_ok=True)

    # Save the model
    try:
        torch.save(model.state_dict(), os.path.join(save_dir, 'TCM1.pth'))
        print(
            f"Model saved successfully to {os.path.join(save_dir, 'TCM1.pth')}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

    # Calculate average test loss and accuracy across all epochs
    avg_test_loss = sum(results["test_loss"]) / epochs
    avg_test_acc = sum(results["test_acc"]) / epochs

    return avg_test_loss, avg_test_acc, results
