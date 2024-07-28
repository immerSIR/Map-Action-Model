import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from typing import Annotated, Tuple, Dict
from zenml import step
from torchvision import transforms


@step(experiment_tracker="mlflow_tracker")
def train_model(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    epochs: int,
    loss_fn: torch.nn.Module
) -> Tuple[
    Annotated[torch.nn.Module, "model"],
    Annotated[Dict, "results"]
]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    model.to(device)

    mlflow.end_run()
    with mlflow.start_run():
        for epoch in range(epochs):
            model.train()
            train_loss, train_acc = 0, 0
            train_batches = 0

            for batch, (X, y) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                y_pred_class = torch.argmax(
                    torch.softmax(y_pred, dim=1), dim=1)
                train_acc += (y_pred_class == y).sum().item() / len(y_pred)
                train_batches += 1

            train_loss /= train_batches
            train_acc /= train_batches

            # Validation phase
            model.eval()
            val_loss, val_acc = 0, 0
            val_batches = 0

            with torch.no_grad():
                for batch, (X, y) in enumerate(tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]")):
                    X, y = X.to(device), y.to(device)

                    y_pred = model(X)
                    loss = loss_fn(y_pred, y)

                    val_loss += loss.item()
                    y_pred_class = torch.argmax(
                        torch.softmax(y_pred, dim=1), dim=1)
                    val_acc += (y_pred_class == y).sum().item() / len(y_pred)
                    val_batches += 1

            val_loss /= val_batches
            val_acc /= val_batches

            # Log results
            print(
                f"Epoch: {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

        mlflow.pytorch.log_model(model, "model")

    return model, results
