import torch
from tqdm import tqdm
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

class TrainingPipeline:
    def __init__(self, model: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader,
                 optimizer: Optimizer, loss_fn: nn.Module, epochs: int):
        """
        Initialize the TrainingPipeline.

        Args:
            model (nn.Module): The neural network model.
            train_dataloader (DataLoader): DataLoader for the training dataset.
            test_dataloader (DataLoader): DataLoader for the testing dataset.
            optimizer (Optimizer): The optimizer for model training.
            loss_fn (nn.Module): The loss function.
            epochs (int): Number of training epochs.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.results = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }

    def training_step(self):
        """
        Perform a training step.

        Returns:
            Tuple[float, float]: Training loss and accuracy.
        """
        self.model.train()
        train_loss, train_acc = 0, 0

        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.cuda(), y.cuda()
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            train_loss += loss.cpu().item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        train_loss = train_loss / len(self.train_dataloader)
        train_acc = train_acc / len(self.train_dataloader)
        return train_loss, train_acc

    def test_step(self):
        """
        Perform a testing step.

        Returns:
            Tuple[float, float]: Testing loss and accuracy.
        """
        self.model.eval()
        test_loss, test_acc = 0, 0

        with torch.inference_mode():
            for batch, (X, y) in enumerate(self.test_dataloader):
                X, y = X.cuda(), y.cuda()
                test_pred_logits = self.model(X)
                loss = self.loss_fn(test_pred_logits, y)
                test_loss += loss.cpu().item()
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

        test_loss = test_loss / len(self.test_dataloader)
        test_acc = test_acc / len(self.test_dataloader)
        return test_loss, test_acc

    def train(self):
        """
        Execute the training pipeline.

        Returns:
            dict: Dictionary containing training and testing results.
        """
        self.model.cuda()

        for epoch in tqdm(range(self.epochs)):
            train_loss, train_acc = self.training_step()
            test_loss, test_acc = self.test_step()

            print(
                f"Epoch: {epoch + 1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
            )

            self.results["train_loss"].append(train_loss)
            self.results["train_acc"].append(train_acc)
            self.results["test_loss"].append(test_loss)
            self.results["test_acc"].append(test_acc)

        return self.results
