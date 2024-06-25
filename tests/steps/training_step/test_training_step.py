import pytest
from unittest.mock import patch, MagicMock
import torch
from torch.utils.data import DataLoader
# Adjust the import based on your directory structure
from code.steps.training_step.training_step import train_model

# Patch mlflow and tqdm used within the train_model step


@patch('code.steps.training_step.training_step.mlflow')
@patch('code.steps.training_step.training_step.tqdm')
@patch('code.steps.training_step.training_step.torch.cuda.is_available', return_value=False)
@patch('code.steps.training_step.training_step.torch.optim.SGD')
@patch('code.steps.training_step.training_step.torch.device')
def test_train_model(mock_device, mock_SGD, mock_is_available, mock_tqdm, mock_mlflow):
    # Arrange
    mock_device.return_value = torch.device('cpu')
    mock_optimizer = MagicMock()
    mock_SGD.return_value = mock_optimizer
    mock_tqdm_range = MagicMock()
    mock_tqdm.return_value = mock_tqdm_range

    # Mock model
    class MockModel(torch.nn.Module):
        def forward(self, x):
            # Assuming binary classification with logits for 2 classes
            return torch.randn(x.size(0), 2)

    model = MockModel()

    # Mock DataLoader that returns batches of example data and labels
    # Example tensor shape for 4 samples, assuming 3x224x224 images
    sample_data = torch.randn(4, 3, 224, 224)
    # Example labels for binary classification
    sample_labels = torch.randint(0, 2, (4,))

    train_dataloader = MagicMock(spec=DataLoader)
    train_dataloader.__iter__.return_value = [(sample_data, sample_labels)]

    # Mock loss function
    class MockLossFn(torch.nn.Module):
        def forward(self, y_pred, y):
            return torch.tensor(1.0)

    loss_fn = MockLossFn()

    epochs = 3

    # Act
    trained_model, results = train_model(
        model, train_dataloader, epochs, loss_fn)

    # Assert
    assert len(results["train_loss"]) == epochs
    assert len(results["train_acc"]) == epochs
    assert mock_mlflow.log_metric.call_count == 2 * epochs
    mock_mlflow.pytorch.log_model.assert_called_once_with(model, "model")
    assert mock_optimizer.step.call_count == (epochs * len(train_dataloader))
    assert mock_optimizer.zero_grad.call_count == (
        epochs * len(train_dataloader))
    assert model.to.call_count == 1


if __name__ == '__main__':
    pytest.main()
