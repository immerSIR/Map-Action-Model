import pytest
import torch
from torch.utils.data import DataLoader
from unittest.mock import patch, MagicMock
from code.steps.model_eval import test_step


@patch('code.steps.model_eval.mlflow')
@patch('code.steps.model_eval.tqdm')
@patch('code.steps.model_eval.torch.save')
@patch('code.steps.model_eval.torch.device')
def test_test_step(mock_device, mock_torch_save, mock_tqdm, mock_mlflow):
    # Arrange
    mock_device.return_value = torch.device('cpu')
    mock_tqdm_range = MagicMock()
    mock_tqdm.return_value = mock_tqdm_range

    model = MagicMock()
    test_dataloader = MagicMock(spec=DataLoader)
    loss_fn = MagicMock()
    results = {"test_loss": [], "test_acc": []}
    epochs = 1
    # Example tensor shape for 4 samples, assuming 3x224x224 images
    sample_data = torch.randn(4, 3, 224, 224)
    # Example labels for binary classification
    sample_labels = torch.randint(0, 2, (4,))

    # Configure the dataloader to return a sample data batch
    test_dataloader.__iter__.return_value = [(sample_data, sample_labels)]

    def mock_model_forward(x):
        # Example output logits for 4 samples and 2 classes
        return torch.randn(4, 2)

    model.side_effect = mock_model_forward
    mock_loss_value = torch.tensor(1.0)
    loss_fn.return_value = mock_loss_value

    # Act
    test_loss, test_acc, results = test_step(
        model, test_dataloader, loss_fn, results, epochs)

    # Assert
    assert test_loss == 1.0
    assert 0.0 <= test_acc <= 1.0
    assert 'test_loss' in results
    assert 'test_acc' in results
    assert len(results['test_loss']) == epochs
    assert len(results['test_acc']) == epochs
    assert mock_mlflow.log_metric.call_count == 2 * epochs
    mock_torch_save.assert_called_once_with(
        model.state_dict(), 'save/TCM1.pth')
    mock_mlflow.pytorch.log_model.assert_called_once_with(model, "model")


if __name__ == '__main__':
    pytest.main()
