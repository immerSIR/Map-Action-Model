import pytest
from unittest.mock import patch, MagicMock
from code.steps.plot_metrics.plot_metrics import plot_loss_curves


@patch('code.steps.plot_metrics.plot_metrics.plt')
def test_plot_loss_curves(mock_plt):
    # Arrange
    results = {
        "train_loss": [0.1, 0.08, 0.06, 0.05],
        "test_loss": [0.1, 0.09, 0.08, 0.07],
        "train_acc": [0.9, 0.92, 0.93, 0.94],
        "test_acc": [0.89, 0.9, 0.91, 0.92]
    }

    # Act
    plot_loss_curves(results)

    # Assert
    # Check that the figure and subplots were created
    mock_plt.figure.assert_called_with(figsize=(15, 7))
    assert mock_plt.subplot.call_count == 2

    # Check the first subplot for loss
    mock_plt.subplot.assert_any_call(1, 2, 1)
    mock_plt.plot.assert_any_call(
        range(4), results["train_loss"], label="train_loss")
    mock_plt.plot.assert_any_call(
        range(4), results["test_loss"], label="test_loss")
    mock_plt.title.assert_any_call("Loss")
    mock_plt.xlabel.assert_any_call("Epochs")
    mock_plt.legend.assert_any_call()

    # Check the second subplot for accuracy
    mock_plt.subplot.assert_any_call(1, 2, 2)
    mock_plt.plot.assert_any_call(
        range(4), results["train_acc"], label="train_accuracy")
    mock_plt.plot.assert_any_call(
        range(4), results["test_acc"], label="test_accuracy")
    mock_plt.title.assert_any_call("Accuracy")
    mock_plt.xlabel.assert_any_call("Epochs")
    mock_plt.legend.assert_any_call()


if __name__ == '__main__':
    pytest.main()
