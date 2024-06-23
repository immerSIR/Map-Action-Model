import pytest
from unittest.mock import patch, MagicMock
from code.pipelines.zenml_pipeline import zenml_training_pipeline


@patch('steps.dagshub_utils.dagshub_data_load.download_and_organize_data')
@patch('steps.create_dataloaders.create_dataloaders')
@patch('steps.m_a_model.m_a_model')
@patch('steps.train_model.train_model')
@patch('steps.test_step.test_step')
@patch('steps.plot_metrics.plot_metrics.plot_loss_curves')
def test_zenml_training_pipeline(
    mock_plot_loss_curves,
    mock_test_step,
    mock_train_model,
    mock_m_a_model,
    mock_create_dataloaders,
    mock_download_and_organize_data
):
    # Mock return values
    mock_download_and_organize_data.return_value = (
        'train_dir', 'valid_dir', 'test_dir', 32)
    mock_create_dataloaders.return_value = (MagicMock(), MagicMock(), 10, 5)
    mock_m_a_model.return_value = (MagicMock(), MagicMock())
    mock_train_model.return_value = (MagicMock(), 'results')
    mock_test_step.return_value = (0.1, 0.9, 'results')

    # Run the pipeline
    zenml_training_pipeline()

    # Assertions to ensure mocks were called
    mock_download_and_organize_data.assert_called_once()
    mock_create_dataloaders.assert_called_once_with(
        train_dir='train_dir', valid_dir='valid_dir', test_dir='test_dir', batch_size=32)
    mock_m_a_model.assert_called_once_with(10)
    mock_train_model.assert_called_once()
    mock_test_step.assert_called_once()
    mock_plot_loss_curves.assert_called_once_with('results')


if __name__ == '__main__':
    pytest.main()
