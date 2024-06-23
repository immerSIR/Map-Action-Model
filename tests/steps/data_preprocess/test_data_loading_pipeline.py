import pytest
from unittest.mock import patch, MagicMock
# Import create_dataloaders from the correct module
from code.steps.data_preprocess.data_loading_pipeline import create_dataloaders


@patch('code.steps.data_preprocess.data_loading_pipeline.datasets.ImageFolder')
@patch('code.steps.data_preprocess.data_loading_pipeline.DataLoader')
@patch('code.steps.data_preprocess.data_loading_pipeline.get_transform')
def test_create_dataloaders(mock_get_transform, mock_DataLoader, mock_ImageFolder):
    # Set up mock return values
    mock_get_transform.return_value = MagicMock()
    mock_train_data = MagicMock()
    mock_test_data = MagicMock()
    mock_ImageFolder.side_effect = [mock_train_data, mock_test_data]

    mock_train_loader = MagicMock()
    mock_test_loader = MagicMock()
    mock_DataLoader.side_effect = [mock_train_loader, mock_test_loader]

    # Mock the classes attribute on the mocked train_data object
    mock_train_data.classes = ['class1', 'class2', 'class3']

    # Call the function with test parameters
    train_dir = "train_dir"
    valid_dir = "valid_dir"
    test_dir = "test_dir"
    batch_size = 32

    training_dataloader, testing_dataloader, num_classes, epochs = create_dataloaders(
        train_dir=train_dir,
        valid_dir=valid_dir,
        test_dir=test_dir,
        batch_size=batch_size
    )

    # Assertions to ensure the function behaves as expected
    mock_ImageFolder.assert_any_call(train_dir, transform=mock_get_transform())
    mock_ImageFolder.assert_any_call(test_dir, transform=mock_get_transform())

    assert mock_ImageFolder.call_count == 2

    mock_DataLoader.assert_any_call(
        mock_train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    mock_DataLoader.assert_any_call(
        mock_test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    assert mock_DataLoader.call_count == 2

    assert training_dataloader == mock_train_loader
    assert testing_dataloader == mock_test_loader
    assert num_classes == len(mock_train_data.classes)
    assert epochs == 5


if __name__ == '__main__':
    pytest.main()
