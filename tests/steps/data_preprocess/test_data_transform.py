import pytest
import torch
from torchvision.transforms import v2 as T
from unittest.mock import patch
from code.steps.data_preprocess.data_transform import get_transform


@patch('code.steps.data_preprocess.data_transform.T.RandomHorizontalFlip')
@patch('code.steps.data_preprocess.data_transform.T.RandomResizedCrop')
@patch('code.steps.data_preprocess.data_transform.T.ToDtype')
@patch('code.steps.data_preprocess.data_transform.T.Compose')
def test_get_transform_train(mock_Compose, mock_ToDtype, mock_RandomResizedCrop, mock_RandomHorizontalFlip):
    # Arrange
    train = True
    mock_RandomHorizontalFlip_instance = mock_RandomHorizontalFlip.return_value
    mock_RandomResizedCrop_instance = mock_RandomResizedCrop.return_value
    mock_ToDtype_instance = mock_ToDtype.return_value

    # Act
    transform = get_transform(train=train)

    # Assert
    mock_RandomHorizontalFlip.assert_called_once_with(0.5)
    mock_RandomResizedCrop.assert_called_once_with(
        size=[224, 224], antialias=True)
    mock_ToDtype.assert_called_once_with(torch.float32, scale=True)

    # Check that Compose receives the correct sequence of transformations
    mock_Compose.assert_called_once_with([
        mock_RandomHorizontalFlip_instance,
        mock_RandomResizedCrop_instance,
        mock_ToDtype_instance
    ])
    assert transform == mock_Compose.return_value


@patch('code.steps.data_preprocess.data_transform.T.ToDtype')
@patch('code.steps.data_preprocess.data_transform.T.Compose')
def test_get_transform_not_train(mock_Compose, mock_ToDtype):
    # Arrange
    train = False
    mock_ToDtype_instance = mock_ToDtype.return_value

    # Act
    transform = get_transform(train=train)

    # Assert
    mock_ToDtype.assert_called_once_with(torch.float32, scale=True)

    # Check that Compose receives the correct sequence of transformations
    mock_Compose.assert_called_once_with([
        mock_ToDtype_instance
    ])
    assert transform == mock_Compose.return_value


if __name__ == '__main__':
    pytest.main()
