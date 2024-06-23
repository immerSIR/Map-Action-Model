import pytest
import torch
from unittest.mock import patch, MagicMock
from code.steps.model import m_a_model


@patch('code.steps.model.vgg16_bn')
@patch('code.steps.model.VGG16_BN_Weights')
@patch('code.steps.model.torch.nn.CrossEntropyLoss')
def test_m_a_model(mock_CrossEntropyLoss, mock_VGG16_BN_Weights, mock_vgg16_bn):
    # Arrange
    num_classes = 10
    mock_vgg_instance = MagicMock()
    mock_vgg16_bn.return_value = mock_vgg_instance
    mock_weight_instance = MagicMock()
    mock_VGG16_BN_Weights.DEFAULT = mock_weight_instance
    mock_loss_fn_instance = MagicMock()
    mock_CrossEntropyLoss.return_value = mock_loss_fn_instance

    # Act
    model, loss_fn = m_a_model(num_classes)

    # Assert
    mock_vgg16_bn.assert_called_once_with(weights=mock_weight_instance)
    for param in mock_vgg_instance.parameters():
        assert param.requires_grad is False

    assert isinstance(model.classifier, torch.nn.Sequential)
    assert len(model.classifier) == 6  # Adjust if the number of layers varies
    assert isinstance(model.classifier[-1], torch.nn.Linear)
    assert model.classifier[-1].out_features == num_classes

    mock_CrossEntropyLoss.assert_called_once()
    assert loss_fn == mock_loss_fn_instance


if __name__ == '__main__':
    pytest.main()
