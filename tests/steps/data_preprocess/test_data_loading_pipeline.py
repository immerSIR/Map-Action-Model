import unittest
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader
from torchvision import datasets
from .test_data_transform import get_transform
from code import create_dataloaders as data_loading_pipeline

# add doctrings to the test class
class TestDataLoadingPipeline(unittest.TestCase):
    @patch('torchvision.datasets.ImageFolder')
    def test_data_loading_pipeline(self, mock_image_folder):
        # Mock the ImageFolder method
        mock_image_folder.return_value = MagicMock(classes=['class1', 'class2', 'class3'])

        # Define the directories and batch size
        train_dir = 'train_dir'
        test_dir = 'test_dir'
        batch_size = 20

        # Run the function
        training_dataloader, testing_dataloader, num_classes, num_epochs = data_loading_pipeline(train_dir, test_dir, batch_size)

        # Check the data loaders
        self.assertIsInstance(training_dataloader, DataLoader)
        self.assertIsInstance(testing_dataloader, DataLoader)

        # Check the number of classes
        self.assertEqual(num_classes, 3)

        # Check the number of epochs
        self.assertEqual(num_epochs, 10)

if __name__ == '__main__':
    unittest.main()