import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Annotated, Tuple
from zenml import step


@step
def create_dataloaders(
    train_dir: str,
    valid_dir: str,
    test_dir: str,
    batch_size: int
) -> Tuple[
    Annotated[DataLoader, "training_dataloader"],
    Annotated[DataLoader, "validation_dataloader"],
    Annotated[DataLoader, "testing_dataloader"],
    Annotated[int, "num_classes"],
    Annotated[int, "epochs"],
]:
    """
    Create PyTorch data loaders for training, validation, and testing datasets.

    Args:
    train_dir (str): Directory containing the training dataset.
    valid_dir (str): Directory containing the validation dataset.
    test_dir (str): Directory containing the testing dataset.
    batch_size (int): Batch size for the data loaders.

    Returns:
    Tuple[DataLoader, DataLoader, DataLoader, int, int]: Tuple containing the training data loader,
    validation data loader, testing data loader, number of classes, and the number of epochs.
    """
    NUM_WORKERS = 2

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Use ImageFolder to create datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    # Get class names
    class_names = train_data.classes
    num_classes = len(class_names)

    # Turn images into data loaders
    training_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    validation_dataloader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    testing_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    epochs = 10  # You can adjust this or make it a parameter

    return training_dataloader, validation_dataloader, testing_dataloader, num_classes, epochs
