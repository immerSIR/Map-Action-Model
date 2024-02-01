import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from typing import Annotated, Optional, Tuple, List, Dict
from zenml import step
from zenml.pipelines import pipeline


@step
def create_dataloaders(train_dir, valid_dir, test_dir, transform, batch_size, num_workers) -> Tuple[
    Annotated[DataLoader, "trainin_dataloader"],
    Annotated[DataLoader, "testting_dataloader"],
    Annotated[List, "class_names"]
]:
    
    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    
    # Get class names
    class_names = train_data.classes
    num_classes = len(class_names)
    
    # Turn images into data loaders
    training_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    validation_dataloader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    testing_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 20
    
    return training_dataloader, testing_dataloader, num_classes, optimizer, loss_fn, epochs