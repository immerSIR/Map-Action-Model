import torch
from torchvision.transforms import v2 as T
from typing import Annotated, Optional, Tuple

def get_transform(train: bool) -> T.Compose:
    """
    Get image transformations based on whether it's for training or not.

    This function builds a series of image transformations using torchvision v2 API,
    tailored for either training or testing phase. The transformations for training 
    include random horizontal flipping and resizing with cropping, followed by normalization.
    For both training and testing, the images are converted to torch.float32 and scaled.

    Args:
        train (bool): True if transformations are for training, False otherwise.

    Returns:
        T.Compose: A torchvision.transforms.Compose object that represents the composition 
        of image transformations.
    """
    transforms = []  # Initialize an empty list to store transformations

    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))  # Randomly flip the image horizontally
        transforms.append(T.RandomResizedCrop(size=[224, 224], antialias=True))  # Resize and crop

    transforms.append(T.ToDtype(torch.float32, scale=True))  # Convert to float32 and scale

    return T.Compose(transforms)  # Combine transformations into a Compose object
