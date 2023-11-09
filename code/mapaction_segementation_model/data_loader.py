import torch
import os
import utils
from utils import MapActionDataset
from data_transform import get_transform
from torch.utils.data import DataLoader, Subset


NUM_WORKERS = os.cpu_count()


mapaction_dataset = MapActionDataset('', get_transform(train=True))
mapaction_test_dataset = MapActionDataset('', get_transform(train=False))


indices = torch.randperm(len(mapaction_dataset)).tolist()
dataset = Subset(mapaction_dataset, indices[:70])
test_dataset = Subset(mapaction_test_dataset, indices[70:])




mapction_data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_worker=NUM_WORKERS,
    collate_fn=utils.collate_fn
)

mapaction_test_data_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_worker=NUM_WORKERS,
    collate_fn=utils.collate_fn
)