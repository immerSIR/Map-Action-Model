import os
import torch
from torch import nn
from zenml.steps import step, Output, BaseStepConfig
from zenml.pipelines import pipeline
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
from zenml_pipeline import zenml_training_pipeline
from ..step.data_loading_pipeline import create_dataloaders
from ..step.dagshub_data_load import download_and_organize_data
from ..step.data_transform import get_transform
from ..step.m_a_model import m_a_model
from ..step.training_step import train_model
from ..step.evaluation import test_step

NUM_WORKERS = os.cpu_count()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)

training = zenml_training_pipeline(
    get_data = download_and_organize_data,
    data_preparation = create_dataloaders,
    model = m_a_model,
    optimizer = optimizer,
    loss_fn = loss_fn,
    batch_size = 20,
    num_workers = NUM_WORKERS,
    epochs = 20,
    train_model = train_model,
    evaluate_model = evaluate_model()
)
    
training.run()