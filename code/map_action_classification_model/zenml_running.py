import os
import torch
from torch import nn
import zenml
from zenml.steps import step, Output
from zenml.pipelines import pipeline
from zenml.integrations.mlflow.experiment_trackers import (
    MLFlowExperimentTracker,
)
from pipelines.zenml_pipeline import zenml_training_pipeline


training = zenml_training_pipeline().with_options(enable_cache=False)
    
training()