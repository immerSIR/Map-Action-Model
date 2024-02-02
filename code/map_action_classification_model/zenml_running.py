from zenml import step, pipeline

from pipelines.zenml_pipeline import zenml_training_pipeline

training = zenml_training_pipeline().with_options(enable_cache=False)

training()