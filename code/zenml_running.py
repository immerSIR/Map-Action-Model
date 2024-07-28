from zenml import step, pipeline
from pipelines.zenml_pipeline import zenml_training_pipeline

if __name__ == "__main__":
    training = zenml_training_pipeline()
