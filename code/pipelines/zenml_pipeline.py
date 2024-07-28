from zenml import pipeline
from steps.dagshub_utils import download_and_organize_data
from steps.data_preprocess.data_loading_pipeline import create_dataloaders
from steps.model import m_a_model
from steps.training_step import train_model
from steps.model_eval import test_step
from steps.plot_metrics import plot_loss_curves


@pipeline
def zenml_training_pipeline():
    train_dir, valid_dir, test_dir, batch_size = download_and_organize_data()

    training_dataloader, validation_dataloader, testing_dataloader, num_classes, epochs = create_dataloaders(
        train_dir=train_dir,
        valid_dir=valid_dir,
        test_dir=test_dir,
        batch_size=batch_size
    )

    model, loss_fn = m_a_model(num_classes)

    model, results = train_model(
        model=model,
        train_dataloader=training_dataloader,
        val_dataloader=validation_dataloader,
        epochs=epochs,
        loss_fn=loss_fn
    )

    test_loss, test_acc, results = test_step(
        model=model,
        test_dataloader=testing_dataloader,
        loss_fn=loss_fn,
        results=results,
        epochs=epochs
    )

    plot_loss_curves(results)

    return model, results
