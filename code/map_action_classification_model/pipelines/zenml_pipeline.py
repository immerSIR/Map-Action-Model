import os
from zenml.steps import step, Output, BaseStepConfig
from zenml.pipelines import pipeline
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow


def zenml_training_pipeline(get_data, data_preparation ,model,
                      train_model ,batch_size ,num_workers ,evaluate_model,optimizer ,loss_fn, epochs):
    
    
    get_data = download_and_organize_data()
    training_dataloader, testing_dataloader, class_names = create_dataloaders(get_data['train_dir'],
                                                                              get_data['valid_dir'], 
                                                                              get_data['test_dir'], 
                                                                              transform = get_transform(train=True), 
                                                                              batch_size = batch_size, 
                                                                              num_workers=num_workers)
    model = model(len(class_names))
    model, results = train_model(model ,training_dataloader , optimizer, loss_fn, epochs)
    test_loss, test_acc = evaluate_model(model, testing_dataloader, loss_fn)
    
    
    