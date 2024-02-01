import os
import torch
import zenml
from zenml import pipeline
from steps import download_and_organize_data
from steps import get_transform
from steps import m_a_model
from steps import train_model
from steps import test_step
from steps import create_dataloaders



@pipeline
def zenml_training_pipeline():
    
    train_dir, valid_dir, test_dir, transform, batch_size, NUM_WORKERS = download_and_organize_data()
    training_dataloader, testing_dataloader, num_classes, optimizer, loss_fn, epochs = create_dataloaders(train_dir = train_dir,
                                                                              valid_dir = valid_dir, 
                                                                              test_dir = test_dir, 
                                                                              transform = transform, 
                                                                              batch_size = batch_size, 
                                                                              num_workers=NUM_WORKERS)
    model = m_a_model(num_classes)
    model, results = train_model(model ,training_dataloader , optimizer, loss_fn, epochs)
    test_loss, test_acc = evaluate_model(model, testing_dataloader, loss_fn)
    
    
    