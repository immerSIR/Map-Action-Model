import torch
from data_transform import get_transform
from data_loading_pipeline import create_dataloaders
from m_a_model import m_a_model
from training_utils import TrainingPipeline


training_dataloader, testing_dataloader, class_names = create_dataloaders(train_dir=train_dir, 
                                                                          test_dir=test_dir, 
                                                                          valid_dir=valid_dir, 
                                                                          batch_size=16, 
                                                                          num_workers=os.cpu_count(), 
                                                                          transform=get_transform)

model = m_a_model(len(class_names))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)

trainer = TrainingPipeline(model=model, 
                       train_dataloader=training_dataloader,
                       test_dataloader=testing_dataloader, 
                       epochs=20, 
                       loss_fn=loss_fn, 
                       optimizer=optimizer)

results = trainer.train()