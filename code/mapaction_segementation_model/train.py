import torch
from training_pipeline import train
from data_loader import mapaction_test_data_loader, mapction_data_loader
import model

save_path = "/home/mapaction/mapaction_env/Map-Action-Model/model/test.h5"
num_classes = 2
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

Model =model.mapaction_instance_segmentation_model(num_classes)


loss_fn = torch.nn.CrossEntropyLoss()
params = [p for p in Model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

results = train(Model,train_dataloader=mapction_data_loader,
                test_dataloader= mapaction_test_data_loader,
                epochs=20,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device)

torch.save(Model, save_path)