import os
import torch
from tqdm import tqdm
from data_loader import mapaction_test_data_loader, mapction_data_loader
import model
from typing import Tuple, List, Dict

num_classes = 2
Model =model.mapaction_instance_segmentation_model(num_classes)
Device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
Loss_fn = torch.nn.CrossEntropyLoss()

params = [p for p in Model.parameters() if p.requires_grad]


Optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    model.train()
    train_loss, train_acc = 0.0, 0.0
    
    for X, y in tqdm(dataloader):
        X = list(image.to(device) for image in X)
        y = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in y]
        
        optimizer.zero_grad()
        #print("Targets:", y)
        loss_dict = model(X, y)
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        #y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        #train_acc += (y_pred_class == y).sum().item() / len(y_pred)
        
        # Accumulate loss
        train_loss  = loss
    
    # Average loss and accuracy over the entire training set
    #train_loss /= len(dataloader)
    #train_acc /= len(dataloader)
    
    return train_loss
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    model.eval()
    test_loss, test_acc = 0.0, 0.0
    
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X = X.to(torch.uint8)
            y = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in y]
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss +=loss.cpu().item()
            
            # Calculate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)
            
            # Accumulate loss

    
    # Average loss and accuracy over the entire test set
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    
    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List[float]]:
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}
    
    model.to(device)
    
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(Model,
                                           mapction_data_loader,
                                           Loss_fn,
                                           Optimizer,
                                           Device)
        test_loss , test_acc = test_step(Model,
                                        mapaction_test_data_loader,
                                        Loss_fn,
                                        Device)
        
        print(f"Epoch: {epoch + 1} | "
              f"train_loss: {train_loss:.4f} | "
              f"test_loss: {test_loss:.4f} | "
              f"test_acc: {test_acc:.4f}"
              )
        
        results["train_loss"].append(train_loss)
        #results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
    return results
