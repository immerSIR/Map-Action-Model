import torch
from torchvision.models import vgg16_bn, VGG16_BN_Weights


def m_a_model(num_classes):
    # Load VGG16 model with batch normalization weights
    vgg16_bn_weights = VGG16_BN_Weights.DEFAULT
    model = vgg16_bn(weights=vgg16_bn_weights)
    
    # Freeze all parameters in the model
    for params in model.parameters():
        params.requires_grad = False
        
    # Modify the classifier to adapt to the number of classes
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Linear(num_features, len(num_classes))])
    model.classifier = nn.Sequential(*features)
    
    return model

        