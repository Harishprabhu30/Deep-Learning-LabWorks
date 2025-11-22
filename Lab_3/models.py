import torch.nn as nn
from torchvision import models
import pretrainedmodels

def build_model(name, num_classes=2):
    if name == "mobilenet":
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif name == "efficientnet":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif name == "nasanet":
        model = pretrainedmodels.__dict__["nasnetamobile"](num_classes=1000, pretrained="imagenet")
        in_features = model.last_linear.in_features
        model.last_linear = nn.Linear(in_features, num_classes)

    else:
        raise ValueError("Model not supported.")
    
    return model
