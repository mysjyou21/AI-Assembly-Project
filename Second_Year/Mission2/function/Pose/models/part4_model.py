import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def Part4DirModel():
    model_name = 'vgg11_bn'
    model = models.vgg11_bn()
    num_classes = 2
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model
