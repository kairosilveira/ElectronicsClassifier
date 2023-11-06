import torch.nn as nn
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models import resnet18


class ElectronicsClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Linear(512, num_classes)


    def forward(self, x):
        return self.base_model(x)
