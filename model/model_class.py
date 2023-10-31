import torch.nn as nn
import torchvision.models as models


class ElectronicClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.base_model(x)