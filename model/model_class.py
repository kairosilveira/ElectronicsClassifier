import torch.nn as nn
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models import resnet18
import torch
from PIL import Image
from data.transforms import get_transform

class ElectronicsClassifier(nn.Module):
    def __init__(self, classes, n_layers_unfrozen = 10):
        super().__init__()
        self.base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Linear(512, len(classes))

        self.classes = classes

        total_layers = sum(len(list(child.parameters())) for child in self.base_model.children())
        layers_unfrozen = 0

        for child in self.base_model.children():
            for param in child.parameters():
                param.requires_grad = False
                layers_unfrozen += 1
                if layers_unfrozen == total_layers - n_layers_unfrozen:
                    break
            if layers_unfrozen == total_layers - n_layers_unfrozen:
                break

    def forward(self, x):
        return self.base_model(x)    

    def predict(self, input):
        self.eval()
        with torch.no_grad():
            output = self.forward(input)
            index = torch.argmax(output).item()
        return self.classes[index]
    
    def predict_proba(self, input):
        self.eval()
        with torch.no_grad():
            output = self.forward(input)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            prediction = {
                'class':self.classes[torch.argmax(probabilities).item()],
                'prob': torch.max(probabilities).item()
            }
            
        return prediction
