import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data.preprocess_data import AutoOrient, MakeSquare, ReduceImage
from model.model_class import ElectronicsClassifier

def train_model(data_dir, train_dir, test_dir, num_epochs, batch_size, lr, scale_factor, loss_function, optimizer):
    # Define data transforms including the variable scale factor
    transform = transforms.Compose([
        AutoOrient(),  # Automatically orient the image
        MakeSquare(),  # Center crop to make it square
        ReduceImage(scale_factor),  # Variable scale factor
        transforms.Resize((224, 224)),  # Resize to a standard size
        transforms.ToTensor(),  # Convert to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    num_classes = len(train_dataset.classes)
    model = ElectronicsClassifier(num_classes)  # Import the model class

    # Define loss function and optimizer
    criterion = loss_function
    optimizer = optimizer(model.parameters(), lr=lr)

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.2f}")

# Example usage:
data_dir = 'Resistores'
train_dir = 'train'
test_dir = 'test'
num_epochs = 10
batch_size = 16
lr = 0.001
scale_factor = 0.50  # You can adjust this value
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam

train_model(data_dir, train_dir, test_dir, num_epochs, batch_size, lr, scale_factor, loss_function, optimizer)
