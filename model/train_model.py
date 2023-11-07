import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data.preprocess_data import AutoOrient, MakeSquare, ReduceImage
from model.model_class import ElectronicsClassifier

def train_model(train_dataset, test_dataset, num_epochs, batch_size, lr, loss_function, optimizer, device):  
    """
    Train a neural network model on a given training dataset and evaluate it on a test dataset.

    Args:
    train_dataset (torch.utils.data.Dataset): The training dataset.
    test_dataset (torch.utils.data.Dataset): The test dataset.
    num_epochs (int): The number of training epochs.
    batch_size (int): The batch size for training.
    lr (float): The learning rate for optimization.
    loss_function (torch.nn.Module): The loss function to use.
    optimizer (torch.optim.Optimizer): The optimizer for model weights adjustment.
    device (torch.device): The device (e.g., 'cpu' or 'cuda') on which to perform the training.

    Returns:
    float: The test accuracy achieved by the model after training.

    The function trains a neural network model on the provided training dataset for the specified number of epochs
    using the specified hyperparameters. After training, it evaluates the model's performance on the test dataset
    and returns the test accuracy.

    Example:
    >>> train_dataset = CustomDataset(train_data, train_labels)
    >>> test_dataset = CustomDataset(test_data, test_labels)
    >>> num_epochs = 10
    >>> batch_size = 32
    >>> lr = 0.001
    >>> loss_function = nn.CrossEntropyLoss()
    >>> optimizer = optim.Adam
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>> test_accuracy = train_model(train_dataset, test_dataset, num_epochs, batch_size, lr, loss_function, optimizer, device)
    >>> print(f"Test Accuracy: {test_accuracy:.2f}")

    """
    
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    num_classes = len(train_dataset.classes)
    model = ElectronicsClassifier(num_classes)  # Import the model class

    # Define loss function and optimizer
    criterion = loss_function
    optimizer = optimizer(model.parameters(), lr=lr)

    # Training loop
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

    # Evaluation on the test dataset
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

    return accuracy
