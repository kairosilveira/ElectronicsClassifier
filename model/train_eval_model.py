import torch

import numpy as np
import torchvision.datasets as datasets
from model.model_class import ElectronicsClassifier
from collections import namedtuple
from experiment.config import N_CORES

def train_model(train_dir, num_epochs, batch_size, lr, optimizer, transform_train, device, learning_curve=False, val_dir=None, transform_val=None):
    """
    Train EletronicClassifier model on a given training dataset.

    Args:
    train_dir (str): The directory path to the training dataset.
    num_epochs (int): The number of training epochs.
    batch_size (int): The batch size for training.
    lr (float): The learning rate for optimization.
    optimizer (torch.optim.Optimizer): The optimizer for model weights adjustment.
    transform_train (torchvision.transforms.Compose): The data transformation for training.
    device (torch.device): The device (e.g., 'cpu' or 'cuda') on which to perform the training.
    learning_curve (bool): Whether to return the learning curve information.

    Returns:
    torch.nn.Module or Tuple[torch.nn.Module, List[float]]: The trained model. If learning_curve is True, 
    it also returns a list of training loss values for each epoch.

    This function trains a neural network model on the provided training dataset for the specified number of epochs
    using the specified hyperparameters and returns the trained model.

    Example:
    >>> train_dir = 'path_to_training_data'
    >>> num_epochs = 10
    >>> batch_size = 32
    >>> lr = 0.001
    >>> optimizer = Adam
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>> trained_model = train_model(train_dir, num_epochs, batch_size, lr, optimizer, transform_train, device)
    """
    train_result = namedtuple(
        'train_result',
        ['model', 'train_loss_values', 'val_loss_values']
    )
    # Create datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=N_CORES)
    if learning_curve:
        val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=N_CORES)
    
    # Initialize the model
    num_classes = len(train_dataset.classes)
    model = ElectronicsClassifier(num_classes)  # Import the model class

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optimizer(model.parameters(), lr=lr)

    # Training loop
    model.to(device)

    if learning_curve:
        train_loss_values = []
        val_loss_values = []

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

        if learning_curve:
            train_loss_values.append(running_loss / len(train_loader))
            val_loss = get_validation_loss(model, criterion, val_loader, device)
            val_loss_values.append(val_loss)

    if learning_curve:
        return train_result(
            model=model,
            train_loss_values=train_loss_values, 
            val_loss_values=val_loss_values)
    else:
        return train_result(
            model=model,
            train_loss_values=None, 
            val_loss_values=None)


def get_validation_loss(model, criterion, val_loader, device):
    """
    Calculate validation loss for the provided model.

    Args:
    model (torch.nn.Module): The trained model.
    criterion (torch.nn.Module): The loss criterion.
    val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    device (torch.device): The device (e.g., 'cpu' or 'cuda').

    Returns:
    float: Validation loss.
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    return val_loss / len(val_loader)


def eval_model(model, test_dir, transform_test, batch_size, device):
    """
    Evaluate a neural network model on a test dataset.

    Args:
    model (torch.nn.Module): The trained neural network model.
    test_dir (str): The directory path to the test dataset.
    transform_test (torchvision.transforms.Compose): The data transformation for testing.
    batch_size (int): The batch size for evaluation.
    device (torch.device): The device (e.g., 'cpu' or 'cuda') on which to perform the evaluation.

    Returns:
    float: The accuracy of the model on the test dataset.
    numpy.ndarray: The confusion matrix.

    This function evaluates a trained neural network model on the provided test dataset and returns the accuracy
    as well as the confusion matrix.

    Example:
    >>> test_dir = 'path_to_test_data'
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>> accuracy, confusion = eval_model(trained_model, test_dir, transform_test, batch_size, device)
    """
    metrics = namedtuple(
        'metrics',
        ['accuracy', 'confusion_matrix']
    )

    # Evaluation on the test dataset
    model.eval()

    test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=N_CORES)

    # Get the number of samples in the test dataset
    num_samples = len(test_dataset)

    # Preallocate NumPy arrays for true and predicted labels
    y_true = np.empty(num_samples, dtype=int)
    y_pred = np.empty(num_samples, dtype=int)

    current_index = 0  # To keep track of the current index in the arrays

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Calculate the indices to update in the arrays
            start_idx = current_index
            end_idx = current_index + labels.size(0)
            y_true[start_idx:end_idx] = labels.cpu().numpy()
            y_pred[start_idx:end_idx] = predicted.cpu().numpy()

            current_index = end_idx

    acc = accuracy(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)

    return metrics(accuracy=acc, confusion_matrix=confusion)



def accuracy(y_true, y_pred):
    """
    Calculate the classification accuracy.

    Args:
    y_true (numpy.ndarray): True labels.
    y_pred (numpy.ndarray): Predcted labels.

    Returns:
    float: Classification accuracy.
    """
    total = len(y_true)
    correct = np.sum(y_true == y_pred)
    return correct / total


def confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix.

    Args:
    y_true (numpy.ndarray): True labels.
    y_pred (numpy.ndarray): Predicted labels.

    Returns:
    numpy.ndarray: Confusion matrix as a NumPy array.
    """
    num_classes = max(max(y_true), max(y_pred)) + 1
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        confusion_matrix[t, p] += 1

    return confusion_matrix




