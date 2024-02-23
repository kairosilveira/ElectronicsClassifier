import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from experiment.config import ROOT_DIR_PATH
import os

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, figsize=(8, 6)):
    """
    Plot the confusion matrix.

    Args:
    - cm (numpy.ndarray): Confusion matrix.
    - classes (list): List of class labels.
    - title (str): Title for the plot.
    - cmap: Colormap for the plot.
    """
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    # Add text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')

    img_path = os.path.join(ROOT_DIR_PATH, 'utils/plots/confusion_matrix.png')
    plt.savefig(img_path)


def plot_normalized_confusion_matrix(cm, classes, title='Normalized Confusion Matrix', cmap=plt.cm.Blues, figsize=(8, 6)):
    """
    Plot the normalized confusion matrix.

    Args:
    - cm (numpy.ndarray): Confusion matrix.
    - classes (list): List of class labels.
    - title (str): Title for the plot.
    - cmap: Colormap for the plot.
    """
    # Calculate normalized confusion matrix
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized_cm = cm / row_sums.astype(float)

    plt.figure(figsize=figsize)
    plt.imshow(normalized_cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    # Add text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, f'{normalized_cm[i, j]:.2f}', ha='center', va='center', color='white' if normalized_cm[i, j] > 0.5 else 'black')

    img_path = os.path.join(ROOT_DIR_PATH, 'utils/plots/normalized_confusion_matrix.png')
    plt.savefig(img_path)


def plot_learning_curve(train_losses, test_losses, title='Learning Curve', xlabel='Epoch', ylabel='Loss',figsize=(8, 6)):
    """
    Plot the learning curve with training and test loss values.

    Args:
    - train_losses (list or numpy.ndarray): List or array of training loss values.
    - test_losses (list or numpy.ndarray): List or array of test loss values.
    - title (str): Title for the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=figsize)
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, test_losses, label='Test Loss', marker='o')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    img_path = os.path.join(ROOT_DIR_PATH, 'utils/plots/learning_curve.png')
    plt.savefig(img_path)