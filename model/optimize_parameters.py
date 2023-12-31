
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.transforms as transforms
from hyperopt import hp, tpe, fmin
from model.train_eval_model import train_model, eval_model
from data.preprocess_data import AutoOrient, MakeSquare, ReduceImage
from functools import partial 
from time import time


# Define the objective function to minimize (maximize validation accuracy)
def objective(params, train_dir, val_dir, device):
    
    start = time()
    # train params
    num_epochs = int(params['num_epochs'])
    batch_size = params['batch_size']
    lr = params['lr']
    optimizer = params['optimizer']

    # Transform params
    scale_factor = params['scale_factor']
    degrees = params['degrees']
    brightness_factor = params['brightness_factor']
    contrast_factor = params['contrast_factor']
    saturation_factor = params['saturation_factor']
    hue_factor = params['hue_factor']

    # Define the data augmentation steps for the training transform
    transform_train = transforms.Compose([
        AutoOrient(),  # Automatically orient the image
        MakeSquare(),  # Center crop to make it square
        transforms.RandomRotation(degrees=degrees),  # Random rotation within +/- degrees
        ReduceImage(scale_factor),  # Apply reduction after rotation
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.ColorJitter(
            brightness=brightness_factor,
            contrast=contrast_factor,
            saturation=saturation_factor,
            hue=hue_factor
        ),  # Color jitter
        transforms.Resize((224, 224)),  # Resize to a standard size
        transforms.ToTensor(),  # Convert to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    # Define the test transform (without data augmentation)
    transform_test = transforms.Compose([
        AutoOrient(),  # Automatically orient the image
        MakeSquare(),  # Center crop to make it square
        ReduceImage(scale_factor),  # Apply reduction
        transforms.Resize((224, 224)),  # Resize to a standard size
        transforms.ToTensor(),  # Convert to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    # Call the train_model function with the specified hyperparameters and validation set
    model = train_model( 
                train_dir, 
                num_epochs, 
                batch_size, 
                lr,
                optimizer, 
                transform_train,  
                device)
    
    accuracy, _ = eval_model(
                model, 
                val_dir,
                transform_test,
                batch_size,
                device)

    # Hyperopt minimizes, so negate the accuracy to maximize it
    end = time()
    print("params:{}".format(params))
    run_time = end-start
    print("=======================")
    print(f"Run Time:{run_time}")
    
    return -accuracy


def find_best_params(space, train_dir, val_dir, device, max_evals= 30):
    best = fmin(
        fn=partial(objective, train_dir=train_dir, val_dir=val_dir, device = device), 
        space=space, 
        algo=tpe.suggest, 
        max_evals=max_evals
    )
    return best
