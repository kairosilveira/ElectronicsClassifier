

import torch.nn as nn
import torch.optim as optim
import torch
from hyperopt import hp, tpe, fmin

from data.split_data import split_data_into_train_val_test
from model.optimize_parameters import find_best_params

# Define your data directories
data_dir = "data/data_electronic/raw_test"
data_dir = "/home/silveira/Documents/EletronicsClassifier/data/data_electronic/raw_test"

train_dir = "/home/silveira/Documents/EletronicsClassifier/data/data_electronic/train"
test_dir = "/home/silveira/Documents/EletronicsClassifier/data/data_electronic/test"
val_dir = "/home/silveira/Documents/EletronicsClassifier/data/data_electronic/val"
split_data_into_train_val_test(data_dir, train_dir, val_dir, test_dir)

# # Define the search space for hyperparameters
# space = {
#     'num_epochs': hp.quniform('num_epochs', 5, 20, 1),
#     'batch_size': hp.choice('batch_size', [16, 32, 64]),
#     'lr': hp.loguniform('lr', -5, -2),
#     'loss_function': hp.choice('loss_function', [nn.CrossEntropyLoss(), nn.MSELoss()]),
#     'optimizer': hp.choice('optimizer', [optim.SGD, optim.Adam, optim.RMSprop]),
#     'scale_factor': hp.uniform('scale_factor', 0.1, 1.0),
#     'degrees': hp.uniform('degrees', 0, 45),
#     'brightness_factor': hp.uniform('brightness_factor', 0.5, 2.0),
#     'contrast_factor': hp.uniform('contrast_factor', 0.5, 2.0),
#     'saturation_factor': hp.uniform('saturation_factor', 0.5, 2.0),
#     'hue_factor': hp.uniform('hue_factor', 0.0, 0.5),
# }



# # Define the device (e.g., 'cuda' or 'cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Your objective function and find_best_params function

# # Define param dictionary for find_best_params
# param = {
#     'space': space,
#     'train_dir': train_dir,
#     'val_dir': val_dir,
#     'device': device,
#     'max_evals': 30,  # Adjust based on your requirements
# }

# best_params = find_best_params(**param)

# print("Best hyperparameters:", best_params)
