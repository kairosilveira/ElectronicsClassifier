import torch.optim as optim
from hyperopt import hp
from multiprocessing import cpu_count
import torch
import os

N_CORES = cpu_count()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SPLIT_RATIO = (0.6,0.2,0.2)

SPACE_PARAMS = {
    'num_epochs': hp.choice('num_epochs', [1]),
    # 'num_epochs': hp.quniform('num_epochs', 3, 20, 1),
    # 'batch_size': hp.choice('batch_size', [32]),
    'batch_size': hp.choice('batch_size', [16,32,64]),
    'lr': hp.loguniform('lr', -5, -2),
    # 'lr': hp.choice('lr', [0.001]),
    'optimizer': hp.choice('optimizer', [optim.SGD, optim.Adam, optim.RMSprop]),
    # 'optimizer': hp.choice('optimizer', [optim.SGD]),
    'scale_factor': hp.uniform('scale_factor', 0.1, 1.0),
    'degrees': hp.uniform('degrees', 0, 45),
    'brightness_factor': hp.uniform('brightness_factor', 0.5, 2.0),
    'contrast_factor': hp.uniform('contrast_factor', 0.5, 2.0),
    'saturation_factor': hp.uniform('saturation_factor', 0.5, 2.0),
    'hue_factor': hp.uniform('hue_factor', 0.0, 0.5),
}

MAX_EVALS = 1

ROOT_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_LAYERS_UNFROZEN = 17