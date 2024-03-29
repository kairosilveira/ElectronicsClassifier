from hyperopt import hp, tpe, fmin
from model.train_eval_model import train_model, eval_model
from data.transforms import get_transform
from functools import partial 


import torch.optim as optim


# Define the objective function to minimize (maximize validation accuracy)
def objective(params, train_dir, val_dir, device):
    

    # train params
    num_epochs = int(params['num_epochs'])
    batch_size = params['batch_size']
    lr = params['lr']
    optimizer = params['optimizer']

    transform_train = get_transform('train', params)
    transform_test = get_transform('test', params)


    # Call the train_model function with the specified hyperparameters and validation set
    train_result = train_model( 
                train_dir, 
                num_epochs, 
                batch_size, 
                lr,
                optimizer, 
                transform_train,  
                device)
    model = train_result.model
    
    metrics_result = eval_model(
                model, 
                val_dir,
                transform_test,
                batch_size,
                device)
    accuracy = metrics_result.accuracy

    # Hyperopt minimizes, so negate the accuracy to maximize it

    return -accuracy


def find_best_params(space, train_dir, val_dir, device, max_evals= 30):
    best = fmin(
        fn=partial(objective, train_dir=train_dir, val_dir=val_dir, device = device), 
        space=space, 
        algo=tpe.suggest, 
        max_evals=max_evals
    )

    best = fix_params(best)
    return best

def fix_params(params):
    optmizers = {
        0:optim.SGD, 
        1:optim.Adam, 
        2:optim.RMSprop
    }

    batch_sizes = {
        0:16,
        1:32,
        2:64
    }
    params['optimizer'] = optmizers[params['optimizer']] 
    params['batch_size'] = batch_sizes[params['batch_size']]
    params['num_epochs'] = int(params['num_epochs'])
    return params