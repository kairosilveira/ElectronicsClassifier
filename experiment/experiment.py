import os
from utils.plots import plot_confusion_matrix, plot_normalized_confusion_matrix, plot_learning_curve
from data.transforms import get_transform
from data.split_data import split_data, merge_val_into_train
from model.optimize_parameters import find_best_params
from model.train_eval_model import train_model, eval_model
from .config import *
from time import time
import mlflow


class Experiment:
    def __init__(self, data_path, experiment_name = 'resistor_classifier') -> None:
        self.data_path = os.path.join(ROOT_DIR_PATH,data_path)
        self.experiment_name = experiment_name

        self.train_path = None
        self.test_path = None
        self.val_path = None 

        self._set_data_paths()

    def run(self, run_name=None):
        start_time = time()
        with mlflow.start_run(run_name=run_name):
            print("spliting data")
            split_data(self.data_path, 
                    self.train_path, 
                    self.test_path,
                    self.val_path,
                    split_ratio=SPLIT_RATIO)
            
            print("starting optimization")
            print("NÃO DESLIGUE O COMPUTADOR")
            hyper_opt_params = self._get_hyper_opt_params()
            best_hyper_params = find_best_params(**hyper_opt_params) 

            # Log hyperparameters
            mlflow.log_params(best_hyper_params)

            #train a model with best params and evaluate
            merge_val_into_train(self.train_path, self.val_path)
            self.train_params = self._get_train_params(best_hyper_params)

            print("starting training")
            train_results = train_model(**self.train_params)
            model = train_results.model
            # Log model
            mlflow.pytorch.log_model(model, "model")
            
            train_losses = train_results.train_loss_values
            test_losses = train_results.val_loss_values

            
            plot_learning_curve(train_losses,test_losses)
            mlflow.log_artifact(os.path.join('utils', 'plots', 'learning_curve.png'))

            eval_params = self._get_eval_params(model) 
            eval_metrics = eval_model(**eval_params)

            accuracy = eval_metrics.accuracy 
            confusion_matrix = eval_metrics.confusion_matrix
            mlflow.log_metrics({'accuracy': accuracy})

            plot_confusion_matrix(confusion_matrix, model.classes) 
            plot_normalized_confusion_matrix(confusion_matrix, model.classes) 

            mlflow.log_artifact(os.path.join('utils', 'plots', 'confusion_matrix.png'))
            mlflow.log_artifact(os.path.join('utils', 'plots', 'normalized_confusion_matrix.png'))


            end_time = time()
            run_time = end_time-start_time
            mlflow.log_param('run_time',run_time)





    def _set_data_paths(self):
        # Constructing paths for train, test, and validation data directories
        self.train_path = os.path.join(ROOT_DIR_PATH, "data", "data_electronic", "train")
        self.test_path = os.path.join(ROOT_DIR_PATH, "data", "data_electronic", "test")
        self.val_path = os.path.join(ROOT_DIR_PATH, "data", "data_electronic", "val")

    def _get_hyper_opt_params(self):
        hyper_opt_params = {
            'space': SPACE_PARAMS,
            'train_dir': self.train_path,
            'val_dir': self.val_path,
            'device': DEVICE,
            'max_evals': MAX_EVALS,  # Adjust based on your requirements
        }
        return hyper_opt_params

    def _get_train_params(self, best_hyper_params):

        train_transform = get_transform('train', best_hyper_params)
        test_transform = get_transform('test', best_hyper_params)

        train_params =  {
            'train_dir':self.train_path,
            'num_epochs':best_hyper_params['num_epochs'],
            'batch_size':best_hyper_params['batch_size'],
            'lr':best_hyper_params['lr'],
            'optimizer':best_hyper_params['optimizer'],
            'transform_train':train_transform, 
            'device':DEVICE, 
            'learning_curve':True, 
            'val_dir':self.test_path, 
            'transform_val':test_transform
        }

        return train_params
    
    def _get_eval_params(self, model):
        eval_params = {
            'model':model, 
            'test_dir':self.test_path, 
            'transform_test': self.train_params['transform_val'], 
            'batch_size': self.train_params['batch_size'], 
            'device': DEVICE
        }

        return eval_params