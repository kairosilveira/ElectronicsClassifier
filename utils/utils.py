import mlflow
import torch
from PIL import Image
from data.transforms import get_transform

def convert_str_to_number(value):
    try:
        # Try converting to number (either int or float)
        number = float(value)
        if number.is_integer():
            return int(number)
        return number
    except ValueError:
        # Return as string if not convertible to number
        return value
    

def open_PIL_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image
    

def get_best_run_id(metric):
    runs = mlflow.search_runs(order_by=[f"metrics.{metric} DESC"])
    if not runs.empty:
        best_run_id = runs.iloc[0]["run_id"]
        return best_run_id
    else:
        raise ValueError("No runs found")
    
def load_model(run_id):
    model_path = f'mlruns/0/{run_id}/artifacts/model'
    best_model = mlflow.pytorch.load_model(model_path,map_location=torch.device('cpu'))
    return best_model


def load_run_params(run_id):
    params_dict = mlflow.get_run(run_id).to_dictionary()['data']['params']
    converted_params = {key: convert_str_to_number(value) for key, value in params_dict.items()}
    return converted_params

def load_best_model_and_transform():
    best_run_id = get_best_run_id('accuracy')
    best_model = load_model(best_run_id)
    params = load_run_params(best_run_id)
    transform = get_transform('test', params)
    return best_model, transform 


def get_transform_img():
    best_run_id = get_best_run_id('accuracy')
    params = load_run_params(best_run_id)
    transform = get_transform('img', params)
    return transform

