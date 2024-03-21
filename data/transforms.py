
import torchvision.transforms as transforms
from data.preprocess_data import AutoOrient, MakeSquare, ReduceImage

def get_transform(type, params):

    # Transform params
    scale_factor = params['scale_factor']
    if type == 'train':
        degrees = params['degrees']
        brightness_factor = params['brightness_factor']
        contrast_factor = params['contrast_factor']
        saturation_factor = params['saturation_factor']
        hue_factor = params['hue_factor']
        
        return transforms.Compose([
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
    
    if type == 'test':
        return transforms.Compose([
        AutoOrient(),  # Automatically orient the image
        MakeSquare(),  # Center crop to make it square
        ReduceImage(scale_factor),  # Apply reduction
        transforms.Resize((224, 224)),  # Resize to a standard size
        transforms.ToTensor(),  # Convert to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    if type == 'img':
        return transforms.Compose([
        AutoOrient(),  # Automatically orient the image
        MakeSquare(),  # Center crop to make it square
        ReduceImage(scale_factor),  # Apply reduction
    ])