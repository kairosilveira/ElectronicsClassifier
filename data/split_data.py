import os
import shutil
import random
import torch.optim as optim

def create_directory(directory, overwrite=True):
    """
    Create a directory, optionally overwriting it if it exists.

    Args:
        directory (str): The path to the directory to create.
        overwrite (bool, optional): Whether to overwrite the directory if it already exists. Defaults to True.

    Returns:
        None
    """
    if os.path.exists(directory):
        if overwrite:
            # If overwrite is True, remove the directory and its contents
            shutil.rmtree(directory)
        else:
            # If overwrite is False, do nothing and return
            return

    os.makedirs(directory)


# Split data into train, validation, and test
def split_data(data_dir, train_dir, test_dir, val_dir=None, 
               split_ratio=(0.6, 0.2, 0.2), random_state=42, overwrite_dirs=True):
    """
    Split data into train, validation, and test sets or train and test sets.

    Args:
        data_dir (str): Path to the data directory.
        train_dir (str): Path to the training data directory.
        test_dir (str): Path to the testing data directory.
        val_dir (str, optional): Path to the validation data directory. Set to None to skip validation set creation.
        split_ratio (tuple, optional): A tuple specifying the ratios for train, validation, and test sets. 
                                     Defaults to (0.6, 0.2, 0.2). If val_dir is None, it is treated as (train_ratio, test_ratio).
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        overwrite_dirs (bool, optional): Whether to overwrite existing directories. Defaults to True.

    Raises:
        ValueError: If len(split_ratio) > 2 and val_dir is None.

    Returns:
        None

    The function splits data into train, validation, and test sets according to the provided split ratios. If val_dir is None, it creates only train and test sets. The data is distributed into corresponding class folders within the specified directories.

    Example:
    >>> data_dir = 'data/images'
    >>> train_dir = 'data/train'
    >>> test_dir = 'data/test'
    >>> val_dir = 'data/validation'
    >>> split_data(data_dir, train_dir, test_dir, val_dir, split_ratio=(0.6, 0.2, 0.2))
    >>> # This creates train, validation, and test sets with a 60-20-20 ratio.
    """

    if len(split_ratio) > 2 and val_dir is None:
        raise ValueError("Invalid input: If you provide a split_ratio with more than two elements, you must specify val_dir.")
    
    random.seed(random_state)

    # Create or overwrite train, validation, and test directories
    create_directory(train_dir, overwrite=overwrite_dirs)
    create_directory(test_dir, overwrite=overwrite_dirs)
    
    if val_dir:
        create_directory(val_dir, overwrite=overwrite_dirs)

    # Only split data for each class if val_dir is not None
    if val_dir:
        for class_folder in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_folder)
            if os.path.isdir(class_path):
                train_class_path = os.path.join(train_dir, class_folder)
                val_class_path = os.path.join(val_dir, class_folder)
                test_class_path = os.path.join(test_dir, class_folder)

                os.makedirs(train_class_path, exist_ok=True)
                os.makedirs(val_class_path, exist_ok=True)
                os.makedirs(test_class_path, exist_ok=True)

                class_images = os.listdir(class_path)
                num_images = len(class_images)
                num_train = int(num_images * split_ratio[0])
                num_val = int(num_images * split_ratio[1])

                train_images = random.sample(class_images, num_train)
                remaining_images = list(set(class_images) - set(train_images))
                val_images = random.sample(remaining_images, num_val)

                for image in class_images:
                    source_path = os.path.join(class_path, image)
                    if image in train_images:
                        destination_path = os.path.join(train_class_path, image)
                    elif image in val_images:
                        destination_path = os.path.join(val_class_path, image)
                    else:
                        destination_path = os.path.join(test_class_path, image)
                    shutil.copy(source_path, destination_path)
    else:
        # No validation set, allocate all data to training and testing
        for class_folder in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_folder)
            if os.path.isdir(class_path):
                train_class_path = os.path.join(train_dir, class_folder)
                test_class_path = os.path.join(test_dir, class_folder)

                os.makedirs(train_class_path, exist_ok=True)
                os.makedirs(test_class_path, exist_ok=True)

                class_images = os.listdir(class_path)
                num_images = len(class_images)
                num_train = int(num_images * split_ratio[0])

                train_images = random.sample(class_images, num_train)

                for image in class_images:
                    source_path = os.path.join(class_path, image)
                    if image in train_images:
                        destination_path = os.path.join(train_class_path, image)
                    else:
                        destination_path = os.path.join(test_class_path, image)
                    shutil.copy(source_path, destination_path)



def merge_val_into_train(train_dir, val_dir):
    """
    Merge the validation dataset into the training dataset.

    Args:
        train_dir (str): Path to the training data directory.
        val_dir (str): Path to the validation data directory.

    Returns:
        None

    This function moves all data from the validation directory (val_dir) into the training directory (train_dir).
    After calling this function, the validation directory will be empty, and all data will be in the training directory.

    Example:
    >>> train_dir = 'data/train'
    >>> val_dir = 'data/validation'
    >>> merge_val_into_train(train_dir, val_dir)
    >>> # The validation data is now merged into the training data directory.
    """
    for class_folder in os.listdir(val_dir):
        class_path = os.path.join(val_dir, class_folder)
        if os.path.isdir(class_path):
            train_class_path = os.path.join(train_dir, class_folder)
            os.makedirs(train_class_path, exist_ok=True)

            class_images = os.listdir(class_path)

            for image in class_images:
                source_path = os.path.join(class_path, image)
                destination_path = os.path.join(train_class_path, image)
                shutil.move(source_path, destination_path)

    # Remove the now-empty validation directory
    shutil.rmtree(val_dir)

if __name__ == "__main__":
    data_dir = os.path.join("data/data_electronic", 'raw')
    train_dir = os.path.join("data/data_electronic", 'train')
    val_dir  = os.path.join("data/data_electronic", "validation")
    test_dir = os.path.join("data/data_electronic", 'test')

    split_data(data_dir, train_dir, test_dir)
