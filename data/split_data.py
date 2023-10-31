import os
import shutil
import random


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


# Split data into train and test
def split_data_into_train_test(data_dir, train_dir, test_dir, split_ratio=0.8, random_state=42, overwrite_dirs=True):
    """
    Split data into train and test sets.

    Args:
        data_dir (str): Path to the data directory.
        train_dir (str): Path to the training data directory.
        test_dir (str): Path to the testing data directory.
        split_ratio (float, optional): The ratio of data to be in the training set. Defaults to 0.8.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        overwrite_dirs (bool, optional): Whether to overwrite existing directories. Defaults to True.

    Returns:
        None
    """
    random.seed(random_state)

    # Create or overwrite train and test directories
    create_directory(train_dir, overwrite=overwrite_dirs)
    create_directory(test_dir, overwrite=overwrite_dirs)

    # Split data for each class
    for class_folder in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_folder)
        if os.path.isdir(class_path):
            train_class_path = os.path.join(train_dir, class_folder)
            test_class_path = os.path.join(test_dir, class_folder)
            os.makedirs(train_class_path, exist_ok=True)
            os.makedirs(test_class_path, exist_ok=True)

            class_images = os.listdir(class_path)
            num_images = len(class_images)
            num_train = int(num_images * split_ratio)
            train_images = random.sample(class_images, num_train)

            for image in class_images:
                source_path = os.path.join(class_path, image)
                if image in train_images:
                    destination_path = os.path.join(train_class_path, image)
                else:
                    destination_path = os.path.join(test_class_path, image)
                shutil.copy(source_path, destination_path)


if __name__ == "__main__":
    data_dir = os.path.join("data/data_resistors", 'resistors')
    train_dir = os.path.join("data/data_resistors", 'train')
    test_dir = os.path.join("data/data_resistors", 'test')

    # Create or overwrite train and test directories
    split_data_into_train_test(data_dir, train_dir, test_dir, overwrite_dirs=True)
