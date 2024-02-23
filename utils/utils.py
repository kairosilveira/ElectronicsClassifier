import os

def get_class_names(main_directory):
    """
    Get the class names from a directory containing class folders.

    Args:
    - main_directory (str): Path to the main directory.

    Returns:
    - List of class names.
    """
    classes = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]
    return classes