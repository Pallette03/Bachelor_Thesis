import os
import random
import shutil


def split_dataset(dataset_path, train_output_path, validate_output_path, split_ratio=0.8):
    """Split a dataset into training and validation sets.
    
    Args:
        dataset_path (str): The path to the dataset.
        train_output_path (str): The path to the training set.
        validate_output_path (str): The path to the validation set.
        split_ratio (float): The ratio of the training set size to the dataset size.
    """
    dataset = os.listdir(os.path.join(dataset_path, 'images', 'rgb'))
    random.shuffle(dataset)
    split_index = int(len(dataset) * split_ratio)
    train_set = dataset[:split_index]
    validate_set = dataset[split_index:]
    
    # Clear the output directories
    if os.path.exists(train_output_path):
        print(f"Removing existing training set at {train_output_path}")
        shutil.rmtree(os.path.join(train_output_path, 'images'))
        shutil.rmtree(os.path.join(train_output_path, 'annotations'))
        os.makedirs(os.path.join(train_output_path, 'images'))
        os.makedirs(os.path.join(train_output_path, 'annotations'))
        
    if os.path.exists(validate_output_path):
        print(f"Removing existing validation set at {validate_output_path}")
        shutil.rmtree(os.path.join(validate_output_path, 'images'))
        shutil.rmtree(os.path.join(validate_output_path, 'annotations'))
        os.makedirs(os.path.join(validate_output_path, 'images'))
        os.makedirs(os.path.join(validate_output_path, 'annotations'))
    
    print(f"Copying files to training and validation sets...")
    for filename in train_set:
        annotation_name = filename.replace('.png', '.json')
        shutil.copy(os.path.join(dataset_path, 'images', 'rgb', filename), os.path.join(train_output_path, 'images', filename))
        shutil.copy(os.path.join(dataset_path, 'annotations', annotation_name), os.path.join(train_output_path, 'annotations', annotation_name))
        
    for filename in validate_set:
        annotation_name = filename.replace('.png', '.json')
        shutil.copy(os.path.join(dataset_path, 'images', 'rgb', filename), os.path.join(validate_output_path, 'images', filename))
        shutil.copy(os.path.join(dataset_path, 'annotations', annotation_name), os.path.join(validate_output_path, 'annotations', annotation_name))
        
        
    print(f"Dataset split into training and validation sets. Training set: {len(train_set)} samples, Validation set: {len(validate_set)} samples.")
    
dataset_path = 'C:/Users/paulb/Documents/TUDresden/Bachelor/datasets/cropped_objects'
train_output_path = 'C:/Users/paulb/Documents/TUDresden/Bachelor/datasets/cropped_objects/train'
validate_output_path = 'C:/Users/paulb/Documents/TUDresden/Bachelor/datasets/cropped_objects/validate'

split_dataset(dataset_path, train_output_path, validate_output_path, split_ratio=0.8)
