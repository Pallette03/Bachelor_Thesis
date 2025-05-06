import os
import random
import shutil


def split_dataset(dataset_path, train_output_path, validate_output_path, split_ratio=0.8, with_depth=False):
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
    
        
    if os.path.exists(validate_output_path):
        print(f"Removing existing validation set at {validate_output_path}")
        shutil.rmtree(os.path.join(validate_output_path, 'images'))
        shutil.rmtree(os.path.join(validate_output_path, 'annotations'))
    
    os.makedirs(os.path.join(train_output_path, 'images'))
    os.makedirs(os.path.join(train_output_path, 'annotations'))
    os.makedirs(os.path.join(train_output_path, 'images', 'rgb'))
    if with_depth:
        os.makedirs(os.path.join(train_output_path, 'images', 'depth'))   
    
    os.makedirs(os.path.join(validate_output_path, 'images'))
    os.makedirs(os.path.join(validate_output_path, 'annotations'))
    os.makedirs(os.path.join(validate_output_path, 'images', 'rgb'))
    if with_depth:
        os.makedirs(os.path.join(validate_output_path, 'images', 'depth'))
    
    print(f"Copying files to training and validation sets...")
    progress = 0
    for filename in train_set:
        annotation_name = filename.replace('.png', '.json')
        shutil.copy(os.path.join(dataset_path, 'images', 'rgb', filename), os.path.join(train_output_path, 'images', 'rgb', filename))
        if with_depth:
            depth_name = filename.replace('.png', '_depth.png')
            shutil.copy(os.path.join(dataset_path, 'images', 'depth', depth_name), os.path.join(train_output_path, 'images', 'depth', depth_name))
        shutil.copy(os.path.join(dataset_path, 'annotations', annotation_name), os.path.join(train_output_path, 'annotations', annotation_name))
        progress += 1
        if progress % (len(train_set) % 20) == 0:
            print(f"Progress: {progress}/{len(train_set)}")

    progress = 0
    for filename in validate_set:
        annotation_name = filename.replace('.png', '.json')
        shutil.copy(os.path.join(dataset_path, 'images', 'rgb', filename), os.path.join(validate_output_path, 'images', 'rgb', filename))
        if with_depth:
            depth_name = filename.replace('.png', '_depth.png')
            shutil.copy(os.path.join(dataset_path, 'images', 'depth', depth_name), os.path.join(validate_output_path, 'images', 'depth', depth_name))
        shutil.copy(os.path.join(dataset_path, 'annotations', annotation_name), os.path.join(validate_output_path, 'annotations', annotation_name))
        progress += 1
        if progress % (len(validate_set) % 20) == 0:
            print(f"Progress: {progress}/{len(validate_set)}")
        
        
    print(f"Dataset split into training and validation sets. Training set: {len(train_set)} samples, Validation set: {len(validate_set)} samples.")
    
dataset_path = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'salt_and_pepper')
train_output_path = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'salt_and_pepper', 'train')
validate_output_path = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'salt_and_pepper', 'validate')

split_dataset(dataset_path, train_output_path, validate_output_path, split_ratio=0.9, with_depth=True)
