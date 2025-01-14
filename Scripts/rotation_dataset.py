from torch.utils.data import Dataset
import torch
from PIL import Image
import os
import json


class RotationDataset(Dataset):
    def __init__(self, root_dir, bin_size=5, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.bin_size = bin_size
        self.images = {}
        self.annotations = []
        
        for brick_dir in os.listdir(root_dir):
            brick_dir_path = os.path.join(root_dir, brick_dir)
            for dir in os.listdir(brick_dir_path):
                dir_path = os.path.join(brick_dir_path, dir)
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    if file.endswith('.png'):
                        self.images[file] = file_path
                    elif file.endswith('.json'):
                        with open(file_path, mode='r') as json_file:
                            data = json.load(json_file)
                            self.annotations.extend(data)
                    

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_path = self.images[annotation['img_name']]
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        azimuth_bin, elevation_bin, optical_rotation_bin = self.split_into_bins(annotation['azimuth'], annotation['elevation'], annotation['optical_rotation'], self.bin_size)
        
        return image, torch.tensor([azimuth_bin, elevation_bin, optical_rotation_bin])
    
    def reduce_size(self, size):
        self.annotations = self.annotations[:size]
    
    def split_into_bins(self, azimuth, elevation, optical_rotation, bin_size=5):
        elevation = elevation + 90
        
        azimuth_bin = int(azimuth // bin_size)
        elevation_bin = int(elevation // bin_size)
        optical_rotation_bin = int(optical_rotation // bin_size)
        
        return azimuth_bin, elevation_bin, optical_rotation_bin
    