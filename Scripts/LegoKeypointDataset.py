import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import cv2
import numpy as np
from torchvision import transforms

class LegoKeypointDataset(Dataset):
    def __init__(self, annotations_folder, img_dir, transform=None):
        combined_annotations = []
        for file in os.listdir(annotations_folder):
            if file.endswith(".json"):
                with open(os.path.join(annotations_folder, file), 'r') as f:
                    annotations = json.load(f)
                    combined_annotations.append(annotations)
        self.annotations = combined_annotations
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        camera_matrix = np.array(annotation["camera_matrix"])
        img_path = os.path.join(self.img_dir, annotation["image_id"]) + ".png"
        image = Image.open(img_path).convert("RGB")
        
        keypoints = []
        for brick in annotation["annotations"]:
            for value in brick["keypoints"].values():
                keypoints.append(value)
                
        heatmap = np.zeros((image.height, image.width), dtype=np.uint8)
        for brick in annotation["annotations"]:
            normalized_keypoints = brick["normal_pixel_coordinates"]
            denormalized_keypoints = self.denormalize_keypoints(normalized_keypoints, image.width, image.height)
            for corner_name, corner_vector in denormalized_keypoints.items():
                x, y = corner_vector
                x, y = int(x), int(y)
                heatmap[y, x] = 1
            
        keypoints = [kp[:2] for kp in keypoints]
        
        
        # Flatten keypoints for simplicity (x1, y1, x2, y2, ...)
        #keypoints = torch.tensor([kp for brick in keypoints for kp in brick], dtype=torch.float32)
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        
        
        if self.transform:
            image = self.transform(image)
            heatmap = transforms.ToTensor()(heatmap).float()
        
        # Set the max value of the heatmap to 1
        heatmap = heatmap / heatmap.max()
        
        return image, heatmap
    
    def denormalize_keypoints(self, keypoints, image_width, image_height):
        denormalized_keypoints = {}
        for corner_name, corner_vector in keypoints.items():
            x = corner_vector[0] * image_width
            y = corner_vector[1] * image_height
            denormalized_keypoints[corner_name] = [x, y]
        return denormalized_keypoints
