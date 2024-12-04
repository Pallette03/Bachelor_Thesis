import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

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
        img_path = os.path.join(self.img_dir, annotation["image_id"]) + ".png"
        image = Image.open(img_path).convert("RGB")
        
        keypoints = []
        for brick in annotation["annotations"]:
            keypoints.extend(brick["keypoints"])
            
        # Flatten keypoints for simplicity (x1, y1, x2, y2, ...)
        #keypoints = torch.tensor([kp for brick in keypoints for kp in brick], dtype=torch.float32)
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)

        return image, keypoints
