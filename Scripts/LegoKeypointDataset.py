from concurrent.futures import ThreadPoolExecutor
import json
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class LegoKeypointDataset(Dataset):
    def __init__(self, annotations_folder, img_dir, transform=None, num_workers=4):
        
        
        def load_annotation(file):
            with open(os.path.join(annotations_folder, file), 'r') as f:
                return json.load(f)
        
        combined_annotations = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            annotation_files = [file for file in os.listdir(annotations_folder) if file.endswith(".json")]
            combined_annotations = list(executor.map(load_annotation, annotation_files))
            
        self.transform = transform
        self.annotations = combined_annotations
        self.img_dir = img_dir

    def __len__(self):
        return len(self.annotations)
    

    def denormalize_keypoints(self, keypoints, image_width, image_height):
        denormalized_keypoints = []
        for corner_vector in keypoints:
            x = corner_vector[0] * image_width
            y = corner_vector[1] * image_height
            denormalized_keypoints.append([x, y])
        return denormalized_keypoints

    def reduce_dataset_size(self, size):
        self.annotations = self.annotations[:size]

    def process_annotation(self, annotation):
            normalized_corners = []
            for brick in annotation["annotations"]:
                for corner_name, data in brick["normal_pixel_coordinates"].items():
                    if data[1]:
                        normalized_corners.append(data[0])
            
            normalized_corners = np.array(normalized_corners)

            return normalized_corners
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.img_dir, annotation["image_id"]) + ".png"
        image = Image.open(img_path).convert("RGB")
            
        if self.transform:
            image = self.transform(image)
            
        normalized_corners = self.process_annotation(annotation)

        return {"image": image, "norm_corners": normalized_corners, "image_path": img_path}
