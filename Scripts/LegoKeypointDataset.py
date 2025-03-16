from concurrent.futures import ThreadPoolExecutor
import json
import time
import torch
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
        # Use ThreadPoolExecutor with 8 workers
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


            # # Denormalize corner coordinates
            # img_width, img_height = self.image_size
            # denormalized_corners = self.denormalize_keypoints(normalized_corners, img_width, img_height)


            # for corner in denormalized_corners:
            #     x, y = int(corner[0]), int(corner[1])

            #     # Skip invalid or out-of-bound corners
            #     if x < 0 or x >= img_width or y < 0 or y >= img_height:
            #         print(f"Invalid corner: {corner}")
            #         continue

            return normalized_corners
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.img_dir, annotation["image_id"]) + ".png"
        image = Image.open(img_path).convert("RGB")
            
        # Apply optional transformations to the image
        if self.transform:
            image = self.transform(image)
            
        normalized_corners = self.process_annotation(annotation)

        return {"image": image, "norm_corners": normalized_corners, "image_path": img_path}

# #image_dir = os.path.join(os.path.dirname(__file__), 'validate', 'images')
# #annotation_dir = os.path.join(os.path.dirname(__file__), 'validate', 'annotations')
# image_dir = "C:/Users/paulb/Documents/TUDresden/Bachelor/datasets/cropped_objects/validate/images"
# annotation_dir = "C:/Users/paulb/Documents/TUDresden/Bachelor/datasets/cropped_objects/validate/annotations"
# # Get the first element from the dataset
# start_time = time.time()
# dataset = LegoKeypointDataset(annotation_dir, image_dir, image_size=(600, 600), sigma=0.3)
# print(f"Time taken: {time.time() - start_time:.2f} seconds")
# print(f"Time per image: {(time.time() - start_time) / len(dataset):.4f} seconds")
# print(f"Dataset size: {len(dataset)}")

# start_time = time.time()
# sample = dataset[1]
# print(f"Time taken: {time.time() - start_time:.2f} seconds")
# print(sample["heatmaps"])
