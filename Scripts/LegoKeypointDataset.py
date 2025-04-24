from concurrent.futures import ThreadPoolExecutor
import json
import time
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import os
import numpy as np
import torchvision.transforms as transforms

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


# def collate_fn(batch):
#     images = [item["image"] for item in batch]
#     corners_list = [item["norm_corners"] for item in batch]
#     max_corner_amount = max([norm_corners.shape[0] for norm_corners in corners_list])

#     # Pad the corners
#     for i in range(len(corners_list)):
#         corners = corners_list[i]
        
#         pad_amount = max_corner_amount - corners.shape[0]
#         pad = np.zeros((pad_amount, 2))
        
#         if corners.shape[0] == 0:
#             corners_list[i] = pad
#         else:
#             corners_list[i] = np.concatenate((corners, pad), axis=0)
        
#     images = torch.stack(images)
#     corners_list = torch.stack([torch.tensor(corners, dtype=torch.float32) for corners in corners_list])

#     return {"image": images, "norm_corners": corners_list}

# image_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'with_clutter', 'train', 'images', 'rgb')
# annotation_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'with_clutter', 'train', 'annotations')

# # # Get the first element from the dataset
# # start_time = time.time()

# transform_1 = transforms.Compose([
#             transforms.ToTensor()
#             #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

# dataset = LegoKeypointDataset(annotation_dir, image_dir, transform=transform_1, num_workers=8)
# train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=collate_fn)


# # Load the entire dataset once
# for batch in train_dataloader:
#     images = batch["image"]
#     corners = batch["norm_corners"]

    
    
