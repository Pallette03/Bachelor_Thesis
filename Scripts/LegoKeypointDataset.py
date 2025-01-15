import json
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import cv2
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F

class LegoKeypointDataset(Dataset):
    def __init__(self, annotations_folder, img_dir, image_size=(224,224), sigma=2, transform=None):
        combined_annotations = []
        for file in os.listdir(annotations_folder):
            if file.endswith(".json"):
                with open(os.path.join(annotations_folder, file), 'r') as f:
                    annotations = json.load(f)
                    combined_annotations.append(annotations)
        self.annotations = combined_annotations
        self.img_dir = img_dir
        self.image_size = image_size
        self.sigma = sigma
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def gaussian_2d(self, shape, center, sigma):
        """Generate a 2D Gaussian heatmap."""
        y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        dist_sq = (x - center[0]) ** 2 + (y - center[1]) ** 2
        heatmap = np.exp(-dist_sq / (2 * sigma ** 2))
        return heatmap

    def denormalize_keypoints(self, keypoints, image_width, image_height):
        denormalized_keypoints = []
        for corner_vector in keypoints:
            x = corner_vector[0] * image_width
            y = corner_vector[1] * image_height
            denormalized_keypoints.append([x, y])
        return denormalized_keypoints

    def reduce_dataset_size(self, size):
        self.annotations = self.annotations[:size]

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        camera_matrix = np.array(annotation["camera_matrix"])
        img_path = os.path.join(self.img_dir, annotation["image_id"]) + ".png"
        image = Image.open(img_path).convert("RGB")
        
        if self.image_size is not None:
            image = image.resize(self.image_size)
        
        normalized_corners = []
        for brick in annotation["annotations"]:
            for corner_name, value in brick["normal_pixel_coordinates"].items():
                normalized_corners.append(value)
            
        normalized_corners = np.array(normalized_corners)
        # Denormalize corner coordinates
        img_width, img_height = self.image_size
        denormalized_corners = self.denormalize_keypoints(normalized_corners, img_width, img_height)

        # Generate heatmaps, offsets, and masks
        heatmap = np.zeros((img_height, img_width), dtype=np.float32)
        offset = np.zeros((2, img_height, img_width), dtype=np.float32)
        mask = np.zeros((img_height, img_width), dtype=np.float32)

        radius = 30
        
        for corner in denormalized_corners:
            x, y = int(corner[0]), int(corner[1])

            # Skip invalid or out-of-bound corners
            if x < 0 or x >= img_width or y < 0 or y >= img_height:
                print(f"Invalid corner: {corner}")
                continue

            # Add Gaussian blob to the heatmap
            heatmap += self.gaussian_2d((img_height, img_width), (x, y), self.sigma)

            
            for i in range(max(0, y - radius), min(img_height, y + radius + 1)):
                for j in range(max(0, x - radius), min(img_width, x + radius + 1)):
                    dx = corner[0] - j  # x-offset
                    dy = corner[1] - i  # y-offset

                    # Only update offset and mask if this pixel is within the radius
                    if (dx**2 + dy**2) <= radius**2:  # Check if within the radius
                        offset[0, i, j] = dx
                        offset[1, i, j] = dy
                        mask[i, j] = 1.0  # Mark as valid pixel

        # Normalize heatmap to [0, 1]
        heatmap = np.clip(heatmap, 0, 1)

        # Apply optional transformations to the image
        if self.transform:
            image = self.transform(image)

        # Convert to tensors
        heatmap = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        offset = torch.tensor(offset, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return {"image": image, "heatmaps": heatmap, "offsets": offset, "mask": mask}


# image_dir = 'C:/Users/paulb/Documents/TUDresden/Bachelor/datasets/object_detection/images/rgb'
# annotation_dir = 'C:/Users/paulb/Documents/TUDresden/Bachelor/datasets/object_detection/annotations'
# # Get the first element from the dataset
# dataset = LegoKeypointDataset(annotation_dir, image_dir, image_size=(224, 224), sigma=0.5)
# sample = dataset[1]
# #show the image
# image = sample["image"]
# image.show()
# #show the heatmap
# heatmap = sample["heatmaps"]
# heatmap = heatmap.squeeze(0).numpy()
# plt.imshow(heatmap, cmap="hot")
# plt.axis("off")
# plt.title("Heatmap")
# plt.show()
# #show the offsets
# offsets = sample["offsets"]
# offsets = offsets.squeeze(0).numpy()
# plt.imshow(offsets[0], cmap="hot")
# plt.axis("off")
# plt.title("Offset X")
# plt.show()
# plt.imshow(offsets[1], cmap="hot")
# plt.axis("off")
# plt.title("Offset Y")
# plt.show()
# #show the mask
# mask = sample["mask"]
# mask = mask.squeeze(0).numpy()
# plt.imshow(mask, cmap="gray")
# plt.axis("off")
# plt.title("Mask")
# plt.show()

