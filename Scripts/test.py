import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from LegoKeypointDataset import LegoKeypointDataset
from KeypointDetector import DynamicCornerDetector

annotations_folder = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'cropped_objects', 'validate', 'annotations')
img_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'cropped_objects', 'validate', 'images')
model_path = os.path.join(os.path.dirname(__file__), os.pardir, 'output', 'dynamic_corner_detector_epoch.pth')

def collate_fn(batch):
    images = [item["image"] for item in batch]
    corners_list = [item["norm_corners"] for item in batch]
    max_corner_amount = max([norm_corners.shape[0] for norm_corners in corners_list])

    # Pad the corners
    for i in range(len(corners_list)):
        corners = corners_list[i]
        pad_amount = max_corner_amount - corners.shape[0]
        pad = np.zeros((pad_amount, 2))
        corners_list[i] = np.concatenate((corners, pad), axis=0)
        


    images = torch.stack(images)
    corners_list = torch.stack([torch.tensor(corners, dtype=torch.float32) for corners in corners_list])

    return {"image": images, "norm_corners": corners_list}

def keypoints_to_heatmap(keypoints, heatmap_size=500, image_size=500, sigma=1.0):
    """
    Converts keypoints into a lower-resolution heatmap (e.g., 128×128) for training.
    The heatmap will be upsampled to match the input image size (500×500).
    """
    target_heatmap = torch.zeros((1, heatmap_size, heatmap_size))
    scale = heatmap_size / image_size  # Scale down factor (e.g., 128/500)

    for (x, y) in keypoints:
        x = x * image_size
        y = y * image_size
        x, y = int(x * scale), int(y * scale)  # Scale keypoints
        if 0 <= x < heatmap_size and 0 <= y < heatmap_size:
            for i in range(-2, 3):  # Small 5×5 Gaussian
                for j in range(-2, 3):
                    xi, yj = x + i, y + j
                    if 0 <= xi < heatmap_size and 0 <= yj < heatmap_size:
                        exponent = torch.tensor(-((i**2 + j**2) / (2 * sigma**2)), dtype=torch.float32)
                        target_heatmap[0, yj, xi] += torch.exp(exponent)

    return target_heatmap.clamp(0, 1).detach().cpu().numpy()  # Keep values in [0,1]

def convert_pred_to_heatmap(pred_heatmap, threshold=0.5):
    # for every value in the heatmap, if it is greater than the threshold, set it to 1, else 0
    pred_heatmap[pred_heatmap > threshold] = 1
    pred_heatmap[pred_heatmap <= threshold] = 0
    return pred_heatmap

# Load the model
model = DynamicCornerDetector()
model.load_state_dict(torch.load(model_path))
model.eval()

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

# Load the dataset
dataset = LegoKeypointDataset(annotations_folder, img_dir, image_size=(500, 500), transform=transforms)

dataset_length = len(dataset)
rand_index = np.random.randint(0, dataset_length)
sample = dataset[rand_index]
model_input = sample['image'].unsqueeze(0)

# Predict the keypoints
pred_heatmap = model(model_input)
pred_heatmap = pred_heatmap.squeeze(0).squeeze(0).detach().cpu().numpy()
pred_heatmap = convert_pred_to_heatmap(pred_heatmap)


target_heatmap = keypoints_to_heatmap(sample['norm_corners']).squeeze(0)

# Plot the heatmaps


plt.imshow(pred_heatmap, cmap='hot', interpolation='nearest')
plt.title("Predicted Heatmap")
plt.savefig("predicted_heatmap.png")

plt.imshow(target_heatmap, cmap='hot', interpolation='nearest')
plt.title("Target Heatmap")
plt.savefig("target_heatmap.png")



