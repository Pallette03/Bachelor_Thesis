import os
import json
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch
import torchvision
from LegoKeypointDataset import LegoKeypointDataset
from KeypointDetector import UNet
from models.hourglass.posenet import PoseNet
from models.simpleModel.simple_model import SimpleModel
from PIL import Image

from models.KeyNet.keynet import KeyNet

annotations_folder = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'placeholder', 'test', 'annotations')
img_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'placeholder', 'test', 'images', 'rgb')
model_path = os.path.join(os.path.dirname(__file__), os.pardir, 'output', '137_UNet_gaussian_clutter_lateral.pth')
external_img_path = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'external_images')
results_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'real_world_results_lateral')

global_image_size = (800, 800)

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

def keypoints_to_heatmap(keypoints, image_size=500, sigma=1.0):
    """
    Converts keypoints into a lower-resolution heatmap (e.g., 128×128) for training.
    The heatmap will be upsampled to match the input image size (500×500).
    """
    target_heatmap = torch.zeros((1, image_size, image_size))


    for (x, y) in keypoints:
        x = x * image_size
        y = y * image_size
        x, y = int(x), int(y)  # Scale keypoints

        target_heatmap[0, y, x] = 1

        # if 0 <= x < heatmap_size and 0 <= y < heatmap_size:
        #     for i in range(-2, 3):  # Small 5×5 Gaussian
        #         for j in range(-2, 3):
        #             xi, yj = x + i, y + j
        #             if 0 <= xi < heatmap_size and 0 <= yj < heatmap_size:
        #                 exponent = torch.tensor(-((i**2 + j**2) / (2 * sigma**2)), dtype=torch.float32)
        #                 target_heatmap[0, yj, xi] += torch.exp(exponent)

    return target_heatmap.clamp(0, 1).detach().cpu().numpy()  # Keep values in [0,1]

def convert_pred_to_heatmap(pred_heatmap, threshold=0.5):

    print(f"Pixel Number above threshold: {np.sum(pred_heatmap > threshold)}, Pixel Number below threshold: {np.sum(pred_heatmap < threshold)}")
    
    # for every value in the heatmap, if it is greater than the threshold, set it to 1, else 0
    #pred_heatmap[pred_heatmap > threshold] = 1
    pred_heatmap[pred_heatmap <= threshold] = 0
    return pred_heatmap


use_external_image = True
name_suffix = "unet_gaussian_clutter_lateral"
threshold = 0.25
num_channels = 3

# Load the model
model = UNet(n_channels=3, n_classes=2)
#model = KeyNet(num_filters=8, num_levels=8, kernel_size=5, in_channels=num_channels)
#model = PoseNet(nstack=4, inp_dim=512, oup_dim=1, bn=False, increase=0, input_image_size=global_image_size[0])
#model = SimpleModel(in_channels=num_channels, out_channels=1)

model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(model_path)) 
model.eval()

if num_channels == 1:
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(global_image_size),
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor()
    ])
else:
    transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(global_image_size),
    torchvision.transforms.ToTensor()
    #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


i = 0
if not use_external_image:
    
    # Load the dataset
    dataset = LegoKeypointDataset(annotations_folder, img_dir, transform=transforms)
    dataset_length = len(dataset)
else:
    external_list = os.listdir(external_img_path)
while True:
    if not use_external_image:
        
        rand_index = np.random.randint(0, dataset_length)

        sample = dataset[i]

        print(i)
        
    else:
        # Load an external image
        img_path = os.path.join(external_img_path, external_list.pop())
        image = Image.open(img_path).convert("RGB")
        image = transforms(image)
        sample = {'image': image, 'norm_corners': None}

    model_input = sample['image'].unsqueeze(0)

    input_image = sample['image'].permute(1, 2, 0).cpu().numpy()
    
    # Predict the keypoints
    start_time = time.time()
    pred_heatmaps = model(model_input)
    
    print(pred_heatmaps.shape)
    
    pred_heatmap_0 = pred_heatmaps[:, 0, :, :]
    pred_heatmap_1 = pred_heatmaps[:, 1, :, :]
    
    end_time = time.time()
    print(f"Prediction Time: {end_time - start_time:.4f} seconds")
    pred_heatmap_0 = pred_heatmap_0.squeeze(0).squeeze(0).detach().cpu().numpy()
    pred_heatmap_1 = pred_heatmap_1.squeeze(0).squeeze(0).detach().cpu().numpy()


    # Convert to probability using sigmoid
    prob_heatmap_0 = torch.sigmoid(torch.tensor(pred_heatmap_0)).detach().cpu().numpy()
    prob_heatmap_1 = torch.sigmoid(torch.tensor(pred_heatmap_1)).detach().cpu().numpy()

    # Normalize the heatmap to the range [0, 1]
    prob_heatmap_0 = (prob_heatmap_0 - np.min(prob_heatmap_0)) / (np.max(prob_heatmap_0) - np.min(prob_heatmap_0))
    prob_heatmap_1 = (prob_heatmap_1 - np.min(prob_heatmap_1)) / (np.max(prob_heatmap_1) - np.min(prob_heatmap_1))

    # Save a heatmap where every pixel gets a color from black to red according to its value using OpenCV
    heatmap_0 = cv2.applyColorMap((prob_heatmap_0 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_1 = cv2.applyColorMap((prob_heatmap_1 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    filename_0 = str(i) + "predicted_heatmap_blue_to_red_" + name_suffix + "_0.png"
    filename_1 = str(i) + "predicted_heatmap_blue_to_red_" + name_suffix + "_1.png"
    file_path_0 = os.path.join(results_dir, filename_0)
    file_path_1 = os.path.join(results_dir, filename_1)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    cv2.imwrite(file_path_0, heatmap_0)
    cv2.imwrite(file_path_1, heatmap_1)

    pred_heatmap_0 = convert_pred_to_heatmap(prob_heatmap_0, threshold=threshold)
    pred_heatmap_1 = convert_pred_to_heatmap(prob_heatmap_1, threshold=threshold)

    # Detect blobs in the heatmap and convert them to keypoints
    binary_map_0 = (pred_heatmap_0 > threshold).astype(np.uint8)
    binary_map_1 = (pred_heatmap_1 > threshold).astype(np.uint8)
    num_labels_0, labels_0, stats_0, centroids_0 = cv2.connectedComponentsWithStats(binary_map_0)
    num_labels_1, labels_1, stats_1, centroids_1 = cv2.connectedComponentsWithStats(binary_map_1)

    # Extract keypoint coordinates (ignore the background component at index 0)
    keypoints_0 = centroids_0[1:]
    keypoints_1 = centroids_1[1:]

    # Print keypoints
    print("Detected Bottom Keypoints amount:", len(keypoints_0))
    print("Detected Top Keypoints amount:", len(keypoints_1))

    # Apply max filter to find local peaks
    local_max_0 = scipy.ndimage.maximum_filter(prob_heatmap_0, size=5)  # Adjust size
    peaks_0 = (prob_heatmap_0 == local_max_0) & (prob_heatmap_0 > threshold)
    local_max_1 = scipy.ndimage.maximum_filter(prob_heatmap_1, size=5)  # Adjust size
    peaks_1 = (prob_heatmap_1 == local_max_1) & (prob_heatmap_1 > threshold)

    # Get peak coordinates
    y_coords_0, x_coords_0 = np.where(peaks_0)
    y_coords_1, x_coords_1 = np.where(peaks_1)
    keypoints_0 = np.column_stack((x_coords_0, y_coords_0))
    keypoints_1 = np.column_stack((x_coords_1, y_coords_1))

    print("Filtered Bottom Keypoints amount:", len(keypoints_0))
    print("Filtered Top Keypoints amount:", len(keypoints_1))

    #Save input image with predicted keypoints using opencv
    cv_image = (input_image * 255).astype(np.uint8)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite("model_input_image.png", cv_image)

    for x, y in keypoints_0:
        cv2.circle(cv_image, (int(x), int(y)), radius=1, color=(0, 255, 255), thickness=3)
        
    for x, y in keypoints_1:
        cv2.circle(cv_image, (int(x), int(y)), radius=1, color=(255, 0, 255), thickness=3)
    filename = str(i) + "input_image_with_predicted_keypoints_" + name_suffix + ".png"
    file_path = os.path.join(results_dir, filename)
    cv2.imwrite(file_path, cv_image)

    i = i + 1

