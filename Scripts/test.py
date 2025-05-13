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
model_path_top = os.path.join(os.path.dirname(__file__), os.pardir, 'output', '139_UNet_gaussian_clutter_lateral_top.pth')
model_path_bottom = os.path.join(os.path.dirname(__file__), os.pardir, 'output', '138_UNet_gaussian_clutter_lateral_bottom.pth')
external_img_path = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'external_images')
results_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'real_world_results_2_models_2')

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
name_suffix = "hourglass_mixed_gaussian_clutter_external"
threshold = 0.25
num_channels = 3

# Load the model
model_top = UNet(n_channels=3, n_classes=1)
model_bottom = UNet(n_channels=3, n_classes=1)
#model = KeyNet(num_filters=8, num_levels=8, kernel_size=5, in_channels=num_channels)
#model = PoseNet(nstack=4, inp_dim=512, oup_dim=1, bn=False, increase=0, input_image_size=global_image_size[0])
#model = SimpleModel(in_channels=num_channels, out_channels=1)

model_top = torch.nn.DataParallel(model_top)
model_bottom = torch.nn.DataParallel(model_bottom)
model_top.load_state_dict(torch.load(model_path_top)) 
model_bottom.load_state_dict(torch.load(model_path_bottom))
model_top.eval()
model_bottom.eval()

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
    pred_heatmap_top = model_top(model_input)
    pred_heatmap_bottom = model_bottom(model_input)
    end_time = time.time()
    print(f"Prediction Time: {end_time - start_time:.4f} seconds")
    pred_heatmap_top = pred_heatmap_top.squeeze(0).squeeze(0).detach().cpu().numpy()
    pred_heatmap_bottom = pred_heatmap_bottom.squeeze(0).squeeze(0).detach().cpu().numpy()


    # Convert to probability using sigmoid
    prob_heatmap_top = torch.sigmoid(torch.tensor(pred_heatmap_top)).detach().cpu().numpy()
    prob_heatmap_bottom = torch.sigmoid(torch.tensor(pred_heatmap_bottom)).detach().cpu().numpy()

    # Normalize the heatmap to the range [0, 1]
    prob_heatmap_top = (prob_heatmap_top - np.min(prob_heatmap_top)) / (np.max(prob_heatmap_top) - np.min(prob_heatmap_top))
    prob_heatmap_bottom = (prob_heatmap_bottom - np.min(prob_heatmap_bottom)) / (np.max(prob_heatmap_bottom) - np.min(prob_heatmap_bottom))

    # Save a heatmap where every pixel gets a color from black to red according to its value using OpenCV
    heatmap_top = cv2.applyColorMap((prob_heatmap_top * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_bottom = cv2.applyColorMap((prob_heatmap_bottom * 255).astype(np.uint8), cv2.COLORMAP_JET)
    filename_top = str(i) + "predicted_heatmap_blue_to_red_" + name_suffix + "_top.png"
    filename_bottom = str(i) + "predicted_heatmap_blue_to_red_" + name_suffix + "_bottom.png"
    file_path_top = os.path.join(results_dir, filename_top)
    file_path_bottom = os.path.join(results_dir, filename_bottom)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    cv2.imwrite(file_path_top, heatmap_top)
    cv2.imwrite(file_path_bottom, heatmap_bottom)


    


    pred_heatmap_top = convert_pred_to_heatmap(prob_heatmap_top, threshold=threshold)
    pred_heatmap_bottom = convert_pred_to_heatmap(prob_heatmap_bottom, threshold=threshold)

    # Detect blobs in the heatmap and convert them to keypoints
    binary_map_top = (pred_heatmap_top > threshold).astype(np.uint8)
    binary_map_bottom = (pred_heatmap_bottom > threshold).astype(np.uint8)
    num_labels_top, labels_top, stats_top, centroids_top = cv2.connectedComponentsWithStats(binary_map_top)
    num_labels_bottom, labels_bottom, stats_bottom, centroids_bottom = cv2.connectedComponentsWithStats(binary_map_bottom)

    # Extract keypoint coordinates (ignore the background component at index 0)
    keypoints_top = centroids_top[1:]
    keypoints_bottom = centroids_bottom[1:]

    # Print keypoints
    print("Detected Top Keypoints amount:", len(keypoints_top))
    print("Detected Bottom Keypoints amount:", len(keypoints_bottom))

    # Apply max filter to find local peaks
    local_max_top = scipy.ndimage.maximum_filter(prob_heatmap_top, size=5)  # Adjust size
    local_max_bottom = scipy.ndimage.maximum_filter(prob_heatmap_bottom, size=5)  # Adjust size
    peaks_top = (prob_heatmap_top == local_max_top) & (prob_heatmap_top > threshold)
    peaks_bottom = (prob_heatmap_bottom == local_max_bottom) & (prob_heatmap_bottom > threshold)

    # Get peak coordinates
    y_coords_top, x_coords_top = np.where(peaks_top)
    y_coords_bottom, x_coords_bottom = np.where(peaks_bottom)
    keypoints_top = np.column_stack((x_coords_top, y_coords_top))
    keypoints_bottom = np.column_stack((x_coords_bottom, y_coords_bottom))

    print("Filtered Top Keypoints amount:", len(keypoints_top))
    print("Filtered Bottom Keypoints amount:", len(keypoints_bottom))

    #Save input image with predicted keypoints using opencv
    cv_image = (input_image * 255).astype(np.uint8)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite("model_input_image.png", cv_image)

    for x, y in keypoints_top:
        cv2.circle(cv_image, (int(x), int(y)), radius=1, color=(0, 255, 255), thickness=3)
        
    for x, y in keypoints_bottom:
        cv2.circle(cv_image, (int(x), int(y)), radius=1, color=(255, 0, 255), thickness=3)
    filename = str(i) + "input_image_with_predicted_keypoints_" + name_suffix + ".png"
    file_path = os.path.join(results_dir, filename)
    cv2.imwrite(file_path, cv_image)

    i = i + 1

