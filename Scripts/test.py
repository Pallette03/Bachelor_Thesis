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
model_path = os.path.join(os.path.dirname(__file__), os.pardir, 'output', '48_Hourglass_mixed_guassian_clutter_and_clutter.pth')
external_img_path = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'external_images')
results_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'real_world_results')

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
#model = UNet(n_channels=3, n_classes=1)
#model = KeyNet(num_filters=8, num_levels=8, kernel_size=5, in_channels=num_channels)
model = PoseNet(nstack=4, inp_dim=512, oup_dim=1, bn=False, increase=0, input_image_size=global_image_size[0])
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
    pred_heatmap = model(model_input)
    end_time = time.time()
    print(f"Prediction Time: {end_time - start_time:.4f} seconds")
    pred_heatmap = pred_heatmap.squeeze(0).squeeze(0).detach().cpu().numpy()


    # Convert to probability using sigmoid
    prob_heatmap = torch.sigmoid(torch.tensor(pred_heatmap)).detach().cpu().numpy()

    # Normalize the heatmap to the range [0, 1]
    prob_heatmap = (prob_heatmap - np.min(prob_heatmap)) / (np.max(prob_heatmap) - np.min(prob_heatmap))

    # Save a heatmap where every pixel gets a color from black to red according to its value using OpenCV
    heatmap = cv2.applyColorMap((prob_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    filename = str(i) + "predicted_heatmap_blue_to_red_" + name_suffix + ".png"
    file_path = os.path.join(results_dir, filename)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    cv2.imwrite(file_path, heatmap)


    


    pred_heatmap = convert_pred_to_heatmap(prob_heatmap, threshold=threshold)

    # Detect blobs in the heatmap and convert them to keypoints
    binary_map = (pred_heatmap > threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map)

    # Extract keypoint coordinates (ignore the background component at index 0)
    keypoints = centroids[1:]

    # Print keypoints
    print("Detected Keypoints amount:", len(keypoints))

    # Apply max filter to find local peaks
    local_max = scipy.ndimage.maximum_filter(prob_heatmap, size=5)  # Adjust size
    peaks = (prob_heatmap == local_max) & (prob_heatmap > threshold)

    # Get peak coordinates
    y_coords, x_coords = np.where(peaks)
    keypoints = np.column_stack((x_coords, y_coords))

    print("Filtered Keypoints amount:", len(keypoints))

    #Save input image with predicted keypoints using opencv
    cv_image = (input_image * 255).astype(np.uint8)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite("model_input_image.png", cv_image)

    for x, y in keypoints:
        cv2.circle(cv_image, (int(x), int(y)), radius=1, color=(0, 255, 255), thickness=3)
    filename = str(i) + "input_image_with_predicted_keypoints_" + name_suffix + ".png"
    file_path = os.path.join(results_dir, filename)
    cv2.imwrite(file_path, cv_image)


    #target_heatmap = keypoints_to_heatmap(sample['norm_corners'], image_size=global_image_size[0]).squeeze(0)



    # Plot the heatmaps


    plt.imshow(pred_heatmap, cmap='hot', interpolation='nearest')
    plt.title("Predicted Heatmap")
    plt.savefig("predicted_heatmap.png")
    plt.close()

    i = i + 1

