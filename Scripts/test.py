import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch
import torchvision
from LegoKeypointDataset import LegoKeypointDataset
from KeypointDetector import UNet
from models.simpleModel.simple_model import SimpleModel
from PIL import Image

from models.KeyNet.keynet import KeyNet

annotations_folder = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'cropped_objects', 'test', 'annotations')
img_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'cropped_objects', 'test', 'images')
model_path = os.path.join(os.path.dirname(__file__), os.pardir, 'output', 'dynamic_corner_detector_epoch.pth')

global_image_size = (1000, 1000)

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


use_external_image = False
threshold = 0.4
num_channels = 3

# Load the model
#model = UNet(n_channels=3, n_classes=1)
model = KeyNet(num_filters=8, num_levels=5, kernel_size=5, in_channels=num_channels)
#model = SimpleModel(in_channels=num_channels, out_channels=1)
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


if not use_external_image:
    # Load the dataset
    dataset = LegoKeypointDataset(annotations_folder, img_dir, transform=transforms)

    dataset_length = len(dataset)
    rand_index = np.random.randint(0, dataset_length)
    sample = dataset[rand_index]
    model_input = sample['image'].unsqueeze(0)
    
    input_image = sample['image'].permute(1, 2, 0).cpu().numpy()
    
    # Predict the keypoints
    pred_heatmap = model(model_input)
    pred_heatmap = pred_heatmap.squeeze(0).squeeze(0).detach().cpu().numpy()


    # Convert to probability using sigmoid
    prob_heatmap = torch.sigmoid(torch.tensor(pred_heatmap)).detach().cpu().numpy()

    # Normalize the heatmap to the range [0, 1]
    prob_heatmap = (prob_heatmap - np.min(prob_heatmap)) / (np.max(prob_heatmap) - np.min(prob_heatmap))

    # Save a heatmap where every pixel gets a color from black to red according to its value
    plt.imshow(prob_heatmap, cmap='Reds', interpolation='nearest')
    plt.title("Predicted Heatmap (Black to Red)")
    plt.colorbar()
    plt.savefig("predicted_heatmap_black_to_red.png")
    plt.close()


    


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

    plt.imshow(input_image, cmap="jet")
    for x, y in keypoints:
        circle = plt.Circle((x, y), radius=3, color="cyan", fill=False, linewidth=1.5)
        plt.gca().add_patch(circle)
    plt.title("Predicted Heatmap with Detected Keypoints")
    plt.colorbar()
    plt.savefig("predicted_heatmap_after_scipy.png")
    plt.close()


    #target_heatmap = keypoints_to_heatmap(sample['norm_corners'], image_size=global_image_size[0]).squeeze(0)



    # Plot the heatmaps


    plt.imshow(pred_heatmap, cmap='hot', interpolation='nearest')
    plt.title("Predicted Heatmap")
    plt.savefig("predicted_heatmap.png")
    

    plt.figure(figsize=(10, 10))
    plt.imshow(input_image)

    plt.title("Input Image with Predicted Keypoints")
    plt.savefig("model_input_image.png")
    plt.close()

else:
    # Load an external image
    img_path = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'external_images', 'lego_brick_3.jpg')
    image = Image.open(img_path).convert("RGB")
    image = image.resize(global_image_size)

    # Apply optional transformations to the image
    if transforms:
        image = transforms(image)
    
    model_input = image.unsqueeze(0)

    # Predict the keypoints
    pred_heatmap = model(model_input)
    pred_heatmap = pred_heatmap.squeeze(0).squeeze(0).detach().cpu().numpy()

    plt.imshow(pred_heatmap, cmap='Reds', interpolation='nearest')
    plt.title("Predicted Heatmap (Black to Red)")
    plt.colorbar()
    plt.savefig("predicted_heatmap_black_to_red.png")
    plt.close()


    pred_heatmap = convert_pred_to_heatmap(pred_heatmap)

    # Plot the heatmaps
    plt.imshow(pred_heatmap, cmap='hot', interpolation='nearest')
    plt.title("Predicted Heatmap")
    plt.savefig("predicted_heatmap_external.png")
    plt.close()

    # Overlay the predicted keypoints onto the input image
    pred_keypoints = np.argwhere(pred_heatmap == 1)
    input_image = image.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(input_image)

    for (y, x) in pred_keypoints:
        plt.scatter(x, y, c='yellow', s=1)

    plt.title("Input Image with Predicted Keypoints")
    plt.savefig("input_image_with_predicted_keypoints.png")
    plt.close()