import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from LegoKeypointDataset import LegoKeypointDataset
from KeypointDetector import UNet
from PIL import Image

annotations_folder = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'cropped_objects', 'validate', 'annotations')
img_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'cropped_objects', 'validate', 'images')
model_path = os.path.join(os.path.dirname(__file__), os.pardir, 'output', 'dynamic_corner_detector_epoch.pth')

global_image_size = (650, 650)

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
    #Output max value of the heatmap
    min_value = np.min(pred_heatmap)
    max_value = np.max(pred_heatmap)
    middle_value = -(abs(min_value) - abs(max_value)) / 2
    middle_value = middle_value

    print(f"Min value: {min_value}, Max value: {max_value}, Middle value: {middle_value}")
    print(f"Pixel Number above middle: {np.sum(pred_heatmap > middle_value)}, Pixel Number below middle: {np.sum(pred_heatmap < middle_value)}")
    
    # for every value in the heatmap, if it is greater than the threshold, set it to 1, else 0
    pred_heatmap[pred_heatmap > middle_value] = 1
    pred_heatmap[pred_heatmap <= middle_value] = 0
    return pred_heatmap

# Load the model
model = UNet(n_channels=3, n_classes=1)
model.load_state_dict(torch.load(model_path))
model.eval()

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

use_external_image = False

if not use_external_image:
    # Load the dataset
    dataset = LegoKeypointDataset(annotations_folder, img_dir, image_size=global_image_size, transform=transforms)

    dataset_length = len(dataset)
    rand_index = np.random.randint(0, dataset_length)
    sample = dataset[rand_index]
    model_input = sample['image'].unsqueeze(0)
    
    
    # Predict the keypoints
    pred_heatmap = model(model_input)
    pred_heatmap = pred_heatmap.squeeze(0).squeeze(0).detach().cpu().numpy()


    # Save a heatmap where every pixel gets a color from black to red according to its value
    plt.imshow(pred_heatmap, cmap='Reds', interpolation='nearest')
    plt.title("Predicted Heatmap (Black to Red)")
    plt.colorbar()
    plt.savefig("predicted_heatmap_black_to_red.png")
    plt.close()



    pred_heatmap = convert_pred_to_heatmap(pred_heatmap)


    target_heatmap = keypoints_to_heatmap(sample['norm_corners'], image_size=global_image_size[0]).squeeze(0)



    # Plot the heatmaps


    plt.imshow(pred_heatmap, cmap='hot', interpolation='nearest')
    plt.title("Predicted Heatmap")
    plt.savefig("predicted_heatmap.png")

    plt.imshow(target_heatmap, cmap='hot', interpolation='nearest')
    plt.title("Target Heatmap")
    plt.savefig("target_heatmap.png")



    # Overlay the predicted keypoints onto the input image
    pred_keypoints = np.argwhere(pred_heatmap == 1)
    input_image = sample['image'].permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(input_image)

    for (y, x) in pred_keypoints:
        plt.scatter(x, y, c='yellow', s=1)

    plt.title("Input Image with Predicted Keypoints")
    plt.savefig("input_image_with_predicted_keypoints.png")
    plt.close()


    # Overlay the target keypoints onto the input image
    target_keypoints = np.argwhere(target_heatmap == 1)
    tageet_image = sample['image'].permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(tageet_image)

    for (y, x) in target_keypoints:
        plt.scatter(x, y, c='yellow', s=1)

    plt.title("Input Image with Target Keypoints")
    plt.savefig("input_image_with_target_keypoints.png")
    plt.close()
else:
    # Load an external image
    img_path = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'external_images', 'lego_brick.jpg')
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