from matplotlib import pyplot as plt
import torch
import numpy as np
import os
import json
from PIL import Image

annotations_folder = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'cropped_objects', 'annotations')
img_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'cropped_objects', 'images', 'rgb')

img_path = os.path.join(img_dir, '25022025-112914-89.png')


def denormalize_keypoints(keypoints, image_width, image_height):
        denormalized_keypoints = {}
        for corner_name, corner_data in keypoints.items():
            x = corner_data[0][0] * image_width
            y = corner_data[0][1] * image_height
            denormalized_keypoints[corner_name] = ([x, y], corner_data[1])
        return denormalized_keypoints

def keypoints_to_heatmap(keypoints, image_size=500, sigma=1.0, gaussian_blur=False, device='cpu'):
    """
    Converts keypoints into a lower-resolution heatmap (e.g., 128×img = Image.open(image_path)28) for training.
    The heatmap will be upsampled to match the input image size (500×500).
    """
    target_heatmap = torch.zeros((1, image_size, image_size)).to(device)

    for (x, y) in keypoints:
        #x = x * image_size
        #y = y * image_size
        x, y = int(x), int(y)  # Scale keypoints
        
        if gaussian_blur:
            if 0 <= x < image_size and 0 <= y < image_size:
                for i in range(-2, 3):  # Small 5×5 Gaussian
                    for j in range(-2, 3):
                        xi, yj = x + i, y + j
                        if 0 <= xi < image_size and 0 <= yj < image_size:
                            exponent = torch.tensor(-((i**2 + j**2) / (2 * sigma**2)), dtype=torch.float32).to(device)
                            target_heatmap[0, yj, xi] += torch.exp(exponent).to(device)
        else:
            target_heatmap[0, y, x] = 1

    
    # if not os.path.exists(os.path.join(os.path.dirname(__file__), os.pardir, 'heatmap.png')):
    #     heatmap = target_heatmap.squeeze().detach().cpu().numpy()
    #     plt.imshow(heatmap, cmap='Reds', interpolation='nearest')
    #     plt.title("Target Heatmap (Black to Red)")
    #     plt.colorbar()
    #     plt.savefig("heatmap.png")
    #     plt.close()

    return target_heatmap


def main(image_path, annotations_folder):

    img = Image.open(image_path)

    file_name = image_path.split("/")[-1].split(".")[0]

    annotations_file_path = os.path.join(annotations_folder, f"{file_name}.json")

    all_corners = []

    with open(annotations_file_path, mode='r') as file:
        data = json.load(file)
        annotations = data['annotations']
        for annotation in annotations:
            obj_name = annotation['brick_type']
            corners = annotation['normal_pixel_coordinates']
            color = annotation['color']

            denormalized_corners = denormalize_keypoints(corners, img.size[0], img.size[1])

            for corner_name, corner_data in denormalized_corners.items():
                x, y = corner_data[0]
                if corner_data[1]:
                    all_corners.append([x, y])

    
    target_heatmap = keypoints_to_heatmap(all_corners, image_size=img.size[0], sigma=1.0, gaussian_blur=True, device='cpu')

    return target_heatmap


heatmap = main(img_path, annotations_folder)


# Save the heatmap
heatmap = heatmap.squeeze().detach().cpu().numpy()
plt.imshow(heatmap, cmap='Reds', interpolation='nearest')
plt.title("Target Heatmap (Black to Red)")
plt.colorbar()
plt.savefig("heatmap.png")
plt.close()