import os
import torch
import torchvision.transforms as transforms
import random

from viewpoint_estimator import ViewpointEstimator
from rotation_dataset import RotationDataset

model_path = "C:\\Users\\paulb\\Documents\\TUDresden\\Bachelor\\output\\viewpoint_estimator.pth"
root_dir = "C:\\Users\\paulb\\Documents\\TUDresden\\Bachelor\\datasets\\rotation_estimation"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization for VGG
])


#Load the state dict of the model
model = ViewpointEstimator(num_azimuth_bins=360 // 5, 
                        num_elevation_bins=180 // 5, 
                        num_rotation_bins=360 // 5).cuda()

model.load_state_dict(torch.load(model_path))

dataset = RotationDataset(root_dir, transform=transform, bin_size=5)

def get_predictions(model, images):
    """
    Get integer predictions for azimuth, elevation, and rotation bins.

    Args:
        model (nn.Module): The trained PyTorch model.
        images (torch.Tensor): A batch of images (shape: [batch_size, 3, H, W]).

    Returns:
        tuple of torch.Tensor: Predicted azimuth, elevation, and rotation bins as integers.
    """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Forward pass
        azimuth_logits, elevation_logits, rotation_logits = model(images)

        # Get the predicted bins as integers (argmax over the logits)
        azimuth_pred = torch.argmax(azimuth_logits, dim=1)
        elevation_pred = torch.argmax(elevation_logits, dim=1)
        rotation_pred = torch.argmax(rotation_logits, dim=1)

    return azimuth_pred, elevation_pred, rotation_pred


#rand_int = random.randint(0, len(dataset))
for i in range(len(dataset)):
    image = dataset[-(i+2202)][0].unsqueeze(0).cuda()
    labels = dataset[-(i+2202)][1].unsqueeze(0).cuda()
    print(labels)
    azimuth_pred, elevation_pred, rotation_pred = get_predictions(model, image)
    print(f"Azimuth: {azimuth_pred.item() * 5}°")
    print(f"Elevation: {elevation_pred.item() * 5}°")
    print(f"Rotation: {rotation_pred.item() * 5}°")
    print(f"True Azimuth: {labels[0][0].item() * 5}°")
    print(f"True Elevation: {labels[0][1].item() * 5}°")
    print(f"True Rotation: {labels[0][2].item() * 5}°")
    input("Press Enter to continue...")