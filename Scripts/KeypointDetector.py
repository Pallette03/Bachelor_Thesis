import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import cv2
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from LegoKeypointDataset import LegoKeypointDataset
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models
import time


class DynamicCornerDetector(nn.Module):
    def __init__(self, input_channels=3, num_classes=1):
        super(DynamicCornerDetector, self).__init__()
        # Initial Convolutional Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces spatial size by 2
        )
        
        # Intermediate Blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces spatial size by 2
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Output Heatmap Head
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0),  # Final heatmap
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)  # Upsample to 4x
        )
        
        # Output Offset Head
        self.offset_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0),  # 2 channels: dx and dy
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)  # Upsample to 4x
        )

    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Output heads
        heatmap = self.heatmap_head(x)
        offsets = self.offset_head(x)

        return heatmap, offsets

class WeightedBCELoss(nn.Module):
    def __init__(self, weight_corner, weight_non_corner):
        super(WeightedBCELoss, self).__init__()
        self.weight_corner = weight_corner
        self.weight_non_corner = weight_non_corner

    def forward(self, preds, targets):
        max_val = torch.max(targets)
        weights = torch.where(targets == max_val, self.weight_corner, self.weight_non_corner)
        loss = nn.BCELoss(reduction='none')(preds, targets)
        weighted_loss = (weights * loss).mean()
        return weighted_loss

def compute_class_weights(mask):
    total_pixels = mask.numel()
    corner_pixels = mask.sum().item()
    non_corner_pixels = total_pixels - corner_pixels

    weight_corner = total_pixels / (2 * corner_pixels)
    weight_non_corner = total_pixels / (2 * non_corner_pixels)
    
    return weight_corner, weight_non_corner

def heatmap_loss(predicted_heatmaps, target_heatmaps):
    return nn.MSELoss()(predicted_heatmaps, target_heatmaps)

def offset_loss(predicted_offsets, target_offsets, mask):
    # Mask ensures loss is only computed for valid corner locations
    return nn.MSELoss()(predicted_offsets * mask, target_offsets * mask)

# Training Loop
def train_model(model, dataloader, num_epochs=5, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Loss functions
    heatmap_criterion = nn.MSELoss()
    offset_criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_heatmap_loss = 0.0
        total_offset_loss = 0.0
        
        for batch in dataloader:
            images = batch["image"].to(device)  # Shape: [batch_size, 3, H, W]
            target_heatmaps = batch["heatmaps"].to(device)  # Shape: [batch_size, 1, H, W]
            target_offsets = batch["offsets"].to(device)  # Shape: [batch_size, 2, H, W]
            mask = batch["mask"].to(device)  # Shape: [batch_size, 1, H, W], 1 for valid corner locations
            predicted_heatmaps, predicted_offsets = model(images)
            
            #print(f"predicted_offsets: {predicted_offsets.shape}, target_offsets: {target_offsets.shape}, mask: {mask.shape}")
            
            heatmap_loss = heatmap_criterion(predicted_heatmaps, target_heatmaps)
            offset_loss = offset_criterion(predicted_offsets * mask, target_offsets * mask)
            
            # Total loss
            total_loss = heatmap_loss + offset_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses for logging
            total_heatmap_loss += heatmap_loss.item()
            total_offset_loss += offset_loss.item()
        
        # Update learning rate
        #scheduler.step()

        # Log epoch stats
        print(f"Took {time.time() - start_time:.2f} seconds for epoch {epoch + 1}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Heatmap Loss: {total_heatmap_loss / len(dataloader):.4f}, "
            f"Offset Loss: {total_offset_loss / len(dataloader):.4f}")
    return model

def convert_tensor_to_image(tensor):
    tensor = tensor.cpu().detach()
    to_pil = transforms.ToPILImage()
    return to_pil(tensor)

# Visualize Predictions
def visualize_predictions(image, model, threshold=0.5, image_size=(224, 224)):
    """
    Visualizes the predicted corners on the input image.
    
    Args:
        image (PIL.Image or tensor): The input image.
        model (torch.nn.Module): The trained corner detection model.
        threshold (float): Heatmap threshold to filter corner predictions.
        image_size (tuple): The size to resize the input image.
        
    Returns:
        None: Displays the image with predictions.
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Transform and preprocess the image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if isinstance(image, Image.Image):
        original_image = image.copy()
        image = F.resize(image, image_size)
        image_tensor = F.to_tensor(image).unsqueeze(0).to(next(model.parameters()).device)  # Add batch dim
    if isinstance(image, torch.Tensor):
        image_tensor = image.clone().detach().unsqueeze(0).to(next(model.parameters()).device)
        original_image = convert_tensor_to_image(image)
    else:
        raise ValueError(f"Unsupported image type. Provide a PIL.Image or NumPy array. Got {type(image)}.")
    
    # Forward pass to get predictions
    with torch.no_grad():
        predicted_heatmaps, predicted_offsets = model(image_tensor)
    
    # Process the heatmaps
    predicted_heatmaps = predicted_heatmaps.squeeze(0).cpu().numpy()  # Shape: [1, H, W]
    heatmap = predicted_heatmaps[0]  # Extract single channel heatmap
    
    # Apply thresholding to detect keypoints
    keypoints = np.argwhere(heatmap > threshold)  # [y, x] positions
    keypoints = keypoints[:, [1, 0]]  # Convert to [x, y]
    
    # Rescale keypoints back to the original image size
    keypoints = keypoints * np.array(original_image.size) / np.array(heatmap.shape)
    
    
    # Convert image to displayable format
    original_image = np.array(original_image)
    
    # Show the image without keypoints
    plt.figure(figsize=(8, 8))
    plt.imshow(original_image)
    plt.axis("off")
    plt.title("Original Image")
    plt.show()
    
    # Plot the image and overlay keypoints
    plt.figure(figsize=(8, 8))
    plt.imshow(original_image)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c="red", s=50, label="Predicted Corners")
    plt.legend()
    plt.axis("off")
    plt.title("Predicted Corners")
    plt.show()
# Paths
model_path = 'C:/Users/paulb/Documents/TUDresden/Bachelor/output/detector.pth'
image_dir = 'C:/Users/paulb/Documents/TUDresden/Bachelor/datasets/object_detection/images/rgb'
annotation_dir = 'C:/Users/paulb/Documents/TUDresden/Bachelor/datasets/object_detection/annotations'
temp_dir = 'C:/Users/paulb/Documents/TUDresden/Bachelor/datasets/temp_dataset'

just_visualize = False

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

# Dataset and DataLoader
print("Loading dataset...")
dataset = LegoKeypointDataset(annotation_dir, image_dir, transform=transform, sigma=0.5)
dataset.reduce_dataset_size(3000)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Model, Optimizer, and Loss
model = DynamicCornerDetector()

if not just_visualize:
    # Train the model
    print("Training the model...")
    model = train_model(model, train_dataloader, num_epochs=15, lr=1e-3)
else:
    # Load the model
    print("Loading the model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    

# Save the model
print("Saving the model...")
torch.save(model.state_dict(), model_path)

# Visualize results on an image
#image, _ = dataset[0]

# Create a loop that goes through the dataset and visualizes the predictions at the press of a button
for batch in val_dataset:
    visualize_predictions(batch["image"], model, threshold=0.3, image_size=(224, 224))
    if input("Press 'q' to quit, or any other key to continue: ") == 'q':
        break
    else:
        plt.close()
        