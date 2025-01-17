import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from LegoKeypointDataset import LegoKeypointDataset
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import time


class DynamicCornerDetector(nn.Module):
    def __init__(self, input_channels=3, num_classes=1):
        super(DynamicCornerDetector, self).__init__()
        # Initial Convolutional Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Intermediate Block 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Strided conv reduces size by 2
            nn.ReLU(inplace=True)
        )
        
        # Intermediate Block 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Dilated Convolution Block
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )
        
        # Output Heatmap Head
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0),  # Final heatmap
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # Upsample to match input resolution
        )

    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dilated_conv(x)

        # Output heads
        heatmap = self.heatmap_head(x)

        return heatmap

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

def weighted_mse_loss(predicted, target, weight):
    return ((predicted - target) ** 2 * weight).mean()

def heatmap_loss(predicted_heatmaps, target_heatmaps):
    return nn.MSELoss()(predicted_heatmaps, target_heatmaps)

def offset_loss(predicted_offsets, target_offsets, mask):
    # Mask ensures loss is only computed for valid corner locations
    return nn.MSELoss()(predicted_offsets * mask, target_offsets * mask)

# Training Loop
def train_model(model, dataloader, val_dataloader, temp_model_path, num_epochs=5, lr=1e-3):
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
        counter = 0
        
        for batch in dataloader:
            images = batch["image"].to(device)  # Shape: [batch_size, 3, H, W]
            target_heatmaps = batch["heatmaps"].to(device)  # Shape: [batch_size, 1, H, W]
            #target_offsets = batch["offsets"].to(device)  # Shape: [batch_size, 2, H, W]
            #mask = batch["mask"].to(device)  # Shape: [batch_size, 1, H, W], 1 for valid corner locations
            predicted_heatmaps = model(images)#, predicted_offsets
            
            #print(f"predicted_offsets: {predicted_offsets.shape}, target_offsets: {target_offsets.shape}, mask: {mask.shape}")
            # Assign higher weights to corner regions (positive pixels) and lower to the background
            positive_weight = 10.0  # Weight for corner regions
            negative_weight = 1.0   # Weight for background regions

            # Compute weights based on target heatmap values
            weight = torch.where(target_heatmaps > 0.1, positive_weight, negative_weight)
            
            
            heatmap_loss = weighted_mse_loss(predicted_heatmaps, target_heatmaps, weight)
            #offset_loss = offset_criterion(predicted_offsets * mask, target_offsets * mask)
            
            # Total loss
            total_loss = heatmap_loss# + offset_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses for logging
            total_heatmap_loss += heatmap_loss.item()
            #total_offset_loss += offset_loss.item()
            
            counter += 1
            # Check the progress through the batch and print every 10 percent
            if counter % (len(dataloader) // 20) == 0:
                print(f"At Batch {counter}/{len(dataloader)} for Epoch {epoch + 1}")
                
        
        # Update learning rate
        #scheduler.step()


        validate_model(model, val_dataloader)

        # Log epoch stats
        print(f"Took {time.time() - start_time:.2f} seconds for epoch {epoch + 1}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Heatmap Loss: {total_heatmap_loss / len(dataloader):.4f}")
        
        # Save model after each epoch
        print(f"Saving model to {temp_model_path}")
        torch.save(model.state_dict(), temp_model_path)
    return model

def convert_tensor_to_image(tensor):
    tensor = tensor.cpu().detach()
    to_pil = transforms.ToPILImage()
    return to_pil(tensor)

def validate_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Validating on {device}")
    model.eval()
    total_heatmap_loss = 0.0
    total_offset_loss = 0.0
    
    heatmap_criterion = nn.MSELoss()
    
    for batch in dataloader:
        images = batch["image"].to(device)  # Shape: [batch_size, 3, H, W]
        target_heatmaps = batch["heatmaps"].to(device)  # Shape: [batch_size, 1, H, W]
        #target_offsets = batch["offsets"].to(device)  # Shape: [batch_size, 2, H, W]
        #mask = batch["mask"].to(device)  # Shape: [batch_size, 1, H, W], 1 for valid corner locations
        predicted_heatmaps = model(images)#, predicted_offsets
        
        #print(f"predicted_offsets: {predicted_offsets.shape}, target_offsets: {target_offsets.shape}, mask: {mask.shape}")
        # Assign higher weights to corner regions (positive pixels) and lower to the background
        positive_weight = 10.0  # Weight for corner regions
        negative_weight = 1.0   # Weight for background regions

        # Compute weights based on target heatmap values
        weight = torch.where(target_heatmaps > 0.1, positive_weight, negative_weight)
        
        heatmap_loss = weighted_mse_loss(predicted_heatmaps, target_heatmaps, weight)
        #offset_loss = offset_criterion(predicted_offsets * mask, target_offsets * mask)
        
        # Accumulate losses for logging
        total_heatmap_loss += heatmap_loss.item()
        #total_offset_loss += offset_loss.item()
    
    print(f"Validation Loss: {total_heatmap_loss / len(dataloader):.4f}")
    
    return total_heatmap_loss / len(dataloader)

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
        predicted_heatmaps = model(image_tensor)
    
    # Process the heatmaps
    predicted_heatmaps = predicted_heatmaps.squeeze(0).cpu().numpy()  # Shape: [1, H, W]
    heatmap = predicted_heatmaps[0]  # Extract single channel heatmap
    
    print(f"Max value: {heatmap.max()}")
    
    
    
    
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
#model_path = 'C:/Users/paulb/Documents/TUDresden/Bachelor/output/detector.pth'
temp_model_path = 'C:/Users/paulb/Documents/TUDresden/Bachelor/output/temp_detector.pth'
model_path = temp_model_path
image_dir = 'C:/Users/paulb/Documents/TUDresden/Bachelor/datasets/object_detection/images/rgb'
annotation_dir = 'C:/Users/paulb/Documents/TUDresden/Bachelor/datasets/object_detection/annotations'
temp_dir = 'C:/Users/paulb/Documents/TUDresden/Bachelor/datasets/temp_dataset'
cropped_dir = 'C:/Users/paulb/Documents/TUDresden/Bachelor/datasets/cropped_objects/images/rgb'
annotation_cropped_dir = 'C:/Users/paulb/Documents/TUDresden/Bachelor/datasets/cropped_objects/annotations'
train_dir = 'C:/Users/paulb/Documents/TUDresden/Bachelor/datasets/cropped_objects/train'
validate_dir = 'C:/Users/paulb/Documents/TUDresden/Bachelor/datasets/cropped_objects/validate'

just_visualize = True
batch_size = 8
global_image_size = (500, 500)

transform = transforms.Compose([
        transforms.Resize(global_image_size),
        transforms.ToTensor()
    ])

# Dataset and DataLoader
print("Loading dataset...")
train_dataset = LegoKeypointDataset(os.path.join(train_dir, 'annotations'), os.path.join(train_dir, 'images'), image_size=global_image_size,transform=transform, sigma=0.3)
train_dataset.reduce_dataset_size(4000)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#dataset.reduce_dataset_size(3000)

val_dataset = LegoKeypointDataset(os.path.join(validate_dir, 'annotations'), os.path.join(validate_dir, 'images'), image_size=global_image_size, transform=transform, sigma=0.3)
val_dataset.reduce_dataset_size(500)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, Optimizer, and Loss
model = DynamicCornerDetector()

if not just_visualize:
    # Train the model
    print("Training the model...")
    model = train_model(model, train_dataloader, val_dataloader, temp_model_path, num_epochs=5, lr=1e-3)
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
#validate_model(model, val_dataloader)
# Create a loop that goes through the dataset and visualizes the predictions at the press of a button
for batch in val_dataset:
    visualize_predictions(batch["image"], model, threshold=0.3, image_size=global_image_size)
    if input("Press 'q' to quit, or any other key to continue: ") == 'q':
        break
    else:
        plt.close()
        