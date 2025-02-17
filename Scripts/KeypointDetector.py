import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from LegoKeypointDataset import LegoKeypointDataset
import torchvision.transforms as transforms
import time
import torch.nn.functional as F

os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'

class DynamicCornerDetector(nn.Module):
    def __init__(self):
        super(DynamicCornerDetector, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Heatmap output (1 channel for corner probability)
        self.corner_head = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        heatmap = self.corner_head(x)  # Output shape: (batch, 1, 31, 31)
        return heatmap



def keypoints_to_heatmap(keypoints, heatmap_size=31, image_size=500, sigma=1.5, device="cpu"):
    """
    Converts a list of keypoints into a heatmap representation.

    keypoints: List of (x, y) coordinates (scaled for the original image size).
    heatmap_size: Size of the output heatmap.
    image_size: Original image size before downsampling.
    sigma: Spread of the Gaussian response.

    Returns:
    - target_heatmap: A tensor of shape (1, heatmap_size, heatmap_size)
    """
    target_heatmap = torch.zeros((1, heatmap_size, heatmap_size))

    scale = heatmap_size / image_size  # Downscale factor (e.g., 31/500)
    for (x, y) in keypoints:
        x, y = int(x * scale), int(y * scale)  # Scale to heatmap size
        if 0 <= x < heatmap_size and 0 <= y < heatmap_size:
            # Create a small Gaussian response around the keypoint
            for i in range(-3, 4):  # Gaussian spread in a 7x7 window
                for j in range(-3, 4):
                    xi, yj = x + i, y + j
                    if 0 <= xi < heatmap_size and 0 <= yj < heatmap_size:
                        target_heatmap[0, yj, xi] += torch.exp(torch.tensor(-((i**2 + j**2) / (2 * sigma**2))))


    return target_heatmap.clamp(0, 1).to(device) # Clamp to 0-1 range

def heatmap_loss(pred_heatmap, keypoints_list, heatmap_size=31, image_size=500, device='cpu'):
    """
    Computes loss between predicted heatmap and target heatmap.

    pred_heatmap: Tensor of shape (batch, 1, heatmap_size, heatmap_size)
    keypoints_list: List of keypoints for each image in batch (batch_size lists)

    Returns:
    - BCE Loss between predicted and ground truth heatmaps
    """
    batch_size = pred_heatmap.shape[0]
    target_heatmaps = torch.stack([keypoints_to_heatmap(kp, heatmap_size, image_size, device=device) for kp in keypoints_list])
    
    # Compute Binary Cross-Entropy loss
    loss = F.binary_cross_entropy(pred_heatmap, target_heatmaps)
    
    return loss

def extract_coordinates(heatmap, threshold=0.5):
    heatmap = heatmap.squeeze().detach().cpu().numpy()  # Remove batch & channel dims
    corners = torch.nonzero(heatmap > threshold)  # Get (y, x) positions
    corners = corners.numpy() * 16  # Upscale to original image size (500x500)
    return corners  # List of (y, x) positions

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
    corners_list = torch.tensor(corners_list, dtype=torch.float32)

    return {"image": images, "norm_corners": corners_list}

# Training Loop
def train_model(model, dataloader, val_dataloader, epoch_model_path, num_epochs=5, lr=1e-3, global_image_size=(500, 500)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        batch_start_time = time.time()
        model.train()
        total_corner_loss = 0.0
        counter = 0
        
        for batch in dataloader:
            images = batch["image"].to(device)  # Shape: [batch_size, 3, H, W]
            target_corners = batch["norm_corners"].to(device)  # Shape: [batch_size, num_corners_in_batch, 2]

            predicted_corners = model(images)
            
            corner_loss = heatmap_loss(predicted_corners, target_corners, image_size=global_image_size[0], device=device)
            last_loss = corner_loss.item()


            # Backward pass
            optimizer.zero_grad()
            corner_loss.backward()
            optimizer.step()
            
            # Accumulate losses for logging
            total_corner_loss += corner_loss.item()
          
            counter += 1
            # Check the progress through the batch and print every 5 percent
            if counter % (len(dataloader) // 20) == 0:
                print(f"At Batch {counter}/{len(dataloader)} for Epoch {epoch + 1} taking {time.time() - batch_start_time:.2f} seconds since last checkpoint. Last Loss: {last_loss:.4f}")
                batch_start_time = time.time()


        #Clear vram memory before validation
        torch.cuda.empty_cache()

        validate_model(model, val_dataloader, global_image_size)

        # Log epoch stats
        print(f"Took {time.time() - start_time:.2f} seconds for epoch {epoch + 1}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Heatmap Loss: {total_corner_loss / len(dataloader):.4f}")
        
        # Save model after each epoch
        print(f"Saving model to {epoch_model_path}")
        torch.save(model.state_dict(), epoch_model_path)
    return model

def validate_model(model, dataloader, global_image_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Validating on {device}")
    model.eval()
    total_corner_loss = 0.0
    
    for batch in dataloader:
        images = batch["image"].to(device)  # Shape: [batch_size, 3, H, W]
        target_corners = batch["norm_corners"].to(device)  # Shape: [batch_size, 1, H, W]
        #target_offsets = batch["offsets"].to(device)  # Shape: [batch_size, 2, H, W]
        #mask = batch["mask"].to(device)  # Shape: [batch_size, 1, H, W], 1 for valid corner locations
        predicted_corners = model(images)#, predicted_offsets
        
        # Use normal MSE loss for now but ignore 0
        corner_loss = heatmap_loss(predicted_corners, target_corners, image_size=global_image_size[0], device=device)
        
        # Accumulate losses for logging
        total_corner_loss += corner_loss.item()
    
    print(f"Validation Loss: {total_corner_loss / len(dataloader):.4f}")
    
    return total_corner_loss / len(dataloader)

# Paths
model_path = os.path.join(os.path.dirname(__file__), os.pardir, 'output', 'dynamic_corner_detector.pth')
epoch_model_path = os.path.join(os.path.dirname(__file__), os.pardir, 'output', 'dynamic_corner_detector_epoch.pth')
train_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'cropped_objects', 'train')
validate_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'cropped_objects', 'validate')

print(f"Paths: {model_path}, {epoch_model_path}, {train_dir}, {validate_dir}")

batch_size = 25
global_image_size = (500, 500)

transform = transforms.Compose([
        transforms.Resize(global_image_size),
        transforms.ToTensor()
    ])

# Dataset and DataLoader
print("Loading dataset...")
train_dataset = LegoKeypointDataset(os.path.join(train_dir, 'annotations'), os.path.join(train_dir, 'images'), image_size=global_image_size,transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
#dataset.reduce_dataset_size(3000)

val_dataset = LegoKeypointDataset(os.path.join(validate_dir, 'annotations'), os.path.join(validate_dir, 'images'), image_size=global_image_size, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Model, Optimizer, and Loss
model = DynamicCornerDetector()

# Train the model
print("Training the model...")
model = train_model(model, train_dataloader, val_dataloader, epoch_model_path, num_epochs=5, lr=1e-3, global_image_size=global_image_size)

# Save the model
print("Saving the model...")
torch.save(model.state_dict(), model_path)

        