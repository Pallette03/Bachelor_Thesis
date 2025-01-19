import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from LegoKeypointDataset import LegoKeypointDataset
import torchvision.transforms as transforms
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

# Training Loop
def train_model(model, dataloader, val_dataloader, epoch_model_path, num_epochs=5, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        batch_start_time = time.time()
        model.train()
        total_heatmap_loss = 0.0
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
            weight = torch.where(target_heatmaps > 0.3, positive_weight, negative_weight)
            
            
            heatmap_loss = weighted_mse_loss(predicted_heatmaps, target_heatmaps, weight)
            
            # Total loss
            total_loss = heatmap_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses for logging
            total_heatmap_loss += heatmap_loss.item()
          
            counter += 1
            # Check the progress through the batch and print every 5 percent
            if counter % (len(dataloader) // 20) == 0:
                print(f"At Batch {counter}/{len(dataloader)} for Epoch {epoch + 1} taking {time.time() - batch_start_time:.2f} seconds since last checkpoint")
                batch_start_time = time.time()


        validate_model(model, val_dataloader)

        # Log epoch stats
        print(f"Took {time.time() - start_time:.2f} seconds for epoch {epoch + 1}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Heatmap Loss: {total_heatmap_loss / len(dataloader):.4f}")
        
        # Save model after each epoch
        print(f"Saving model to {epoch_model_path}")
        torch.save(model.state_dict(), epoch_model_path)
    return model

def validate_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Validating on {device}")
    model.eval()
    total_heatmap_loss = 0.0
    
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
        weight = torch.where(target_heatmaps > 0.3, positive_weight, negative_weight)
        
        heatmap_loss = weighted_mse_loss(predicted_heatmaps, target_heatmaps, weight)
        
        # Accumulate losses for logging
        total_heatmap_loss += heatmap_loss.item()
    
    print(f"Validation Loss: {total_heatmap_loss / len(dataloader):.4f}")
    
    return total_heatmap_loss / len(dataloader)

# Paths
model_path = os.path.join(os.path.dirname(__file__), 'output', 'dynamic_corner_detector.pth')
epoch_model_path = os.path.join(os.path.dirname(__file__), 'output', 'dynamic_corner_detector_epoch.pth')
train_dir = os.path.join(os.path.dirname(__file__), 'train')
validate_dir = os.path.join(os.path.dirname(__file__), 'validate')

print(f"Paths: {model_path}, {epoch_model_path}, {train_dir}, {validate_dir}")

batch_size = 32
global_image_size = (500, 500)

transform = transforms.Compose([
        transforms.Resize(global_image_size),
        transforms.ToTensor()
    ])

# Dataset and DataLoader
print("Loading dataset...")
train_dataset = LegoKeypointDataset(os.path.join(train_dir, 'annotations'), os.path.join(train_dir, 'images'), image_size=global_image_size,transform=transform, sigma=0.3)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#dataset.reduce_dataset_size(3000)

val_dataset = LegoKeypointDataset(os.path.join(validate_dir, 'annotations'), os.path.join(validate_dir, 'images'), image_size=global_image_size, transform=transform, sigma=0.3)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, Optimizer, and Loss
model = DynamicCornerDetector()

# Train the model
print("Training the model...")
model = train_model(model, train_dataloader, val_dataloader, epoch_model_path, num_epochs=5, lr=1e-3)

# Save the model
print("Saving the model...")
torch.save(model.state_dict(), model_path)

        