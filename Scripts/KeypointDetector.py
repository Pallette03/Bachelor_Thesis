import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from LegoKeypointDataset import LegoKeypointDataset
import torchvision.transforms as transforms
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from unet_model import UNet



def keypoints_to_heatmap(keypoints, heatmap_size=500, image_size=500, sigma=1.0, device='cpu'):
    """
    Converts keypoints into a lower-resolution heatmap (e.g., 128×128) for training.
    The heatmap will be upsampled to match the input image size (500×500).
    """
    target_heatmap = torch.zeros((1, heatmap_size, heatmap_size)).to(device)
    scale = heatmap_size / image_size  # Scale down factor (e.g., 128/500)

    for (x, y) in keypoints:
        x = x * image_size
        y = y * image_size
        x, y = int(x * scale), int(y * scale)  # Scale keypoints

        #target_heatmap[0, y, x] = 1

        if 0 <= x < heatmap_size and 0 <= y < heatmap_size:
            for i in range(-2, 3):  # Small 5×5 Gaussian
                for j in range(-2, 3):
                    xi, yj = x + i, y + j
                    if 0 <= xi < heatmap_size and 0 <= yj < heatmap_size:
                        exponent = torch.tensor(-((i**2 + j**2) / (2 * sigma**2)), dtype=torch.float32).to(device)
                        target_heatmap[0, yj, xi] += torch.exp(exponent).to(device)

    return target_heatmap.clamp(0, 1)  # Keep values in [0,1]

def heatmap_loss(pred_heatmaps, keypoints_list, heatmap_size=500, image_size=500, device='cpu', critereon=None):

    target_heatmaps = torch.stack([keypoints_to_heatmap(kp, heatmap_size=heatmap_size, image_size=image_size, device=device) for kp in keypoints_list]).to(device)

    loss = critereon(pred_heatmaps, target_heatmaps)
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
    corners_list = torch.stack([torch.tensor(corners, dtype=torch.float32) for corners in corners_list])

    return {"image": images, "norm_corners": corners_list}

# Training Loop
def train_model(model, dataloader, val_dataloader, epoch_model_path, num_epochs=5, lr=1e-3, global_image_size=(500, 500)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    critereon = nn.BCEWithLogitsLoss()
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
            
            corner_loss = heatmap_loss(predicted_corners, target_corners, image_size=global_image_size[0], device=device, critereon=critereon)
            #corner_loss = heatmap_loss(predicted_corners, target_corners, image_size=global_image_size[0], device=device)
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

        #validate_model(model, val_dataloader, global_image_size, critereon=critereon)

        # Log epoch stats
        print(f"Took {time.time() - start_time:.2f} seconds for epoch {epoch + 1}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Heatmap Loss: {total_corner_loss / len(dataloader):.4f}")
        
        # Save model after each epoch
        print(f"Saving model to {epoch_model_path}")
        torch.save(model.state_dict(), epoch_model_path)
    return model

def validate_model(model, dataloader, global_image_size, critereon=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Validating on {device}")
    model.eval()
    model = model.to(device)
    total_corner_loss = 0.0
    val_counter = 0
    val_batch_start_time = time.time()

    for batch in dataloader:
        images = batch["image"].to(device)  # Shape: [batch_size, 3, H, W]
        target_corners = batch["norm_corners"].to(device)  # Shape: [batch_size, 1, H, W]
        #target_offsets = batch["offsets"].to(device)  # Shape: [batch_size, 2, H, W]
        #mask = batch["mask"].to(device)  # Shape: [batch_size, 1, H, W], 1 for valid corner locations
        predicted_corners = model(images)#, predicted_offsets
        
        # Use normal MSE loss for now but ignore 0
        corner_loss = corner_loss = heatmap_loss(predicted_corners, target_corners, image_size=global_image_size[0], device=device, critereon=critereon)
        val_last_loss = corner_loss.item()
        
        # Accumulate losses for logging
        total_corner_loss += corner_loss.item()

        val_counter += 1
        # Check the progress through the batch and print every 5 percent
        if val_counter % (len(dataloader) // 20) == 0:
            print(f"At Batch {val_counter}/{len(dataloader)} for Validation taking {time.time() - val_batch_start_time:.2f} seconds since last checkpoint. Last Loss: {val_last_loss:.4f}")
            val_batch_start_time = time.time()
    
    print(f"Validation Loss: {total_corner_loss / len(dataloader):.4f}")
    
    return total_corner_loss / len(dataloader)

if __name__ == "__main__":
    # Paths
    model_path = os.path.join(os.path.dirname(__file__), os.pardir, 'output', 'dynamic_corner_detector.pth')
    epoch_model_path = os.path.join(os.path.dirname(__file__), os.pardir, 'output', 'dynamic_corner_detector_epoch.pth')
    train_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'cropped_objects', 'train')
    validate_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'cropped_objects', 'validate')

    only_validate = True

    print(f"Paths: {model_path}, {epoch_model_path}, {train_dir}, {validate_dir}")

    batch_size = 8
    val_batch_size = 4
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
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate_fn)

    # Model, Optimizer, and Loss
    model = UNet(n_channels=3, n_classes=1)

    # Train the model
    print("Training the model...")
    if not only_validate:
        model = train_model(model, train_dataloader, val_dataloader, epoch_model_path, num_epochs=5, lr=1e-3, global_image_size=global_image_size)
    else:
        model.load_state_dict(torch.load(epoch_model_path))
        validate_model(model, val_dataloader, global_image_size, critereon=nn.BCEWithLogitsLoss())

    # Save the model
    print("Saving the model...")
    torch.save(model.state_dict(), model_path)

        