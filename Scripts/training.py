import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from LegoKeypointDataset import LegoKeypointDataset
from KeypointDetector import KeypointDetector
import torch.nn as nn

# Parameters
num_keypoints = 8  # Adjust based on your annotations
batch_size = 16
num_epochs = 20
learning_rate = 0.001

render_images_folder = 'C:/Users/paulb/Documents/TUDresden/Bachelor/dataset/images'
annotations_folder = 'C:/Users/paulb/Documents/TUDresden/Bachelor/dataset/annotations'
model_path = 'C:/Users/paulb/Documents/TUDresden/Bachelor/output/detector.pth'

# Data Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def collate_fn(batch):
    images, keypoints = zip(*batch)  # Separate images and keypoints

    # Determine the maximum number of keypoints in this batch
    max_len = max(kp.size(0) for kp in keypoints)

    # Pad keypoints to the maximum length
    padded_keypoints = torch.full((len(keypoints), max_len, 2), -1, dtype=torch.float32)
    for i, kp in enumerate(keypoints):
        padded_keypoints[i, :kp.size(0), :] = kp

    # Stack images into a single tensor
    images = torch.stack(images)  # Assumes images are already transformed

    return images, padded_keypoints

def compute_loss(outputs, keypoints, mask):
    # Expand the mask to match the shape of outputs and keypoints
    mask = mask.unsqueeze(-1).expand_as(outputs)  # Shape: [Batch, MaxKeypoints, 2]

    # Apply the mask to outputs and keypoints
    valid_outputs = outputs[mask].view(-1, 2)  # [Valid_N, 2]
    valid_keypoints = keypoints[mask].view(-1, 2)  # [Valid_N, 2]

    # Mean Squared Error Loss
    criterion = nn.MSELoss()
    loss = criterion(valid_outputs, valid_keypoints)
    return loss



# Dataset and DataLoader
dataset = LegoKeypointDataset(annotations_folder, render_images_folder, transform=transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)



# Model, Loss, Optimizer
model = KeypointDetector().to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, keypoints in dataloader:
        images = images.to('cuda')
        keypoints = keypoints.to('cuda')

        # Create a mask for non-padded keypoints
        mask = keypoints[:, :, 0] != -1  # Mask for valid keypoints
        max_keypoints = keypoints.size(1)  # Max number of keypoints in the batch

        # Forward pass
        outputs = model(images, max_keypoints)  # [Batch, MaxKeypoints, 2]
        loss = compute_loss(outputs, keypoints, mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}")

model.eval()  # Set model to evaluation mode
total_loss = 0
criterion = nn.MSELoss()

with torch.no_grad():  # Disable gradient computation
    for images, keypoints in dataloader:
        images = images.to('cuda')
        keypoints = keypoints.to('cuda')

        # Create a mask for non-padded keypoints
        mask = keypoints[:, :, 0] != -1  # Mask for valid keypoints
        max_keypoints = keypoints.size(1)  # Max number of keypoints in the batch

        # Forward pass
        outputs = model(images, max_keypoints)  # [Batch, MaxKeypoints, 2]

        # Compute loss with masking
        mask_expanded = mask.unsqueeze(-1).expand_as(outputs)
        valid_outputs = outputs[mask_expanded].view(-1, 2)  # [Valid_N, 2]
        valid_keypoints = keypoints[mask_expanded].view(-1, 2)  # [Valid_N, 2]
        loss = criterion(valid_outputs, valid_keypoints)

        total_loss += loss.item()

average_loss = total_loss / len(dataloader)
print(f"Validation Loss: {average_loss:.4f}")

# Save model if validation loss improves
if average_loss < best_val_loss:
    best_val_loss = average_loss
    torch.save(model.state_dict(), model_path)
    print("Saved new best model!")