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


# Harris Corner Detector
def harris_corner_detector(image, threshold=0.01):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    image[dst > threshold * dst.max()] = [0, 0, 255]
    return image


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.decoder(x)
        return x

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

# Training Loop
def train_model(model, dataloader, num_epochs=5, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            
            weight_corner, weight_non_corner = compute_class_weights(masks)
            criterion = WeightedBCELoss(weight_corner, weight_non_corner)
            
            preds = model(images)
            preds_resized = F.interpolate(preds, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(preds_resized, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}")
    return model

def convert_tensor_to_image(tensor):
    tensor = tensor.cpu().detach()
    to_pil = transforms.ToPILImage()
    return to_pil(tensor)

# Visualize Predictions
def visualize_predictions(model, image, transform):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    image = convert_tensor_to_image(image)
    
    input_image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(input_image).squeeze(0).cpu().numpy()
    
    preds_binary = (preds > 0.95).astype(np.uint8)
    for y, x in zip(*np.where(preds_binary[0] == 1)):
        plt.plot(x, y, 'ro', markersize=2)

    plt.imshow(image)
    plt.show()
# Paths
model_path = 'C:/Users/paulb/Documents/TUDresden/Bachelor/output/detector.pth'
image_dir = 'C:/Users/paulb/Documents/TUDresden/Bachelor/dataset/images/rgb'
annotation_dir = 'C:/Users/paulb/Documents/TUDresden/Bachelor/dataset/annotations'

just_visualize = True

transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

# Dataset and DataLoader
print("Loading dataset...")
dataset = LegoKeypointDataset(annotation_dir, image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Model, Optimizer, and Loss
model = UNet()

if not just_visualize:
    # Train the model
    print("Training the model...")
    model = train_model(model, dataloader, num_epochs=10, lr=1e-3)
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
for image, _ in dataset:
    visualize_predictions(model, image, transform)
    if input("Press 'q' to quit, or any other key to continue: ") == 'q':
        break
    else:
        plt.close()
        