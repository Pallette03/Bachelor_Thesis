import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from bobe import LegoDetector
from LegoKeypointDataset import LegoKeypointDataset
from bobe import compute_loss

# Define the training loop
def train_model(model, dataset, epochs, batch_size, learning_rate, device):
    # DataLoader for batch processing
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0.0

        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            bbox_preds, keypoint_preds = model(images)
            
            # Split targets into bbox_targets and keypoint_targets
            bbox_targets = targets[:, :bbox_preds.shape[1], :, :]  # Assuming bbox targets are at the start
            keypoint_targets = targets[:, bbox_preds.shape[1]:, :, :]  # Keypoint targets follow
            
            # Compute loss
            loss = compute_loss(bbox_preds, bbox_targets, keypoint_preds, keypoint_targets)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Log epoch loss
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

# Parameters
img_dir = 'C:/Users/paulb/Documents/TUDresden/Bachelor/dataset/images'
annotations_folder = 'C:/Users/paulb/Documents/TUDresden/Bachelor/dataset/annotations'
model_path = 'C:/Users/paulb/Documents/TUDresden/Bachelor/output/detector.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset with transforms
transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])
dataset = LegoKeypointDataset(annotations_folder, img_dir, transform=transform)

# Model
model = LegoDetector(num_anchors=9, num_keypoints=8)

# Training
train_model(
    model=model,
    dataset=dataset,
    epochs=10,
    batch_size=4,
    learning_rate=0.001,
    device=device
)
