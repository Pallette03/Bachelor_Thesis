import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

from rotation_dataset import RotationDataset
from viewpoint_estimator import ViewpointEstimator

root_dir='C:\\Users\\paulb\\Documents\\TUDresden\\Bachelor\\datasets\\rotation_estimation'
model_path = "C:\\Users\\paulb\\Documents\\TUDresden\\Bachelor\\output\\viewpoint_estimator.pth"
epochs=15

def evaluate(model, dataloader):
    model.eval()
    correct_azimuth, correct_elevation, correct_rotation = 0, 0, 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.cuda(), labels.cuda()
            azimuth_labels, elevation_labels, rotation_labels = labels[:, 0], labels[:, 1], labels[:, 2]

            azimuth_pred, elevation_pred, rotation_pred = model(images)
            _, azimuth_pred = torch.max(azimuth_pred, 1)
            _, elevation_pred = torch.max(elevation_pred, 1)
            _, rotation_pred = torch.max(rotation_pred, 1)
            
            correct_azimuth += (azimuth_pred == azimuth_labels).sum().item()
            correct_elevation += (elevation_pred == elevation_labels).sum().item()
            correct_rotation += (rotation_pred == rotation_labels).sum().item()
            total += labels.size(0)

    print(f"Azimuth Accuracy: {correct_azimuth / total:.4f}")
    print(f"Elevation Accuracy: {correct_elevation / total:.4f}")
    print(f"Rotation Accuracy: {correct_rotation / total:.4f}")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization for VGG
])

# Initialize dataset, model, and optimizer
dataset = RotationDataset(root_dir, transform=transform, bin_size=5)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = ViewpointEstimator(num_azimuth_bins=360 // 5, 
                        num_elevation_bins=180 // 5, 
                        num_rotation_bins=360 // 5).cuda()

model.load_state_dict(torch.load(model_path))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
#scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Training loop
for epoch in range(epochs):
    model.train()
    for images, labels in dataloader:
        images, labels = images.cuda(), labels.cuda()
        azimuth_labels, elevation_labels, rotation_labels = labels[:, 0], labels[:, 1], labels[:, 2]
        
        # Forward pass
        azimuth_pred, elevation_pred, rotation_pred = model(images)

        # Compute losses
        loss_azimuth = criterion(azimuth_pred, azimuth_labels)
        loss_elevation = criterion(elevation_pred, elevation_labels)
        loss_rotation = criterion(rotation_pred, rotation_labels)
        loss = loss_azimuth + loss_elevation + loss_rotation

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    evaluate(model, val_loader)

print("Training complete")
print("Saving model to disk")
torch.save(model.state_dict(), model_path)
evaluate(model, val_loader)
