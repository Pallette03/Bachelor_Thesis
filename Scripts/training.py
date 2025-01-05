import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from LegoKeypointDataset import LegoKeypointDataset
import torch.nn as nn

num_keypoints = 8
batch_size = 16
num_epochs = 5
learning_rate = 0.001

render_images_folder = 'C:/Users/paulb/Documents/TUDresden/Bachelor/dataset/images'
annotations_folder = 'C:/Users/paulb/Documents/TUDresden/Bachelor/dataset/annotations'
model_path = 'C:/Users/paulb/Documents/TUDresden/Bachelor/output/detector.pth'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


dataset = LegoKeypointDataset(annotations_folder, render_images_folder, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


model = KeypointDetector().to('cuda')
#model = LegoKeypointDetector().to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, keypoints in dataloader:
        images = images.to('cuda')
        keypoints = keypoints.to('cuda')

        outputs = model(images, 10*8)
        loss = criterion(outputs, keypoints)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}")

model.eval() 
total_loss = 0
criterion = nn.MSELoss()

with torch.no_grad(): 
    for images, keypoints in dataloader:
        images = images.to('cuda')
        keypoints = keypoints.to('cuda')

        outputs = model(images, 10*8) 

        loss = criterion(outputs, keypoints)
        

        total_loss += loss.item()

average_loss = total_loss / len(dataloader)
print(f"Validation Loss: {average_loss:.4f}")

if average_loss < best_val_loss:
    best_val_loss = average_loss
    torch.save(model.state_dict(), model_path)
    print("Saved new best model!")