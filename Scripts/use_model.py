import torch
from torchvision import transforms
from KeypointDetector import KeypointDetector
from PIL import Image
import matplotlib.pyplot as plt

image_name = '01122024-183039-173.png'
model_path = 'C:/Users/paulb/Documents/TUDresden/Bachelor/output/detector.pth'
image_base_path = 'C:/Users/paulb/Documents/TUDresden/Bachelor/dataset/images/'

# Define the model structure (must match the trained model)
model = KeypointDetector(input_features=64 * 28 * 28)


# Load the saved weights
model.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = Image.open(image_base_path + image_name).convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

with torch.no_grad():
    output = model(input_tensor, max_keypoints=64)

keypoints = output[0]  # Remove batch dimension: shape [MaxKeypoints, 2]

# Filter out invalid keypoints
valid_keypoints = keypoints[keypoints[:, 0] != -1]  # Only keep valid (non-placeholder) keypoints

# Convert keypoints back to image coordinates (if necessary)
width, height = image.size
valid_keypoints[:, 0] *= width   # Scale x-coordinates
valid_keypoints[:, 1] *= height  # Scale y-coordinates

# Plot the image and keypoints
plt.imshow(image)
for x, y in valid_keypoints:
    plt.scatter(x.item(), y.item(), color="red", s=10)
plt.show()