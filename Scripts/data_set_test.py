from LegoKeypointDataset import LegoKeypointDataset
import matplotlib.pyplot as plt

image_dir = 'C:/Users/paulb/Documents/TUDresden/Bachelor/dataset/images/rgb'
annotation_dir = 'C:/Users/paulb/Documents/TUDresden/Bachelor/dataset/annotations'

# Prepare dataset and dataloader
dataset = LegoKeypointDataset(annotation_dir, image_dir, None)

# Get the first image and heatmap
image, heatmap = dataset[0]

# Visualize the image and heatmap
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image.permute(1, 2, 0))
plt.title('Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(heatmap.squeeze())
plt.title('Heatmap')
plt.axis('off')
plt.show()
#
