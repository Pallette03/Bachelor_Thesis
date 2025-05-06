from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def validate_image(img_path):
    """
    Validate a single image file.
    """
    try:
        with Image.open(img_path) as img:
            img.verify()  # Verify that the image is not corrupted
        return img_path, None  # Return file path and no error
    except (IOError, SyntaxError) as e:
        return img_path, str(e)  # Return file path and the error

def validate_images_parallel(image_dir, num_workers=4):
    """
    Validate all images in a directory in parallel.
    """
    image_paths = [os.path.join(image_dir, img_file) for img_file in os.listdir(image_dir)]
    corrupted_images = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_image = {executor.submit(validate_image, img_path): img_path for img_path in image_paths}

        for future in as_completed(future_to_image):
            img_path, error = future.result()
            if error:  # Log corrupted images
                print(f"Corrupted image: {img_path} - Error: {error}")
                corrupted_images.append(img_path)

    return corrupted_images

# Specify your directory path and number of threads
train_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'gaussian_noise', 'train', 'images', 'rgb')
validate_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'gaussian_noise', 'validate', 'images', 'rgb')
train_annotations_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'gaussian_noise', 'train', 'annotations')
validate_annotations_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'gaussian_noise', 'validate', 'annotations')
num_workers = 4  # Adjust based on your CPU core count
corrupted_images = validate_images_parallel(train_dir, num_workers)
validate_corrupted_images = validate_images_parallel(validate_dir, num_workers)

print(f"Total corrupted images: {len(corrupted_images)}")
print(f"Total corrupted images in validation set: {len(validate_corrupted_images)}")

print("Removing corrupted images...")
for img_path in corrupted_images:
    print(f"Removing corrupted image: {img_path}")
    os.remove(img_path)
    # Optionally, you can also remove the corresponding annotation files if needed
    annotation_path = os.path.join(train_annotations_dir, os.path.basename(img_path).replace('.png', '.json'))
    os.remove(annotation_path)
    
for img_path in validate_corrupted_images:
    print(f"Removing corrupted image: {img_path}")
    os.remove(img_path)
    # Optionally, you can also remove the corresponding annotation files if needed
    annotation_path = os.path.join(validate_annotations_dir, os.path.basename(img_path).replace('.png', '.json'))
    os.remove(annotation_path)
    
    
    
corrupted_images = validate_images_parallel(train_dir, num_workers)
validate_corrupted_images = validate_images_parallel(validate_dir, num_workers)
print(f"Total corrupted images after removal: {len(corrupted_images)}")
print(f"Total corrupted images in validation set after removal: {len(validate_corrupted_images)}")
