
import os
from PIL import Image
import shutil


image_dir = 'C:/Users/paulb/Documents/TUDresden/Bachelor/datasets/object_detection/images/rgb'
annotation_dir = 'C:/Users/paulb/Documents/TUDresden/Bachelor/datasets/object_detection/annotations'
annotation_cropped_dir = 'C:/Users/paulb/Documents/TUDresden/Bachelor/datasets/cropped_objects/annotations'
cropped_dir = 'C:/Users/paulb/Documents/TUDresden/Bachelor/datasets/cropped_objects/images/rgb'

def crop_images(image_dir, cropped_dir, annotation_dir, annotation_cropped_dir, new_size=(224, 224)):
    # file_amount = len(os.listdir(image_dir))
    # print(f'Found {file_amount} files in {image_dir}')
    # counter = 0
    # for filename in os.listdir(annotation_dir):
    #     src_file = os.path.join(annotation_dir, filename)
    #     dest_file = os.path.join(annotation_cropped_dir, filename)
        
    #     # Check if it is a file
    #     if os.path.isfile(src_file):
    #         shutil.copy(src_file, dest_file)
    #         counter += 1
    #     if counter % 50 == 0:
    #         print(f'Processed {counter}/{file_amount} annotations.')
        
    # print(f"Annotations copied to {annotation_cropped_dir}")
    processed_files = os.listdir(cropped_dir)
    
    file_amount = len(os.listdir(image_dir))
    print(f'Found {file_amount} files in {image_dir}')
    counter = 0
    for filename in os.listdir(image_dir):
        if filename in processed_files:
            print(f"Skipping {filename}")
            counter += 1
            continue
        elif filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            image = Image.open(image_path)
            image = image.resize(new_size)
            image.save(os.path.join(cropped_dir, filename))
            counter += 1
            # Close the image
            image.close()
            
        if counter % 50 == 0:
            print(f'Processed {counter}/{file_amount} images.')
            
            
crop_images(image_dir, cropped_dir, annotation_dir, annotation_cropped_dir, new_size=(600, 600))