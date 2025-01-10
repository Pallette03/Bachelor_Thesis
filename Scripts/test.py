import os
import json
import numpy as np
import cv2

annotations_folder = 'C:/Users/paulb/Documents/TUDresden/Bachelor/dataset/annotations'
img_dir = 'C:/Users/paulb/Documents/TUDresden/Bachelor/dataset/images/rgb'

# Harris Corner Detector
def harris_corner_detector(image, threshold=0.01):
    # Convert PIL image to OpenCV image
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    image[dst > threshold * dst.max()] = [0, 0, 255]
    return image

def denormalize_keypoints(keypoints, image_width, image_height):
    denormalized_keypoints = {}
    for corner_name, corner_vector in keypoints.items():
        x = corner_vector[0] * image_width
        y = corner_vector[1] * image_height
        denormalized_keypoints[corner_name] = [x, y]
    return denormalized_keypoints

def demormalize_coordinates(normal_pixel_coordinates, image_width, image_height):
    x = normal_pixel_coordinates[0] * image_width
    y = normal_pixel_coordinates[1] * image_height
    return [x, y]

def draw_points_on_rendered_image(file_name):
    
    # Get the annotations
    annotations_file_path = os.path.join(os.path.abspath(annotations_folder), f"{file_name}.json")
    with open(annotations_file_path, mode='r') as file:
        data = json.load(file)
        annotations = data['annotations']
        camera_matrix = data['camera_matrix']
        # in the form of [[-1.0, 0.0, -8.742277657347586e-08], [0.0, 1.0, 0.0], [8.742277657347586e-08, 0.0, -1.0], [0.0, 0.0, 0.0]] convert to numpy matrix
        camera_matrix = np.array(camera_matrix)
        image = cv2.imread(f"C:/Users/paulb/Documents/TUDresden/Bachelor/dataset/images/rgb/{file_name}.png")
        
        for annotation in annotations:
            obj_name = annotation['brick_type']
            corners = annotation['keypoints']
            normal_pixel_coordinates = annotation['normal_pixel_coordinates']
            denormalized_keypoints = denormalize_keypoints(normal_pixel_coordinates, image.shape[1], image.shape[0])
            color = annotation['color']
            
            bb_box = annotation['bb_box']
            
            # Draw bounding box
            top_left = demormalize_coordinates(bb_box["top_left"], image.shape[1], image.shape[0])
            bottom_right = demormalize_coordinates(bb_box["bottom_right"], image.shape[1], image.shape[0])
            cv2.rectangle(image, (int(top_left[0]), int(top_left[1])), (int(bottom_right[0]), int(bottom_right[1])), (0, 255, 0), 2)
        
        cv2.imshow('image', image)
        cv2.waitKey(0)
   
#draw_points_on_rendered_image("10012025-115014-192")

for f in os.listdir(img_dir):
    file_name = f.split(".")[0]
    draw_points_on_rendered_image(file_name)
    continue