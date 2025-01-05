import os
import json
import numpy as np
import cv2

annotations_folder = 'C:/Users/paulb/Documents/TUDresden/Bachelor/dataset/annotations'

def denormalize_keypoints(keypoints, image_width, image_height):
    denormalized_keypoints = {}
    for corner_name, corner_vector in keypoints.items():
        x = corner_vector[0] * image_width
        y = corner_vector[1] * image_height
        denormalized_keypoints[corner_name] = [x, y]
    return denormalized_keypoints

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
            
            heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            for corner_name, corner_vector in denormalized_keypoints.items():
                x, y = corner_vector
                x, y = int(x), int(y)
                heatmap[y, x] = 1.0
                # Draw a circle on the heatmap
                cv2.circle(image, (x, y), radius=3, color=(255, 0, 0), thickness=-1)
                
                
            # Show the heatmap
            cv2.imshow("Heatmap", heatmap)
            cv2.imshow("Image", image)
            cv2.waitKey(0)
            
            
draw_points_on_rendered_image("04012025-005453-98")