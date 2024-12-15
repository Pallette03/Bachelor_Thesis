import os
import json
import numpy as np

annotations_folder = 'C:/Users/paulb/Documents/TUDresden/Bachelor/dataset/annotations'

def draw_points_on_rendered_image(file_name):
    
    # Get the annotations
    annotations_file_path = os.path.join(os.path.abspath(annotations_folder), f"{file_name}.json")
    with open(annotations_file_path, mode='r') as file:
        data = json.load(file)
        annotations = data['annotations']
        camera_matrix = data['camera_matrix']
        # in the form of [[-1.0, 0.0, -8.742277657347586e-08], [0.0, 1.0, 0.0], [8.742277657347586e-08, 0.0, -1.0], [0.0, 0.0, 0.0]] convert to numpy matrix
        camera_matrix = np.array(camera_matrix)

        for annotation in annotations:
            obj_name = annotation['brick_type']
            corners = annotation['keypoints']
            color = annotation['color']
            
            for name, coordinates in corners.items():
                coordinates = np.array(coordinates)
                coordinates = np.dot(camera_matrix, coordinates)
                print(f"{name}: {coordinates}")
            
            
draw_points_on_rendered_image("13122024-181208-433")