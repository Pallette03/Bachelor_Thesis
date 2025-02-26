import os
import cv2
from PIL import Image
import json


annotations_folder = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'cropped_objects', 'annotations')
img_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets', 'cropped_objects', 'images', 'rgb')

img_path = os.path.join(img_dir, '26022025-123822-502.png')

def denormalize_keypoints(keypoints, image_width, image_height):
        denormalized_keypoints = {}
        for corner_name, corner_data in keypoints.items():
            x = corner_data[0][0] * image_width
            y = corner_data[0][1] * image_height
            denormalized_keypoints[corner_name] = ([x, y], corner_data[1])
        return denormalized_keypoints

def draw_points_on_rendered_image(image_path, annotations_folder):
        # Load the image
        img_cv2 = cv2.imread(image_path)
        img = Image.open(image_path)

        #img height
        img_height = img.size[1]

        file_name = image_path.split("/")[-1].split(".")[0]

        # Get the annotations
        annotations_file_path = os.path.join(annotations_folder, f"{file_name}.json")
        with open(annotations_file_path, mode='r') as file:
            data = json.load(file)
            annotations = data['annotations']
            for annotation in annotations:
                obj_name = annotation['brick_type']
                corners = annotation['normal_pixel_coordinates']
                color = annotation['color']

                denormalized_corners = denormalize_keypoints(corners, img.size[0], img.size[1])

                for corner_name, corner_data in denormalized_corners.items():
                    x, y = corner_data[0]
                    if corner_data[1]:
                        cv2.circle(img_cv2, (int(x), int(y)), 1, (0, 255, 0), -1)
                    else:
                        cv2.circle(img_cv2, (int(x), int(y)), 1, (0, 0, 255), -1)

        # Save the image
        cv2.imwrite(image_path, img_cv2)


draw_points_on_rendered_image(img_path, annotations_folder)

