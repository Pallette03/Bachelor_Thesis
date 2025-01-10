import json
import math
import mathutils
import random
import bpy # type: ignore
import bpy_extras # type: ignore
import os
import sys

dir = os.path.dirname(bpy.data.filepath)
dir = os.path.join(dir, 'Scripts')
if not dir in sys.path:
    sys.path.append(dir )

import util_functions

uf = util_functions.Util_functions()

rotation_estimation_folder = '//datasets/rotation_estimation'
hdri_folder = '//hdri'

collection_name = 'Parts'
camera_collection_name = 'In_Camera'
line_collection_name = 'Lines'

rendered_images_amount = 1
will_render_image = True


bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080
bpy.context.scene.use_nodes = True

# Get the collection
collection = bpy.data.collections.get(collection_name)
camera_collection = bpy.data.collections.get(camera_collection_name)
line_collection = bpy.data.collections.get(line_collection_name)
# Get the Camera
camera = bpy.data.objects.get('Camera')

image_extensions = ('.hdr', '.exr')
background_images = [f for f in os.listdir(bpy.path.abspath(hdri_folder)) if f.lower().endswith(image_extensions)]

def main(renderes_images_amount=1):
    

    hdri_image_path = os.path.join(bpy.path.abspath(hdri_folder), random.choice(background_images))
    uf.load_hdri_image(hdri_image_path)

    if collection and camera_collection and camera and line_collection:
        for obj in collection.objects:
            print(f"Object found: {obj.name}")
            # Clear the camera collection
            for cam_obj in camera_collection.objects:
                camera_collection.objects.unlink(cam_obj)
                bpy.data.objects.remove(cam_obj)

            # Clear the line collection
            for line_obj in line_collection.objects:
                line_collection.objects.unlink(line_obj)
                bpy.data.objects.remove(line_obj)
                
            new_obj = obj.copy()
            new_obj.data = obj.data.copy()
            # Set new object location to origin
            new_obj.location = (0, 0, 0)
            camera_collection.objects.link(new_obj)
            bpy.context.view_layer.update()
            
            create_rotation_images(new_obj, renderes_images_amount, camera, rotation_estimation_folder)
            
            
def create_rotation_images(obj, renderes_images_amount, camera, rotation_estimation_folder):
    print(f"Creating {renderes_images_amount} images for object {obj.name}")
    brick_type = obj.name.split(".")[0]
    images_folder = os.path.join(bpy.path.abspath(rotation_estimation_folder), brick_type, 'images')
    annotations_folder = os.path.join(bpy.path.abspath(rotation_estimation_folder), brick_type, 'annotations')
    
    name_number = 0
    for f in os.listdir(images_folder):
        if f.endswith('.png'):
            name_number += 1
    
    for i in range(renderes_images_amount):
        print(f"Rendering image {i+1} of {renderes_images_amount}")
        
        file_name = f"{brick_type}_{name_number}.png"
        bpy.context.scene.render.filepath = os.path.join(bpy.path.abspath(images_folder), file_name)
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        
        azimuth, elevation, optical_rotation = place_camera_randomly(obj, camera)
        create_annotation_file(brick_type, file_name, annotations_folder, azimuth, elevation, optical_rotation)
        
        bpy.ops.render.render(write_still=True)
        
        name_number += 1
        
        
def place_camera_randomly(obj, camera, min_distance=0.2, max_distance=2):
    """
    Places the camera at a random position around the object with the optical axis pointing at the object's origin.

    Args:
        obj (bpy.types.Object): The target object.
        camera (bpy.types.Object): The camera object.
        min_distance (float): Minimum distance from the object.
        max_distance (float): Maximum distance from the object.
    """
    # Object origin
    obj_location = obj.location
    
    # Random distance from the object
    distance = random.uniform(min_distance, max_distance)
    
    # Random azimuth (angle around the object) and elevation
    azimuth = random.uniform(0, 2*math.pi)  # 0 to 360 degrees
    elevation = random.uniform(-math.pi/2, math.pi/2)  # -90 to 90 degrees
    
    # Spherical to Cartesian conversion
    x = distance * math.cos(elevation) * math.cos(azimuth)
    y = distance * math.cos(elevation) * math.sin(azimuth)
    z = distance * math.sin(elevation)
    
    # Set camera location
    camera.location = obj_location + mathutils.Vector((x, y, z))
    
    # Point the camera at the object's origin
    direction = obj_location - camera.location
    rotation = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rotation.to_euler()
    
    # Randomly rotate the camera around the optical axis
    optical_rotation = random.uniform(0, 2*math.pi)
    camera.rotation_euler.rotate_axis('Z', optical_rotation)
    
    return math.degrees(azimuth), math.degrees(elevation), math.degrees(optical_rotation)
    
def split_into_bins(azimuth, elevation, optical_rotation, bin_size=5):
    elevation = elevation + 90
    
    azimuth_bin = int(azimuth // bin_size)
    elevation_bin = int(elevation // bin_size)
    optical_rotation_bin = int(optical_rotation // bin_size)
    
    return azimuth_bin, elevation_bin, optical_rotation_bin
     
     
def create_annotation_file(brick_type, img_name, annotations_folder, azimuth, elevation, optical_rotation):
    annotation_file_path = os.path.join(annotations_folder, f"{brick_type}.json")
    
    data = {
        "img_name": img_name,
        "azimuth": azimuth,
        "elevation": elevation,
        "optical_rotation": optical_rotation,
    }
    
    # If file exists, append to it
    if os.path.exists(annotation_file_path):
        with open(annotation_file_path, mode='r') as file:
            annotations = json.load(file)
            annotations.append(data)
            
        with open(annotation_file_path, mode='w') as file:
            json.dump(annotations, file, indent=4)
    else:
        with open(annotation_file_path, mode='w') as file:
            json.dump([data], file, indent=4)
        
    print(f"Annotation file saved at {annotation_file_path}")
    
main(renderes_images_amount=rendered_images_amount)
