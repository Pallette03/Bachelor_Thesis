import bpy
import random
import os
import mathutils
import math
import uuid

print('----------------')

# Replace 'CollectionName' with the name of your collection
collection_name = 'Parts'
camera_collection_name = 'In_Camera'
line_collection_name = 'Lines'
# Backgrounds images folder path
backgrounds_folder_path = '//Backgrounds'

# Minimum and Maximum amount of items to be placed in the camera collection
min_items = 1
max_items = 1


# Get the collection
collection = bpy.data.collections.get(collection_name)
camera_collection = bpy.data.collections.get(camera_collection_name)
line_collection = bpy.data.collections.get(line_collection_name)

# Get the Camera
camera = bpy.data.objects.get('Camera')

min_z = 0.3  # Distance from camera to near plane
max_z = 0.9  # Distance to far plane (adjust as needed)

# Get the amount of images in the backgrounds folder
image_extensions = ('.png', '.jpg', '.jpeg')
background_images = [f for f in os.listdir(bpy.path.abspath(backgrounds_folder_path)) if f.lower().endswith(image_extensions)]

# Set defining the items to be removed due to occlusion or other reasons
to_be_removed = set()


def main():
    if collection and camera_collection and camera and line_collection:
        print(f"Everything found.")
        
        # Set random Background Image
        random_background = random.choice(background_images)
        img = bpy.data.images.load(os.path.join(bpy.path.abspath(backgrounds_folder_path), random_background))
        camera.data.background_images.clear()
        bg = camera.data.background_images.new()
        bg.image = img
        camera.data.show_background_images = True
        
        # Clear the camera collection
        for obj in camera_collection.objects:
            camera_collection.objects.unlink(obj)
            bpy.data.objects.remove(obj)
            
        # Clear the line collection
        for obj in line_collection.objects:
            line_collection.objects.unlink(obj)
            bpy.data.objects.remove(obj)
        
        # Get a random amount of objects from the Parts collection
        random_objects = []
        for i in range(random.randint(min_items, max_items)):
            if i > 100:
                break
            random_objects.append(random.choice(collection.objects))
        
        # Copy the object to the camera collection
        for obj in random_objects:
            new_obj = obj.copy()
            new_obj.data = obj.data.copy()
            camera_collection.objects.link(new_obj)
            # Set the location of the new object within the camera's view frustum 
            random_attributes_object(new_obj)
        
        # Force update
        bpy.context.view_layer.update()
        
        for obj in camera_collection.objects:
            occluding_objects = get_occluding_objects(camera, obj)
            for occluding_obj in occluding_objects:
                to_be_removed.add(occluding_obj)
            
        remove_objects(to_be_removed)
                
        for obj in camera_collection.objects:
            # Get the object's bounding box corners in world space
            corners = get_corners_of_object(obj)
            print(f"Object: {obj.name}")
            is_point_in_camera_view(camera, obj.location)
            
            
            
            
            
            
        
    else:
        if not collection:
            print(f"Collection '{collection_name}' not found.")
        if not camera_collection:
            print(f"Collection '{camera_collection_name}' not found.")
        if not camera:
            print(f"Camera not found.")
        if not line_collection:
            print(f"Collection '{line_collection_name}' not found.")
    

def draw_bounding_box(obj, box_name="BoundingBox"):
    # Get the bounding box corners in world space
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]

    # Create a new mesh and object for the bounding box
    mesh = bpy.data.meshes.new(box_name)
    bbox_obj = bpy.data.objects.new(box_name, mesh)
    line_collection.objects.link(bbox_obj)

    # Define vertices and edges for the bounding box
    vertices = bbox_corners
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    # Create the mesh
    mesh.from_pydata(vertices, edges, [])
    mesh.update()

    return bbox_obj


def get_camera_view_bounds(scene, camera_obj, depth):
    """
    Get the 4 corners of the camera view at a certain depth.
    """
    cam_data = camera_obj.data
    frame = cam_data.view_frame(scene=scene)
    
    # Adjust the frame based on depth
    frame = [camera_obj.matrix_world @ (v * depth) for v in frame]
    
    # Extract the corners
    lower_left = frame[0]
    lower_right = frame[1]
    upper_right = frame[2]
    upper_left = frame[3]
    
    return lower_left, lower_right, upper_right, upper_left

def is_point_in_camera_view(camera, point_world):
    # Transform the point to the camera's local space
    point_camera_space = camera.matrix_world.inverted() @ point_world
    
    # Get camera data
    cam_data = camera.data
    horizontal_fov, vertical_fov = get_camera_opening_angle(camera)
    print(f"Horizontal FOV: {horizontal_fov}, Vertical FOV: {vertical_fov}")
    
    beta = -(vertical_fov / 2) + 90
    c_y = point_camera_space.z / math.sin(beta) # correct
    a_y = math.sqrt(-point_camera_space.z**2 + c_y**2) # correct
    border_y = camera.location.y + a_y # incorrect
    
    print(f"Beta: {beta}, C_y: {c_y}, A_y: {a_y}")
    
    alpha = -(horizontal_fov / 2) + 90
    c_x = point_camera_space.z / math.sin(alpha) # correct
    a_x = math.sqrt(-point_camera_space.z**2 + c_x**2) # correct
    border_x = camera.location.x + a_x # incorrect
    print(f"Alpha: {alpha}, C_x: {c_x}, A_x: {a_x}")
    
    print(f"Point: {point_world}, Camera: {camera.location}, Border X: {border_x}, Border Y: {border_y}")
    
    

def get_camera_opening_angle(camera):
    # Check if the object is a camera
    if camera.type != 'CAMERA':
        raise ValueError("The object is not a camera")

    # Get the camera data
    cam_data = camera.data
    aspect_ratio = cam_data.sensor_width / cam_data.sensor_height
    
    # Check sensor fit and get field of view (FOV)
    if cam_data.sensor_fit == 'VERTICAL':
        vertical_fov = cam_data.angle_y
        horizontal_fov = 2 * math.atan(math.tan(vertical_fov / 2) * aspect_ratio)
    else:  # 'HORIZONTAL' or automatic fit
        horizontal_fov = cam_data.angle_x
        vertical_fov = 2 * math.atan(math.tan(horizontal_fov / 2) / aspect_ratio)

    # Convert radians to degrees if needed
    horizontal_fov_deg = math.degrees(horizontal_fov)
    vertical_fov_deg = math.degrees(vertical_fov)
    
    return horizontal_fov_deg, vertical_fov_deg


def random_point_in_camera_view(scene, camera_obj, depth):
    """
    Get a random point within the camera frustum at a given depth.
    """
    lower_left, lower_right, upper_right, upper_left = get_camera_view_bounds(scene, camera_obj, depth)
    # Random interpolation of the edges of the view
    random_x = random.uniform(0, 1)
    random_y = random.uniform(0, 1)
    
    # Linear interpolation between the corners
    point_on_left = lower_left.lerp(upper_left, random_y)
    point_on_right = lower_right.lerp(upper_right, random_y)
    
    # Final point inside the frustum
    random_point = point_on_left.lerp(point_on_right, random_x)
    return random_point

def random_attributes_object(obj):
    """
    Randomly rotate an object.
    """
    # Random location
    obj.location = random_point_in_camera_view(bpy.context.scene, camera, random.uniform(min_z, max_z))
    
    # Random rotation
    obj.rotation_euler = mathutils.Euler((random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi)))
    
    colours = [(255,0,0), (0,255,0), (0,0,255)]
    
    # Random colour adjust material
    mat = copy_simple_mat(random.choice(colours), obj.data.materials[0])
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    
    return obj

def to_blender_color(c):    # gamma correction
    c = min(max(0, c), 255) / 255
    return c / 12.92 if c < 0.04045 else math.pow((c + 0.055) / 1.055, 2.4)

# function to create a material that not assign to any object
def copy_simple_mat(rgb, material):
    new_mat = material.copy()
    unique_id = str(uuid.uuid4())
    new_mat.name = material.name + '_copy_' + unique_id
    
    
    # Set the new colour
    new_mat.diffuse_color = (to_blender_color(rgb[0]), to_blender_color(rgb[1]), to_blender_color(rgb[2]), 1)
    
    return new_mat


def get_occluding_objects(camera, target, draw_line=False):
    # Get the position of the camera and target
    cam_location = camera.location
    target_location = target.location

    # Vector from camera to target
    cam_to_target_vec = target_location - cam_location
    cam_to_target_distance = cam_to_target_vec.length
    

    objects_in_between = []

    # Loop through all objects in the scene
    for obj in camera_collection.objects:
        # Ignore the camera and target itself
        if obj == camera or obj == target:
            continue

        # Vector from camera to the current object
        cam_to_obj_vec = obj.location - cam_location

        # Project the object onto the camera-to-target line
        projection_length = cam_to_obj_vec.dot(cam_to_target_vec.normalized())
        
        # Check if the object is in between camera and target along the line
        if 0 < projection_length < cam_to_target_distance:
            # Check if the object is close to the camera-target line
            distance_to_line = (cam_to_obj_vec - cam_to_target_vec.normalized() * projection_length).length
            
            # Adjust this threshold for tolerance in distance to the line
            if distance_to_line < 0.05:
                objects_in_between.append(obj)

    if draw_line and objects_in_between:
        # Draw a line from the camera to the target
        line = draw_line_meth(cam_location, cam_to_target_vec, cam_to_target_distance, line_name=f"CameraToTarget_{target.name}")
        line_collection.objects.link(line)

    return objects_in_between


def draw_line_meth(start, direction, length, line_name="Line"):
    # Calculate the end point of the line
    end = start + direction.normalized() * length

    # Create mesh and object
    mesh = bpy.data.meshes.new(line_name)
    obj = bpy.data.objects.new(line_name, mesh)

    # Define vertices and edges
    vertices = [start, end]
    edges = [(0, 1)]
    
    # Create the mesh
    mesh.from_pydata(vertices, edges, [])
    mesh.update()
    
    # Change the line color to red
    mat = bpy.data.materials.new(name="LineMaterial")
    mat.diffuse_color = (1, 0, 0, 1)
    obj.data.materials.append(mat)
    
    return obj

def remove_objects(obj_set: set):
    for obj in obj_set:
        print(f"Removing object: {obj.name}")
        camera_collection.objects.unlink(obj)
        bpy.data.objects.remove(obj)
    
    obj_set.clear()
    
    
    
    
def position_relative_to_camera(camera, obj):
    # Convert the object's world location to the camera's local space
    relative_position = camera.matrix_world.inverted() @ obj.location
    return relative_position

def get_corners_of_object(obj):
    # Get the object's bounding box corners in world space and put them in a dictionary
    corners = {f"Corner_{i}": obj.matrix_world @ mathutils.Vector(corner) for i, corner in enumerate(obj.bound_box)}
    return corners
    

main()
remove_objects(to_be_removed)


