import bpy
import random
import os
import mathutils
import math
import uuid
import time

print('----------------')

# Replace 'CollectionName' with the name of your collection
collection_name = 'Parts'
camera_collection_name = 'In_Camera'
line_collection_name = 'Lines'
# Backgrounds images folder path
backgrounds_folder_path = '//Backgrounds'

# Minimum and Maximum amount of items to be placed in the camera collection
min_items = 1
max_items = 10


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
    
    start_time = time.time()
    
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
            for corner in corners.values():
                if is_point_in_camera_view(camera, corner):
                    continue
                else:
                    to_be_removed.add(obj)
                    break
        
        remove_objects(to_be_removed)
        
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        
        
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
    upper_right = frame[0]
    lower_right = frame[1]
    lower_left = frame[2]
    upper_left = frame[3]
    
    return lower_left, lower_right, upper_right, upper_left

def is_point_in_camera_view(camera, point_world):
    # Transform the point to the camera's local space
    point_camera_space = camera.matrix_world.inverted() @ point_world
    
    # Get camera data
    cam_data = camera.data
    
    # Create Plane objects for each side of the camera frustum
    lower_left, lower_right, upper_right, upper_left = get_camera_view_bounds(bpy.context.scene, camera, 1)
    bottom_plane_data = define_plane_from_vertices(lower_left, lower_right, camera.location)
    right_plane_data = define_plane_from_vertices(lower_right, upper_right, camera.location)
    top_plane_data = define_plane_from_vertices(upper_right, upper_left, camera.location)
    left_plane_data = define_plane_from_vertices(upper_left, lower_left, camera.location)
    
    planes = [bottom_plane_data, right_plane_data, top_plane_data, left_plane_data]
    
    for plane in planes:
        normal, point = plane
        if is_point_above_plane(point_camera_space, point, normal):
            return False
    
    return True

def is_point_above_plane(point, plane_point, normal):
    # Calculate the vector from the point on the plane to the point of interest
    vector = [point[i] - plane_point[i] for i in range(3)]
    # Dot product with the plane's normal vector
    dot_product = sum(vector[i] * normal[i] for i in range(3))
    return dot_product > 0  # Returns True if above, False if below or on the plane

def create_plane_with_normal(normal, point, name="CustomPlane"):
    # Normalize the normal vector
    normal = mathutils.Vector(normal).normalized()
    point = mathutils.Vector(point)

    # Create a new plane
    bpy.ops.mesh.primitive_plane_add(size=5)
    plane = bpy.context.object
    plane.name = name

    # Set the plane location to the given point
    plane.location = point

    # Calculate the rotation to align the plane's normal to the given normal vector
    up = mathutils.Vector((0, 0, 1))
    rotation = up.rotation_difference(normal).to_euler()
    plane.rotation_euler = rotation
    
    # unlink the plane from the scene collection
    bpy.context.collection.objects.unlink(plane)
    
    line_collection.objects.link(plane)
    return plane
    
def define_plane_from_vertices(v1, v2, v3):
    
    # Calculate two vectors in the plane
    vec1 = v2 - v1
    vec2 = v3 - v1
    
    # Calculate the normal vector (cross product of vec1 and vec2)
    normal = vec1.cross(vec2)
    A, B, C = normal
    
    # Calculate D using the point-normal form of the plane equation
    D = -normal.dot(v1)
    
    return normal, v1

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
        horizontal_fov = cam_data.angle_x
    else:  # 'HORIZONTAL' or automatic fit
        horizontal_fov = cam_data.angle_x
        vertical_fov = cam_data.angle_y

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
        # print(f"Removing object: {obj.name}")
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


