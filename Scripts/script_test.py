import bpy
import bpy_extras
import random
import os
import mathutils
import numpy as np
import cv2
import math
import uuid
import time
import json

debug = False
print('----------------')

if debug:
    import debugpy
    try:
        debugpy.listen(("localhost", 5678))
        print('Listening')
    except RuntimeError as e:
        if not str(e).__contains__('debugpy.listen() has already been called on this process'):
            raise e
        else:
            print('Already listening')
    print('Waiting for debugger attach')
    debugpy.wait_for_client()
    print('Debugger attached')
    
# Create setup method 
# - set up camera location
# - load parts collection
# - create composition nodes

collection_name = 'Parts'
camera_collection_name = 'In_Camera'
line_collection_name = 'Lines'

# Minimum and Maximum amount of items to be placed in the camera collection
min_items = 1
max_items = 10
rendered_images_amount = 1

# Set the render resolution
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080

# Set output file path and format
will_render_image = True
draw_on_image = False
fill_to_max_items = False
render_images_folder = '//dataset/images/rgb'
annotations_folder = '//dataset/annotations'
hdri_folder = '//hdri'

bpy.context.scene.use_nodes = True


# Get the collection
collection = bpy.data.collections.get(collection_name)
camera_collection = bpy.data.collections.get(camera_collection_name)
line_collection = bpy.data.collections.get(line_collection_name)

# Get the Camera
camera = bpy.data.objects.get('Camera')

min_z = 0.3  # Distance from camera to near plane
max_z = 0.9  # Distance to far plane (adjust as needed)

# Get the amount of images in the backgrounds folder
image_extensions = ('.hdr', '.exr')
background_images = [f for f in os.listdir(bpy.path.abspath(hdri_folder)) if f.lower().endswith(image_extensions)]

# Set defining the items to be removed due to occlusion or other reasons
to_be_removed = set()


def main(file_name="rendered_image.png", fill_to_max_items=False):

    bpy.context.scene.render.filepath = os.path.join(bpy.path.abspath(render_images_folder), file_name)
    bpy.context.scene.render.image_settings.file_format = 'PNG'

    # Load the HDRI image
    hdri_image_path = os.path.join(bpy.path.abspath(hdri_folder), random.choice(background_images))
    load_hdri_image(hdri_image_path)

    if collection and camera_collection and camera and line_collection:
        print(f"Everything found.")

        # Randomly rotate the camera
        camera.rotation_euler = mathutils.Euler((random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi)))
        bpy.context.view_layer.update()
        
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

        clean_scene()

        if fill_to_max_items:
            while len(camera_collection.objects) < max_items:
                random_objects = []
                for i in range(max_items - len(camera_collection.objects)):
                    random_objects.append(random.choice(collection.objects))

                for obj in random_objects:
                    new_obj = obj.copy()
                    new_obj.data = obj.data.copy()
                    camera_collection.objects.link(new_obj)
                    # Set the location of the new object within the camera's view frustum 
                    random_attributes_object(new_obj)

                bpy.context.view_layer.update()

                clean_scene()

    else:
        if not collection:
            print(f"Collection '{collection_name}' not found.")
        if not camera_collection:
            print(f"Collection '{camera_collection_name}' not found.")
        if not camera:
            print(f"Camera not found.")
        if not line_collection:
            print(f"Collection '{line_collection_name}' not found.")


def clean_scene():
    # Checks for occlusions and objects out of bounds
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


def world_to_camera_coords(camera, world_coords):
    if not isinstance(world_coords, mathutils.Vector):
        world_coords = mathutils.Vector(world_coords)
    cam_matrix_world = camera.matrix_world
    cam_matrix_world_inv = cam_matrix_world.inverted()
    camera_relative_coords = cam_matrix_world_inv @ world_coords
    return camera_relative_coords

def world_to_pixel(scene, camera_obj, world_coords):
    # Ensure we have a valid camera
    if camera_obj.type != 'CAMERA':
        raise ValueError("Object is not a camera")
    
    # Get the camera's intrinsic matrix
    cam_data = camera_obj.data
    render = scene.render
    
    # Projection matrix (4x4)
    proj_matrix = camera_obj.calc_matrix_camera(
        render.resolution_x, 
        render.resolution_y, 
        render.pixel_aspect_x / render.pixel_aspect_y
    )
    
    # World-to-camera matrix (view matrix)
    world_to_camera_matrix = camera_obj.matrix_world.inverted()
    
    # Transform the world coordinates to camera coordinates
    camera_coords = world_to_camera_matrix @ world_coords
    
    # Transform camera coordinates to clip space
    clip_coords = proj_matrix @ camera_coords
    
    # Normalize clip coordinates to NDC
    if clip_coords.w != 0:
        ndc_coords = mathutils.Vector((
            clip_coords.x / clip_coords.w,
            clip_coords.y / clip_coords.w,
            clip_coords.z / clip_coords.w
        ))
    else:
        raise ValueError("Invalid clip coordinates, w = 0")
    
    # Convert NDC to pixel coordinates
    pixel_x = (ndc_coords.x + 1) / 2.0 * render.resolution_x
    pixel_y = (1 - ndc_coords.y) / 2.0 * render.resolution_y  # Flip y-axis for Blender's pixel space
    
    return (pixel_x, pixel_y)

def normalize_keypoints(keypoints, image_width, image_height):
    normalized_keypoints = {}
    for corner_name, corner_vector in keypoints.items():
        x = corner_vector[0] / image_width
        y = corner_vector[1] / image_height
        normalized_keypoints[corner_name] = [x, y]
    return normalized_keypoints

def load_hdri_image(img_path):
    hdri_image = bpy.ops.image.open(filepath=img_path)
    hdri_image = bpy.data.images.get(os.path.basename(img_path))
    
    # Environment Texture node
    env_node = bpy.context.scene.world.node_tree.nodes.get('Environment Texture')
    
    if not env_node:
        print("Environment Texture node not found, creating a new one")
        env_node = bpy.context.scene.world.node_tree.nodes.new('ShaderNodeTexEnvironment')
        env_node.location = (-300, 300)
        env_node.name = 'Environment Texture'
        # Connect to Background node
        bg_node = bpy.context.scene.world.node_tree.nodes.get('Background')
        if bg_node:
            bpy.context.scene.world.node_tree.links.new(bg_node.inputs[0], env_node.outputs[0])
        else:
            raise ValueError("Background node not found")
    
    env_node.image = hdri_image

def denormalize_keypoints(keypoints, image_width, image_height):
    denormalized_keypoints = {}
    for corner_name, corner_vector in keypoints.items():
        x = corner_vector[0] * image_width
        y = corner_vector[1] * image_height
        denormalized_keypoints[corner_name] = [x, y]
    return denormalized_keypoints

def write_annotations_to_file(file_name):

    with open(os.path.join(bpy.path.abspath(annotations_folder), f"{file_name}.json"), mode='a') as json_file:

        image_id = file_name
        json_file.write(f'{{"image_id": "{image_id}", ')
        json_file.write('"camera_matrix": [')
        for vector in camera.matrix_world:
            json.dump((vector[0], vector[1], vector[2]), json_file)
            if vector != camera.matrix_world[-1]:
                json_file.write(', ')
            else:
                json_file.write('], \n')
        json_file.write('"annotations": [\n')

        for obj in camera_collection.objects:
            corners = get_corners_of_object(obj)
            brick_type = obj.name.split('.')[0]
            color = obj.data.materials[0].node_tree.nodes.get("Group").inputs[0].default_value

            json_file.write(f'{{"brick_type": "{brick_type}",\n "color": "{[color[0], color[1], color[2]]}",\n "keypoints": ')

            serialized_corners = {}
            camera_corners = {}
            for corner_name, corner_vector in corners.items():
                serialized_corners.update(convert_coordinates(corner_name, corner_vector))
                temp_tuple = ()
                for val in world_to_camera_coords(camera, corner_vector):
                    temp_tuple += (val,)
                camera_corners[corner_name] = temp_tuple
            
            # serialized_corners = {}
            # for corner_name, corner_vector in corners.items():
            #     serialized_corners.update(convert_coordinates(corner_name, corner_vector))

            normalized_corners = normalize_keypoints(serialized_corners, bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y)
            #normalized_corners = normalize_keypoints(serialized_corners, bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y)
            
            json.dump(camera_corners, json_file)
            json_file.write(',\n')
            
            json_file.write('"normal_pixel_coordinates": ')
            json.dump(normalized_corners, json_file)
            
            
            if obj == camera_collection.objects[-1]:
                json_file.write('}\n')
            else:
                json_file.write('},\n')

        json_file.write(']\n}\n')
        json_file.close()

    print(f"Annotations written to file {file_name}.json")

def draw_corners(obj):
    # Get the object's bounding box corners in world space
    corners = get_corners_of_object(obj)

    # Create a new mesh object for the corners
    mesh = bpy.data.meshes.new("Corners")
    obj_corners = bpy.data.objects.new("Corners", mesh)
    line_collection.objects.link(obj_corners)

    # Define vertices and edges
    vertices = list(corners.values())
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]

    # Create the mesh
    mesh.from_pydata(vertices, edges, [])
    mesh.update()

    # Change the line color to red
    mat = bpy.data.materials.new(name="CornersMaterial")
    mat.diffuse_color = (1, 0, 0, 1)
    obj_corners.data.materials.append(mat)

    return obj_corners


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
    bottom_test = is_point_above_plane(lower_left, lower_right, camera.location, point_world)
    right_test = is_point_above_plane(lower_right, upper_right, camera.location, point_world)
    top_test = is_point_above_plane(upper_right, upper_left, camera.location, point_world)
    left_test = is_point_above_plane(upper_left, lower_left, camera.location, point_world)
    
    
    if bottom_test and right_test and top_test and left_test:
        return True
    else:
        return False

def is_point_above_plane(p1, p2, p3, test_point):
    """
    Determine on which side of the plane the test_point lies.

    :param p1: First point defining the plane (mathutils.Vector)
    :param p2: Second point defining the plane (mathutils.Vector)
    :param p3: Third point defining the plane (mathutils.Vector)
    :param test_point: Point to test (mathutils.Vector)
    """
    # Calculate the normal of the plane
    normal = (p2 - p1).cross(p3 - p1).normalized()
    
    # Calculate a vector from one plane point to the test point
    vector_to_point = test_point - p1
    
    # Compute the dot product
    dot_product = normal.dot(vector_to_point)
    
    # Determine the side
    if dot_product < 0:
        return True
    else:
        return False

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
    # Random location
    obj.location = random_point_in_camera_view(bpy.context.scene, camera, random.uniform(min_z, max_z))

    # Random rotation
    obj.rotation_euler = mathutils.Euler((random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi)))

    # Random color
    colours = [(255,0,0), (0,255,0), (0,0,255)]
    rgb = random.choice(colours)

    new_mat = obj.data.materials[0].copy()
    unique_id = str(uuid.uuid4())
    new_mat.name = obj.data.materials[0].name + '_copy_' + unique_id

    node_tree = new_mat.node_tree
    nodes = node_tree.nodes

    # get the node with the name "Group"
    group_node = nodes.get("Group")

    group_node.inputs[0].default_value = (to_blender_color(rgb[0]), to_blender_color(rgb[1]), to_blender_color(rgb[2]), 1)

    obj.data.materials.clear()
    obj.data.materials.append(new_mat)

    return obj

def to_blender_color(c):    # gamma correction
    c = min(max(0, c), 255) / 255
    return c / 12.92 if c < 0.04045 else math.pow((c + 0.055) / 1.055, 2.4)


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

def convert_coordinates(corner_name, vector):
    # Get the camera object
    camera = bpy.data.objects.get('Camera')

    # Convert the world coordinates to camera view coordinates

    co_2d = bpy_extras.object_utils.world_to_camera_view(bpy.context.scene, camera, vector)

    # Convert the normalized coordinates to pixel coordinates
    render = bpy.context.scene.render
    x = co_2d.x * render.resolution_x
    y = co_2d.y * render.resolution_y

    # Flip the y coordinate
    y = render.resolution_y - y

    return {corner_name: (x, y)}

def get_2d_bound_box(obj):
    # Get the object's bounding box corners in world space
    corners = get_corners_of_object(obj)
    # Convert the corners to camera view coordinates
    corners_2d = {corner_name: convert_coordinates(corner_name, corner_vector) for corner_name, corner_vector in corners.items()}
    return corners_2d

def draw_points_on_rendered_image(image_path, file_name):
    # Load the image
    img_cv2 = cv2.imread(image_path)
    img = bpy.data.images.load(image_path)

    #img height
    img_height = img.size[1]

    # Get the annotations
    annotations_file_path = os.path.join(bpy.path.abspath(annotations_folder), f"{file_name}.json")
    with open(annotations_file_path, mode='r') as file:
        data = json.load(file)
        annotations = data['annotations']
        for annotation in annotations:
            obj_name = annotation['brick_type']
            corners = annotation['keypoints']
            color = annotation['color']

            denormalized_corners = denormalize_keypoints(corners, img.size[0], img.size[1])

            for corner_name, corner_vector in denormalized_corners.items():
                x, y = corner_vector
                cv2.circle(img_cv2, (int(x), int(y)), 3, (0, 255, 0), -1)

            #lowest x and highest y
            top_left_corner = (int(min(denormalized_corners.values(), key=lambda x: x[0])[0]), int(max(denormalized_corners.values(), key=lambda x: x[1])[1]))
            #highest x and lowest y
            bottom_right_corner = (int(max(denormalized_corners.values(), key=lambda x: x[0])[0]), int(min(denormalized_corners.values(), key=lambda x: x[1])[1]))

            # Draw the object's name
            # Get the center of the bounding box
            center = (int((top_left_corner[0] + bottom_right_corner[0]) / 2), int((top_left_corner[1] + bottom_right_corner[1]) / 2))


            # Get the color
            if color == "Blue":
                bgr = (255, 0, 0)
            elif color == "Green":
                bgr = (0, 255, 0)
            elif color == "Red":
                bgr = (0, 0, 255)
            else:
                bgr = (255, 255, 255)  # Default color

            # Draw the object's name
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            cv2.putText(img_cv2, obj_name, center, font, font_scale, bgr, font_thickness)
            cv2.rectangle(img_cv2, top_left_corner, bottom_right_corner, bgr, 1)



    # Save the image
    img.save_render(image_path)
    cv2.imwrite(image_path, img_cv2)


start_time = time.time()

for i in range(rendered_images_amount):
    time_for_name = time.strftime("%d%m%Y-%H%M%S") + f"-{int(time.time() * 1000) % 1000}"
    image_name = time_for_name + '.png'
    main(image_name, fill_to_max_items)
    remove_objects(to_be_removed)
    if will_render_image:
        if camera_collection.objects:
            write_annotations_to_file(time_for_name)
            print(f"Rendering image {i+1}/{rendered_images_amount}")
            bpy.ops.render.render(write_still=True, use_viewport=True)
            if draw_on_image:
                draw_points_on_rendered_image(bpy.context.scene.render.filepath, time_for_name)
        else:
            print(f"No objects in camera collection. Not saving image.")
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
print(f"Average time per image: {(end_time - start_time) / rendered_images_amount} seconds")