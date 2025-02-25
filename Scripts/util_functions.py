import bpy # type: ignore
import bpy_extras # type: ignore
import json
import math
import os
import random
import uuid
import cv2
import mathutils


class Util_functions:
    def __init__(self):
        pass

    def clean_scene(self, camera_collection, line_collection, camera, to_be_removed):
        # Checks for occlusions and objects out of bounds
        for obj in camera_collection.objects:
            occluding_objects = self.get_occluding_objects(camera, obj, camera_collection, line_collection)
            for occluding_obj in occluding_objects:
                to_be_removed.add(occluding_obj)

        self.remove_objects(to_be_removed, camera_collection)

        for obj in camera_collection.objects:
            # Get the object's bounding box corners in world space
            corners = self.get_corners_of_object(obj, camera, camera_collection)
            for corner_data in corners.values():
                if self.is_point_in_camera_view(camera, corner_data[0]):
                    continue
                else:
                    to_be_removed.add(obj)
                    break

        self.remove_objects(to_be_removed, camera_collection)
    
    def world_to_camera_coords(self, camera, world_coords):
        if not isinstance(world_coords, mathutils.Vector):
            world_coords = mathutils.Vector(world_coords)
        cam_matrix_world = camera.matrix_world
        cam_matrix_world_inv = cam_matrix_world.inverted()
        camera_relative_coords = cam_matrix_world_inv @ world_coords
        return camera_relative_coords
    
    def world_to_pixel(self, scene, camera_obj, world_coords):
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
    
    def normalize_keypoints(self, keypoints, image_width, image_height):
        normalized_keypoints = {}
        for corner_name, corner_data in keypoints.items():
            x = corner_data[0][0] / image_width
            y = corner_data[0][1] / image_height
            normalized_keypoints[corner_name] = ([x, y], corner_data[1])
        return normalized_keypoints
    
    def normalize_coordinate(self, coord, image_width, image_height):
        x = coord[0] / image_width
        y = coord[1] / image_height
        return (x, y)
    
    def load_hdri_image(self, img_path):
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
        
        
    def denormalize_keypoints(self, keypoints, image_width, image_height):
        denormalized_keypoints = {}
        for corner_name, corner_data in keypoints.items():
            x = corner_data[0][0] * image_width
            y = corner_data[0][1] * image_height
            denormalized_keypoints[corner_name] = ([x, y], corner_data[1])
        return denormalized_keypoints
    
    def draw_corners(self, obj, line_collection, camera_collection):
        # Get the object's bounding box corners in world space
        corners = self.get_corners_of_object(obj, bpy.context.scene.camera, camera_collection)

        for corner_name, corner_data in corners.items():
            corners[corner_name] = corner_data[0]

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
    
    def get_camera_view_bounds(self, scene, camera_obj, depth):
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
    
    def is_point_in_camera_view(self, camera, point_world):
        # Transform the point to the camera's local space
        point_camera_space = camera.matrix_world.inverted() @ point_world

        # Get camera data
        cam_data = camera.data

        # Create Plane objects for each side of the camera frustum
        lower_left, lower_right, upper_right, upper_left = self.get_camera_view_bounds(bpy.context.scene, camera, 1)
        bottom_test = self.is_point_above_plane(lower_left, lower_right, camera.location, point_world)
        right_test = self.is_point_above_plane(lower_right, upper_right, camera.location, point_world)
        top_test = self.is_point_above_plane(upper_right, upper_left, camera.location, point_world)
        left_test = self.is_point_above_plane(upper_left, lower_left, camera.location, point_world)
        
        
        if bottom_test and right_test and top_test and left_test:
            return True
        else:
            return False
        
        
    def is_point_above_plane(self, p1, p2, p3, test_point):
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
        
        
    def random_point_in_camera_view(self, scene, camera_obj, depth):
        """
        Get a random point within the camera frustum at a given depth.
        """
        lower_left, lower_right, upper_right, upper_left = self.get_camera_view_bounds(scene, camera_obj, depth)
        # Random interpolation of the edges of the view
        random_x = random.uniform(0, 1)
        random_y = random.uniform(0, 1)

        # Linear interpolation between the corners
        point_on_left = lower_left.lerp(upper_left, random_y)
        point_on_right = lower_right.lerp(upper_right, random_y)

        # Final point inside the frustum
        random_point = point_on_left.lerp(point_on_right, random_x)
        return random_point
    
    def random_attributes_object(self, obj, camera, min_z, max_z):
        # Random location
        obj.location = self.random_point_in_camera_view(bpy.context.scene, camera, random.uniform(min_z, max_z))

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

        group_node.inputs[0].default_value = (self.to_blender_color(rgb[0]), self.to_blender_color(rgb[1]), self.to_blender_color(rgb[2]), 1)

        obj.data.materials.clear()
        obj.data.materials.append(new_mat)

        return obj
    
    def to_blender_color(self, c):    # gamma correction
        c = min(max(0, c), 255) / 255
        return c / 12.92 if c < 0.04045 else math.pow((c + 0.055) / 1.055, 2.4)
    
    def get_occluding_objects(self, camera, target, camera_collection, line_collection, draw_line=False):
        # Get the position of the camera and target
        cam_location = camera.location
        obj_flag = False

        if not isinstance(target, mathutils.Vector):
            target_location = target.location
            obj_flag = True
        else:
            target_location = target
            obj_flag = False

        # Vector from camera to target
        cam_to_target_vec = target_location - cam_location
        cam_to_target_distance = cam_to_target_vec.length


        objects_in_between = []

        # Loop through all objects in the scene
        for obj in camera_collection.objects:
            # Ignore the camera and target itself
            if obj == camera:
                continue
            if obj == target and obj_flag:
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
            line = self.draw_line_meth(cam_location, cam_to_target_vec, cam_to_target_distance, line_name=f"CameraToTarget_{target.name}")
            line_collection.objects.link(line)

        return objects_in_between
    
    def draw_line_meth(self, start, direction, length, line_name="Line"):
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
    
    def remove_objects(self, obj_set: set, camera_collection):
        for obj in obj_set:
            # print(f"Removing object: {obj.name}")
            camera_collection.objects.unlink(obj)
            bpy.data.objects.remove(obj)

        obj_set.clear()
        
    def position_relative_to_camera(self, camera, obj):
        # Convert the object's world location to the camera's local space
        relative_position = camera.matrix_world.inverted() @ obj.location
        return relative_position
    
    def get_corners_of_object(self, obj, camera, camera_collection):
        # Get the object's bounding box corners in world space and put them in a dictionary
        corners = {f"Corner_{i}": obj.matrix_world @ mathutils.Vector(corner) for i, corner in enumerate(obj.bound_box)}# In the form: {"Corner_0": Vector((x, y, z)), ...}

        # Check the visibility of the corners
        for corner_name, corner_vector in corners.items():
            # Cast a ray from the camera to the corner
            is_visible = self.is_visible(camera, corner_vector)
            # Check if the ray intersected with any object
            if is_visible:
                corners[corner_name] = (corner_vector, True)
            else:
                corners[corner_name] = (corner_vector, False)

        return corners
    
    def convert_coordinates(self, corner_name, vector, scene, camera, visible=None):

        # Convert the world coordinates to camera view coordinatesget_occluding_objects(camera, corner_vector, camera_collection, None, draw_line=False)

        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, vector)
        # Convert the normalized coordinates to pixel coordinates
        x = co_2d.x * scene.render.resolution_x
        y = co_2d.y * scene.render.resolution_y

        # Flip the y coordinate
        y = scene.render.resolution_y - y

        if visible is not None:
            return {corner_name: ((x, y), visible)}
        else:
            return {corner_name: (x, y)}
    
    def get_2d_bound_box(self, obj, scene, camera, camera_collection):
        # Get the object's bounding box corners in world space
        corners = self.get_corners_of_object(obj, camera, camera_collection)

        for corner_name, corner_data in corners.items():
            corner_vector, visible = corner_data
            corners[corner_name] = corner_vector

        # Convert the corners to camera view coordinates
        corners_2d = {corner_name: self.convert_coordinates(corner_name, corner_vector, scene, camera)[corner_name] for corner_name, corner_vector in corners.items()}
        
        # Get the top left and bottom right coordinates
        top_left = min(corners_2d.values(), key=lambda x: x[0])[0], max(corners_2d.values(), key=lambda x: x[1])[1]
        bottom_right = max(corners_2d.values(), key=lambda x: x[0])[0], min(corners_2d.values(), key=lambda x: x[1])[1]
        return top_left, bottom_right
    
    def draw_points_on_rendered_image(self, image_path, annotations_folder):
        # Load the image
        img_cv2 = cv2.imread(image_path)
        img = bpy.data.images.load(image_path)

        #img height
        img_height = img.size[1]

        file_name = image_path.split("/")[-1].split(".")[0]

        # Get the annotations
        annotations_file_path = os.path.join(bpy.path.abspath(annotations_folder), f"{file_name}.json")
        with open(annotations_file_path, mode='r') as file:
            data = json.load(file)
            annotations = data['annotations']
            for annotation in annotations:
                obj_name = annotation['brick_type']
                corners = annotation['normal_pixel_coordinates']
                color = annotation['color']

                denormalized_corners = self.denormalize_keypoints(corners, img.size[0], img.size[1])

                for corner_name, corner_data in denormalized_corners.items():
                    x, y = corner_data[0]
                    if corner_data[1]:
                        cv2.circle(img_cv2, (int(x), int(y)), 3, (0, 255, 0), -1)
                    else:
                        cv2.circle(img_cv2, (int(x), int(y)), 3, (0, 0, 255), -1)

                #lowest x and highest y
                top_left_corner = (int(min(denormalized_corners.values(), key=lambda x: x[0][0])[0]), int(max(denormalized_corners.values(), key=lambda x: x[0][1])[1]))
                #highest x and lowest y
                bottom_right_corner = (int(max(denormalized_corners.values(), key=lambda x: x[0][0])[0]), int(min(denormalized_corners.values(), key=lambda x: x[0][1])[1]))

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
        
    def crop_image(self, image_path, bottom_left, top_right):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image at path '{image_path}' could not be found or loaded.")
        
        # Use bottom_left and top_right to crop the image
        x1, y1 = bottom_left
        x2, y2 = top_right
        
        # Ensure the coordinates are within the image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
        
        
        cropped_image = image[y2:y1, x1:x2]
        

        cv2.imwrite(image_path, cropped_image)
        
        #cv2.imwrite(image_path, cropped_image)

    def is_visible(self, camera, target, exclude_objects=[]):
        """
        Checks if there are any objects between the camera and the target point.
        
        :param camera: The camera object
        :param target: A mathutils.Vector representing the target point
        :param exclude_objects: A list of objects to ignore in the occlusion check
        :return: True if there is an occlusion, False otherwise
        """
        scene = bpy.context.scene
        depsgraph = bpy.context.evaluated_depsgraph_get()
        
        # Get camera world position
        camera_location = camera.location
        
        # Direction vector from camera to target
        direction = (target - camera_location).normalized()
        
        # Perform ray casting
        result, location, normal, index, obj, matrix = scene.ray_cast(depsgraph, camera_location, direction)
        
        # If we hit an object, check if it's in the exclude list
        if result and obj not in exclude_objects:
            return False  # There is an occlusion
        
        return True  # No occlusion found
        
        