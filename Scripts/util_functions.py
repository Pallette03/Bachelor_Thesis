import time
import numpy as np
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

    def clean_scene(self, camera_collection, line_collection, clutter_collection, camera, to_be_removed):
        for obj in camera_collection.objects:
            # Get the object's bounding box corners in world space
            corners = self.get_corners_of_object(obj, camera)
            for corner_data in corners.values():
                if self.is_point_in_camera_view(camera, corner_data[0]) and self.is_visible(camera, corner_data[0], exclude_objects=[obj]):
                    continue
                else:
                    to_be_removed.add(obj)
                    break

        self.remove_objects(to_be_removed, camera_collection)
        
        for mesh in bpy.data.meshes:
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
    
    def world_to_camera_coords(self, camera, world_coords):
        if not isinstance(world_coords, mathutils.Vector):
            world_coords = mathutils.Vector(world_coords)
        cam_matrix_world = camera.matrix_world
        cam_matrix_world_inv = cam_matrix_world.inverted()
        camera_relative_coords = cam_matrix_world_inv @ world_coords
        return camera_relative_coords
    
    def world_to_pixel(self, scene, camera_obj, world_coords):
        if camera_obj.type != 'CAMERA':
            raise ValueError("Object is not a camera")
        
        cam_data = camera_obj.data
        render = scene.render
        
        # Projection matrix (4x4)
        proj_matrix = camera_obj.calc_matrix_camera(
            render.resolution_x, 
            render.resolution_y, 
            render.pixel_aspect_x / render.pixel_aspect_y
        )
        
        world_to_camera_matrix = camera_obj.matrix_world.inverted()
        
        camera_coords = world_to_camera_matrix @ world_coords
        
        clip_coords = proj_matrix @ camera_coords
        
        if clip_coords.w != 0:
            ndc_coords = mathutils.Vector((
                clip_coords.x / clip_coords.w,
                clip_coords.y / clip_coords.w,
                clip_coords.z / clip_coords.w
            ))
        else:
            raise ValueError("Invalid clip coordinates, w = 0")
        
        pixel_x = (ndc_coords.x + 1) / 2.0 * render.resolution_x
        pixel_y = (1 - ndc_coords.y) / 2.0 * render.resolution_y  # Flip y-axis for Blender's pixel space
        
        return (pixel_x, pixel_y)
    
    def normalize_keypoints(self, keypoints, image_width, image_height):
        normalized_keypoints = {}
        for corner_name, corner_data in keypoints.items():
            x = corner_data[0][0] / image_width
            y = corner_data[0][1] / image_height
            normalized_keypoints[corner_name] = ([x, y], corner_data[1], corner_data[2])
        return normalized_keypoints
    
    def normalize_coordinate(self, coord, image_width, image_height):
        x = coord[0] / image_width
        y = coord[1] / image_height
        return (x, y)
    
    def load_hdri_image(self, img_path):
        hdri_image = bpy.ops.image.open(filepath=img_path)
        hdri_image = bpy.data.images.get(os.path.basename(img_path))
        
        env_node = bpy.context.scene.world.node_tree.nodes.get('Environment Texture')
        
        if not env_node:
            print("Environment Texture node not found, creating a new one")
            env_node = bpy.context.scene.world.node_tree.nodes.new('ShaderNodeTexEnvironment')
            env_node.location = (-300, 300)
            env_node.name = 'Environment Texture'
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
        corners = self.get_corners_of_object(obj, bpy.context.scene.camera)

        for corner_name, corner_data in corners.items():
            corners[corner_name] = corner_data[0]

        mesh = bpy.data.meshes.new("Corners")
        obj_corners = bpy.data.objects.new("Corners", mesh)
        line_collection.objects.link(obj_corners)

        vertices = list(corners.values())
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)]

        mesh.from_pydata(vertices, edges, [])
        mesh.update()

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

        frame = [camera_obj.matrix_world @ (v * depth) for v in frame]

        upper_right = frame[0]
        lower_right = frame[1]
        lower_left = frame[2]
        upper_left = frame[3]

        return lower_left, lower_right, upper_right, upper_left
    
    def is_point_in_camera_view(self, camera, point_world):
        point_camera_space = camera.matrix_world.inverted() @ point_world

        cam_data = camera.data

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
        normal = (p2 - p1).cross(p3 - p1).normalized()
        
        vector_to_point = test_point - p1
        
        dot_product = normal.dot(vector_to_point)
        
        if dot_product < 0:
            return True
        else:
            return False
        
        
    def random_point_in_camera_view(self, scene, camera_obj, depth):
        """
        Get a random point within the camera frustum at a given depth.
        """
        lower_left, lower_right, upper_right, upper_left = self.get_camera_view_bounds(scene, camera_obj, depth)
        random_x = random.uniform(0, 1)
        random_y = random.uniform(0, 1)

        point_on_left = lower_left.lerp(upper_left, random_y)
        point_on_right = lower_right.lerp(upper_right, random_y)

        random_point = point_on_left.lerp(point_on_right, random_x)
        return random_point
    
    def random_attributes_object(self, obj, camera, min_z, max_z, is_clutter=False):
        # Random location
        obj.location = self.random_point_in_camera_view(bpy.context.scene, camera, random.uniform(min_z, max_z))

        # Random rotation
        obj.rotation_euler = mathutils.Euler((random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi)))

        if not is_clutter:
            # Random color
            colours = [(255,0,0), (0,255,0), (0,0,255)]
            rgb = random.choice(colours)

            new_mat = obj.data.materials[0].copy()
            unique_id = str(uuid.uuid4())
            new_mat.name = obj.data.materials[0].name + '_copy_' + unique_id

            node_tree = new_mat.node_tree
            nodes = node_tree.nodes

            group_node = nodes.get("Group")

            group_node.inputs[0].default_value = (self.to_blender_color(rgb[0]), self.to_blender_color(rgb[1]), self.to_blender_color(rgb[2]), 1)

            obj.data.materials.clear()
            obj.data.materials.append(new_mat)

        return obj
    
    def clutter_scene(self, clutter_collection, camera_clutter_collection, camera, min_z, max_z, min_clutter_items, max_clutter_items):
        clutter_objects = []
        for i in range(random.randint(min_clutter_items, max_clutter_items)):
            if i > 100:
                break
            clutter_objects.append(random.choice(clutter_collection.objects))
            
        for obj in clutter_objects:
            new_obj = obj.copy()
            new_obj.data = obj.data.copy()
            camera_clutter_collection.objects.link(new_obj)
            new_obj.hide_render = False
            self.random_attributes_object(new_obj, camera, min_z, max_z, is_clutter=True)
    
    def to_blender_color(self, c):    # gamma correction
        c = min(max(0, c), 255) / 255
        return c / 12.92 if c < 0.04045 else math.pow((c + 0.055) / 1.055, 2.4)
    
    
    def draw_line_meth(self, start, direction, length, line_name="Line"):
        end = start + direction.normalized() * length

        mesh = bpy.data.meshes.new(line_name)
        obj = bpy.data.objects.new(line_name, mesh)

        vertices = [start, end]
        edges = [(0, 1)]

        mesh.from_pydata(vertices, edges, [])
        mesh.update()

        mat = bpy.data.materials.new(name="LineMaterial")
        mat.diffuse_color = (1, 0, 0, 1)
        obj.data.materials.append(mat)

        return obj
    
    def remove_objects(self, obj_set: set, camera_collection):
        for obj in obj_set:
            camera_collection.objects.unlink(obj)
            bpy.data.objects.remove(obj)

        obj_set.clear()
        
    def position_relative_to_camera(self, camera, obj):
        relative_position = camera.matrix_world.inverted() @ obj.location
        return relative_position
    
    def get_corners_of_object(self, obj, camera):
        all_corners = {f"Corner_{i}": mathutils.Vector(corner) for i, corner in enumerate(obj.bound_box)}

        stud_corrected_corners = {}
        
        min_z = min(c.z for c in all_corners.values())
        max_z = max(c.z for c in all_corners.values())  # This includes studs

        face_heights = [
            (face.center).z
            for face in obj.data.polygons
            if len(face.vertices) == 4 and abs(face.normal.z) > 0.9
        ]
        
        unique_heights = sorted(set(face_heights))
        if len(unique_heights) > 1:
            real_top_z = unique_heights[-1]  # Second-highest surface -> top of brick body
        else:
            real_top_z = max_z

        for name, value in all_corners.items():
            if value.z == max_z:
                stud_corrected_corners[name] = (obj.matrix_world @ mathutils.Vector((value.x, value.y, real_top_z)), "top")
            else:
                stud_corrected_corners[name] = (obj.matrix_world @ value, "bottom")
                    
        # Check the visibility of the corners
        for corner_name, corner_data in stud_corrected_corners.items():
            stud_corrected_corners[corner_name] = (corner_data[0], self.is_visible(camera, corner_data[0]), corner_data[1])

        return stud_corrected_corners
    
    def convert_coordinates(self, corner_name, vector, scene, camera, visible=None, lateral_category=None):

        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, vector)
        x = co_2d.x * scene.render.resolution_x
        y = co_2d.y * scene.render.resolution_y

        y = scene.render.resolution_y - y

        if visible is not None and lateral_category is not None:
            return {corner_name: ((x, y), visible, lateral_category)}
        else:
            return {corner_name: (x, y)}
    
    def get_2d_bound_box(self, obj, scene, camera, camera_collection):
        corners = self.get_corners_of_object(obj, camera)

        for corner_name, corner_data in corners.items():
            corner_vector, visible, lateral_category = corner_data
            corners[corner_name] = corner_vector

        corners_2d = {corner_name: self.convert_coordinates(corner_name, corner_vector, scene, camera)[corner_name] for corner_name, corner_vector in corners.items()}
        
        top_left = min(corners_2d.values(), key=lambda x: x[0])[0], max(corners_2d.values(), key=lambda x: x[1])[1]
        bottom_right = max(corners_2d.values(), key=lambda x: x[0])[0], min(corners_2d.values(), key=lambda x: x[1])[1]
        return top_left, bottom_right
    
    def draw_points_on_rendered_image(self, image_path, annotations_folder):
        img_cv2 = cv2.imread(image_path)
        img = bpy.data.images.load(image_path)

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

                top_left_corner = (int(min(denormalized_corners.values(), key=lambda x: x[0][0])[0]), int(max(denormalized_corners.values(), key=lambda x: x[0][1])[1]))
                bottom_right_corner = (int(max(denormalized_corners.values(), key=lambda x: x[0][0])[0]), int(min(denormalized_corners.values(), key=lambda x: x[0][1])[1]))

                center = (int((top_left_corner[0] + bottom_right_corner[0]) / 2), int((top_left_corner[1] + bottom_right_corner[1]) / 2))

                if color == "Blue":
                    bgr = (255, 0, 0)
                elif color == "Green":
                    bgr = (0, 255, 0)
                elif color == "Red":
                    bgr = (0, 0, 255)
                else:
                    bgr = (0, 0, 255)

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                cv2.putText(img_cv2, obj_name, center, font, font_scale, bgr, font_thickness)
                cv2.rectangle(img_cv2, top_left_corner, bottom_right_corner, bgr, 1)



        img.save_render(image_path)
        cv2.imwrite(image_path, img_cv2)
        
    def crop_image(self, image_path, bottom_left, top_right):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image at path '{image_path}' could not be found or loaded.")
        
        x1, y1 = bottom_left
        x2, y2 = top_right

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
        
        
        cropped_image = image[y2:y1, x1:x2]
        
        cv2.imwrite(image_path, cropped_image)


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
        
        camera_location = camera.location
        
        direction = (target - camera_location).normalized()
        
        # Perform ray casting
        result, location, normal, index, obj, matrix = scene.ray_cast(depsgraph, camera_location, direction)
        
        if result and obj not in exclude_objects:
            target_distance = (target - camera_location).length
            hit_distance = (location - camera_location).length
            return hit_distance >= target_distance
        
        return True  # No occlusion found

    
    def add_gaussian_noise_to_image(self, image_path, mean=0.0, var=0.01):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not load image at: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        sigma = var ** 0.5
        
        noise = np.random.normal(loc=mean, scale=sigma, size=img_rgb.shape).astype(np.float32)
        
        noisy_img = np.clip(img_rgb + noise, 0.0, 1.0)
        
        noisy_img_uint8 = (noisy_img * 255).astype(np.uint8)
        cv2.imwrite(image_path, cv2.cvtColor(noisy_img_uint8, cv2.COLOR_RGB2BGR))
        return noisy_img_uint8
    
    def add_salt_and_pepper_noise(self, image_path, prob=0.5):
        img = cv2.imread(image_path)
        output = np.copy(img)

        black = np.array([0, 0, 0], dtype='uint8')
        white = np.array([255, 255, 255], dtype='uint8')

        probs = np.random.random(output.shape[:2])
        output[probs < (prob / 2)] = black
        output[probs > 1 - (prob / 2)] = white
        
        cv2.imwrite(image_path, output)
        return output
        
    def save_depth_map(self, depth_map_path, scene, file_name):
        original_use_nodes = scene.use_nodes
        scene.use_nodes = True 
        
        node_tree = scene.node_tree
        nodes = node_tree.nodes
        links = node_tree.links

        for layer in scene.view_layers:
            layer.use_pass_z = True

        nodes_created = []
        
        file_node = nodes.new(type="CompositorNodeOutputFile")
        file_node.name = "TempDepthOutput"
        file_node.label = "Temp Depth Output"
        nodes_created.append(file_node)

        base_path = bpy.path.abspath("//") + depth_map_path
        file_node.base_path = base_path
        complete_file_name = file_name + "_depth"
        file_node.file_slots[0].path = complete_file_name

        file_node.format.file_format = 'PNG'
        file_node.format.color_mode = 'RGB'
 
        render_layers = None
        for node in nodes:
            if node.type == 'R_LAYERS':
                render_layers = node
                break
        render_layers_created = False
        if render_layers is None:
            render_layers = nodes.new(type="CompositorNodeRLayers")
            render_layers_created = True
            nodes_created.append(render_layers)

        while file_node.inputs[0].links:
            links.remove(file_node.inputs[0].links[0])

        links.new(render_layers.outputs['Depth'], file_node.inputs[0])

        bpy.ops.render.render(write_still=True, use_viewport=True)
        
        frame_str = f"{scene.frame_current:04d}"
        extension = ".png"
        old_filename = complete_file_name + frame_str + extension
        old_filepath = os.path.join(base_path, old_filename)

        new_filename = complete_file_name + extension
        new_filepath = os.path.join(base_path, new_filename)

        try:
            os.rename(old_filepath, new_filepath)
        except Exception as e:
            print("Error renaming file:", e)

        # Clean up: remove only the nodes created
        for node in nodes_created:
            nodes.remove(node)
        
        # Restore the original state
        scene.use_nodes = original_use_nodes
        
    def preload_clutter(self, object_folder=bpy.path.abspath("//") + "google_scanned_objects",  start_index=0, amount=50, clutter_collection_name="Clutter"):
        object_list = os.listdir(object_folder)
        
        start_index = int(start_index)
        end_index = int(start_index + amount)
        if end_index > len(object_list):
            start_index = 0
            end_index = int(amount)
            
        object_list = object_list[start_index:end_index]
        
        def find_layer_collection(layer_collection, name):
            if layer_collection.collection.name == name:
                return layer_collection
            for child in layer_collection.children:
                result = find_layer_collection(child, name)
                if result:
                    return result
            return None

        view_layer = bpy.context.view_layer

        layer_collection = find_layer_collection(view_layer.layer_collection, clutter_collection_name)

        if layer_collection:
            for obj in layer_collection.collection.objects:
                layer_collection.collection.objects.unlink(obj)
                bpy.data.objects.remove(obj)
            
            view_layer.active_layer_collection = layer_collection
            print(f"Active collection set to: {clutter_collection_name}")
        else:
            print(f"Collection '{clutter_collection_name}' not found.")
        
        for folder in object_list:
            if os.path.isdir(os.path.join(object_folder, folder)):
                obj_file = os.path.join(object_folder, folder, "meshes", "model.obj")
                if os.path.isfile(obj_file):
                    
                    bpy.ops.wm.obj_import(filepath=obj_file)
                    obj = bpy.data.objects.get("model")
                    if not obj:
                        continue
                    obj.name = folder
                    obj.hide_render = True
                    obj.hide_set(True)

                    obj.scale[0] = 0.25
                    obj.scale[1] = 0.25
                    obj.scale[2] = 0.25
        
        