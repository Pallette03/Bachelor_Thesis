import argparse
import bpy # type: ignore
import bpy_extras # type: ignore
import random
import os
import mathutils
import math
import time
import json
import sys

dir = os.path.dirname(bpy.data.filepath)
dir = os.path.join(dir, 'Scripts')
if not dir in sys.path:
    sys.path.append(dir )

import util_functions

print('----------------')

bpy.context.scene.use_nodes = True

uf = util_functions.Util_functions()

def main(params, render_images_folder, hdri_folder, file_name, camera, camera_collection, line_collection, clutter_collection, camera_clutter_collection, parts_collection, to_be_removed):

    min_items = params.get('min_items')
    max_items = params.get('max_items')
    min_clutter_items = params.get('min_clutter_items')
    max_clutter_items = params.get('max_clutter_items')
    min_z = params.get('min_z')
    max_z = params.get('max_z')
    
    bpy.context.scene.render.filepath = os.path.join(bpy.path.abspath(render_images_folder), file_name)
    bpy.context.scene.render.image_settings.file_format = 'PNG'

    # Loading the HDRI image
    image_extensions = ('.hdr', '.exr')
    background_images = [f for f in os.listdir(bpy.path.abspath(hdri_folder)) if f.lower().endswith(image_extensions)]
    hdri_image_path = os.path.join(bpy.path.abspath(hdri_folder), random.choice(background_images))
    uf.load_hdri_image(hdri_image_path)

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
    
    # Clear the camera clutter collection
    for obj in camera_clutter_collection.objects:
        camera_clutter_collection.objects.unlink(obj)
        bpy.data.objects.remove(obj)

    # Get a random amount of objects from the Parts collection
    random_objects = []
    for i in range(random.randint(min_items, max_items)):
        if i > 100:
            break
        random_objects.append(random.choice(parts_collection.objects))

    # Copy the object to the camera collection
    for obj in random_objects:
        new_obj = obj.copy()
        new_obj.data = obj.data.copy()
        camera_collection.objects.link(new_obj)
        # Set the location of the new object within the camera's view frustum 
        uf.random_attributes_object(new_obj, camera, min_z, max_z)

    # Force update
    bpy.context.view_layer.update()

    uf.clean_scene(camera_collection, line_collection, camera, to_be_removed)

    if params.get('fill_to_max_items'):
        while len(camera_collection.objects) < max_items:
            random_objects = []
            for i in range(max_items - len(camera_collection.objects)):
                random_objects.append(random.choice(parts_collection.objects))

            for obj in random_objects:
                new_obj = obj.copy()
                new_obj.data = obj.data.copy()
                camera_collection.objects.link(new_obj)
                # Set the location of the new object within the camera's view frustum 
                uf.random_attributes_object(new_obj, camera, min_z, max_z)

            bpy.context.view_layer.update()
            
            uf.clean_scene(camera_collection, line_collection, camera, to_be_removed)
            
    # Add clutter to the scene
    clutter_time = time.time()
    if params.get('add_clutter'):
        if clutter_collection and camera_clutter_collection:
            uf.clutter_scene(clutter_collection, camera_clutter_collection, camera, min_z, max_z, min_clutter_items, max_clutter_items)
        else:
            print(f"Clutter collections not found. Skipping clutter generation.")
    print(f"Clutter generation time: {time.time() - clutter_time} seconds")

        
            

def write_annotations_to_file(params, file_name, annotations_folder, camera, camera_collection):

    with open(os.path.join(bpy.path.abspath(annotations_folder), f"{file_name}.json"), mode='a') as json_file:

        image_id = file_name
        json_file.write(f'{{"image_id": "{image_id}", ')
        json_file.write(f'"gaussian_noise_flag": {"true" if params.get("gaussian_noise") else "false"}, "salt_and_pepper_noise_flag": {"true" if params.get("salt_and_pepper_noise") else "false"}, "depth_flag": {"true" if params.get("depth_output") else "false"}, "clutter_flag": {"true" if params.get("add_clutter") else "false"}, ')
        json_file.write('"camera_matrix": [')
        for vector in camera.matrix_world:
            json.dump((vector[0], vector[1], vector[2]), json_file)
            if vector != camera.matrix_world[-1]:
                json_file.write(', ')
            else:
                json_file.write('], \n')
        json_file.write('"annotations": [\n')

        for obj in camera_collection.objects:
            corners = uf.get_corners_of_object(obj, camera, camera_collection)
            brick_type = obj.name.split('.')[0]
            color = obj.data.materials[0].node_tree.nodes.get("Group").inputs[0].default_value

            json_file.write(f'{{"brick_type": "{brick_type}",\n "color": "{[color[0], color[1], color[2]]}",\n "keypoints": ')

            serialized_corners = {}
            camera_corners = {}
            for corner_name, corner_data in corners.items():
                serialized_corners.update(uf.convert_coordinates(corner_name, corner_data[0], bpy.context.scene, camera, corner_data[1]))
                temp_tuple = ()
                for val in uf.world_to_camera_coords(camera, corner_data[0]):
                    temp_tuple += (val,)
                camera_corners[corner_name] = temp_tuple
            
            # serialized_corners = {}
            # for corner_name, corner_vector in corners.items():
            #     serialized_corners.update(convert_coordinates(corner_name, corner_vector, bpy.context.scene, camera))

            normalized_corners = uf.normalize_keypoints(serialized_corners, bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y)
            #normalized_corners = uf.normalize_keypoints(serialized_corners, bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y)
            
            json.dump(camera_corners, json_file)
            json_file.write(',\n')
            
            json_file.write('"normal_pixel_coordinates": ')
            json.dump(normalized_corners, json_file)
            json_file.write(',\n')
            
            json_file.write('"bb_box": ')
            top_left, bottom_right = uf.get_2d_bound_box(obj, bpy.context.scene, camera, camera_collection)
            json.dump({"top_left": uf.normalize_coordinate(top_left, bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y), "bottom_right": uf.normalize_coordinate(bottom_right, bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y)}, json_file)
            
            if obj == camera_collection.objects[-1]:
                json_file.write('}\n')
            else:
                json_file.write('},\n')

        json_file.write(']\n}\n')
        json_file.close()

    print(f"Annotations written to file {file_name}.json")

def main_main(params):
    
    gaussian_mean = 0
    gaussian_var = 0.25

    salt_pepper_prob = 0.5
    
    will_render_image = params.get('will_render_image')
    draw_on_image = False
    fill_to_max_items = params.get('fill_to_max_items')
    add_gaussian_noise = params.get('gaussian_noise')
    add_salt_and_pepper_noise = params.get('salt_and_pepper_noise')
    depth_output = params.get('depth_output')
    add_clutter = params.get('add_clutter')
    
    render_images_folder = f"{params.get('output_folder')}/images/rgb"
    if depth_output:
        depth_map_folder = f"{params.get('output_folder')}/images/depth"
        os.makedirs(bpy.path.abspath(depth_map_folder), exist_ok=True)
    annotations_folder = f"{params.get('output_folder')}/annotations"
    
    hdri_folder = f"{params.get('hdri_folder')}/"
    if not os.path.exists(bpy.path.abspath(hdri_folder)):
        print(f"HDIR folder {hdri_folder} not found. Exiting.")
        return
        
    os.makedirs(bpy.path.abspath(render_images_folder), exist_ok=True)
    os.makedirs(bpy.path.abspath(annotations_folder), exist_ok=True)
        
    camera = bpy.data.objects.get('Camera')
    
    parts_collection_name = 'Parts'
    camera_collection_name = 'In_Camera'
    line_collection_name = 'Lines'
    clutter_collection_name = 'Clutter'
    camera_clutter_name = 'Camera_Clutter'
    
    parts_collection = bpy.data.collections.get(parts_collection_name)
    camera_collection = bpy.data.collections.get(camera_collection_name)
    line_collection = bpy.data.collections.get(line_collection_name)
    clutter_collection = bpy.data.collections.get(clutter_collection_name)
    camera_clutter_collection = bpy.data.collections.get(camera_clutter_name)
    
    to_be_removed = set()
    
    if not parts_collection:
        print(f"Collection '{parts_collection_name}' not found. Creating a new collection.")
        parts_collection = bpy.data.collections.new(parts_collection_name)
        bpy.context.scene.collection.children.link(parts_collection)
             
    if not camera_collection:
        print(f"Collection '{camera_collection_name}' not found. Creating a new collection.")
        camera_collection = bpy.data.collections.new(camera_collection_name)
        bpy.context.scene.collection.children.link(camera_collection)
        camera_collection.objects.link(camera)
        
    if not camera:
        print(f"Camera not found. Cant render images.")
        return
    
    if not line_collection:
        print(f"Collection '{line_collection_name}' not found. Creating a new collection.")
        line_collection = bpy.data.collections.new(line_collection_name)
        bpy.context.scene.collection.children.link(line_collection)

    min_items = params.get('min_items')
    max_items = params.get('max_items')
    rendered_images_amount = params.get('rendered_images_amount')

    min_clutter_items = params.get('min_clutter_items')
    max_clutter_items = params.get('max_clutter_items')

    bpy.context.scene.render.resolution_x = params.get('rendered_image_resolution')
    bpy.context.scene.render.resolution_y = params.get('rendered_image_resolution')
    
    start_time = time.time()
    bpy.context.preferences.edit.use_global_undo = False

    preload_amount = 50 # Defines the amount of objects to be preloaded. The Higher the amount, the more memory is used.
    seperating_img_amount = math.ceil(rendered_images_amount / (1000 / preload_amount))
    
    for i in range(rendered_images_amount):
        
        if add_clutter and i % seperating_img_amount == 0:
            start_index = (i/seperating_img_amount) * preload_amount
            preload_time = time.time()
            uf.preload_clutter(start_index=start_index, amount=preload_amount, clutter_collection_name=clutter_collection_name)
            print(f"Preloaded {preload_amount} objects in {time.time() - preload_time} seconds")
        
        time_for_name = time.strftime("%d%m%Y-%H%M%S") + f"-{int(time.time() * 1000) % 1000}"
        image_name = time_for_name + '.png'
        main(params, render_images_folder, hdri_folder, image_name, camera, camera_collection, line_collection, clutter_collection, camera_clutter_collection, parts_collection, to_be_removed)
        uf.remove_objects(to_be_removed, camera_collection)
        if will_render_image:
            if camera_collection.objects:
                write_annotations_to_file(params, time_for_name, annotations_folder, camera, camera_collection)
                print(f"Rendering image {i+1}/{rendered_images_amount}")

                if depth_output:
                    uf.save_depth_map(depth_map_folder, bpy.context.scene, time_for_name)
                else:
                    bpy.ops.render.render(write_still=True, use_viewport=True)

                bpy.data.orphans_purge()  # Purges unused data
                
                if add_gaussian_noise:
                    uf.add_gaussian_noise_to_image(bpy.context.scene.render.filepath, gaussian_mean, gaussian_var)
                    
                if add_salt_and_pepper_noise:
                    uf.add_salt_and_pepper_noise(bpy.context.scene.render.filepath, salt_pepper_prob)

                if draw_on_image:
                    uf.draw_points_on_rendered_image(bpy.context.scene.render.filepath, annotations_folder)
            else:
                print(f"No objects in camera collection. Not saving image.")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Average time per image: {(end_time - start_time) / rendered_images_amount} seconds")
    
    
if __name__ == "__main__":
    
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser(description="Generate a dataset of images with annotations.")
    parser.add_argument("--rendered_images_amount", type=int, default=1, help="Number of images to render.")
    parser.add_argument("--output_folder", type=str, default="//datasets/normal", help="Folder to save the images and annotations.")
    parser.add_argument("--hdri_folder", type=str, default="//HDRI", help="Folder containing HDRI images.")
    parser.add_argument("--gaussian_noise", action="store_true", help="Add Gaussian noise to the images.")
    parser.add_argument("--salt_and_pepper_noise", action="store_true", help="Add salt and pepper noise to the images.")
    parser.add_argument("--depth_output", action="store_true", help="Output depth maps.")
    parser.add_argument("--add_clutter", action="store_true", help="Add clutter to the images.")
    parser.add_argument("--fill_to_max_items", action="store_true", help="Fill the camera collection to the maximum number of items.")
    parser.add_argument("--will_render_image", action="store_true", help="Render the images.")
    parser.add_argument("--min_items", type=int, default=1, help="Minimum number of items to place in the camera collection.")
    parser.add_argument("--max_items", type=int, default=15, help="Maximum number of items to place in the camera collection.")
    parser.add_argument("--min_clutter_items", type=int, default=5, help="Minimum number of clutter items to place in the camera collection.")
    parser.add_argument("--max_clutter_items", type=int, default=15, help="Maximum number of clutter items to place in the camera collection.")
    parser.add_argument("--min_z", type=float, default=0.3, help="Minimum distance from camera to near plane.")
    parser.add_argument("--max_z", type=float, default=0.9, help="Maximum distance to far plane.")
    parser.add_argument("--rendered_image_resolution", type=int, default=1000, help="Resolution of the rendered images. In tuple format (width, height).")
    
    args = parser.parse_args(argv)
    params = vars(args)
    main_main(params)