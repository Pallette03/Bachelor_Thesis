import os
import bpy # type: ignore

def preload_objects(object_folder=os.path.join("G:\GoogleScannedObjects\extracted_files"),  start_index=0, amount=50):
    object_list = os.listdir(object_folder)
    
    if (start_index + amount) > len(object_list):
        start_index = 0
    
    object_list = object_list[start_index:(start_index + amount)]
    
    collection_name = "Clutter"
    
    def find_layer_collection(layer_collection, name):
        if layer_collection.collection.name == name:
            return layer_collection
        for child in layer_collection.children:
            result = find_layer_collection(child, name)
            if result:
                return result
        return None

    # Get the active view layer
    view_layer = bpy.context.view_layer

    # Find the desired collection
    layer_collection = find_layer_collection(view_layer.layer_collection, collection_name)

    # Set it as active if found
    if layer_collection:
        view_layer.active_layer_collection = layer_collection
        print(f"Active collection set to: {collection_name}")
    else:
        print(f"Collection '{collection_name}' not found.")
    
    for folder in object_list:
        if os.path.isdir(os.path.join(object_folder, folder)):
            obj_file = os.path.join(object_folder, folder, "meshes", "model.obj")
            if os.path.isfile(obj_file):
                
                # Load the object
                bpy.ops.wm.obj_import(filepath=obj_file)
                # Get the loaded object by name since its always model
                obj = bpy.data.objects.get("model")
                if not obj:
                    print(f"Object not found: {obj_file}")
                    input("Press Enter to continue...")
                    continue
                obj.name = folder
                obj.hide_render = True
                obj.hide_set(True)
                
                # Rescale the object to one quarter of its size
                obj.scale[0] = 0.25
                obj.scale[1] = 0.25
                obj.scale[2] = 0.25
                

                print(f"Preloaded object: {obj.name}")
                
                
object_folder = os.path.join("G:\GoogleScannedObjects\extracted_files")

preload_objects(object_folder)