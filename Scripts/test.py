import os
import json

annotations_folder = 'C:/Users/paulb/Documents/TUDresden/Bachelor/dataset/annotations'

print(os.listdir(annotations_folder))

with open(os.path.join(annotations_folder, '29112024-002317-179.json')) as f:
    annotations = json.load(f)
    print(annotations)