# import the necessary packages
from sklearn.calibration import LabelEncoder
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import csv
import cv2


# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

# define the base path to the input dataset and then use it to derive
# the path to the input images and annotation CSV files
BASE_PATH = os.path.sep.join([parent_dir, "dataset"])
IMAGES_PATH = os.path.sep.join([BASE_PATH, "images"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "annotations"])
# define the path to the base output directory
BASE_OUTPUT = os.path.sep.join([parent_dir, "output"])

class CustomTensorDataset(Dataset):
	# initialize the constructor
    def __init__(self, tensors, transforms=None):
        self.tensors = tensors
        self.transforms = transforms

    def __getitem__(self, index):
        # grab the image, label, and its bounding box coordinates
        image = self.tensors[0][index]
        label = self.tensors[1][index]
        corners = self.tensors[2][index]
        color = self.tensors[3][index]
        # transpose the image such that its channel dimension becomes
        # the leading one
        image = image.permute(2, 0, 1)
        # check to see if we have any image transformations to apply
        # and if so, apply them
        if self.transforms:
            image = self.transforms(image)
        # return a tuple of the images, labels, and bounding
        # box coordinates
        return (image, label, corners, color)
    
    def __len__(self):
        # return the size of the dataset
        return self.tensors[0].size(0)
    
    
data = []
labels = []
corners = []
colors = []
    

for csv_file in os.listdir(ANNOTS_PATH):
    if csv_file.endswith(".csv"):
        csv_path = os.path.join(ANNOTS_PATH, csv_file)
        with open(csv_path, mode='r') as file:
            image_name = csv_file.split(".")[0]
            
            reader = csv.reader(file)
            
            image_path = os.path.sep.join([IMAGES_PATH, f"{image_name}.png"])
            
            # load the input image from disk
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            img_labels = []
            img_corners = []
            img_colors = []
            
            #Skip the header
            next(reader)
            
            for row in reader:
                label = row[0]
                corner_0 = (int(row[1].split(",")[0]), int(row[1].split(",")[1]))
                corner_1 = (int(row[2].split(",")[0]), int(row[2].split(",")[1]))
                corner_2 = (int(row[3].split(",")[0]), int(row[3].split(",")[1]))
                corner_3 = (int(row[4].split(",")[0]), int(row[4].split(",")[1]))
                corner_4 = (int(row[5].split(",")[0]), int(row[5].split(",")[1]))
                corner_5 = (int(row[6].split(",")[0]), int(row[6].split(",")[1]))
                corner_6 = (int(row[7].split(",")[0]), int(row[7].split(",")[1]))
                corner_7 = (int(row[8].split(",")[0]), int(row[8].split(",")[1]))
                color = row[9]
                
                # update the list of data, labels, bounding boxes, and
                # colors
                img_labels.append(label)
                img_corners.append((corner_0, corner_1, corner_2, corner_3, corner_4, corner_5, corner_6, corner_7))
                img_colors.append(color)
                
            le = LabelEncoder()
            img_labels = le.fit_transform(img_labels)
            img_colors = le.fit_transform(img_colors)
                
            data.append(image)
            labels.append(img_labels)
            corners.append(img_corners)
            colors.append(img_colors)
            
                
data = np.array(data, dtype="float32")
labels = np.array(labels)
corners = np.array(corners, dtype="float32")
colors = np.array(colors)



# convert the labels to a tensor
labels = torch.tensor(labels, dtype=torch.int64)
data = torch.tensor(data, dtype=torch.float32)
corners = torch.tensor(corners, dtype=torch.float32)
colors = torch.tensor(colors, dtype=torch.float32)

# construct the custom dataset
dataset = CustomTensorDataset(tensors=(data, labels, corners, colors))

print(dataset.__len__())
