'''
@File   :   maskGeneration.py
@Date   :   30/09/2022
@Author :   María de los Ángeles Contreras Anaya
@Version:   2.0
@Desc:   Program that generates masks from the annotations exported as JSON from the VGG image annotator app.
'''
import os
import cv2
import json
import numpy as np 

source_folder = os.path.join(os.getcwd(), "CESM") # name of the folder with ground thruth images
json_file = "masks.json" # JSON file with annotations from VGG
saved_images = 0
coordinates = {}								

# open JSON file and save content on the annotations variable
with open(json_file) as f:
  annotations = json.load(f)

def addCoordinates(data, row, image_name, count):
    """
    Extracts X and Y coordinates from the annotations and saves them in dictionary.   
    
    Parameters:
        data(JSON object): Annotations information from the JSON
        row(string): ID on JSON object of the image from where to generate masks
        image_name(string): Name of the image
        count(int): number of mask
    """
    try:
        x_points = data[row]["regions"][count]["shape_attributes"]["all_points_x"]
        y_points = data[row]["regions"][count]["shape_attributes"]["all_points_y"]
    except:
        print(image_name + " has no coordinates, thus skipping.")
        return
    
    # construct pair of coordinates 
    all_points = []
    for i, x in enumerate(x_points):
        all_points.append([x, y_points[i]])
    
    coordinates[image_name] = all_points

# Read annotations from JSON file
for row in annotations:
    image_name = annotations[row]["filename"][:-4] 
    a_count = 0
    num_annotations = len(annotations[row]["regions"])
    # if the image has multiple annotations
    if  num_annotations > 1:
        for _ in range(num_annotations):
            key = image_name + "*" + str(a_count+1)
            addCoordinates(annotations, row, key, a_count)
            a_count += 1
    else:
        addCoordinates(annotations, row, image_name, 0)
        
# Generate folder structure
for image_filename in os.listdir(source_folder):
    save_folder = os.path.join(source_folder, image_filename[:-4])
    masks_folder = os.path.join(save_folder, "masks")
    current_img = os.path.join(source_folder, image_filename)
    new_path = os.path.join(save_folder, image_filename)
    # Create directories
    os.mkdir(save_folder)
    os.mkdir(masks_folder)
    # Copy image to new location
    os.rename(current_img, new_path)

# Generate masks and save them in folder
for row in coordinates:
    folder_name = row.split("*")
    save_folder = os.path.join(source_folder, folder_name[0])
    mask_folder = os.path.join(save_folder, "masks")
    image_path = save_folder+"/" + folder_name[0]+".jpg"
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        H, W = image.shape # get width and height from ground truth image
        mask = np.zeros((H, W))
        arr = np.array(coordinates[row])
        print(arr)
    except:
        print("Not found:", row)
        continue
    saved_images += 1
    # create annotation
    cv2.fillPoly(mask, [arr], color=(255))
    
    # Add a differentiating number to the mask if multiple masks are generated per image
    if len(folder_name) > 1:
        cv2.imwrite(os.path.join(mask_folder, row.replace("*", "_") + ".png") , mask)    
    else:
        cv2.imwrite(os.path.join(mask_folder, row + ".png") , mask)
        
print("# of saved images:", saved_images)