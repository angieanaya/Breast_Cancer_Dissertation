'''
@File   :   combineMasks.py
@Date   :   30/09/2022
@Author :   María de los Ángeles Contreras Anaya
@Version:   1.0
@Desc:   Program that takes all masks from a particular image and merges them all into one single mask image.
'''
import os
import cv2
import numpy as np

PATH = "CESM_&_MASKS" # name of dir where images with multiple masks are saved
image_ids = next(os.walk(PATH))[1] #list all images from directory

for id in image_ids:
    path = os.path.join( PATH, id, 'masks/')
    ground_truth_img = os.path.join( PATH, id, id + ".jpg")
    image = cv2.imread(ground_truth_img, cv2.IMREAD_GRAYSCALE)
    H, W = image.shape
    mask = np.zeros((H, W, 1)) # generate a black image of the same width and height of the original one
    mask_num = next(os.walk(path))[2]

    # it there are multiple masks
    if(len(mask_num)>1):
        for mask_file in next(os.walk(path))[2]:
            mask_ = cv2.imread(os.path.join(path, mask_file))
            mask = np.maximum(mask, mask_) # save the maximum value --> 255 (white pixel)
            cv2.imwrite(os.path.join(path, id + ".png") , mask)