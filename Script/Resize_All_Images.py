# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 08:58:47 2019

@author: uidv7259
"""
import os
from skimage import io
from skimage.transform import resize, rescale
from numpy import uint8

folder_path = "c:/Users/uidv7259/Pictures/Regalo_Resized/"

def resize_image(img, nx = 80, ny = 80):
    new_image = resize(img, (nx, ny))
    resized_image = new_image * 255
    new_image = resized_image.astype(uint8)
    return new_image

def rescale_image(img, factor = 0.5):
    new_image = rescale(img, factor, anti_aliasing = False)
    rescaled_image = new_image * 255
    new_image = rescaled_image.astype(uint8)
    return new_image

directory = os.fsencode(folder_path)
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".jpg") or filename.endswith(".JPG"):
         new_img = io.imread(folder_path + filename)
         resized_img = resize_image(new_img, 80, 80)
#         rescaled_img = rescale_image(new_img, 0.5)
         io.imsave(folder_path + filename, resized_img)
