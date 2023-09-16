import argparse

from two_dimensions.filters import Filters

import skimage as sk
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


image_path = "D:\\data_sets\\CV\\MyImages"
save_path = "D:\\data_sets\\CV\\Results"
plot_path = "D:\\data_sets\\CV\\Plots"

file_type = ".jpg"

for folder in os.listdir(image_path):
    img = Filters(os.path.join(image_path, folder), file_type)
    combined, masks = img.infinity_focus("sobel", 35, 15, 'segment')
    img.help.plot_gaussian_pyramid(masks, folder, plot_path)
    sk.io.imsave(os.path.join(save_path,f'stacked_image_{folder}.jpg'),  (combined * 255).astype(np.uint8))

pan_img = Filters(save_path, file_type)
panorama_image = pan_img.opencv_stitch()
cv2.imwrite(os.path.join(save_path,f'panorama.jpg'), panorama_image)