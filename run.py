import argparse

from two_dimensions.filters import Filters
from two_dimensions.processing import ImageProcessing
from two_dimensions.helper import Helpers

import skimage as sk
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

image_path = "D:\\data_sets\\CV\\blending\\focus_stack"
save_path = "D:\\data_sets\\CV\\blending\\Results"
plot_path = "D:\\data_sets\\CV\\blending\\Plots"
file_type = ".jpg"

for folder in os.listdir(image_path):
    img = Filters(os.path.join(image_path, folder), file_type)
    combined, masks = img.infinity_focus("sobel", 20, 10, 'segment')
    img.help.plot_gaussian_pyramid(masks, folder, plot_path)
    sk.io.imsave(os.path.join(save_path,f'stacked_image_{folder}.jpg'), combined)

# image_path = ".\\Panorama\\Images"
# file_type = ".jpg"

# img = Filters(image_path, file_type)

# panorama_image = img.panorama(scale=0.25, num_iter=50000, threshold=10)
# img.help.plot_gaussian_pyramid(panorama_image)
# for i, image in enumerate(panorama_image):
#     sk.io.imsave(f'panorama_image_{i}.jpg', image)

# panorama_image = img.opencv_stitch()
# cv2.imwrite('panorama_image.jpg', panorama_image)