import argparse

from two_dimensions.filters import Filters
from two_dimensions.processing import ImageProcessing
from two_dimensions.helper import Helpers

import skimage as sk
import matplotlib.pyplot as plt
import numpy as np

# image_path = ".\\Stacking\\Images"
# file_type = ".jpg"

# img = Filters(image_path, file_type)

# combined, masks = img.infinity_focus("sobel", 10, 5, 'segment')
# img.help.plot_gaussian_pyramid(masks)
# sk.io.imsave('stacked_image.jpg', combined)

image_path = ".\\Panorama\\Images"
file_type = ".jpg"

img = Filters(image_path, file_type)

panorama_image = img.panorama()
plt.figure(figsize=(20, 20)); plt.imshow(panorama_image)
sk.io.imsave('panorama_image.jpg', panorama_image)