import argparse

from two_dimensions.filters import Filters
from two_dimensions.processing import ImageProcessing
from two_dimensions.helper import Helpers

import skimage as sk
import matplotlib.pyplot as plt
import numpy as np

image_path = ".\Test\\Images"
file_type = ".jpg"

img = Filters(image_path, file_type)

combined, masks = img.infinity_focus("sobel", 20, 10, 'segment')
img.help.plot_gaussian_pyramid(masks)
sk.io.imsave('stacked_image.jpg', combined)

