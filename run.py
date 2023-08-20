import argparse

from two_dimensions.filters import Filters
from two_dimensions.processing import ImageProcessing
from two_dimensions.helper import Helpers

image_path = "D:\\data_sets\\CV\\blending"
file_type = ".jpg"

img = Filters(image_path, file_type, 10, 4)
img.infinity_focus(0.15)