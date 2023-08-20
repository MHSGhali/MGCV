import os

from .processing import ImageProcessing
from .helper import Helpers
import numpy as np
import skimage as sk

class Filters():
    def __init__(self, image_path, file_type, sigma=10, truncate = 4):
        self.help = Helpers(image_path, file_type)
        self.process = ImageProcessing()
        self.images = self.help.load_images()
        self.sigma, self.truncate = sigma, truncate

    def infinity_focus(self, threshold):
        filters = []
        for image in self.images:
            image = sk.color.rgb2gray(image)
            filtered = self.process.gaussian(image, self.sigma, self.truncate)
            residual = abs(image - filtered)
            normalized = sk.measure.find_contours(residual/np.max(residual), 0.5)
            filters.append(normalized)
        self.help.show_images(self.images, filters)