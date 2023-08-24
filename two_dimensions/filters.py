import os

from .processing import ImageProcessing
from .helper import Helpers
import numpy as np
import skimage as sk

class Filters():
    def __init__(self, image_path, file_type):
        self.help = Helpers(image_path, file_type)
        self.process = ImageProcessing()
        self.images = self.help.load_images()

    def infinity_focus(self):
        masks = []
        results = self.images.copy()
        for i, img in enumerate(self.images):
            img = sk.color.rgb2gray(img)
            filtered, _ = self.process.sobel(img)
            normalized = self.process.normalize_image(filtered)
            binary = self.process.get_threshold(normalized)
            hull = self.process.convex_hull_window(binary, 35, 35)
            filled_hull = self.process.fill_voids(hull)
            void_indices = np.argwhere(np.logical_not(filled_hull))
            for idx in void_indices:
                results[i][idx[0], idx[1], :] = [0, 0, 0]
            masks.append(hull)

        combined_image = self.process.combine_masks(results, masks)
        sk.io.imsave('stacked_image.jpg', combined_image)