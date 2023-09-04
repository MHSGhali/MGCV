import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import cv2
from PIL import ExifTags, Image

class Helpers():
    def __init__(self, image_path, file_type):
        self.image_path, self.file_type = image_path, file_type            
    
    def read_image_with_rotation(self, image_path):
        image = sk.io.imread(image_path)
        try:
            with Image.open(image_path) as img:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        exif = dict(img._getexif().items())
                        if orientation in exif:
                            if exif[orientation] == 3:
                                image = sk.transform.rotate(image, 180, resize=True)
                            elif exif[orientation] == 6:
                                image = sk.transform.rotate(image, 270, resize=True)
                            elif exif[orientation] == 8:
                                image = sk.transform.rotate(image, 90, resize=True)
        except (AttributeError, KeyError, IndexError):
            # Handle exceptions when there is no Exif data or if the image format doesn't support Exif
            pass
        return image

    def load_images(self):
        images = []
        for filename in sorted(os.listdir(self.image_path)):
            if filename.endswith(self.file_type):
                img_path = os.path.join(self.image_path, filename)
                img = self.read_image_with_rotation(img_path)
                if img is not None:
                    images.append(img)
        return np.array(images)
    
    def scale_image(self, img, scale_percent):
        width = int(img.shape[1] * scale_percent)
        height = int(img.shape[0] * scale_percent)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return resized

    def plot_gaussian_pyramid(self, pyramid):
        num_levels = len(pyramid)
        rows = int(np.ceil(np.sqrt(num_levels)))
        cols = int(np.ceil(num_levels / rows))

        plt.figure(figsize=(12, 8))

        for i, level in enumerate(pyramid):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(level)
            plt.title(f'Level {i + 1}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def show_images(self, images, result):
        fig, axes = plt.subplot_mosaic("ABC;DDD")
        fig.tight_layout()
        axes["A"].imshow(images[0])
        axes["A"].axis('off')
        axes["B"].imshow(images[1])
        axes["B"].axis('off')
        axes["C"].imshow(images[2])
        axes["C"].axis('off')
        axes["D"].imshow(result)
        axes["D"].axis('off')
        plt.show()