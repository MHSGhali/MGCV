import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk

class Helpers():
    def __init__(self, image_path, file_type):
        self.image_path, self.file_type = image_path, file_type            
    
    def load_images(self):
        images = []
        for filename in sorted(os.listdir(self.image_path)):
            if filename.endswith(self.file_type):
                img = sk.io.imread(os.path.join(self.image_path,filename))
                if img is not None:
                    images.append(img)
        return np.array(images)
    
        # image_dict = {}
        # for filename in sorted(os.listdir(self.image_path)):
        #     if filename.endswith(self.file_type):
        #         prefix = filename.split("_")[0]
        #         img = sk.io.imread(os.path.join(self.image_path, filename))
        #         if img is not None:
        #             if prefix not in image_dict:
        #                 image_dict[prefix] = []
        #             image_dict[prefix].append(img)
        # return image_dict
    
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