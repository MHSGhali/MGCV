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
    
    def show_images(self, images, results):
        n_images = len(images)
        fig, axes = plt.subplots(2, n_images, figsize=(15, 5))
        fig.tight_layout()
        if n_images == 1:  # Handle the case of a single image
            axes[0].imshow(images[0])
            axes.axis('off')
            axes[1].imshow(results[0])
            axes.axis('off')
        else:
            for i in range(n_images):
                axes[0,i].imshow(images[i])
                axes[0,i].axis('off')
                # axes[1,i].imshow(results[i])
                # axes[1,i].axis('off')
                for contour in results[i]:
                    axes[1,i].plot(contour[:, 1], contour[:, 0], linewidth=2)
                    axes[1,i].axis('off')
        plt.show()