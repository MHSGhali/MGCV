import os

import skimage as sk
import matplotlib.pyplot as plt
import numpy as np

class InfinityFocus():
    def __init__(self, image_path, file_type, sigma=2, truncate=4, smallestDimension=100):
        self.image_path, self.file_type = image_path, file_type
        self.sigma, self.truncate, self.smallestDimension= sigma, truncate, smallestDimension
        images = self.load_images()
        mask = []    
        for image in images:
            _, gau_p = self.laplacianPyramid(image)
            mask.append(self.pyramidToMask(image,gau_p))
        self.show_images(mask)
            
    def load_images(self):
        images = []
        for filename in sorted(os.listdir(self.image_path)):
            if filename.endswith(self.file_type):
                img = sk.io.imread(os.path.join(self.image_path,filename))
                if img is not None:
                    images.append(img)
        return np.array(images)

    def gaussianPyramid(self, image):
        pyramid = []
        pyramid.append(image)
        while max(np.shape(image)) > self.smallestDimension:
            image = sk.filters.gaussian(image, sigma = self.sigma, truncate = self.truncate, channel_axis=-1)[::2, ::2, :]
            pyramid.append(image)
        return pyramid

    def laplacianPyramid(self, image):
        image = image.astype(float) / 255
        image_pyramid = []
        gaussian_pyramid = []
        image_pyramid.append(image)
        while max(np.shape(image)) > self.smallestDimension:
            temp = sk.filters.gaussian(image, sigma = self.sigma, truncate = self.truncate, channel_axis=-1)
            residual = image - temp 
            gaussian_pyramid.append(residual)
            image = image[::2, ::2, :]
            image_pyramid.append(image)
            
        return image_pyramid, gaussian_pyramid

    def pyramidToMask(self, image, pyramid):
        focus_mask = np.zeros_like(image, dtype=np.float64)
        for level in pyramid:
            level_laplacian = sk.transform.resize(level, focus_mask.shape)
            focus_mask += level_laplacian/np.max(level_laplacian)
        normalized_mask = (focus_mask - np.min(focus_mask)) / (np.max(focus_mask) - np.min(focus_mask))
        grayscale_mask = sk.color.rgb2gray(normalized_mask) 
        binary_mask = (grayscale_mask >= 0.5).astype(np.uint8)
        return binary_mask

    def combine_images(self, images, pyramids, normalized_mask, masks):
        blended_image = np.zeros_like(images[0])
        for i, pyramid_level in enumerate(pyramids):
            blended_image += pyramid_level * (normalized_mask / masks[i])
        return blended_image

    def show_images(self, images, result = None):
        n_images = len(images)
        fig, axes = plt.subplots(1, n_images, figsize=(15, 5))
        fig.tight_layout()
        if n_images == 1:  # Handle the case of a single image
            axes.imshow(images[0])
            axes.axis('off')
        else:
            for i in range(n_images):
                axes[i].imshow(images[i])
                axes[i].axis('off')
        plt.show()


