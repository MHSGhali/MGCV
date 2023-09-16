import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
from itertools import combinations, groupby

class ImageProcessing():
    def __init__(self):      
        pass
    
    def gaussian(self, image, sigma_ = 10, truncate_ = 4):
        filtered_image = sk.filters.gaussian(image, sigma=sigma_, truncate=truncate_, channel_axis = -1)
        residual_image = image - filtered_image
        return filtered_image, residual_image
    
    def laplacian(self, image, k_size=3):
        filtered_image =  sk.filters.laplace(image, ksize=k_size)
        residual_image = image - filtered_image
        return filtered_image, residual_image

    def hysteresis(self, edges, low, high):
        filtered_image = sk.filters.apply_hysteresis_threshold(edges, low, high)
        return filtered_image
    
    def sobel(self, image):
        filtered_image = sk.filters.sobel(image)
        residual_image = image - filtered_image
        return filtered_image, residual_image
    
    def get_threshold(self, image):
        threshold = sk.filters.threshold_otsu(image)
        binary =  image >= threshold
        return binary

    def normalize_image(self, image):
        min_value, max_value = 0, 255
        min_pixel, max_pixel = np.min(image), np.max(image)
        normalized_image = (image - min_pixel) / (max_pixel - min_pixel)
        normalized_image = normalized_image * (max_value - min_value) + min_value
        return normalized_image.astype(np.uint8)

    def laplacianPyramid(self, image, sigma_ = 10, truncate_ = 4, smallest_dim = 360):
        image = image.astype(float) / 255
        gaussian_pyramid = []
        while max(np.shape(image)) > smallest_dim:
            _, res = self.gaussian(image, sigma_, truncate_)
            gaussian_pyramid.append(res)
            image = image[::2, ::2]
        return image.astype(np.uint8), gaussian_pyramid

    def masked_images(self, images, masks):
        results = images.copy()
        for i, mask in enumerate(masks):
            void_indices = np.argwhere(np.logical_not(mask))
            for idx in void_indices:
                results[i][idx[0], idx[1], :] = [0, 0, 0]
        return results

    def stacked_image(self, images, results, masks):
        combined_image = np.copy(images[0]).astype(np.float32)
        for masked_image, binary_mask in zip(results, masks):
            row_indices, col_indices = np.where(binary_mask)
            combined_image[row_indices, col_indices, :] = masked_image[row_indices, col_indices, :]
        return combined_image
    
    def neighborhood(self, index, neighborhood_radius = 1):
            x, y = index
            return (x // neighborhood_radius, y // neighborhood_radius)

    def calculate_distance(self, index1, index2):
        x1, y1 = index1
        x2, y2 = index2
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def clean_masks(self, images, masks, method = 'segment'):
        for mask_indices in combinations(range(len(images)), 2):
            mask1 = masks[mask_indices[0]]
            mask2 = masks[mask_indices[1]]
            image1,_ = self.sobel(images[mask_indices[0]])
            image2,_ = self.sobel(images[mask_indices[1]])
                
            if method == 'group':
                intersection_indices = np.where(np.logical_and(mask1, mask2))
                sorted_indices = sorted(zip(*intersection_indices), key=self.neighborhood)
                for _, group in groupby(sorted_indices, self.neighborhood):
                    group_indices = list(group)
                    group_indices = tuple(zip(*group_indices)) 
                    if np.mean(image1[group_indices]) > np.mean(image2[group_indices]):
                        mask2[group_indices] = False
                    else:
                        mask1[group_indices] = False
            
            elif method == 'segment':
                labeled_intersection = sk.measure.label(np.logical_and(mask1, mask2), connectivity=2)
                props = sk.measure.regionprops(labeled_intersection)
                for prop in props:
                    cluster_indices = prop.coords
                    mean_intensity_1 = np.mean(image1[cluster_indices[:, 0], cluster_indices[:, 1]])
                    mean_intensity_2 = np.mean(image2[cluster_indices[:, 0], cluster_indices[:, 1]])
                    if mean_intensity_1 > mean_intensity_2:
                        mask2[cluster_indices[:, 0], cluster_indices[:, 1]] = False
                    else:
                        mask1[cluster_indices[:, 0], cluster_indices[:, 1]] = False

        combined_mask = np.logical_or.reduce(masks)
        not_masked = np.logical_not(combined_mask)

        masks = self.distibute_unmasked_regions(images, not_masked, masks)

        return masks
    
    def distibute_unmasked_regions(self, images, unmasked, masks):
        edge = []

        for image in images:
            _, img_edge = self.normalize_image(self.gaussian(image))
            edge.append(img_edge)

        unmasked = unmasked.astype(np.uint8)
        unmasked_regions = sk.measure.regionprops_table(unmasked, properties=('coords', 'image'))
        
        for cluster_indices, _ in zip(unmasked_regions['coords'], unmasked_regions['image']):
            mean_intensities = []
            for img in edge:
                mean_intensities.append(np.mean(img[cluster_indices[:, 0], cluster_indices[:, 1]]))
            index = np.array(mean_intensities).argmax()
            masks[index][cluster_indices[:, 0], cluster_indices[:, 1]] = True

        return masks

    def convex_hull_window(self, image, window_div, stride_div):
        window_size = min([x//window_div for x in np.shape(image)])
        stride = window_size//stride_div
        num_rows = (image.shape[0] - window_size) // stride + 1
        num_cols = (image.shape[1] - window_size) // stride + 1
        combined_mask = np.zeros(image.shape, dtype=bool)
        for row in range(num_rows):
            for col in range(num_cols):
                window = image[row * stride:row * stride + window_size, col * stride:col * stride + window_size]
                non_zero_indices = np.transpose(np.nonzero(window))
                if len(non_zero_indices) > 0:
                    hull_mask = sk.morphology.convex_hull_image(window)
                    combined_mask[row * stride:row * stride + window_size, col * stride:col * stride + window_size] |= hull_mask
        return combined_mask
    
    def fill_voids(self, binary_mask):
        seed = np.copy(binary_mask)
        seed[1:-1, 1:-1] = binary_mask.max()
        filled_mask = sk.morphology.reconstruction(seed, binary_mask, method='erosion')
        return filled_mask
        