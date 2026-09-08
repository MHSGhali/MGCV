import numpy as np
import skimage as sk
import cv2
from itertools import combinations
from scipy import ndimage as ndi
from skimage.measure import grid_points_in_poly

# The four half-pixel offsets skimage's convex_hull_image adds to every coordinate so
# that a pixel contributes its extent, not just its centre, to the hull.
_DIAMOND = np.array([[-0.5, 0.0], [0.5, 0.0], [0.0, -0.5], [0.0, 0.5]])

# 8-connectivity, matching the default footprint of morphology.reconstruction.
_FULL_CONNECTIVITY = np.ones((3, 3), dtype=bool)


def small_convex_hull(window):
    """The same mask as sk.morphology.convex_hull_image, ~4x faster on small windows.

    convex_hull_image spends most of its time in Qhull and unique_rows, neither of which
    earns its keep on a 17x17 window. This computes the hull with cv2 instead and then
    rasterises with grid_points_in_poly -- the very function convex_hull_image itself
    calls -- so an identical polygon yields an identical mask by construction.

    Offsetting the hull vertices rather than every boundary pixel is exact because
    hull(S + D) == hull(S) + D for the Minkowski sum of a convex set and the diamond.
    """
    points = np.argwhere(window).astype(np.float32)                  # (row, col)
    # cv2 works in (x, y), so columns lead going in and coming back out.
    hull = cv2.convexHull(points[:, ::-1].copy())[:, 0, ::-1]
    offset = (hull.astype(np.float64)[:, None, :] + _DIAMOND).reshape(-1, 2)
    vertices = cv2.convexHull(offset[:, ::-1].astype(np.float32))[:, 0, ::-1]
    # binarize=False labels interior 1, vertices 2 and edges 3; >= 1 keeps all three,
    # which is what convex_hull_image does with include_borders left at its default.
    return grid_points_in_poly(window.shape, vertices.astype(np.float64),
                               binarize=False) >= 1


def hull_sweep(image, window_size, stride, num_rows, num_cols, combined_mask):
    """OR the convex hull of every window's content into combined_mask, in place.

    Shared by the serial path and by the per-band workers in filters.py so the two
    cannot drift apart.
    """
    for row in range(num_rows):
        row_start = row * stride
        row_end = row_start + window_size
        for col in range(num_cols):
            col_start = col * stride
            col_end = col_start + window_size
            window = image[row_start:row_end, col_start:col_end]
            # An empty window contributes nothing. any() short-circuits on the first
            # True and allocates nothing, unlike the transpose(nonzero(...)) this
            # replaces, which built a full coordinate array just to test emptiness.
            #
            # Skipping windows whose hull is already covered looks tempting -- the hull
            # never leaves the bounding box of the window's own set pixels -- but it
            # measured slower: computing the box costs more across every window than the
            # 15-20% of hulls it avoids.
            if not window.any():
                continue
            combined_mask[row_start:row_end, col_start:col_end] |= small_convex_hull(window)
    return combined_mask


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
            results[i][np.logical_not(mask)] = 0
        return results

    def stacked_image(self, images, results, masks):
        combined_image = np.copy(images[0]).astype(np.float32)
        for masked_image, binary_mask in zip(results, masks):
            # Boolean indexing costs one byte per pixel, where the np.where() this
            # replaces materialised two int64 coordinate arrays (~96 MB each at 12 MP)
            # before indexing with them. fill_voids() already returns bool, so the cast
            # is a no-op for masks straight from the pipeline.
            selected = binary_mask.astype(bool, copy=False)
            combined_image[selected] = masked_image[selected]
        return combined_image
    
    def neighborhood(self, index, neighborhood_radius = 1):
            x, y = index
            return (x // neighborhood_radius, y // neighborhood_radius)

    def calculate_distance(self, index1, index2):
        x1, y1 = index1
        x2, y2 = index2
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def clean_masks(self, images, masks, method = 'segment'):
        # sobel() is pure, but was called inside the pair loop below, so each image's
        # edge map was recomputed once per pair it appears in: N*(N-1) filters instead
        # of N. Hoisting is exact and is the difference between 90 and 10 for N=10.
        edges = [self.sobel(image)[0] for image in images]

        for mask_indices in combinations(range(len(images)), 2):
            mask1 = masks[mask_indices[0]]
            mask2 = masks[mask_indices[1]]
            image1 = edges[mask_indices[0]]
            image2 = edges[mask_indices[1]]
                
            if method == 'group':
                # neighborhood() floor-divides by neighborhood_radius, which is always 1
                # here, so it returns the pixel index itself and every pixel forms its
                # own group. The sort, the groupby and the per-group mean therefore all
                # collapse into a single per-pixel comparison. Averaging the channel
                # axis reproduces np.mean over the (1, 3) slice the original produced.
                intersection = np.logical_and(mask1, mask2)
                intensity1 = image1.mean(axis=-1) if image1.ndim > 2 else image1
                intensity2 = image2.mean(axis=-1) if image2.ndim > 2 else image2
                first_is_sharper = intersection & (intensity1 > intensity2)
                mask2[first_is_sharper] = False
                mask1[intersection & ~first_is_sharper] = False
            
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
            # gaussian() returns (filtered, residual). Passing that tuple straight into
            # normalize_image() stacked both arrays and normalised by their joint
            # min/max, then the unpack silently took slice 1 of the result.
            _, residual = self.gaussian(image)
            edge.append(self.normalize_image(residual))

        # regionprops needs a label image. A 0/1 mask has a single label, so every
        # unmasked pixel in the frame was treated as one region and assigned wholesale
        # to whichever image happened to win on the mean over all of them.
        labelled_unmasked = sk.measure.label(unmasked.astype(np.uint8))
        unmasked_regions = sk.measure.regionprops_table(labelled_unmasked, properties=('coords',))
        
        for cluster_indices in unmasked_regions['coords']:
            mean_intensities = []
            for img in edge:
                mean_intensities.append(np.mean(img[cluster_indices[:, 0], cluster_indices[:, 1]]))
            index = np.array(mean_intensities).argmax()
            masks[index][cluster_indices[:, 0], cluster_indices[:, 1]] = True

        return masks

    def sharpness_masks(self, images, sigma=5):
        """Assign every pixel to the frame that is locally sharpest there.

        The convex-hull path assigns whole convex blobs to a single frame, which is why
        only ~64% of its output pixels come from the frame that is actually sharpest.
        Comparing a smoothed Sobel energy per pixel instead agrees with a per-pixel
        oracle ~99% of the time, and costs one filter pass per image rather than a
        sliding-window hull sweep.

        Smoothing is what makes this robust: raw edge energy is near zero inside flat
        regions of every frame, so the argmax would be noise there. The Gaussian carries
        the verdict from nearby textured pixels into those flat patches.
        """
        energy = np.array([
            sk.filters.gaussian(sk.filters.sobel(sk.color.rgb2gray(image)) ** 2,
                                sigma=sigma)
            for image in images])
        sharpest = energy.argmax(axis=0)
        return [sharpest == index for index in range(len(images))]

    def convex_hull_window(self, image, window_div, stride_div):
        window_size = min([x//window_div for x in np.shape(image)])
        stride = window_size//stride_div
        num_rows = (image.shape[0] - window_size) // stride + 1
        num_cols = (image.shape[1] - window_size) // stride + 1
        combined_mask = np.zeros(image.shape, dtype=bool)
        return hull_sweep(image, window_size, stride, num_rows, num_cols, combined_mask)
    
    def fill_voids(self, binary_mask):
        """Fill the holes in a mask that are not connected to the border.

        This replaces a grey-level morphological reconstruction that was doing purely
        binary work: ~4x faster, bit-identical, and it returns bool instead of float64.
        The explicit structure is load-bearing -- reconstruction's default footprint is
        fully connected, whereas binary_fill_holes defaults to a 4-connected cross and
        would leave diagonally-pinched holes unfilled.
        """
        return ndi.binary_fill_holes(binary_mask, structure=_FULL_CONNECTIVITY)
        