import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from .processing import ImageProcessing, hull_sweep
from .homography import Homography
from .helper import Helpers
from .timing import stage
import numpy as np
import skimage as sk
import cv2


def _threshold_one_image(payload):
    """Edge-detect and threshold one image. Module-level so it can be pickled."""
    image, edge_method = payload
    process = ImageProcessing()

    image = sk.color.rgb2gray(image)

    if edge_method == "sobel":
        filtered, _ = process.sobel(image)
    elif edge_method == "gaussian":
        _, filtered = process.gaussian(image)

    return process.get_threshold(process.normalize_image(filtered))


def _hull_band(payload):
    """Union the window hulls for one horizontal band of an image.

    The band already contains every row its windows touch, so it needs no data from
    the rest of the image. Bands overlap by (window_size - stride) rows, but the
    combining operation is OR, which is commutative and associative -- so splitting
    the row range and ORing the pieces back together gives the same mask as the
    serial sweep in ImageProcessing.convex_hull_window.
    """
    band, window_size, stride, num_cols = payload
    num_rows = (band.shape[0] - window_size) // stride + 1
    combined = np.zeros(band.shape, dtype=bool)
    return hull_sweep(band, window_size, stride, num_rows, num_cols, combined)


def _band_tasks(binaries, window_num, stride_window, bands_per_image):
    """Split each thresholded image into row bands, mirroring convex_hull_window."""
    for index, binary in enumerate(binaries):
        window_size = min([x // window_num for x in np.shape(binary)])
        stride = window_size // stride_window
        num_rows = (binary.shape[0] - window_size) // stride + 1
        num_cols = (binary.shape[1] - window_size) // stride + 1

        edges = np.linspace(0, num_rows, bands_per_image + 1).astype(int)
        for first, last in zip(edges[:-1], edges[1:]):
            if last <= first:
                continue
            # Rows this band's windows span: the first window's top through the last
            # window's bottom.
            row_offset = first * stride
            row_limit = (last - 1) * stride + window_size
            yield index, row_offset, (binary[row_offset:row_limit], window_size, stride, num_cols)


class Filters():
    def __init__(self, image_path, file_type):
        with stage('Initialization'):
            self.help = Helpers(image_path, file_type)
            self.process = ImageProcessing()
            self.homography = Homography()
            self.images = self.help.load_images()

    def compensate_focal_distance(self):
        for i,img in enumerate(self.images):
            img = sk.img_as_ubyte(sk.color.rgb2gray(img))
            if i == 0:
                img1 = img
            else:
                homography = self.homography.macthed_image(img,img1)
                if homography is None:
                    # Too few matches to register this frame; leave it as-is and keep
                    # the previous reference rather than crashing in warpPerspective.
                    print(f'Warning: no homography for image {i}, leaving it unwarped')
                    img1 = img
                    continue
                self.images[i] = cv2.warpPerspective(self.images[i], homography, (img1.shape[1], img1.shape[0]),flags=cv2.INTER_LINEAR)
                img1 = cv2.warpPerspective(img, homography, (img1.shape[1], img1.shape[0]),flags=cv2.INTER_LINEAR)
    
    def infinity_focus(self, edge_method, window_num, stride_window, cluster_method = 'group',
                       workers = None, select = 'hull', sharpness_sigma = 5):
        """workers: process count for the masking stage. None uses every core,
        1 forces the serial path.

        The masking stage dominates runtime and its work is independent per image and
        per row band, so this is the main lever on total time. Results are identical
        either way.

        select picks how a pixel is attributed to a frame:
          'hull'      -- threshold edges, union the convex hulls of sliding windows,
                         then resolve overlaps between frames. The original behaviour.
          'sharpness' -- take each pixel from whichever frame is locally sharpest.
                         Sharper output and far cheaper, but different output, so it is
                         opt-in. It needs no overlap resolution: the masks it returns
                         already partition the frame exactly.
        """
        if select not in ('hull', 'sharpness'):
            raise ValueError(f"select must be 'hull' or 'sharpness', got {select!r}")
        with stage('Homography Matching'):
            self.compensate_focal_distance()

        if workers is None:
            workers = os.cpu_count() or 1

        with stage('Edge Detection and Masking'):
            if select == 'sharpness':
                masks = self.process.sharpness_masks(self.images, sigma=sharpness_sigma)
            elif workers > 1:
                try:
                    masks = self._masks_parallel(
                        edge_method, window_num, stride_window, workers)
                except BrokenProcessPool:
                    # macOS spawns workers by re-importing __main__, which fails outright
                    # when the entry point is not a real file -- a REPL, a heredoc, a
                    # notebook. Losing the speed-up beats losing the run.
                    print('Warning: process pool unavailable (is __main__ a real file, '
                          'under an "if __name__" guard?), falling back to serial masking')
                    masks = self._masks_serial(edge_method, window_num, stride_window)
            else:
                masks = self._masks_serial(edge_method, window_num, stride_window)
        
        # sharpness_masks() already yields one owner per pixel, so there is nothing to
        # arbitrate and no unclaimed region to distribute.
        if select == 'hull':
            with stage('Mask Cleaning'):
                masks = self.process.clean_masks(self.images, masks, method = cluster_method)

        # masked_images() zeroes every pixel outside each mask, but stacked_image()
        # reads each result only *inside* that same mask -- exactly where the pixels are
        # left untouched, i.e. equal to the originals. Its output was therefore always
        # discarded, so passing the originals is identical and skips both the
        # full-stack copy and the per-pixel zeroing.
        with stage('Image Stacking'):
            stacked_image = self.process.stacked_image(self.images, self.images, masks)
        return stacked_image, masks
    
    def _masks_serial(self, edge_method, window_num, stride_window):
        masks = []
        for img in self.images:
            binary = _threshold_one_image((img, edge_method))
            hull = self.process.convex_hull_window(binary, window_num, stride_window)
            masks.append(self.process.fill_voids(hull))
        return masks

    def _masks_parallel(self, edge_method, window_num, stride_window, workers):
        """Same masks as _masks_serial, computed across processes.

        Work is split per (image, row band) rather than per image: the images differ
        several-fold in cost, so per-image tasks alone leave most cores idle waiting on
        the slowest image.
        """
        with ProcessPoolExecutor(max_workers=workers) as pool:
            # map() preserves order, so binaries stay aligned with self.images.
            binaries = list(pool.map(_threshold_one_image,
                                     [(img, edge_method) for img in self.images]))

            # Aim for a few bands per worker so uneven bands still balance out.
            bands_per_image = max(1, -(-workers * 3 // max(1, len(binaries))))
            tasks = list(_band_tasks(binaries, window_num, stride_window, bands_per_image))

            hulls = [np.zeros(binary.shape, dtype=bool) for binary in binaries]
            results = pool.map(_hull_band, [payload for _, _, payload in tasks])
            for (index, row_offset, payload), band in zip(tasks, results):
                hulls[index][row_offset:row_offset + band.shape[0]] |= band

        return [self.process.fill_voids(hull) for hull in hulls]

    # Static because it reads the directory itself and touches no instance state.
    # Constructing Filters() just to reach it would decode every image a second time,
    # since __init__ eagerly loads the same folder that this method re-reads.
    @staticmethod
    def opencv_stitch(image_path, file_type, scale=None, compose_resol=None):
        """Stitch every matching image in image_path into one panorama.

        scale and compose_resol both trade output for speed and are off by default:
        scale shrinks the inputs before stitching, compose_resol caps the resolution
        Stitcher composites at (megapixels; -1 means full resolution).
        """
        paths = [os.path.join(image_path, name)
                 for name in sorted(os.listdir(image_path))
                 if name.endswith(file_type)]

        with stage('Panorama Decode'):
            # imread releases the GIL, so threads decode genuinely in parallel -- and
            # unlike processes they hand back the pixels without pickling them. Order
            # follows `paths`, because Stitcher's output depends on input order.
            with ThreadPoolExecutor(max_workers=min(8, len(paths) or 1)) as pool:
                images = [img for img in pool.map(cv2.imread, paths) if img is not None]

        if not images:
            raise ValueError(f'No {file_type} images found in {image_path}')

        if scale is not None:
            helper = Helpers(image_path, file_type)
            images = [helper.scale_image(img, scale) for img in images]

        with stage('Panorama Stitching'):
            stitching = cv2.Stitcher.create()
            stitching.setPanoConfidenceThresh(0.7)
            stitching.setWaveCorrection(False)
            if compose_resol is not None:
                stitching.setCompositingResol(compose_resol)
            status, pano_im = stitching.stitch(images)
            # result = Homography().remove_void_regions(pano_im)

        if status != cv2.Stitcher_OK:
            # Without this the failure surfaces later as an opaque error from imwrite,
            # because pano_im is None.
            raise RuntimeError(
                f'Stitching failed with status {status} '
                f'(need more overlap between the {len(images)} input images)')
        return pano_im
