import argparse
import os

from two_dimensions.filters import Filters

import skimage as sk
import cv2
import numpy as np
from PIL import Image


DEFAULT_IMAGE_PATH = "D:\\data_sets\\CV\\MyImages"
DEFAULT_SAVE_PATH = "D:\\data_sets\\CV\\Results"
DEFAULT_PLOT_PATH = "D:\\data_sets\\CV\\Plots"


def to_uint8(image):
    """Convert a stacked image to 8-bit for saving.

    stacked_image() preserves whatever range the loaded frames had: 0-255 for the
    usual uint8 input, but 0-1 for the float images sk.transform.rotate produces when
    EXIF orientation is applied. Scaling unconditionally by 255 saturates the former.
    """
    peak = float(np.nanmax(image)) if image.size else 0.0
    if peak <= 1.0:
        image = image * 255
    return np.clip(image, 0, 255).astype(np.uint8)


def save_stacked(image, path, save_format):
    """Write the stack, defaulting to JPEG for backwards compatibility.

    The default encoder settings cost more accuracy than the stacking algorithm does --
    on the benchmark stack the round-trip moves RMSE-vs-ideal from 13.3 to 19.4 -- so
    JPEG is written at quality 95, and png is offered for a lossless result.
    """
    destination = f'{path}.{save_format}'
    if save_format == 'jpg':
        # Straight to Pillow: skimage.io deprecated forwarding plugin kwargs like
        # `quality`, and its default (75) is what makes the round-trip so lossy.
        Image.fromarray(image).save(destination, quality=95)
    else:
        sk.io.imsave(destination, image)
    return destination


def stack_folders(args):
    for folder in sorted(os.listdir(args.image_path)):
        source = os.path.join(args.image_path, folder)
        if not os.path.isdir(source):
            continue
        # Skip sibling folders that hold no matching images rather than aborting the
        # whole batch on the first one.
        if not any(name.endswith(args.file_type) for name in os.listdir(source)):
            continue
        print(f'Stacking {folder}')
        img = Filters(source, args.file_type)
        combined, masks = img.infinity_focus(
            "sobel", args.window_num, args.stride_window, 'segment', workers=args.workers,
            select=args.select, sharpness_sigma=args.sharpness_sigma)
        img.help.plot_gaussian_pyramid(masks, folder, args.plot_path)
        save_stacked(to_uint8(combined),
                     os.path.join(args.save_path, f'stacked_image_{folder}'),
                     args.save_format)


def stitch_panorama(args):
    # Called on the class: constructing Filters() would eagerly decode every image in
    # save_path, which opencv_stitch then reads again for itself.
    panorama_image = Filters.opencv_stitch(args.save_path, args.file_type,
                                           scale=args.pano_scale,
                                           compose_resol=args.pano_compose_resol)
    cv2.imwrite(os.path.join(args.save_path, 'panorama.jpg'), panorama_image)


def main():
    parser = argparse.ArgumentParser(description='MGCV image processing pipeline')
    parser.add_argument('--image-path', default=DEFAULT_IMAGE_PATH)
    parser.add_argument('--save-path', default=DEFAULT_SAVE_PATH)
    parser.add_argument('--plot-path', default=DEFAULT_PLOT_PATH)
    parser.add_argument('--file-type', default='.jpg')
    parser.add_argument('--window-num', type=int, default=35)
    parser.add_argument('--stride-window', type=int, default=15)
    parser.add_argument('--workers', type=int, default=None,
                        help='processes for the masking stage (1 = serial, default auto)')
    parser.add_argument('--select', choices=('hull', 'sharpness'), default='hull',
                        help='how pixels are attributed to a frame (default hull)')
    parser.add_argument('--sharpness-sigma', type=float, default=5,
                        help='smoothing for --select sharpness (default 5)')
    parser.add_argument('--save-format', choices=('jpg', 'png'), default='jpg',
                        help='stacked-image format; png is lossless (default jpg)')
    # Both of these change the panorama, so they stay off unless asked for.
    parser.add_argument('--pano-scale', type=float, default=None,
                        help='shrink inputs before stitching, e.g. 0.5 (default off)')
    parser.add_argument('--pano-compose-resol', type=float, default=None,
                        help='megapixels Stitcher composites at, -1 = full (default off)')
    parser.add_argument('--stack', action='store_true', help='run focus stacking')
    parser.add_argument('--stitch', action='store_true', help='run panorama stitching')
    args = parser.parse_args()

    if not (args.stack or args.stitch):
        parser.error('choose at least one of --stack / --stitch')
    if args.stack:
        stack_folders(args)
    if args.stitch:
        stitch_panorama(args)


# Required: infinity_focus() uses a process pool, and macOS spawns workers by
# re-importing this module. Without the guard each worker would rerun the pipeline.
if __name__ == '__main__':
    main()
