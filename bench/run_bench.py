"""Benchmark + bit-exactness harness for the MGCV focus-stacking pipeline.

Workflow:
    python bench/run_bench.py --golden    # before any change: record reference output
    python bench/run_bench.py --check     # after each change: assert output is identical

The golden check is the gate for every optimization: a change that alters `masks` or
`stacked_image` by even one value is not output-identical and must be reverted.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import skimage as sk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from two_dimensions import timing
from two_dimensions.filters import Filters

CLUSTER_METHODS = ('group', 'segment')


def synth_stack(num_images, height, width, seed=0):
    """A deterministic focus stack: one scene, each frame sharp in a different band.

    The scene is deliberately texture-rich so SIFT finds enough keypoints for
    compensate_focal_distance to produce a homography.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)

    base = np.empty((height, width, 3))
    base[..., 0] = xx / width
    base[..., 1] = yy / height
    base[..., 2] = 0.5
    for _ in range(400):
        y0 = rng.integers(0, height - 8)
        x0 = rng.integers(0, width - 8)
        base[y0:y0 + rng.integers(4, 40), x0:x0 + rng.integers(4, 40)] = rng.random(3)
    base = np.clip(base + rng.normal(0, 0.05, base.shape), 0, 1)

    blurred = sk.filters.gaussian(base, sigma=4, channel_axis=-1)
    band = height / (2.5 * num_images)

    images = []
    for i in range(num_images):
        centre = (i + 0.5) / num_images * height
        weight = np.exp(-0.5 * ((yy - centre) / band) ** 2)[..., None]
        frame = weight * base + (1 - weight) * blurred
        images.append((frame * 255).astype(np.uint8))
    return images


def ensure_images(image_dir, num_images, height, width, seed):
    """Write the synthetic stack once and reuse it, so every run reads identical bytes."""
    stamp = os.path.join(image_dir, f'.stamp-{num_images}x{height}x{width}x{seed}')
    if os.path.exists(stamp):
        return
    os.makedirs(image_dir, exist_ok=True)
    for name in os.listdir(image_dir):
        os.remove(os.path.join(image_dir, name))
    for i, img in enumerate(synth_stack(num_images, height, width, seed)):
        sk.io.imsave(os.path.join(image_dir, f'frame_{i:03d}.png'), img, check_contrast=False)
    open(stamp, 'w').close()


def run_pipeline(image_dir, cluster_method, args):
    timing.reset_stages()
    start = time.perf_counter()
    pipeline = Filters(image_dir, '.png')
    try:
        stacked, masks = pipeline.infinity_focus(
            args.edge_method, args.window_num, args.stride_window, cluster_method,
            workers=args.workers, select=args.select)
    except TypeError:
        # Pre-optimization code has no `workers` parameter. Falling back lets this
        # harness benchmark an older checkout (git stash / git worktree) for an
        # apples-to-apples baseline.
        stacked, masks = pipeline.infinity_focus(
            args.edge_method, args.window_num, args.stride_window, cluster_method)
    total = time.perf_counter() - start
    return np.asarray(stacked), np.stack(masks), total, timing.snapshot()


def golden_path(out_dir, cluster_method):
    return os.path.join(out_dir, f'golden_{cluster_method}.npz')


def report(cluster_method, total, stages):
    print(f'\n--- {cluster_method}: {total:.2f} s total ---')
    for name, seconds in sorted(stages.items(), key=lambda kv: -kv[1]):
        print(f'  {seconds * 1000:9.1f} ms  {name}')


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--golden', action='store_true', help='record reference output')
    mode.add_argument('--check', action='store_true', help='assert output matches reference')
    parser.add_argument('--image-dir', default=None,
                        help='use real images instead of the synthetic stack')
    parser.add_argument('--out-dir', default=os.path.join(os.path.dirname(__file__), 'goldens'))
    parser.add_argument('--num-images', type=int, default=4)
    parser.add_argument('--height', type=int, default=600)
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--edge-method', default='sobel')
    parser.add_argument('--window-num', type=int, default=35)
    parser.add_argument('--stride-window', type=int, default=5)
    parser.add_argument('--methods', nargs='+', default=list(CLUSTER_METHODS))
    parser.add_argument('--workers', type=int, default=None,
                        help='processes for the masking stage (1 = serial, default auto)')
    parser.add_argument('--select', choices=('hull', 'sharpness'), default='hull',
                        help='mask strategy; sharpness changes output, so give it its '
                             'own --out-dir')
    parser.add_argument('--quiet', action='store_true', help='silence per-stage prints')
    args = parser.parse_args()

    timing.VERBOSE = not args.quiet
    os.makedirs(args.out_dir, exist_ok=True)

    image_dir = args.image_dir
    if image_dir is None:
        image_dir = os.path.join(os.path.dirname(__file__), 'images')
        ensure_images(image_dir, args.num_images, args.height, args.width, args.seed)

    timings = {}
    failures = []
    for cluster_method in args.methods:
        stacked, masks, total, stages = run_pipeline(image_dir, cluster_method, args)
        report(cluster_method, total, stages)
        timings[cluster_method] = {'total': total, 'stages': stages}

        path = golden_path(args.out_dir, cluster_method)
        if args.golden:
            np.savez_compressed(path, stacked=stacked, masks=masks)
            print(f'  wrote {path}')
        else:
            if not os.path.exists(path):
                failures.append(f'{cluster_method}: no golden at {path} (run --golden first)')
                continue
            with np.load(path) as ref:
                for name, actual in (('stacked', stacked), ('masks', masks)):
                    expected = ref[name]
                    if actual.shape != expected.shape:
                        failures.append(
                            f'{cluster_method}/{name}: shape {actual.shape} != {expected.shape}')
                    elif not np.array_equal(actual, expected):
                        diff = np.count_nonzero(actual != expected)
                        worst = np.abs(actual.astype(np.float64)
                                       - expected.astype(np.float64)).max()
                        failures.append(
                            f'{cluster_method}/{name}: {diff} of {actual.size} values differ '
                            f'(max abs diff {worst})')
                    else:
                        print(f'  OK  {cluster_method}/{name} bit-identical')

    with open(os.path.join(args.out_dir, 'timings.json'), 'w') as handle:
        json.dump(timings, handle, indent=2)

    if failures:
        print('\nFAILED:')
        for failure in failures:
            print('  ' + failure)
        return 1
    print('\nAll checks passed.' if args.check else '\nGoldens recorded.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
