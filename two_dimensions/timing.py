"""Lightweight stage timing shared by the pipeline and the benchmark harness."""

import time
from contextlib import contextmanager

# Accumulated wall time per stage name, in seconds. Reset via reset_stages().
STAGE_TIMES = {}

# Set False to silence the per-stage prints (the benchmark prints its own table).
VERBOSE = True


def reset_stages():
    STAGE_TIMES.clear()


def snapshot():
    return dict(STAGE_TIMES)


@contextmanager
def stage(name):
    """Time a pipeline stage, accumulating into STAGE_TIMES under `name`."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        STAGE_TIMES[name] = STAGE_TIMES.get(name, 0.0) + elapsed
        if VERBOSE:
            print(f'{name} time = {int(round(1000 * elapsed))} mili-seconds')
