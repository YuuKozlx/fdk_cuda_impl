"""Seed-assisted bead extraction for strongly merged simulated projections.

Known simulation geometry is used only to place one seed near each bead. Pixel
ownership and final sub-pixel centers are measured from the RAW intensities.
This is intended to separate segmentation failure from calibration failure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cho_calibration import ChoConfig, phantom_points


def detector_frame(angle: float, tilt_u: float, tilt_v: float, tilt_n: float):
    """Mirror the simulator flat-detector frame convention."""
    frame_angle = angle - 0.5 * np.pi
    er = np.array([np.cos(frame_angle), np.sin(frame_angle), 0.0])
    u0 = np.array([-np.sin(frame_angle), np.cos(frame_angle), 0.0])
    v0 = np.array([0.0, 0.0, 1.0])
    n0 = -er
    base = np.column_stack([u0, v0, n0])
    # The simulator applies local detector pose rotations. For this dataset
    # only tilt_n is nonzero, so retain the full XYZ form for reuse.
    local = Rotation.from_euler("xyz", [tilt_u, tilt_v, tilt_n]).as_matrix()
    axes = base @ local
    return axes[:, 0], axes[:, 1], axes[:, 2]


def simulated_seeds(view: int, views: int = 360) -> np.ndarray:
    cfg = ChoConfig(ring_radius_mm=50.0, ring_half_spacing_mm=50.0,
                    beads_per_ring=12, pixel_size_mm=(0.417, 0.417))
    local = phantom_points(cfg)
    # Phantom forward pose: offset first, then XYZ rotation.
    phantom_r = Rotation.from_euler("xyz", [np.deg2rad(3.0), 0.0, 0.0]).as_matrix()
    world = (phantom_r @ (local + np.array([0.0, 10.0, 0.0])).T).T

    angle = 2.0 * np.pi * view / views
    frame_angle = angle - 0.5 * np.pi
    ca, sa = np.cos(frame_angle), np.sin(frame_angle)
    # Simulator source_offset_y is a rotating local tangential displacement.
    source = np.array([440.0 * ca - 10.0 * sa,
                       440.0 * sa + 10.0 * ca, 0.0])
    eu, ev, normal = detector_frame(angle, 0.0, 0.0, np.deg2rad(3.0))
    detector_center = source + 770.0 * normal + 2.085 * eu + 4.17 * ev
    out = []
    for point in world:
        ray = point - source
        scale = np.dot(detector_center - source, normal) / np.dot(ray, normal)
        hit = source + scale * ray
        relative = hit - detector_center
        u_mm = np.dot(relative, eu)
        v_mm = np.dot(relative, ev)
        out.append([u_mm / 0.417 + 511.5, v_mm / 0.417 + 511.5])
    return np.asarray(out)


def extract_centers(image: np.ndarray, seeds: np.ndarray, threshold: float = 5.0,
                    radius_px: float = 18.0) -> np.ndarray:
    """Partition bright pixels among nearby seeds and calculate centroids."""
    ys, xs = np.nonzero(image > threshold)
    pixels = np.column_stack([xs, ys]).astype(float)
    weights = np.maximum(image[ys, xs].astype(float) - threshold, 0.0)
    # Chunk distance evaluation to keep memory bounded.
    owner = np.empty(len(pixels), dtype=np.int32)
    best_distance = np.empty(len(pixels), dtype=float)
    for start in range(0, len(pixels), 100000):
        stop = min(start + 100000, len(pixels))
        distance = np.sum((pixels[start:stop, None, :] - seeds[None, :, :]) ** 2, axis=2)
        owner[start:stop] = np.argmin(distance, axis=1)
        best_distance[start:stop] = np.min(distance, axis=1)
    valid = best_distance <= radius_px ** 2
    output = np.empty_like(seeds)
    for bead in range(len(seeds)):
        selected = valid & (owner == bead)
        if np.count_nonzero(selected) < 5:
            raise ValueError(f"bead {bead} has insufficient pixels")
        w = weights[selected]
        output[bead] = np.sum(pixels[selected] * w[:, None], axis=0) / np.sum(w)
    return output


def extract_stack(raw_path: Path, output: Path):
    shape = (1024, 1024)
    count = shape[0] * shape[1]
    frame_bytes = count * 4
    stack = []
    seed_errors = []
    with raw_path.open("rb") as stream:
        for view in range(360):
            stream.seek(view * frame_bytes)
            image = np.fromfile(stream, np.float32, count).reshape(shape)
            seeds = simulated_seeds(view)
            centers = extract_centers(image, seeds)
            stack.append(centers)
            seed_errors.append(np.linalg.norm(centers - seeds, axis=1))
    stack = np.asarray(stack)
    np.save(output, stack)
    errors = np.asarray(seed_errors)
    print({"shape": stack.shape,
           "seed_error_mean_px": float(errors.mean()),
           "seed_error_max_px": float(errors.max())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    extract_stack(args.raw, args.output)
