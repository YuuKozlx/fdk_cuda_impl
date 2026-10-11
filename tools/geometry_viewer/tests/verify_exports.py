"""Run the native viewer and verify physical geometry and VTK file round-trips.

Usage: python verify_exports.py --exe path/to/geometry_viewer --output out/checks
Requires the optional vtk Python package for independent file reading only.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOImage import vtkPNGReader
from vtkmodules.vtkIOXML import vtkXMLImageDataReader, vtkXMLMultiBlockDataReader


def run(exe, *args, success=True):
    result = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True)
    if (result.returncode == 0) != success:
        raise AssertionError(f"Unexpected exit {result.returncode}: {result.stdout}\n{result.stderr}")
    return result


def image(path):
    reader = vtkXMLImageDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    data = reader.GetOutput()
    assert data.GetNumberOfPoints() > 0
    return data


def blocks(path):
    reader = vtkXMLMultiBlockDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    data = reader.GetOutput()
    assert data.GetNumberOfBlocks() == 6
    return [data.GetBlock(i) for i in range(6)]


def local_frame(data, origin, basis, labels):
    pts = vtk_to_numpy(data.GetPoints().GetData())
    assert pts.shape == (4, 3) and data.GetNumberOfLines() == 3
    np.testing.assert_allclose(pts[0], origin, atol=1e-4)
    vectors = pts[1:] - pts[0]
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 120*0.22, atol=1e-5)
    actual = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    np.testing.assert_allclose(actual @ actual.T, np.eye(3), atol=2e-6)
    np.testing.assert_allclose(np.cross(actual[0], actual[1]), actual[2], atol=2e-6)
    np.testing.assert_allclose(actual, basis, atol=2e-6)
    names = data.GetFieldData().GetAbstractArray("AxisLabels")
    assert [names.GetValue(i) for i in range(3)] == labels


def verify(exe, output):
    modes = ("flat-circular", "cyl-circular", "flat-helical", "cyl-helical",
             "source-orbit", "nikon-tilted")
    frame, views, per_turn = 30, 180, 90
    theta = 2 * math.pi * frame / per_turn
    report = []
    for mode in modes:
        folder = output / mode
        png = output / (mode + ".png")
        result = run(exe, "--geometry", mode, "--frame", frame,
                     "--screenshot", png, "--export", folder)
        assert "Screenshot:" in result.stdout
        bounds, source, detector, trajectory, source_axes, detector_axes = blocks(folder / "geometry.vtm")
        np.testing.assert_allclose(bounds.GetBounds(), [-32, 32, -32, 32, -24, 24], atol=1e-5)
        assert trajectory.GetNumberOfPoints() == views
        pts = vtk_to_numpy(trajectory.GetPoints().GetData())
        if mode == "source-orbit":
            expected = [18 * math.cos(theta), -120, -18 * math.sin(theta)]
            np.testing.assert_allclose(pts[:, 1], -120, atol=1e-4)
            np.testing.assert_allclose(detector.GetBounds(), [-120, 120, 120, 120, -72, 72], atol=1e-4)
        elif mode == "nikon-tilted":
            expected = [-120 * math.cos(math.pi/6) * math.sin(theta),
                        -120 * math.cos(math.pi/6) * math.cos(theta), -60]
        else:
            z = -0.5 * 36 * (views-1) / per_turn + 36 * frame / per_turn if "helical" in mode else 0
            expected = [120 * math.sin(theta), -120 * math.cos(theta), z]
        np.testing.assert_allclose(source.GetCenter(), expected, atol=1e-4)
        np.testing.assert_allclose(pts[frame], expected, atol=1e-4)
        if mode.startswith("cyl-"):
            assert detector.GetNumberOfPolys() == 64
            dp = vtk_to_numpy(detector.GetPoints().GetData())
            # Default R=SDD means the cylinder axis passes through the source.
            radial = np.linalg.norm(dp[:, :2] - np.asarray(expected[:2]), axis=1)
            np.testing.assert_allclose(radial, 240, atol=1e-4)
        else:
            assert detector.GetNumberOfPolys() == 1
        dp = vtk_to_numpy(detector.GetPoints().GetData())
        if mode.startswith("cyl-"):
            detector_origin = (dp[64]+dp[65])/2
            du, dv = dp[66]-dp[62], dp[65]-dp[64]
        else:
            detector_origin = (dp[0]+dp[3])/2
            du, dv = dp[2]-dp[0], dp[1]-dp[0]
        du, dv = du/np.linalg.norm(du), dv/np.linalg.norm(dv)
        local_frame(detector_axes, detector_origin, [du, dv, np.cross(du,dv)], ["D-U","D-V","D-N"])
        ray = detector_origin-np.asarray(expected)
        ray /= np.linalg.norm(ray)
        su = du - np.dot(du,ray)*ray
        su /= np.linalg.norm(su)
        local_frame(source_axes, expected, [su, ray, np.cross(su,ray)], ["S-U","S-R","S-V"])
        folder0 = output / (mode + "-frame0")
        run(exe, "--geometry", mode, "--frame", 0, "--offscreen", "--export", folder0)
        _, source0, _, trajectory0, sa0, da0 = blocks(folder0 / "geometry.vtm")
        sp0 = vtk_to_numpy(sa0.GetPoints().GetData())
        dp0 = vtk_to_numpy(da0.GetPoints().GetData())
        np.testing.assert_allclose(sp0[0], source0.GetCenter(), atol=1e-4)
        np.testing.assert_allclose(sp0[0], trajectory0.GetPoint(0), atol=1e-4)
        ray0 = dp0[0]-sp0[0]
        ray0 /= np.linalg.norm(ray0)
        np.testing.assert_allclose((sp0[2]-sp0[0])/(120*0.22), ray0, atol=2e-6)
        assert np.dot(dp0[3]-dp0[0], sp0[0]-dp0[0]) > 0, "Detector normal points away from source"
        assert not np.allclose(sp0, vtk_to_numpy(source_axes.GetPoints().GetData())), "Source frame did not update"
        if mode == "source-orbit":
            np.testing.assert_allclose(dp0, vtk_to_numpy(detector_axes.GetPoints().GetData()), atol=1e-4)
        else:
            assert not np.allclose(dp0, vtk_to_numpy(detector_axes.GetPoints().GetData())), "Detector frame did not update"
        vol = image(folder / "volume.vti")
        assert vol.GetDimensions() == (64, 64, 48)
        np.testing.assert_allclose(vol.GetOrigin(), [-31.5, -31.5, -23.5], atol=1e-6)
        assert vol.GetScalarRange()[1] > 0.04
        png_reader = vtkPNGReader()
        png_reader.SetFileName(str(png))
        png_reader.Update()
        pixels = vtk_to_numpy(png_reader.GetOutput().GetPointData().GetScalars())
        assert png_reader.GetOutput().GetDimensions()[:2] == (1200, 800)
        assert len(np.unique(pixels, axis=0)) > 100
        amber = (pixels[:, 0] > 120) & (pixels[:, 0] > pixels[:, 2] * 1.4)
        assert amber.sum() > 40, "Source/trajectory missing from screenshot"
        report.append({"geometry": mode, "source_mm": expected, "export_readback": "PASS",
                       "render": "PASS", "local_frames": "PASS", "detector_polygons": detector.GetNumberOfPolys()})

    # A non-zero center and unequal spacing must survive VTI and box export;
    # known values prove z/y/x layout and real RAW ingestion, not only dimensions.
    raw = output / "known.raw"
    values = np.arange(8*6*4, dtype=np.float32) / 1000
    values.tofile(raw)
    folder = output / "raw-check"
    run(exe, "--geometry", "flat-circular", "--size", 8, 6, 4,
        "--spacing", 0.7, 1.3, 2.1, "--center", 3, -4, 5,
        "--raw", raw, "--offscreen", "--export", folder)
    vol = image(folder / "volume.vti")
    assert vol.GetDimensions() == (8, 6, 4)
    np.testing.assert_allclose(vol.GetSpacing(), [0.7, 1.3, 2.1], atol=1e-6)
    np.testing.assert_allclose(vol.GetOrigin(), [3-3.5*0.7, -4-2.5*1.3, 5-1.5*2.1], atol=1e-6)
    np.testing.assert_array_equal(vtk_to_numpy(vol.GetPointData().GetScalars()), values)
    bbox = blocks(folder / "geometry.vtm")[0]
    np.testing.assert_allclose(bbox.GetBounds(), [3-4*0.7, 3+4*0.7, -4-3*1.3, -4+3*1.3,
                                               5-2*2.1, 5+2*2.1], atol=1e-5)
    run(exe, "--geometry", "flat-circular", "--size", 4, 4, 4, "--raw", raw, "--offscreen", success=False)
    run(exe, "--views", 0, "--offscreen", success=False)
    run(exe, "--frame", 180, "--offscreen", success=False)
    run(exe, "--sid", 300, "--sdd", 120, "--offscreen", success=False)
    run(exe, "--geometry", "cyl-helical", "--pitch", 0, "--offscreen", success=False)
    result = run(exe, "--self-check", "--offscreen")
    assert "Interaction checks:" in result.stdout and "PASS" in result.stdout
    report.append({"RAW_layout_spacing_center": "PASS", "invalid_inputs": "PASS", "interaction_callbacks": "PASS"})
    (output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: six geometries, physical coordinates, volume/RAW round-trip, rendering, interactions; {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    verify(args.exe.resolve(), args.output.resolve())
