"""Verify fixed-gantry invariants and the exported volume pose independently."""
import argparse
import json
from pathlib import Path
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from verify_exports import run, image, blocks


def points(poly):
    return vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)


def basis(poly):
    p = points(poly)
    directions = p[1:] - p[0]
    return (directions / np.linalg.norm(directions, axis=1)[:, None]).T


def verify(exe, output):
    report = []
    for model in ("flat-circular", "cyl-circular", "flat-helical", "cyl-helical", "nikon-tilted"):
        reference = output / model / "world0"
        run(exe, "--geometry", model, "--frame", 0, "--offscreen", "--export", reference)
        g0 = blocks(reference / "geometry.vtm")
        relative_detector = []
        relative_source = []
        for frame in (0, 30, 61):
            world = output / model / f"world{frame}"
            fixed = output / model / f"fixed{frame}"
            # Non-zero centre and unequal spacing catch rotations about the wrong pivot.
            common = ["--geometry", model, "--frame", frame, "--center", 3, -4, 5,
                      "--spacing", 0.7, 1.3, 2.1, "--offscreen"]
            run(exe, *common, "--export", world)
            run(exe, *common, "--gantry-fixed", "--export", fixed)
            gw = blocks(world / "geometry.vtm")
            gf = blocks(fixed / "geometry.vtm")
            rotation = basis(g0[5]) @ basis(gw[5]).T
            translation = points(g0[5])[0] - rotation @ points(gw[5])[0]
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=2e-6)
            assert np.linalg.det(rotation) > 0.99999
            # Everything in the scene undergoes ONE rigid change of coordinates.
            for index in (0, 2, 4, 5):
                np.testing.assert_allclose(points(gf[index]),
                    points(gw[index]) @ rotation.T + translation, atol=6e-5)
            np.testing.assert_allclose(gf[1].GetCenter(),
                rotation @ np.asarray(gw[1].GetCenter()) + translation, atol=6e-5)
            np.testing.assert_allclose(gf[1].GetCenter(), g0[1].GetCenter(), atol=6e-5)
            np.testing.assert_allclose(points(gf[2]), points(g0[2]), atol=6e-5)
            relative_detector.append(points(gf[2]))
            relative_source.append(np.asarray(gf[1].GetCenter()))
            # VTI carries oriented image geometry, without resampling or changing RAW values.
            iw, ir = image(world / "volume.vti"), image(fixed / "volume.vti")
            np.testing.assert_array_equal(vtk_to_numpy(iw.GetPointData().GetScalars()),
                                          vtk_to_numpy(ir.GetPointData().GetScalars()))
            np.testing.assert_allclose(ir.GetOrigin(), rotation @ np.asarray(iw.GetOrigin())+translation, atol=2e-5)
            direction = np.array([[ir.GetDirectionMatrix().GetElement(r,c) for c in range(3)] for r in range(3)])
            np.testing.assert_allclose(direction, rotation, atol=2e-6)
            np.testing.assert_allclose(ir.GetSpacing(), iw.GetSpacing(), atol=1e-12)
            # Beam-to-voxel distances must remain identical in both displays.
            voxel_world = np.asarray(iw.GetPoint(1234))
            voxel_fixed = np.asarray(ir.GetPoint(1234))
            dw = np.linalg.norm(voxel_world-np.asarray(gw[1].GetCenter()))
            dr = np.linalg.norm(voxel_fixed-np.asarray(gf[1].GetCenter()))
            np.testing.assert_allclose(dr, dw, atol=6e-5)
            assert "reference=gantry-fixed" in (fixed / "scene.txt").read_text()
        np.testing.assert_allclose(relative_source[0], relative_source[-1], atol=6e-5)
        np.testing.assert_allclose(relative_detector[0], relative_detector[-1], atol=6e-5)
        assert not np.allclose(points(blocks(output/model/"fixed0"/"geometry.vtm")[0]),
                               points(blocks(output/model/"fixed30"/"geometry.vtm")[0]))
        report.append({"geometry": model, "fixed_gantry": "PASS", "object_motion": "PASS",
                       "volume_pose": "PASS", "distance_invariance": "PASS"})
    run(exe, "--geometry", "source-orbit", "--gantry-fixed", "--offscreen", success=False)
    result = run(exe, "--geometry", "cyl-helical", "--gantry-fixed", "--self-check", "--offscreen")
    assert "Interaction checks:" in result.stdout and "PASS" in result.stdout
    (output / "relative-verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PASS: five rigid geometries across three frames, fixed gantry, volume pose, distances and mode callback")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    verify(args.exe.resolve(), args.output.resolve())
