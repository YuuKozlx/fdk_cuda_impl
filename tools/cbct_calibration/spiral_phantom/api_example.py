"""Direct-call example for the repository's marked-spiral projection."""
from pathlib import Path

from .workflow import SpiralCalibrationConfig, calibrate_spiral_phantom


def run_repository_example() -> dict:
    root = Path(__file__).resolve().parents[3]
    name = "spiral-marker-1024x1024x360-ou5-ov10-tu1-tv2-tn3-soy10-soz5-px10-py15-pz20-prx1-pry2-prz3-projection.raw"
    return calibrate_spiral_phantom(SpiralCalibrationConfig(
        raw_path=root / "example/multispectrum_sim/outputs/quality" / name,
        output_directory=root / "out/calibration/spiral_phantom",
    ))


if __name__ == "__main__":
    run_repository_example()
