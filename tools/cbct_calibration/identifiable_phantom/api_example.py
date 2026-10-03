"""IDE-callable example for the identifiable phantom workflow."""
from pathlib import Path

from .workflow import IdentifiablePhantomConfig, calibrate_identifiable_phantom


CONFIG = IdentifiablePhantomConfig(
    raw_path=Path(
        r"G:\Code\fanproj\fdk-test\example\multispectrum_sim\outputs\quality"
        r"\double-ring-asymmetric-marker-1024x1024x360-ou5-ov10-tu1-tv2-tn3"
        r"-projection.raw"),
    output_directory=Path("out/cbct_calibration/identifiable_phantom"),
)


def run_example(config: IdentifiablePhantomConfig = CONFIG) -> dict:
    return calibrate_identifiable_phantom(config)
