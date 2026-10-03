"""Small IDE-callable example for the complete Cho workflow."""
from pathlib import Path

from .workflow import ChoWorkflowConfig, calibrate_raw


EXAMPLE = ChoWorkflowConfig(
    raw_path=Path(
        r"G:\Code\fanproj\fdk-test\example\multispectrum_sim\outputs\quality"
        r"\double-ring-px10-py15-pz20-tu1-tv2-tn3-prx2-projection-1024x1024x360-f32.raw"),
    output_directory=Path("out/cbct_calibration/cho/example"),
)


def run_example(config: ChoWorkflowConfig = EXAMPLE) -> dict:
    """Run from Python or an IDE; no command-line arguments are required."""
    return calibrate_raw(config)
