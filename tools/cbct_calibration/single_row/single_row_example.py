"""单排钢珠标定示例。

只修改下面 CONFIG 中的输入文件和输出目录，然后在 IDE 中直接运行本文件。
不需要填写命令行参数。
"""
from pathlib import Path

from .single_row_workflow import SingleRowConfig, calibrate_single_row

CONFIG = SingleRowConfig(
    # little-endian float32 RAW 文件
    raw_path=Path(r"G:\Code\fanproj\fdk-test\out\test-artifacts\bead-phantoms-v3\beads-3mm-10mm-flat-sid440-sdd770-1024x1024x359.raw"),
    # 结果目录：参数、轨迹和重投影误差都会写到这里
    output_directory=Path("out/cbct_calibration/single_row_3mm_10mm"),
    views=359,
    bead_count=20,
    spacing_mm=10.0,
    pixel_size_mm=(0.417, 0.417),
    threshold=100.0,
    max_jump_px=60.0,
    refine=False,
)

if __name__ == "__main__":
    report = calibrate_single_row(CONFIG)
    geometry = report["calibration"]["geometry"]
    print("单排钢珠标定完成")
    print(f"结果目录: {CONFIG.output_directory.resolve()}")
    print(f"SDD = {geometry['sdd_mm']:.3f} mm, SOD = {geometry['sod_mm']:.3f} mm")
    print(f"主点 = ({geometry['u0_px']:.3f}, {geometry['v0_px']:.3f}) px")
