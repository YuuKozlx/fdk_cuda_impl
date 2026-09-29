"""非对称双环加标记球标定示例。

只修改下面 CONFIG 中的输入文件和输出目录，然后在 IDE 中直接运行本文件。
不需要填写命令行参数。
"""
from pathlib import Path

from identifiable_phantom_workflow import IdentifiablePhantomConfig, calibrate_identifiable_phantom

CONFIG = IdentifiablePhantomConfig(
    # little-endian float32 RAW 文件
    raw_path=Path(r"G:\Code\fanproj\fdk-test\example\multispectrum_sim\outputs\quality\double-ring-asymmetric-marker-1024x1024x360-ou5-ov10-tu1-tv2-tn3-px10-py15-pz20-prx2-projection.raw"),
    # 结果目录：25个钢珠轨迹和等效几何都会写到这里
    output_directory=Path("out/cbct_calibration/identifiable_phantom_formal"),
    views=360,
    pixel_size_mm=(0.417, 0.417),
)

if __name__ == "__main__":
    report = calibrate_identifiable_phantom(CONFIG)
    fit = report["joint_fit"]
    sid_mm = report["dlt_source_circle"]["fixed_sid_mm"]
    print("双环钢珠标定完成")
    print(f"结果目录: {CONFIG.output_directory.resolve()}")
    print(f"SID = {sid_mm:.3f} mm, SDD = {fit['sdd_mm']:.3f} mm")
    print(f"offsetU = {fit['offset_u_px']:.3f} px, offsetV = {fit['offset_v_px']:.3f} px")
    print(f"旋转方向 = {fit['inferred_rotation_direction']:+d}")
    print(f"重投影 RMSE = {fit['rmse_px']:.3f} px")
