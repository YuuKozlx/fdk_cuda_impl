# Flat 外置逐帧几何样例

四组 CSV + TOML 可直接用于 CLI 或 UI。CSV 已随源码提供；先运行一次 `prepare-phantoms.ps1`，在本应用的 `out/external-frames/phantoms` 生成小型 uint8 水模体，无需下载数据或安装 Python。

## 运行

从仓库根目录执行。若使用其他构建目录，替换 exe 路径。

```powershell
& ./example/multispectrum_sim/configs/project-and-reconstruct/external-frames/prepare-phantoms.ps1
& ./example/multispectrum_sim/out/bin/multispectrum_sim.exe --gui ./example/multispectrum_sim/configs/project-and-reconstruct/external-frames/helical.toml
```

也可在 UI 中加载配套 TOML，用 View 滑条浏览逐帧几何，再运行前投和重建。去掉 `--gui` 即执行完整 CLI 流程。仅加载 CSV 时，需自行匹配探测器行列数及模体网格。

| TOML / 同名 CSV | 几何 | 帧数 | SID / SDD mm | 探测器 |
| --- | --- | ---: | --- | --- |
| `helical` | 2 周螺旋，180 帧/周，60 mm/周，源 z 从 -60 到约 +60 mm | 360 | 709 / 1143 | 128×128，3 mm |
| `breast-cbct` | 乳腺锥束 CT 的 360°环绕示意，世界 Z 为环绕轴 | 180 | 500 / 800 | 128×128，3 mm |
| `nikon-tilted` | 初始源探姿态绕 X 倾斜 30°，随后绕世界 Z 逆向扫描 | 180 | 500 / 800 | 128×128，3 mm |
| `planar-ct` | 源在 y=-500 平面作半径 200 mm 圆周运动，探测器固定于 y=300 | 180 | 标称 500 / 800 | 128×128，6 mm |

这些是几何/链路示例，参数不是设备标定结果。乳腺例为专用环绕式 CBCT，不是有限角乳腺断层合成（DBT），模体也不是乳腺组织模型。Nikon 命名沿用本库的倾斜 CT 定义，不声称覆盖厂商所有扫描模式。

## 模体与重建

- 64³、4 mm 体素，网格中心位于原点。
- 螺旋/乳腺/倾斜例：半径 64 mm 水球，内部含偏心空气孔。
- 平面例：位于 XZ 平面的水圆片，半径 64 mm，沿 Y 总厚度 32 mm，含偏心空气孔。
- 100 kVp、2 mm Al、每像素 1 条射线，不加散射与量子噪声。
- 水 BHC 在线拟合，70 keV 参考；内部 OS-SART 3 次迭代、12 子集，不加 TV。
- 输出在 `example/multispectrum_sim/out/external-frames/<样例名>/`，包括投影、空气参考、校正缓存和重建体积。

低分辨率、少迭代参数用于快速检查链路。平面/倾斜轨迹仍可能存在信息不足与有限角伪影，运行成功不等于图像质量充分。

## CSV 坐标约定

每行是一个投影帧：`source_xyz, center_xyz, u_xyz, v_xyz`，12 列均为 mm。`center` 是探测器阵列中心，U/V 是一个像素的步长。XYZ 使用右手世界坐标，帧顺序与投影数据一一对应，末帧不重复首帧。CSV 是实际射线几何，不会再次叠加 TOML 中的标称倾角、偏置或螺距。

令 θ=2πi/180，p 为像素尺寸，L=SDD−SID：

- 环绕：S=(SID·sinθ, −SID·cosθ, 0)，D=(−L·sinθ, L·cosθ, 0)，U=(p·cosθ,p·sinθ,0)，V=(0,0,p)。
- 螺旋：沿用环绕式，S/D 的 z 同时替换为 −60+60i/180。
- 倾斜：将初始 S=(0,−SID,0)、D=(0,L,0)、U=(p,0,0)、V=(0,0,p) 先乘 Rx(30°)，再乘 Rz(−θ)。位置和像素方向均执行同一刚性旋转。
- 平面：S=(200cosθ,−500,−200sinθ)，D=(0,300,0)，U=(p,0,0)，V=(0,0,p)。标称 SID 是源平面到原点的距离，实际源到原点距离为 √(500²+200²)。

修改帧数或像素间距时，需要同步修改 CSV 和 TOML；仅修改 TOML 标称轨迹参数不会重生成 CSV。扩大探测器采样数量时也要考虑像素间距和物理覆盖范围。

本组统一采用支持逐帧向量的内部迭代重建。不能直接把所有样例改成 FDK/WFBP：外置解析入口只接受与声明的标准圆/螺旋匹配的几何，平面及倾斜 CSV 应继续用内部迭代法。骨 BHC 重投影、TIGRE、统计重建目前未接入这一入口。

## 已验证范围

四份原始 TOML 均完成 CLI 全流程，生成 64³ 有限值体积。验证覆盖配置读取、CSV 校验、前投、校正和重建，不代表临床图像质量验证。UI 使用同一 CSV 几何入口，支持轨迹和逐帧姿态预览。
