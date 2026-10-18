# 配置索引

运行与数据准备见 [配置说明](README.zh-CN.md)。尺寸为 U×V 和 X×Y×Z；输入路径相对于仓库根目录，TOML 内路径则相对于配置文件。`custom/` 为未附带的专用模体，`imported/` 为需提供的原实验投影。

## projection

| 配置 | 帧数 | 探测器 | 重建网格 | 重建方法 | 输入 |
|---|---:|---|---|---|---|
| [projection/brainweb-full-fdk.toml](projection/brainweb-full-fdk.toml) | 360 | 1024x256 | 91x109x91 | — | `data/phantoms/brainweb/normal-1mm.mat` |
| [projection/calibration/tungsten-bead-spiral.toml](projection/calibration/tungsten-bead-spiral.toml) | 360 | 1024x256 | 280x280x380 | — | `out/prepared-examples/bead-spiral/labels.mat` |
| [projection/icrp145-female-head.toml](projection/icrp145-female-head.toml) | 2160 | 1024x256 | 256x256x100 | — | `out/prepared-examples/icrp145-female-head/labels.mat` |
| [projection/icrp145-male-head.toml](projection/icrp145-male-head.toml) | 2160 | 1024x256 | 256x256x100 | — | `out/prepared-examples/icrp145-male-head/labels.mat` |
| [projection/water-circle.toml](projection/water-circle.toml) | 360 | 1024x256 | 64x64x32 | — | `out/prepared-examples/water-circle/labels.mat` |
| [projection/water-cylinder-small.toml](projection/water-cylinder-small.toml) | 360 | 1024x256 | 64x64x32 | — | `out/prepared-examples/water-circle/labels.mat` |
| [projection/water-helix.toml](projection/water-helix.toml) | 720 | 1024x256 | 64x64x32 | — | `out/prepared-examples/water-helix/labels.mat` |

## reconstruction

| 配置 | 帧数 | 探测器 | 重建网格 | 重建方法 | 输入 |
|---|---:|---|---|---|---|
| [reconstruction/analytic/brainweb-center33-fdk.toml](reconstruction/analytic/brainweb-center33-fdk.toml) | 720 | 1024x256 | 181x217x33 | fdk | `out/config-examples/brainweb-center33-fdk/projection.raw` |
| [reconstruction/analytic/brainweb-full-fdk.toml](reconstruction/analytic/brainweb-full-fdk.toml) | 360 | 1024x256 | 91x109x91 | fdk | `out/config-examples/brainweb-full-fdk/projection.raw` |
| [reconstruction/analytic/icrp145-female-head.toml](reconstruction/analytic/icrp145-female-head.toml) | 2160 | 1024x256 | 256x256x100 | wfbp | `out/config-examples/icrp145-female-head/projection.raw` |
| [reconstruction/analytic/icrp145-male-head.toml](reconstruction/analytic/icrp145-male-head.toml) | 2160 | 1024x256 | 256x256x100 | wfbp | `out/config-examples/icrp145-male-head/projection.raw` |
| [reconstruction/analytic/water-circle.toml](reconstruction/analytic/water-circle.toml) | 360 | 1024x256 | 64x64x32 | fdk | `out/config-examples/water-circle/projection.raw` |
| [reconstruction/analytic/water-cylinder-scatter-bhc.toml](reconstruction/analytic/water-cylinder-scatter-bhc.toml) | 360 | 1024x256 | 1024x1024x30 | fdk | `out/config-examples/water-cylinder-200mm-fdk/projection.raw` |
| [reconstruction/analytic/water-helix.toml](reconstruction/analytic/water-helix.toml) | 720 | 1024x256 | 64x64x32 | wfbp | `out/config-examples/water-helix/projection.raw` |
| [reconstruction/calibration/double-ring-calibrated-fdk.toml](reconstruction/calibration/double-ring-calibrated-fdk.toml) | 360 | 1024x1024 | 512x512x600 | fdk | `out/config-examples/imported/double-ring-upper100-lower80-phase15-marker5-above30-px10-py15-pz20-tu1-tv2-tn3-prx2-projection-1024x1024x360-f32.raw` |
| [reconstruction/calibration/double-ring-cylinder-marker-truth-fdk.toml](reconstruction/calibration/double-ring-cylinder-marker-truth-fdk.toml) | 360 | 1024x1024 | 768x768x600 | fdk | `out/config-examples/imported/double-ring-100mm-spacing100-bead3-cylinder6-6beads-spp10-projection-1024x1024x360-f32.raw` |
| [reconstruction/calibration/double-ring-cylinder-marker-truth-sart.toml](reconstruction/calibration/double-ring-cylinder-marker-truth-sart.toml) | 360 | 1024x1024 | 768x768x600 | sart | `out/config-examples/imported/double-ring-100mm-spacing100-bead3-cylinder6-6beads-spp10-projection-1024x1024x360-f32.raw` |
| [reconstruction/calibration/double-ring-truth-fdk.toml](reconstruction/calibration/double-ring-truth-fdk.toml) | 360 | 1024x1024 | 512x512x600 | fdk | `out/config-examples/imported/double-ring-upper100-lower80-phase15-marker5-above30-px10-py15-pz20-tu1-tv2-tn3-prx2-projection-1024x1024x360-f32.raw` |
| [reconstruction/calibration/double-ring-truth-ossart.toml](reconstruction/calibration/double-ring-truth-ossart.toml) | 360 | 1024x1024 | 512x512x600 | ossart | `out/config-examples/imported/double-ring-upper100-lower80-phase15-marker5-above30-px10-py15-pz20-tu1-tv2-tn3-prx2-projection-1024x1024x360-f32.raw` |
| [reconstruction/experimental/water-cylinder-350mm-transmission-guard.toml](reconstruction/experimental/water-cylinder-350mm-transmission-guard.toml) | 360 | 1024x256 | 1024x1024x30 | fdk | `out/config-examples/water-cylinder-350mm-gamma0/projection.raw` |
| [reconstruction/experimental/water-cylinder-adaptive-scatter.toml](reconstruction/experimental/water-cylinder-adaptive-scatter.toml) | 360 | 1024x256 | 1024x1024x30 | fdk | `out/config-examples/water-cylinder-200mm-fdk/projection.raw` |
| [reconstruction/iterative/brainweb-center33-ossart.toml](reconstruction/iterative/brainweb-center33-ossart.toml) | 720 | 1024x256 | 181x217x33 | ossart | `out/config-examples/brainweb-center33-fdk/projection.raw` |
| [reconstruction/iterative/brainweb-center33-sirt.toml](reconstruction/iterative/brainweb-center33-sirt.toml) | 720 | 1024x256 | 181x217x33 | sirt | `out/config-examples/brainweb-center33-fdk/projection.raw` |

### 外部仿真数据

准备与完整链路见 [外部数据说明](reconstruction/external/README.zh-CN.md)。以下为可编辑模板，几何必须取自实际扫描。

| 配置 | 运行内容 | 输入 |
|---|---|---|
| [mcgpu-total-fdk](reconstruction/external/mcgpu-total-fdk.toml) | 可选散射／水 BHC／骨校正 → FDK | `out/config-examples/imported/mcgpu/total.line_integrals.raw` |
| [mcgpu-primary-ossart](reconstruction/external/mcgpu-primary-ossart.toml) | 可选水 BHC → OS-SART + Huber-TV | `out/config-examples/imported/mcgpu/primary.line_integrals.raw` |

## project-and-reconstruct

| 配置 | 帧数 | 探测器 | 重建网格 | 重建方法 | 输入 |
|---|---:|---|---|---|---|
| [project-and-reconstruct/analytic/brainweb-center33-fdk.toml](project-and-reconstruct/analytic/brainweb-center33-fdk.toml) | 720 | 1024x256 | 181x217x33 | fdk | `out/prepared-examples/brainweb-center33/labels.mat` |
| [project-and-reconstruct/analytic/brainweb-center33-heli-cyl-wfbp.toml](project-and-reconstruct/analytic/brainweb-center33-heli-cyl-wfbp.toml) | 3600 | 768x112 | 181x217x33 | wfbp | `out/prepared-examples/brainweb-center33/labels.mat` |
| [project-and-reconstruct/analytic/brainweb-center33-physics-fdk.toml](project-and-reconstruct/analytic/brainweb-center33-physics-fdk.toml) | 60 | 1024x256 | 181x217x33 | fdk | `out/prepared-examples/brainweb-center33/labels.mat` |
| [project-and-reconstruct/analytic/brainweb-full-fdk.toml](project-and-reconstruct/analytic/brainweb-full-fdk.toml) | 360 | 1024x256 | 91x109x91 | fdk | `data/phantoms/brainweb/normal-1mm.mat` |
| [project-and-reconstruct/analytic/pcb-nikon-tilted-fdk.toml](project-and-reconstruct/analytic/pcb-nikon-tilted-fdk.toml) | 180 | 512x512 | 512x512x48 | fdk | `out/prepared-examples/custom/pcb-test-100mm-0p1mm.raw` |
| [project-and-reconstruct/analytic/pcb-planar-pose-fdk.toml](project-and-reconstruct/analytic/pcb-planar-pose-fdk.toml) | 180 | 1024x1024 | 512x48x512 | fdk | `out/prepared-examples/custom/pcb-test-100mm-0p1mm.raw` |
| [project-and-reconstruct/analytic/water-circle.toml](project-and-reconstruct/analytic/water-circle.toml) | 360 | 1024x256 | 64x64x32 | fdk | `out/prepared-examples/water-circle/labels.mat` |
| [project-and-reconstruct/analytic/water-cylinder-120kv-fullfov.toml](project-and-reconstruct/analytic/water-cylinder-120kv-fullfov.toml) | 360 | 1024x256 | 512x512x256 | fdk | `out/prepared-examples/custom/water-cylinder-512x512x256-u8.raw` |
| [project-and-reconstruct/analytic/water-helix.toml](project-and-reconstruct/analytic/water-helix.toml) | 720 | 1024x256 | 64x64x32 | wfbp | `out/prepared-examples/water-helix/labels.mat` |
| [project-and-reconstruct/calibration/tungsten-bead-double-ring-cylinder-marker.toml](project-and-reconstruct/calibration/tungsten-bead-double-ring-cylinder-marker.toml) | 360 | 1024x256 | 270x270x450 | fdk | `out/prepared-examples/bead-cylinder-marker/labels.mat` |
| [project-and-reconstruct/calibration/tungsten-bead-double-ring.toml](project-and-reconstruct/calibration/tungsten-bead-double-ring.toml) | 360 | 1024x256 | 270x270x450 | fdk | `out/prepared-examples/bead-double-ring/labels.mat` |
| [project-and-reconstruct/calibration/tungsten-bead-line-3mm-spacing10.toml](project-and-reconstruct/calibration/tungsten-bead-line-3mm-spacing10.toml) | 360 | 1024x256 | 128x128x480 | fdk | `out/prepared-examples/bead-line/labels.mat` |
| [project-and-reconstruct/calibration/tungsten-bead-synchronized-ring-12-marker.toml](project-and-reconstruct/calibration/tungsten-bead-synchronized-ring-12-marker.toml) | 360 | 1024x256 | 275x275x450 | fdk | `out/prepared-examples/bead-sync-12/labels.mat` |
| [project-and-reconstruct/calibration/tungsten-bead-synchronized-ring-6-marker.toml](project-and-reconstruct/calibration/tungsten-bead-synchronized-ring-6-marker.toml) | 360 | 1024x256 | 275x275x450 | fdk | `out/prepared-examples/bead-sync-6/labels.mat` |
| [project-and-reconstruct/calibration/water-small-cone-fdk.toml](project-and-reconstruct/calibration/water-small-cone-fdk.toml) | 360 | 1024x256 | 181x217x33 | fdk | `out/prepared-examples/water-small-cone/labels.mat` |
| [project-and-reconstruct/calibration/water-small-cone-heli-cyl-wfbp.toml](project-and-reconstruct/calibration/water-small-cone-heli-cyl-wfbp.toml) | 1800 | 768x112 | 181x217x33 | wfbp | `out/prepared-examples/water-small-cone/labels.mat` |
| [project-and-reconstruct/experimental/brainweb-statistical-algebraic.toml](project-and-reconstruct/experimental/brainweb-statistical-algebraic.toml) | 360 | 1024x256 | 256x256x200 | regularized_algebraic | `data/phantoms/brainweb/normal-1mm.mat` |
| [project-and-reconstruct/icrp/icrp145-female-abdomen.toml](project-and-reconstruct/icrp/icrp145-female-abdomen.toml) | 1800 | 1024x256 | 256x256x128 | wfbp | `out/prepared-examples/icrp145-female-abdomen/labels.mat` |
| [project-and-reconstruct/icrp/icrp145-female-abdomen_no_arm.toml](project-and-reconstruct/icrp/icrp145-female-abdomen_no_arm.toml) | 2160 | 1024x256 | 256x256x128 | wfbp | `out/prepared-examples/icrp145-female-abdomen_no_arm/labels.mat` |
| [project-and-reconstruct/icrp/icrp145-female-chest.toml](project-and-reconstruct/icrp/icrp145-female-chest.toml) | 1800 | 1024x256 | 256x256x153 | wfbp | `out/prepared-examples/icrp145-female-chest/labels.mat` |
| [project-and-reconstruct/icrp/icrp145-female-chest_no_arm.toml](project-and-reconstruct/icrp/icrp145-female-chest_no_arm.toml) | 2520 | 1024x256 | 256x256x153 | wfbp | `out/prepared-examples/icrp145-female-chest_no_arm/labels.mat` |
| [project-and-reconstruct/icrp/icrp145-female-head.toml](project-and-reconstruct/icrp/icrp145-female-head.toml) | 2160 | 1024x256 | 256x256x100 | wfbp | `out/prepared-examples/icrp145-female-head/labels.mat` |
| [project-and-reconstruct/icrp/icrp145-male-abdomen.toml](project-and-reconstruct/icrp/icrp145-male-abdomen.toml) | 1800 | 1024x256 | 256x256x128 | wfbp | `out/prepared-examples/icrp145-male-abdomen/labels.mat` |
| [project-and-reconstruct/icrp/icrp145-male-abdomen_no_arm.toml](project-and-reconstruct/icrp/icrp145-male-abdomen_no_arm.toml) | 2160 | 1024x256 | 256x256x128 | wfbp | `out/prepared-examples/icrp145-male-abdomen_no_arm/labels.mat` |
| [project-and-reconstruct/icrp/icrp145-male-chest.toml](project-and-reconstruct/icrp/icrp145-male-chest.toml) | 1800 | 1024x256 | 256x256x153 | wfbp | `out/prepared-examples/icrp145-male-chest/labels.mat` |
| [project-and-reconstruct/icrp/icrp145-male-chest_no_arm.toml](project-and-reconstruct/icrp/icrp145-male-chest_no_arm.toml) | 2520 | 1024x256 | 256x256x153 | wfbp | `out/prepared-examples/icrp145-male-chest_no_arm/labels.mat` |
| [project-and-reconstruct/icrp/icrp145-male-head.toml](project-and-reconstruct/icrp/icrp145-male-head.toml) | 2160 | 1024x256 | 256x256x100 | wfbp | `out/prepared-examples/icrp145-male-head/labels.mat` |
| [project-and-reconstruct/iterative/brainweb-center33-ossart.toml](project-and-reconstruct/iterative/brainweb-center33-ossart.toml) | 720 | 1024x256 | 181x217x33 | ossart | `out/prepared-examples/brainweb-center33/labels.mat` |
| [project-and-reconstruct/iterative/brainweb-circular-ossart.toml](project-and-reconstruct/iterative/brainweb-circular-ossart.toml) | 360 | 1024x256 | 512x512x400 | ossart | `data/phantoms/brainweb/normal-1mm.mat` |
| [project-and-reconstruct/iterative/brainweb-helical-ossart.toml](project-and-reconstruct/iterative/brainweb-helical-ossart.toml) | 2500 | 1024x256 | 512x512x400 | ossart | `data/phantoms/brainweb/normal-1mm.mat` |
| [project-and-reconstruct/quality/gold-foil-0p05-zmtf.toml](project-and-reconstruct/quality/gold-foil-0p05-zmtf.toml) | 540 | 1536x128 | 512x512x128 | fdk | `out/prepared-examples/custom/gold-foil-0p05.raw` |
| [project-and-reconstruct/quality/low-contrast-21.toml](project-and-reconstruct/quality/low-contrast-21.toml) | 540 | 1536x128 | 512x512x128 | fdk | `out/prepared-examples/custom/low-contrast-21.raw` |
| [project-and-reconstruct/quality/tungsten-wire-0p2.toml](project-and-reconstruct/quality/tungsten-wire-0p2.toml) | 540 | 1536x128 | 512x512x128 | fdk | `out/prepared-examples/custom/tungsten-wire-0p05.raw` |
| [project-and-reconstruct/quality/tungsten-wire-slanted-tan0p42.toml](project-and-reconstruct/quality/tungsten-wire-slanted-tan0p42.toml) | 540 | 1536x128 | 512x512x128 | fdk | `out/prepared-examples/custom/tungsten-wire-slanted-tan0p42.raw` |
| [project-and-reconstruct/quality/water-cylinder-200mm-fdk.toml](project-and-reconstruct/quality/water-cylinder-200mm-fdk.toml) | 360 | 1024x256 | 1024x1024x30 | fdk | `out/prepared-examples/custom/water-cylinder-200mm-pmma5mm.raw` |
| [project-and-reconstruct/quality/water-cylinder-350mm-gamma0.toml](project-and-reconstruct/quality/water-cylinder-350mm-gamma0.toml) | 360 | 1024x256 | 1024x1024x30 | fdk | `out/prepared-examples/custom/water-cylinder-350mm-pmma5mm.raw` |
| [project-and-reconstruct/quality/water-cylinder-low-dose-scatter-v4.toml](project-and-reconstruct/quality/water-cylinder-low-dose-scatter-v4.toml) | 360 | 1024x256 | 1024x1024x30 | fdk | `out/prepared-examples/custom/water-cylinder-200mm-pmma5mm.raw` |

## 旧入口迁移

| 旧入口 | 当前入口 |
|---|---|
| `calibration/geometry/tungsten-bead-double-ring-cylinder-marker.toml` | [project-and-reconstruct/calibration/tungsten-bead-double-ring-cylinder-marker.toml](project-and-reconstruct/calibration/tungsten-bead-double-ring-cylinder-marker.toml) |
| `calibration/geometry/tungsten-bead-double-ring.toml` | [project-and-reconstruct/calibration/tungsten-bead-double-ring.toml](project-and-reconstruct/calibration/tungsten-bead-double-ring.toml) |
| `calibration/geometry/tungsten-bead-line-3mm-spacing10.toml` | [project-and-reconstruct/calibration/tungsten-bead-line-3mm-spacing10.toml](project-and-reconstruct/calibration/tungsten-bead-line-3mm-spacing10.toml) |
| `calibration/geometry/tungsten-bead-spiral.toml` | [projection/calibration/tungsten-bead-spiral.toml](projection/calibration/tungsten-bead-spiral.toml) |
| `calibration/geometry/tungsten-bead-synchronized-ring-12-marker.toml` | [project-and-reconstruct/calibration/tungsten-bead-synchronized-ring-12-marker.toml](project-and-reconstruct/calibration/tungsten-bead-synchronized-ring-12-marker.toml) |
| `calibration/geometry/tungsten-bead-synchronized-ring-6-marker.toml` | [project-and-reconstruct/calibration/tungsten-bead-synchronized-ring-6-marker.toml](project-and-reconstruct/calibration/tungsten-bead-synchronized-ring-6-marker.toml) |
| `calibration/reconstruction/double-ring-calibrated-fdk.toml` | [reconstruction/calibration/double-ring-calibrated-fdk.toml](reconstruction/calibration/double-ring-calibrated-fdk.toml) |
| `calibration/reconstruction/double-ring-cylinder-marker-truth-fdk.toml` | [reconstruction/calibration/double-ring-cylinder-marker-truth-fdk.toml](reconstruction/calibration/double-ring-cylinder-marker-truth-fdk.toml) |
| `calibration/reconstruction/double-ring-cylinder-marker-truth-sart.toml` | [reconstruction/calibration/double-ring-cylinder-marker-truth-sart.toml](reconstruction/calibration/double-ring-cylinder-marker-truth-sart.toml) |
| `calibration/reconstruction/double-ring-truth-fdk.toml` | [reconstruction/calibration/double-ring-truth-fdk.toml](reconstruction/calibration/double-ring-truth-fdk.toml) |
| `calibration/reconstruction/double-ring-truth-ossart.toml` | [reconstruction/calibration/double-ring-truth-ossart.toml](reconstruction/calibration/double-ring-truth-ossart.toml) |
| `calibration/reconstruction/water-small-cone-fdk.toml` | [project-and-reconstruct/calibration/water-small-cone-fdk.toml](project-and-reconstruct/calibration/water-small-cone-fdk.toml) |
| `calibration/reconstruction/water-small-cone-heli-cyl-wfbp.toml` | [project-and-reconstruct/calibration/water-small-cone-heli-cyl-wfbp.toml](project-and-reconstruct/calibration/water-small-cone-heli-cyl-wfbp.toml) |
| `pipelines/analytic/brainweb-center33-fdk.toml` | [project-and-reconstruct/analytic/brainweb-center33-fdk.toml](project-and-reconstruct/analytic/brainweb-center33-fdk.toml) |
| `pipelines/analytic/brainweb-center33-heli-cyl-wfbp.toml` | [project-and-reconstruct/analytic/brainweb-center33-heli-cyl-wfbp.toml](project-and-reconstruct/analytic/brainweb-center33-heli-cyl-wfbp.toml) |
| `pipelines/analytic/pcb-nikon-tilted-fdk.toml` | [project-and-reconstruct/analytic/pcb-nikon-tilted-fdk.toml](project-and-reconstruct/analytic/pcb-nikon-tilted-fdk.toml) |
| `pipelines/analytic/pcb-planar-pose-fdk.toml` | [project-and-reconstruct/analytic/pcb-planar-pose-fdk.toml](project-and-reconstruct/analytic/pcb-planar-pose-fdk.toml) |
| `pipelines/analytic/water-cylinder-120kv-fullfov.toml` | [project-and-reconstruct/analytic/water-cylinder-120kv-fullfov.toml](project-and-reconstruct/analytic/water-cylinder-120kv-fullfov.toml) |
| `pipelines/iterative/brainweb-center33-ossart.toml` | [project-and-reconstruct/iterative/brainweb-center33-ossart.toml](project-and-reconstruct/iterative/brainweb-center33-ossart.toml) |
| `quality/gold-foil-0p05-zmtf.toml` | [project-and-reconstruct/quality/gold-foil-0p05-zmtf.toml](project-and-reconstruct/quality/gold-foil-0p05-zmtf.toml) |
| `quality/low-contrast-21.toml` | [project-and-reconstruct/quality/low-contrast-21.toml](project-and-reconstruct/quality/low-contrast-21.toml) |
| `quality/tungsten-wire-0p2.toml` | [project-and-reconstruct/quality/tungsten-wire-0p2.toml](project-and-reconstruct/quality/tungsten-wire-0p2.toml) |
| `quality/tungsten-wire-slanted-tan0p42.toml` | [project-and-reconstruct/quality/tungsten-wire-slanted-tan0p42.toml](project-and-reconstruct/quality/tungsten-wire-slanted-tan0p42.toml) |
| `quality/water-cylinder-200mm-fdk.toml` | [project-and-reconstruct/quality/water-cylinder-200mm-fdk.toml](project-and-reconstruct/quality/water-cylinder-200mm-fdk.toml) |
| `quality/water-cylinder-350mm-gamma0.toml` | [project-and-reconstruct/quality/water-cylinder-350mm-gamma0.toml](project-and-reconstruct/quality/water-cylinder-350mm-gamma0.toml) |
| `quality/water-cylinder-low-dose-scatter-v4.toml` | [project-and-reconstruct/quality/water-cylinder-low-dose-scatter-v4.toml](project-and-reconstruct/quality/water-cylinder-low-dose-scatter-v4.toml) |
| `ui/brainweb-center33-fullchain.toml` | [project-and-reconstruct/analytic/brainweb-center33-physics-fdk.toml](project-and-reconstruct/analytic/brainweb-center33-physics-fdk.toml) |
| `ui/brainweb-fullchain-new.toml` | [project-and-reconstruct/iterative/brainweb-helical-ossart.toml](project-and-reconstruct/iterative/brainweb-helical-ossart.toml) |
| `ui/brainweb-fullchain.toml` | [project-and-reconstruct/analytic/brainweb-full-fdk.toml](project-and-reconstruct/analytic/brainweb-full-fdk.toml) |
| `ui/brainweb-fullchain2.toml` | [project-and-reconstruct/iterative/brainweb-circular-ossart.toml](project-and-reconstruct/iterative/brainweb-circular-ossart.toml) |
| `ui/brainweb-statistical-gd.toml` | [project-and-reconstruct/experimental/brainweb-statistical-algebraic.toml](project-and-reconstruct/experimental/brainweb-statistical-algebraic.toml) |
| `ui/icrp/icrp145-female-abdomen.toml` | [project-and-reconstruct/icrp/icrp145-female-abdomen.toml](project-and-reconstruct/icrp/icrp145-female-abdomen.toml) |
| `ui/icrp/icrp145-female-abdomen_no_arm.toml` | [project-and-reconstruct/icrp/icrp145-female-abdomen_no_arm.toml](project-and-reconstruct/icrp/icrp145-female-abdomen_no_arm.toml) |
| `ui/icrp/icrp145-female-chest.toml` | [project-and-reconstruct/icrp/icrp145-female-chest.toml](project-and-reconstruct/icrp/icrp145-female-chest.toml) |
| `ui/icrp/icrp145-female-chest_no_arm.toml` | [project-and-reconstruct/icrp/icrp145-female-chest_no_arm.toml](project-and-reconstruct/icrp/icrp145-female-chest_no_arm.toml) |
| `ui/icrp/icrp145-female-head.toml` | [project-and-reconstruct/icrp/icrp145-female-head.toml](project-and-reconstruct/icrp/icrp145-female-head.toml) |
| `ui/icrp/icrp145-male-abdomen.toml` | [project-and-reconstruct/icrp/icrp145-male-abdomen.toml](project-and-reconstruct/icrp/icrp145-male-abdomen.toml) |
| `ui/icrp/icrp145-male-abdomen_no_arm.toml` | [project-and-reconstruct/icrp/icrp145-male-abdomen_no_arm.toml](project-and-reconstruct/icrp/icrp145-male-abdomen_no_arm.toml) |
| `ui/icrp/icrp145-male-chest.toml` | [project-and-reconstruct/icrp/icrp145-male-chest.toml](project-and-reconstruct/icrp/icrp145-male-chest.toml) |
| `ui/icrp/icrp145-male-chest_no_arm.toml` | [project-and-reconstruct/icrp/icrp145-male-chest_no_arm.toml](project-and-reconstruct/icrp/icrp145-male-chest_no_arm.toml) |
| `ui/icrp/icrp145-male-head.toml` | [project-and-reconstruct/icrp/icrp145-male-head.toml](project-and-reconstruct/icrp/icrp145-male-head.toml) |
