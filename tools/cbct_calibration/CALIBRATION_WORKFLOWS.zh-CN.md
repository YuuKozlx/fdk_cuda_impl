# CBCT钢珠标定代码总览

## 最简单的使用方法

不需要记命令，也不需要在终端里拼参数。用 PyCharm、VS Code 或其他 Python IDE 打开下面两个文件之一：

- `single_row/single_row_example.py`：单排等间距钢珠；
- `identifiable_phantom/identifiable_phantom_example.py`：编号可辨识三维模体。
- `cho_reference/cho_raw_example.py`：Cho 完整几何标定。

打开后只改文件顶部 `CONFIG` 里的 `raw_path` 和 `output_directory`，确认 `views`、像素尺寸和模体尺寸，随后直接点击 IDE 的“运行”按钮。程序会自动读取 RAW、完成预处理和标定，并把结果写入输出目录。重点查看 `calibration.json`；单排流程另外生成 `tracks.npz` 和 `reprojection.npz`，双环流程生成 `points.npy`。

如果你不熟悉 Python，可以把这两个示例理解为“配置文件加运行按钮”，不需要把任何命令复制到命令行。

本目录保留两条相互独立的主流程：

| 模体 | Python入口 | 主要结果 |
|---|---|---|
| 单排等间距钢珠 | `single_row/single_row_workflow.py` | 椭圆轨迹法固定几何 |
| 编号可辨识三维模体 | `identifiable_phantom/identifiable_phantom_workflow.py` | 逐帧DLT与重建等效几何 |
| Cho 双环完整几何 | `cho_reference/cho_raw_example.py` | 每帧投影矩阵、共享内参和源轨迹 |

两个示例文件都已经放在 `tools/cbct_calibration` 目录中。

## 统一分层

```text
预处理层：RAW、检测、亚像素质心、跨帧编号、边界/粘连状态
标定层：已编号二维点、模体先验、几何求解、重投影验证
验证层：参数合理性、中间可视化、实际重建
```

预处理失败时不能继续解释参数；低重投影误差也不能证明编号一定正确。

## 结果判断顺序

1. 目标是否检测正确；
2. 跨帧编号是否连续且无跳变；
3. 越界、停用和粘连是否有记录；
4. 重投影误差是否随帧稳定；
5. 坐标系是否满足右手系 `U×V=N`；
6. FDK/OS-SART重建是否闭合且无双边缘。

## 机械参数和等效参数

源偏移、探测器偏移和模体姿态可能存在规范耦合。机械诊断需要额外测量；重建可以采用自洽的等效参数或逐帧投影矩阵。双环当前采用源偏移固定为0的等效规范，输出参数必须整组使用。

详细说明分别见：

- `single_row/SINGLE_ROW_GUIDE.zh-CN.md`
- `identifiable_phantom/IDENTIFIABLE_PHANTOM_GUIDE.zh-CN.md`
- `cho_reference/CHO_GUIDE.zh-CN.md`
