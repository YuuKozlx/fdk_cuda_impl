# Quick Plot：独立快速绘图工具

Windows C++ 可执行程序，复用项目已有 Gnuplot 后端，不启动仿真 UI，不依赖 Python、VTK 或重建库。常规绘图只需改 CSV 数据或 JSON 配置，不必重新编译。不增加新的绘图库；本工具直接调用原有后端，以便控制交互窗口和可靠等待导出结束。

## 最常用操作

```powershell
# 导出配置中的子图阵列
.\quick_plot.exe .\examples\gallery.json

# 导出并弹出窗口
.\quick_plot.exe .\examples\gallery.json --show

# 单独弹出第 3 个子图（编号从 1 开始，按行排列）
.\quick_plot.exe .\examples\gallery.json --panel 3

# 无需配置文件，直接叠加 CSV 两列曲线
.\quick_plot.exe --line .\examples\curves.csv --x x --y sin,cos --output lines.png

# 矩阵快速出图
.\quick_plot.exe --heatmap .\examples\matrix.csv --output heatmap.png
.\quick_plot.exe --surface .\examples\matrix.csv --show

# 生成完整可编辑示例（目录不能已有 gallery.json）
.\quick_plot.exe --examples .\my-examples
```

导出支持 PNG、SVG；已有同名输出会替换。省略 output 时默认弹窗；也可指定 `show: true`。相对数据路径和输出路径相对于 **JSON 所在目录**，直接命令的路径相对于当前目录。中文文件名、中文标题支持 UTF-8。

三维单图窗口：左键拖动旋转，中键拖动缩放；具体快捷键由 Gnuplot Qt 后端提供。命令在窗口打开期间保持运行，关闭窗口后结束。阵列总览可弹窗，但后端不能可靠地对任意子图独立缩放/旋转，因此需用 `--panel N` 单独打开需要操作的子图。单图模式不会覆盖总览输出。当前没有阵列点击选中、拖动重排或参数编辑器。

## 配置文件

```json
{
  "rows": 1,
  "cols": 2,
  "width": 1400,
  "height": 650,
  "font_size": 12,
  "title": "标定比较",
  "output": "comparison.png",
  "show": false,
  "panels": [
    {
      "type": "line",
      "title": "解析与拟合",
      "xlabel": "Al thickness (mm)",
      "ylabel": "p",
      "xlim": [0, 10],
      "series": [
        {"file": "low-al-curves.csv", "x": "al_mm", "y": "analytic_p", "where": {"pmma_mm": 0}, "name": "Analytic", "color": "blue"},
        {"file": "low-al-curves.csv", "x": "al_mm", "y": "fitted_p", "where": {"pmma_mm": 0}, "name": "Fitted", "color": "red", "dash": 2}
      ]
    },
    {
      "type": "surface",
      "title": "三维曲面",
      "file": "matrix.csv",
      "x0": -3,
      "y0": -3,
      "dx": 0.15,
      "dy": 0.15,
      "view": [60, 30]
    }
  ]
}
```

## 支持的图与数据

| type | 数据 | 用途 |
|---|---|---|
| line | 有表头的数值 CSV，x/y 列 | 单条或多条折线 |
| scatter | 同上 | 二维散点 |
| scatter3d | 有表头的数值 CSV，x/y/z 列 | 三维散点或轨迹 |
| heatmap | 无表头的矩形数值 CSV | 二维热图，无拟合插值 |
| contour | 同上 | 等高线 |
| surface | 同上 | 三维规则网格曲面 |

表格 CSV 默认首行为列名；`header: false` 时列用从 0 开始的整数索引。各行必须等长，所有数据列必须是有限数值。支持引号包围的列名，不支持跨行 CSV 单元格。缺失值和 NaN 会报错，不会静默忽略。

矩阵 CSV 的列方向为 X、行方向为 Y；第一行对应 y0，坐标依次增加 dx/dy。默认起点 0、间距 1。不是输入三列 x/y/z 后自动网格化；非规则三维点请用 scatter3d。

通用子图字段：`title/xlabel/ylabel/zlabel`，`xlim/ylim/zlim/clim` 两元素范围，`xlog/ylog` 对数坐标。曲线 series 支持 `name/color/width/dash/style`，style 为 lines、points 或 linespoints。`where` 为列名到数值的等值条件，比较容差为 1e-9 × max(1, |指定值|)，筛选不到数据会报错。contour 的 `levels` 为 2–100。surface 的 `view` 为 `[仰角, 方位角]`。

## 构建和运行时

沿用 `plot/CMakeLists.txt` 的独立构建，目标名 `quick_plot`。这一个目标只编译本地 C++ 和已有 nlohmann JSON 头文件；不链接 Matplot++，但同目录其他绘图目标仍使用它。

运行时优先查找 exe 旁的 `gnuplot/bin/gnuplot.exe`，否则使用构建时已有 Gnuplot 路径。本机发布目录复用现有运行时，不是完全独立的单文件发行包。迁移电脑时应携带完整 Gnuplot 运行时到上述相对位置及其许可证，不能只拷一个 exe。

PNG/SVG 旁的 `.gp` 是程序自动生成的绘图记录，平常不用修改。交互记录放在系统临时目录 `fdk-quick-plot`。界面关闭后临时记录暂不自动清理，便于定位问题。

验证：CTest `quick_plot_smoke` 检查中文路径、PNG 签名、数据筛选、错误列名、非法布局、空筛选、不规则 CSV、NaN 和非法子图编号。生成示例用于检查六类图的阵列导出。
