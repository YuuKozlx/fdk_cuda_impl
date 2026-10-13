# 重建后处理插件

本模块属于 `multispectrum_sim` 应用层，接在重建完成与文件导出之间，不修改 YKCBCT 重建算法。独立于 VTK 和 ImGui，未增加外部依赖。

## 流水线与插件

- `IStage`：通过强制 `info()` 提供自描述，通过统一 `contract()` 声明输入/输出数值域及数据位置，通过 `validateContext()` 检查辅助输入；通过 `prepare / process / reset` 管理计算与工作区。所有处理阶段先执行，再导出。
- `IExporter`：接收只有只读数据指针的 `ReadOnlyVolume`，负责格式化及导出，不能修改流水线的体积。
- `pipeline::FactoryRegistry<Product>`：按 ID 注册工厂，创建指定类型的 Stage 或 Exporter；未知 ID、重复注册、空工厂结果显式报错。注册表由调用者拥有，没有静态全局注册，也不加载外部 DLL。
- `Pipeline`：预检查整条链的数值域、辅助输入和传输能力，顺序执行，报告进度及每阶段耗时；发生异常也会 reset 阶段。每个工作线程使用自己的实例，不并发复用。

Stage 可原地修改体积；失败时不自动回滚数值，避免无条件复制一份体积。需要事务行为的插件应自行使用工作区，并在成功后写回。Exporter 失败可能留下部分文件，不保证多文件原子导出。

## 数据和几何契约

`Volume` 借用 float32 存储，顺序为 `[z][y][x]`。位置标识指出权威数据位于 Host 还是 Device；另一份数据可以已经过期。Stage 不得保留借用指针。跨位置时才调用 `IDataTransfer::ensure`，由调用者提供传输、显存池和 stream；缺失桥接时明确拒绝执行，不自动分配显存或偷偷上传下载。

当前内置 Stage/Exporter 在 CPU 上执行，接入点使用重建原本就下载到内存的结果；没有新增 CUDA 后处理 kernel 或生产用传输桥。连续设备 Stage 的驻留调度由接口支持，已用模拟传输测试验证。日志 `host_ms` 表示主机执行/提交耗时，异步设备插件如需 GPU 完成时间应另用 CUDA events。

`Grid` 包含尺寸、毫米间距、首体素中心位置及方向矩阵。当前数据在扫描器世界坐标系，不能直接冒充 DICOM 患者坐标系。`Context` 保留原始投影、实际送入重建的投影和源配置路径；需要投影的插件在 `validateContext()` 中检查路径。投影路径可供将来的 MAR 等方法使用，但这里尚无重投影或重建反馈调度接口。

衰减值 `AttenuationMmInv` 与 CT 值 `HounsfieldUnits` 明确区分。输出数值域与声明不符会报错。

## 已接入的功能

默认重建输出已通过 Exporter 完成：

1. float32 衰减值 RAW，保持原有体素顺序和 UI 读取约定。
2. 可选三张正交中心切片 BMP，保持原来的灰度映射和布局。

`makeHuConversionStage(mu_water_mm_inv)` 可在独立导出流水线中转换 CT 值；调用者应传入 BHC 拟合参考能量对应的水衰减系数。它不会默认修改原始重建数据。默认 RAW Exporter 只接受衰减值，防止 UI 把 HU 再转换一遍。HU/DICOM 分支应从独立导出缓冲或插件内部按块转换，保留原始衰减体积。

`reconstructWithLibraryFdk` 的 `PostprocessConfigurer` 参数用于安装后处理扩展。例如：

```cpp
reconstructWithLibraryFdk(config, projection, volume, slices, diagnostic, progress,
    [&](postprocess::Pipeline& pipeline) {
        // stages/exporters 是调用者的有类型工厂注册表。
        pipeline.append(stages.create("denoise"));
        pipeline.appendExporter(exporters.create("dicom"));
    });
```

配置回调安装扩展之后，应用会继续添加标准衰减 RAW 和可选 BMP 导出，确保现有显示可以读取。此示例仅说明接入方式；当前未注册或实现 denoise/dicom 插件。独立的 Pipeline 不会自动增加导出格式，可自行构造需要的导出链。

## 后续功能边界

- 降噪：可以直接实现体积 Stage，使用声明的间距和数值域。
- DICOM：实现 Exporter，同时明确患者坐标、必要元数据、UID、像素编码及 HU 标度；当前未实现 DICOM 文件写入。
- 骨硬化与去金属伪影：纯图像方法可以作为 Stage；投影域校正或需要重新重建的方案应接前处理/重建链。保留投影路径不等于这些算法已经实现。

默认不增加新的 TOML 开关或 UI 控件，后续插件有实际实现后再按能力显示配置。

## 公共执行层

Stage、Exporter、状态转换、逆序清理、进度、计时和工厂使用 [统一流水线基础层](../../../include/pipeline/README.md)，不再维护独立的 Stage/Requirements/Registry 契约。`prepare(const Context&)` 通过 `context.grid` 获取当前体积几何；插件不得自行修改 domain/location 标记，转换由流水线提交。

Stage 与 Exporter 通过共同 IPlugin::info() 提供元信息，注册表通过 add(PluginInfo, Factory) 注册，list()/find(id) 查询不会实例化插件；详见统一流水线的静态自描述说明。


强度处理已按标定、校正、分割拆分到 `stages/`。μ↔HU 可配置基准、空气／水 ROI 标定、HU 线性校正和多级标签分割的 CPU/CUDA 接口见 [使用说明](../../../include/recon/postprocess/stages/README.md)。标签域与 HU、衰减域分开，原有衰减 RAW 导出不会接收标签图。
