# 统一流水线基础层

多能前投、重建前处理和重建后处理共享 `Stage.hpp`，这是直接重构后的内部接口，不通过旧接口适配运行。

## 公共契约

- `Location` 是唯一的 Host/Device 位置枚举。业务名称只是这个类型的别名，不再维护三套枚举。
- `IStage<Data, Context, Domain>` 统一提供 `info / contract / validateContext / prepare / process / reset`。
- `Contract<Domain>` 声明输入/输出数值域和数据位置。Stage 只处理数值，数值域与位置标记由执行器提交；Stage 自行修改这些标记会报错。
- `IOutput<Data, Context, Domain>` 统一输出能力检查和 `prepare / write / finish / reset`。处理与导出分开，避免导出时修改数值。
- `execute()` 处理单份数据或连续批次。首先验证整个输入批次的域，必要时整批传输一次，然后按输入顺序处理，验证状态，并提交声明的输出状态。
- `RunScope / onExit / resetReverse` 管理运行标志和逆序清理。`Progress` 管理单调进度。`FactoryRegistry<Product>` 管理有类型的插件工厂。

三条生产链实际调用同一个执行器，不仅是接口命名相似。前投信号处理及输出、前处理校正及输出、后处理及输出均已接入。

## 模板体系：为什么使用，以及使用到哪里

当前共有 11 处模板定义，分成两部分：6 类用于表达有类型的契约/接口/工厂，5 个用于实现公共工具。并非 11 套不同流水线。

| 模板 | 参数与用途 | 保留它的理由 |
|---|---|---|
| `Contract<Domain>` | 业务数值域类型 | 前投能量/计数、投影对数和体积衰减/HU 不应混用；位置统一为 `Location` |
| `SourceContract<Domain>` | 源的输出数值域和数据位置 | 源阶段没有前置同类型数据，需要独立表达生成契约 |
| `ISource<Data, Context, Domain>` | 批次输出数据、源输入上下文、数值域 | 将生成数据的多能前投与修改现有投影的处理阶段区分 |
| `IStage<Data, Context, Domain>` | 数据类型、业务上下文、数值域类型 | 同一份阶段接口分别用于投影帧与体积，插件的 process 参数保持类型安全 |
| `IOutput<Data, Context, Domain>` | 只读输出数据、输出上下文、数值域类型 | 共用输出生命周期，同时保留各自的格式与单位约束 |
| `FactoryRegistry<Product>` | 要创建的插件接口类型 | 注册前投 Stage 的工厂不能误返回体积 Exporter；工厂基础逻辑只实现一次 |
| `execute<Data, Domain, Transfer, Process, Synchronize>` | 数据/数值域，以及三个回调类型 | 同一个执行器处理单份或批次数据；回调类型由调用自动推导，不需要手写参数列表 |
| `Cleanup<F>`、`onExit<F>` | 清理回调类型 | 让捕获资源的 lambda 直接保存在作用域守卫里，不必先转成 std::function |
| `resetReverse<Stages>` | 阶段容器类型 | 同时遍历 unique_ptr 容器和借用指针容器，逆序清理规则相同 |
| `advanceDomain<Domain>` | 业务数值域类型 | 共用数值域校验，保留枚举的类型约束 |

`RunScope`、`Progress`、`Location` 和 `ExecutionTiming` 是普通类型，没有模板。这里使用的是常规类/函数模板，没有递归元编程、SFINAE 或 CRTP。

模板不是全部必需。契约、Source、Stage、Output 和工厂的类型参数直接保护业务边界，价值最明确；清理、逆序遍历和回调执行也可以用普通接口或 std::function 实现。现在这些工具选用模板，是为了保留具体回调类型、减少重复实现，不是为了提高正投 kernel 的计算速度。若后续编译耗时、错误信息或代码体积成为问题，应优先简化辅助工具，保持业务类型约束。

### 实际实例化的业务接口

| 用途 | Data | Context | Domain |
|---|---|---|---|
| 前投 Stage | `ProjectionFrame` | `ProjectionStageContext` | `ProjectionSignalDomain` |
| 前投 Output | `ProjectionFrame` | `ProjectionOutputContext` | `ProjectionSignalDomain` |
| 前处理 Stage/Output | `preprocess::Frame` | `preprocess::Context` | `preprocess::Domain` |
| 后处理 Stage | `postprocess::Volume` | `postprocess::Context` | `postprocess::Domain` |
| 后处理 Output | `postprocess::ReadOnlyVolume` | `postprocess::Context` | `postprocess::Domain` |

前处理和后处理的 `IStage` 是公共接口的类型别名；前投的 `IProjectionStage` 继承公共接口，增加 phase 和空气参考处理。普通插件只继承业务接口，通常不需要直接填写公共模板参数。

### 编译期与运行期各负责什么

- 编译期：确定 process/write 的参数类型、Context 类型和 Domain 类型，防止错传另一条链的数据。
- 运行期：选择插件 ID、构造插件实例、选择阶段数量/顺序、检查实际数值域及位置、通过虚函数执行插件。
- 回调类型在编译期确定，但 transfer/process/synchronize 仍在任务运行时调用。使用模板并不意味着插件列表在编译期固定。

因此这是“泛型接口 + 运行期多态”的组合。扩展一个后处理算法通常只需实现一个 `postprocess::IStage`；算法参数放在插件构造参数或 Context 中，不应继续给公共执行器增加算法专属模板参数。

## 对应的设计模式

这一体系由几种思想组合而成，不能用一个 GoF 模式概括。C++ 的 template 语言机制也不等同于“模板方法”设计模式。

| 模式或惯用法 | 对应实现 | 这里的含义 |
|---|---|---|
| **流水线 / Pipes and Filters**（架构模式） | 三条 Pipeline 与有序 Stage | 数据按阶段顺序经过多个处理模块；这是整体结构 |
| **策略模式 / Strategy** | Stage、Output 的虚接口与具体插件 | Pipeline 依赖接口，具体降噪、校正或输出策略可替换 |
| **固定流程骨架**（模板方法思想） | `execute()` | 共用校验、传输、处理、状态提交和同步流程；具体动作通过回调提供。实现采用组合，不是经典的基类继承式 GoF Template Method |
| **工厂注册表 / 简单工厂** | `FactoryRegistry<Product>` | 按 ID 找到构造函数，创建对应类型的插件。没有通过继承覆写创建方法，不应称作严格的 GoF Factory Method；也不是产品族的 Abstract Factory |
| **依赖注入** | typed Context 与 transfer/process/synchronize 回调 | Backend、工作区、进度和传输能力由调用者提供，公共层不查找全局服务 |
| **RAII / Scope Exit**（C++ 资源管理惯用法） | `RunScope`、`Cleanup`、逆序 reset | 正常与异常退出都清理资源并恢复运行标志 |

阶段默认依次执行，不是“某个处理器处理后就终止传递”的职责链。`append()` 是装配接口，也没有必要把它包装成完整的 Builder 模式。进度使用回调通知，目前没有多订阅者的观察者管理系统。

新增功能时，优先明确它属于数值 Stage、输出策略还是前投 Backend，再实现对应接口；不需要为了增加一个算法再新增一种设计模式或一层模板。

## 业务专用部分

| 链条 | 数据 | 上下文 | 专用调度 |
|---|---|---|---|
| 多能前投 | `ProjectionFrame` 批次 | 扫描配置、前投 Backend | 根据物理 phase 稳定排序；每帧信号后立即处理空气参考；跨批次保留余晖等状态 |
| 重建前处理 | `preprocess::Frame` | 配置、CUDA 像素工作区、输出路径 | 内置校正后追加插件；最终必须回到对数衰减 |
| 重建后处理 | `postprocess::Volume` | 几何、辅助投影路径、传输桥、输出路径 | 所有数值处理后执行多个只读 Exporter |

准备接口统一为 `prepare(const Context&)`。后处理运行时把当前体积 Grid 放入 Context，插件通过 `context.grid` 准备工作区。业务辅助输入由 `validateContext()` 检查；不同链条不共享无类型参数包。

前投 Backend 仍负责正投、批次容量、空气/计数参考和设备存储；输出数据的物理单位不变。不同 Domain 枚举保留不同类型，不能把投影对数、能量、线性衰减或 HU 混成一种数据。

## 性能与生命周期

执行器没有 CUDA 依赖，也不创建显存、不复制业务数据。只有位置不匹配才调用业务 transfer 回调；连续设备 Stage 保持设备驻留。是否等待设备完成由业务 synchronize 回调决定，保持现有前投与前处理同步策略；后处理默认主机执行，不新增设备同步。

阶段过程计时由共同执行器完成；数值取决于业务是否同步设备。没有同步时表示主机提交耗时，不能当作 GPU 完成耗时。文件读取、标定、正投主 kernel 等专有操作继续由业务记录。

prepare 可能部分失败，reset 必须 noexcept，并能清理尚未准备或部分准备的对象；任务正常结束和异常结束都按逆序清理。处理失败不自动回滚原地修改的数值。每个实例只在一个 worker 中使用，公共执行层没有承诺跨线程并发操作同一 Pipeline。

## 插件示例

```cpp
class MyStage final : public preprocess::IStage {
public:
    const pipeline::PluginInfo& info() const noexcept override {
        static const pipeline::PluginInfo description{
            "custom.device_correction", "My device correction",
            "Custom correction operating on resident projection data.", "1",
            pipeline::PluginKind::PreprocessStage};
        return description;
    }
    pipeline::Contract<preprocess::Domain> contract() const override {
        return {preprocess::Domain::LogAttenuation,
                preprocess::Domain::LogAttenuation,
                pipeline::Location::Device,pipeline::Location::Device};
    }
    void process(preprocess::Frame& frame,const preprocess::Context& context) override {
        // 在 context 的设备工作区操作 frame.device_data；不修改 domain/location 标记。
    }
};
```

前投插件使用同一契约并另外声明 `phase()`，需要时实现 `processAir()`；后处理插件使用 Volume 与 Context，输出插件实现 `acceptsDomain()` 和 `write()`。工厂直接使用 `pipeline::FactoryRegistry<业务插件类型>` 的 `add / create`，无需重复实现注册基础设施。

## 可运行的 BHC 示例

完整 CPU 多项式 BHC 插件、构造参数、接入代码和验证说明见 [BHC Stage 示例](../../examples/README.md)。它实际通过前处理 Pipeline 运行，而不是独立的示意接口。

## 插件静态自描述与查询

`IPlugin` 是 Stage 和 Output 的共同顶层接口，强制实现 `const PluginInfo& info() const noexcept`。旧的 name() 已删除，日志及阶段名称统一读取 info().name。描述与 prepare/process/reset 的执行状态独立；通常返回静态常量或实例构造时保存的不可变描述。

| 信息 | 含义及要求 |
|---|---|
| id / name / kind | 必填的稳定 ID、显示名称、插件类别；kind 为前投 Source、前投 Stage、前处理 Stage、后处理 Stage 或 Exporter |
| description / version | 功能说明与实现版本，允许为空；版本不是运行计数 |
| category / tags | 功能分类及检索标签，例如 bhc、scatter、polynomial、cpu/cuda |
| vendor / documentation | 维护来源及文档路径/链接 |
| implementation_status / deprecated_message | Stable、Experimental、Deprecated 及可选弃用说明 |
| parameters | 可选参数说明；包含 key、名称、含义、静态类型和单位，不含当前值或运行统计 |

参数说明不是运行参数存储，也不是数值域/设备能力的替代品。数据约束仍来自 contract；进度、耗时、显存占用和错误属于运行状态。这里不会根据元信息自动生成 UI 控件。

`FactoryRegistry<Product>::add(PluginInfo, Factory)` 直接从描述读取 ID，注册时检查必填字段、枚举、参数键及重复 ID。注册的 Product 必须实现 IPlugin。创建时检查对象返回的完整元信息与注册描述一致，避免注册内容和对象描述分叉。

```cpp
postprocess::StageRegistry stages;
const auto& description=pipeline::builtinPluginInfo(pipeline::BuiltinPlugin::HuConversion);
stages.add(description, [] {return postprocess::makeHuConversionStage(.019285);});

for(const auto& info:stages.list()) {
    // 列出这个注册表实际注册的插件，返回独立描述快照。
    std::cout << info.id << ": " << info.description << '\n';
}
if(const auto* info=stages.find("postprocess.hu")) {
    // 仅查询描述；不创建算法实例或设备资源。
}
auto plugin=stages.create("postprocess.hu"); // 这里才调用工厂。
```

find() 对未知 ID 返回 nullptr；返回的只读指针由注册表拥有，不能在注册表销毁后保留。list() 返回描述副本，修改副本不影响注册表。create() 对未知 ID、空结果和元信息不一致显式报错。

`BuiltinPluginInfo.hpp` 提供当前 23 份内置描述（含 CPU/CUDA 实现与 BHC 教学示例），可通过 builtinPluginDescriptions() 和 findBuiltinPluginInfo(id) 查询，无需实例化插件。该目录描述已知实现，不代表某个构建已经编入或注册了全部工厂；判断某个注册表可创建哪些插件，应以该注册表的 list/find 为准。CPU 与 CUDA 实现使用不同 ID，功能分类保持一致。

元信息查询只涉及普通字符串、容器和静态描述初始化，不读取投影/体积文件，也不初始化 CUDA。专用元信息测试目标不链接 CUDA，并验证 list/find 不触发工厂构造。

## 多能前投源 Stage

`MultiSpecFpStage` 实现公共 `ISource` 的投影专用接口 `IProjectionSource`，作为整条链的起点：

体积/材料/能谱/几何 -> MultiSpecFpStage -> 散射/噪声/串扰/余晖/DAS -> Output。

prepare 的 SpectralSourceContext 借用当前配置、标签体积及谱模型，只在准备调用期间有效，源不保留该 Context 的引用。源持有 Backend 并控制准备、复用与清理；融合路径遍历和多能积分 kernel 不拆分，不增加中间路径数组。

- contract() 在 prepare 前即可查询，声明输出数值域和位置。
- prepare() 设置本次任务的尺寸、视角范围和 Backend 批次容量；准备失败立即清理部分资源。
- generate(first,count,frames) 产生连续视角批次，并验证尺寸、数量、视角编号、域和设备位置。Host 输出在这里建立信号和空气参考状态。
- batchCapacity() 仅在准备后有效，保持 Backend 原有内存预算，尾批次自然缩小。
- 设备帧直接交给后续处理，不在源阶段下载；ensureLocation/finalizeBatch/synchronize 等专用服务由源转交 Backend。
- reset() 幂等；成功、失败和重新运行都由源统一管理 Backend 清理，不再由 Pipeline 同时清理 Backend。

正投 Pipeline 只接受 IProjectionSource，不保留接收 Backend 的旧构造入口。业务 StageContext 也只持有 source，不直接访问 Backend。Pipeline::sourceInfo() 可以查询当前源的自描述。

CUDA 源通过 `primary_detail::makeCudaProjectionSourceRegistry()` 注册：

```cpp
auto sources=primary_detail::makeCudaProjectionSourceRegistry();
const auto* description=sources.find("projection.source.multi_spec_fp"); // 不构造 Backend
SpectralProjectionPipeline pipeline(sources.create("projection.source.multi_spec_fp"));
// pipeline.run(config,labels,model,output,error);
```

源不是普通投影 Stage，不需要虚构 input_domain 或实现空 process。它共享 IPlugin 元信息和生命周期规范，保留自己的有类型输入和批次生成接口。
