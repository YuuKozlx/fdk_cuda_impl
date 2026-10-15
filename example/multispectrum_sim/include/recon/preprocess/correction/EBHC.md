# EBHC 实验性骨硬化校正

## 原论文三项模型实现（当前入口）

已核对本地原论文：Kyriakou et al., Medical Physics 2010, pp.5181–5183, Eq.(4)/(5), II.A–II.E。
PDF：`C:/Users/zlx/Zotero/storage/A8HZIDTQ/Kyriakou et al_2010_Empirical beam hardening correction (EBHC) for CT.pdf`。

此前二项原型漏掉了 c01*f01，不能视为原文复现。以下三项版本独立保存，不覆盖旧实验，不默认接入 UI。

1. 输入为水预校正后的投影 p0 和重建 f0（本次还先做了散射校正）。
2. f01=f0*wbone(HU)，wbone 在 T1/T2 间线性从 0 过渡至 1。保留衰减幅值，不用二值路径。
3. 3D Joseph 对 f01 正投为 p2。毫米几何与 mm^-1 衰减，得到无量纲线积分。
4. 重建 f11=R(p0*p2)、f02=R(p2²)，使用与 f0 相同的 WFBP 配置。
5. 求三个系数，最小化整个体积的 TV(f0+c01*f01+c11*f11+c02*f02)。
6. 使用原始 f01 组合，不用 R(A(f01)) 替换它，因此按论文式(5)提供图像域 Stage。

工厂：`postprocess::makeEbhcVolumeStage`；元信息：`postprocess.ebhc_three_term`。
契约：设备 AttenuationMmInv → AttenuationMmInv，原位相加；三份基图由显存池管理。
估计器：`fitEbhcFullVolumeTv`，独立于 Stage，基图与配置必须属于同一扫描。
优化期间基线与三份基图驻显存，总计约 1.56 GiB（512×512×400），不预占全部显存。
本文只声称已实现可见公式与步骤，不声称获得作者源代码或完整复现结果。

### 原文未完全规定的数值选择

- Joseph：按体素坐标主轴的中心平面步进，另两轴双线性插值；体积外取零；入/出边界截断积分长度。
- TV：梯度向量的欧氏模；向负向相邻体素作差并除以体素间距，最小索引边界导数为零。
  输出成本除以体素数，只改变整体比例，不改变极小值；未作区域筛选、降采样、Huber 化或灰度惩罚。
- simplex：采用 Nelder–Mead；从零系数和三个偏移点 (.05,0,0)、(0,.01,0)、(0,0,.01) 开始。
  系数单纯形直径 <1e-5 且目标跨度 <1e-7×初始目标时停止，最多约 600 次评价。没有人为系数范围限制。
- 本文正文举 T2≈500 HU，图3举 T2=1500 HU。本实验分别验证这两个值，T1=100 HU。
- 使用现有 WFBP，未实现论文引用的 EPBP。论文称方法不依赖具体重建算法，但此差异仍需记录。

### 运行

构建 `multispectrum_ebhc_paper`，再执行：

```text
multispectrum_ebhc_paper basis basis-job.json
multispectrum_sim cross/reconstruct.toml
multispectrum_sim square/reconstruct.toml
multispectrum_ebhc_paper fit fit-job.json
```

完整参数 JSON、TOML、输入路径与优化轨迹保存在：
`G:/Code/fanproj/fdk-test/out/brainweb-helical-wfbp-bone-singlepass-mas5-20261013/ebhc-paper`。
1500 HU 对照在同级 `ebhc-paper-t1500`。
这些实验仅使用实际初始图和投影估计系数；单能参考与材料真值只用于独立评价。
输出 reconstruction.raw 是候选，优化收敛不代表质量合格，必须检查对应 evaluation.json。

### 首轮 500 HU 结果

三系数 (-0.0111139448, -0.2697743908, 0.1783101403)，174 次目标评价收敛。
TV 0.00236239893→0.00191863859（下降约18.8%），但骨内均值与结构明显退化，不启用。
独立 CPU 全体积复算与 GPU 成本相符，不能归因为 reduction 或 TV 数值实现错误。
已有物理模型校正的 TV 为0.00236319274，略高于基线；“TV 越低”等价于“骨伪影越少”在该完整物理过程数据上未成立。
这不证明原文无效，仍需区分复现未公开细节、重建/正投一致性和扫描噪声等因素，不能擅自添加约束后宣称原文成功。

### 500 / 1500 HU 全体积验收

两组均采用完整 2500 帧、400 层，无参考图像参与参数估计；均未通过质量验收。

| T2 | TV 下降 | 骨间暗误差 RMS | 骨内部误差 RMS |
|---|---:|---:|---:|
| 未作 EBHC 的基线 | — | 8.87 HU | 51.31 HU |
| 500 HU | 18.78% | 71.23 HU | 1366.34 HU |
| 1500 HU | 18.75% | 66.53 HU | 1370.92 HU |

1500 HU 系数 (-0.06440518,-0.46486670,0.58956161)，182 次评价收敛。
基图正投约 63 s，两次 WFBP 各约 23 s；拟合与组合约 19–21 s。并非严格性能基准。
两个配置都在降低 TV 的同时压低骨值；这排除了“只需改用文中另一个示例阈值”的解释。
尚不能区分论文未公开数值细节、重投影/重建一致性和完整物理过程数据特点的影响，不能据此否定原文。
两个结果目录的 quality-review.json 均标记 accepted=false；不默认使用候选结果。

测试覆盖：Joseph 轴向/反向/斜向解析路径、无交射线、CPU/CUDA 一致性；
各向异性网格的 CPU/CUDA 全体积 TV、三系数已知解恢复、Stage 组合及重复 prepare/reset。

## 历史：此前二项简化原型（已拒绝）

状态：原型已实现，当前头部实验未通过质量验收，不默认接入 UI 或重建链条。

参考 Kyriakou 等，Empirical beam hardening correction (EBHC) for CT，2010，
DOI: https://doi.org/10.1118/1.3477088 。两基函数构造参见
https://archive.rsna.org/2010/9008439.html 。本实现使用二值骨路径和自定义稳健平坦度目标，
不是原论文全部分割、权重与目标函数的逐项复现。

## 数学与输入约定

输入 p 为已经完成基础校正、用于初次重建的负对数投影。本次头部实验的基础校正是散射校正加水 BHC。
对初次重建分割得到的骨掩膜作确定性中心射线重投影，得到路径 L（mm）。
令 q=L/100 mm，生成 q²、p q 两组基投影，使用与基线完全相同的线性 WFBP/FDK 重建为 B1、B2。

拟合 f=f0+a B1+b B2，投影校正为 p'=p+a q²+b p q。系数包含正负号。
路径尺度、分割、扫描几何、基础校正以及重建滤波设置必须与拟合时一致。
这是逐扫描系数，不可直接当作跨扫描的设备标定表。
无需能谱积分、材料衰减系数或 PMMA/Al 模体。不重复施加散射、噪声、随机射线。

不对校正投影截断，否则不再与基图线性组合等价。带 TV、非负约束等非线性重建不能直接套用该等价关系。

## 插件与估计器分离

- `makeEmpiricalBhcStage` 返回现有预处理 `IStage`。契约为设备上的 LogAttenuation → LogAttenuation，原位 CUDA 校正。
- `IBoneThicknessProvider` 为 2DCalBH 和 EBHC 共用的路径供应接口，可重投影或读取缓存。目标缓冲区来自显存池，默认流。
- Stage 只负责应用参数，不持有重建器，不自动拟合、分割或加载参考图像。
- `fitEmpiricalBhc` 只接受基线与基图的差分样本，使用两参数 Huber IRLS 和弱对角正则，拒绝退化基函数。
- 元信息 ID 为 `preprocess.empirical_bhc`，状态为 Experimental。

帧级插件自身不可能检查跨体积的质量或判断基础校正是否已应用。调用方必须确保输入与拟合一致、只应用一次。

## 可复现实验入口

显式构建目标：`multispectrum_ebhc`、`multispectrum_fit_ebhc`。

```text
multispectrum_ebhc basis config.toml segmentation.raw baseline-log.raw basis-directory
multispectrum_sim square/reconstruct.toml
multispectrum_sim cross/reconstruct.toml
multispectrum_fit_ebhc experiment.json
multispectrum_ebhc apply config.toml segmentation.raw baseline-log.raw output-directory coefficients.json
```

两个重建 TOML 从基线的重建配置复制，分别将输入改成 basis-directory/square.raw、cross.raw，关闭前处理，保持几何与滤波不变。
分割为重建网格上的 uint8：0 空气、1 非骨组织、2 骨。辅助工具目前支持 flat CBCT、flat/cyl helical。
RAW 为本机小端 float32，体积顺序 z/y/x，投影顺序 view/v/u。

experiment.json 示例（路径建议绝对路径）：

```json
{
  "dimensions_xyz": [512, 512, 400],
  "water_mu_mm_inv": 0.019285194762051105,
  "baseline": "G:/experiment/baseline/reconstruction.raw",
  "square": "G:/experiment/square/reconstruction.raw",
  "cross": "G:/experiment/cross/reconstruction.raw",
  "segmentation": "G:/experiment/segmented-u8.raw",
  "output_directory": "G:/experiment/fitted"
}
```

实验估计器要求三个维度可被 4 整除。4×4×4 块平均只用于系数估计，最终组合原始全分辨率体积。
块内必须全部为非骨组织，再腐蚀一个块，排除骨与空气边界；HU 范围 -100..180。
使用相隔两个块的三轴差分，基线差分超过 30 HU 的样本不参与；Huber 阈值 10 HU。
按空间块划分训练和留出区域。没有加载单能参考或原始材料真值。

输出包含候选 reconstruction.raw 和 coefficients.json。候选仍可能错误，不代表通过验收。
实验性漂移检查要求：留出差分 RMS 不升高、非骨组织平均修正绝对值不超过 5 HU、骨区域平均修正绝对值不超过 50 HU。
这些阈值不是普适物理标准，通过也不保证准确。正式 `apply` 拒绝缺少 accepted=true 的参数；
只有明确的 `apply-candidate` 诊断模式才允许应用失败候选。
底层 Stage 接口接收显式参数，质量验收由上层调用者负责。

## 2026-10-14 头部实验结论：拒绝该候选

沿用 2500 帧、5 周螺旋、SID709/SDD1143、1024×256@0.6 mm、512×512×400@0.5 mm、5 mAs 的已有完整物理过程仿真。
基线为散射＋水 BHC，重建为固定 RamLak WFBP。
结果根目录：`G:/Code/fanproj/fdk-test/out/brainweb-helical-wfbp-bone-singlepass-mas5-20261013/ebhc`。

拟合 a=0.3793932420，b=-0.1146422729，基函数相关系数 0.7488。
训练 540353 个差分，留出 138768 个；留出差分 RMS 9.151→8.993 HU。
但分割区域平均修正为非骨组织 -6.13 HU、骨 -95.87 HU，漂移检查拒绝。

独立单能参考仅用于全 400 层验收：

| 指标（HU） | 基线 | EBHC 候选 |
|---|---:|---:|
| 全体积非骨低频误差 RMS | 6.82 | 10.95 |
| 颅底非骨低频误差 RMS | 8.95 | 11.09 |
| 骨间 ROI 暗误差 RMS | 8.87 | 13.27 |
| 骨内部误差 RMS | 51.31 | 156.12 |

指标使用已有评价脚本的固定掩膜与 ROI；2 mm 低通只作用在误差分析，不滤波最终结果。
低频指标不代表所有条纹都来自束硬化。现有噪声、采样与几何伪影仍会影响拟合。

独立验证：CUDA 校正投影再 WFBP 与直接组合基图，全 104857600 体素差异 RMS 0.001094 HU，最大 0.008693 HU。
因此目前失败的主要证据指向拟合目标/材料表达，而非应用公式与重建线性组合实现。
不能据此否定 EBHC 方法本身，也不能将该候选作为有效骨硬化结果。
后续需研究更合适的材料分割、组织边界权重以及保灰度约束，并继续用独立参考验收。

本机耗时：生成两组基投影含 IO 24.6 s（当前分别计算两次路径），两次 WFBP 各约 24 s，拟合与全体积组合约 5.3 s。
基投影输出约 5.24 GB；三份全分辨率 float 体积合计约 1.17 GiB 主存。
基图缓存不需全部驻显存，投影设备批缓存限制为 256 MiB，另有分割与单帧路径缓冲。

## 验证

`multispectrum_ebhc_test` 覆盖独立合成系数恢复、离群点鲁棒性、退化输入拒绝、CPU/CUDA 公式一致性、无骨射线、非法路径和重复 prepare/reset。
`multispectrum_plugin_metadata_test` 检查插件查询元信息。


## 补充失败归因：2026-10-14

不能将本次结果表述为“EBHC 算法不适用”。当前实现是二值骨路径、两个二次基函数和自定义组织差分目标的简化原型。

1. 当前目标只优化腐蚀后非骨组织的差分，骨区域完全不参与目标；系数的弱正则也不是灰度保持约束。
   因此优化可以牺牲骨值以换取很小的组织平坦度收益。事后拒绝阈值不会改变优化方向。
   在独立评价用的骨内部掩膜每轴隔点抽样上，平方项贡献均值 +115.27 HU，交叉项 -229.40 HU，净变化 -114.14 HU。
   这与此前完整分割骨区域 -95.87 HU 使用不同掩膜，不应混用。
2. 为分离表达能力与参数估计，另外用已有物理模型校正的差值作目标，诊断性求解两个系数。
   这是使用已知目标的 oracle 分析，不是可部署自动算法，也不是严格的理想上界。
   对非骨与骨内部均衡加权得到 a=0.0577845、b=0.00474178。
   全 400 层评价：骨间暗误差 8.87→7.48 HU，骨内部 RMS 51.31→40.80 HU。
   说明两个基函数有一定校正能力，但自动估计选错了方向；同时表达能力也有限。
   若仅拟合非骨区，物理模型修正量的拟合残差 RMS 可从 6.25 降至 2.64 HU，
   但骨内预测平均修正变为 +164.68 HU，而目标仅约 +20.71 HU，说明两区域存在明显折中。
3. 骨掩膜只保留 0/1 路径，丢失重建骨区衰减/部分容积变化；是否为主要影响需独立实验，尚未证实。
   完整物理过程的噪声、散射残差、采样误差也可能影响自动目标，尚未通过逐项消融定量分离。
4. 现有 0.0011 HU 线性一致性检查只验证 Stage 公式与 WFBP 基图组合，不证明骨分割或正投几何绝对正确。

诊断产物在 ebhc/diagnosis.json、ebhc/oracle-evaluation.json、ebhc/oracle-model-fit/。
未用 oracle 系数替换正式拟合，也没有更改默认链条。
后续优先补全原论文具体目标和材料表达，做无噪声/无散射的水骨模体验证，再返回全物理头部数据。
ABHC 的相关研究明确讨论了梯度目标降低高衰减材料对比的问题，并加入灰度变化惩罚；
这是后续方法的参考，不能声称原 EBHC 使用完全相同的目标：
https://pmc.ncbi.nlm.nih.gov/articles/PMC8095714/ 。
