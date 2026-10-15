# 两遍锥束伪影校正（实验性，未通过质量验收）

本工具依据 Hsieh 两遍方法的公开专利流程探索圆轨迹 CBCT 校正，未接入默认 pipeline 或 UI。
参考：https://patents.google.com/patent/US6266388B1/en 。未取得 SPIE 论文完整正文，不声称完整复现。

构建目标：`multispectrum_cone_two_pass`（EXCLUDE_FROM_ALL）。
运行入口：`multispectrum_cone_two_pass project job.json` 与 `multispectrum_cone_two_pass combine job.json`。
项目参数及必填字段见同目录 ConeArtifactTwoPass.cu 的 main；中间骨投影由正式 multispectrum_sim 按相同几何和滤波器重建。

流程：从初始 FDK 按 225 HU 提取骨并保留衰减值 → 确定性 Joseph 重投影（像素 2×2 积分）→ 相同 FDK 重建骨投影 → 从原图扣除 R(A(b))-G(b)。G 为中心点源估计的高斯 PSF，可额外平滑误差。

2026-10-15 头部验证：720 帧、1024×768 探测器、70 keV、512×512×400 重建。全体积非骨 RMSE 83.61→82.77 HU，骨内 RMSE 70.06→56.07 HU；但轴向边缘非骨 RMSE 273.25→280.93 HU，条带仍存在，故未通过质量验收。

主要局限：中心高斯 PSF 不能表示位置相关响应；原仿真 Siddon 50 射线与模型 Joseph 4 射线不同；骨分割和响应幅度有偏差。数值自测（均匀体投影最大误差 4.92e-8、误差扣除符号）通过，不代表图像质量合格。未实现 Han 多遍算法，也不能补全圆轨迹缺失数据。
