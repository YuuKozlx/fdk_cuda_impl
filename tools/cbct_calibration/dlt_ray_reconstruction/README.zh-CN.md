# 基于 DLT 投影矩阵的射线驱动 FP/BP 测试

入口是 `test_yang_dlt_fp_bp.py`，顶部 `CONFIG` 可直接在 IDE 中修改。

## 计算内容

1. 从 Yang 报告读取每帧 `3x4` 投影矩阵 `P_i`。
2. 从 `P_i` 求相机中心 `C_i`：

   ```text
   P_i = [M_i | p_i]
   C_i = -M_i^-1 p_i
   ```

3. 对每个探测器采样点 `(u,v)` 求射线方向：

   ```text
   d = normalize(M_i^-1 [u,v,1]^T)
   ```

4. 用 Siddon 计算射线穿过每个体素的长度 `l_ri`，形成稀疏矩阵 `A`。
5. 正投影：`g = A @ f`。
6. 反投影：`f_bp = A.T @ g`。
7. 用 LSQR 解小型验证问题 `min ||A f - g||`。

这里不是体素中心点投影。每条射线对体素的贡献是实际穿过长度，因此 BP 是 FP 的严格稀疏转置。

## 当前演示规模

默认使用 36 个稳定视角、48x48 探测器采样、32x32x32 体素。这样是为了快速验证几何和算子闭环，不代表最终临床重建分辨率。输出包括：

- `forward_projection.npy`
- `backprojection.npy`
- `reconstruction.npy`
- `phantom_slice.png`
- `backprojection_slice.png`
- `reconstruction_slice.png`
- `fp_bp_report.json`

`adjoint_relative_error` 应接近机器精度，说明 FP/BP 使用了同一组射线路径长度。

