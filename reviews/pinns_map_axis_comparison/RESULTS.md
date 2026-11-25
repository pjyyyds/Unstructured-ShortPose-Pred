# PINNS 鸟瞰地图坐标纵轴修复配对复算结果

- **云端版本与输入锁**：Kaggle `w1-cpu1-pinns-bev_axis`（Version 2，`pinns_map_axis_comparison_results.zip` SHA-256 `fb46281856ef7beab9ffc2e771a90fd41d96e3bc5b5f63b2a830c3590e007853`），输入包 `pinns_map_axis_inputs.bin` 指纹核验一致，旧版预测重算最大绝对差 `0.0 m`。
- **测试集全窗聚合指标（497 窗，21 段录像）**：
  - 五点线性外推 `ols_five`：`0.5430 m` ADE / `1.1692 m` 末有效步 FDE（233 窗固定 3.0 s FDE `1.3980 m`）
  - 纯运动岭回归 `motion`：`0.5842 m` ADE / `1.2336 m` 末有效步 FDE（固定 3.0 s FDE `1.2672 m`）
  - 仅来源可用标志 `availability`：旧版 `0.6002 m` $\to$ 修复版 `0.6001 m` ADE
  - **因果图像轮廓几何 `image_geometry`**：旧反转纵轴 `0.6234 m` $\to$ **修复正向纵轴 `0.6048 m` ADE**（改善 **`-0.0186 m`**；末有效步 FDE 从 `1.3267 m` 降至 `1.2816 m`）
  - 历史行人占用凸包 `occupancy_geometry`：`0.6098 m` $\to$ `0.6097 m` ADE
- **结论**：修复 `PINNS` 鸟瞰像素向地面纵轴映射符号并严格因果截断后，静态图像轮廓相对仅可用标志的额外误差从 `+0.0232 m` 收窄至 **`+0.0047 m`**（消除约 80% 反转伪退化），证明历史版本中静态轮廓的显著负效应主要源于纵轴反转缺陷。
