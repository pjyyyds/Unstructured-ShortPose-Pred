# ETH University 开发片段的匀速基线

在固定机位的 ETH University 片段（`seq_eth`）中，用每个窗口开头两个相隔 0.4 秒的地面位置估计速度，外推后续七步。**343 名行人贡献 828 个不重叠窗口，ADE 为 0.475251 米，2.8 秒处的 FDE 为 0.850817 米。** ADE 在每窗七个预测时刻取平均，FDE 只取第七步，各窗口等权。[汇总结果](eth_development_cv.json)

ETH University 是 PINNS 四地点之外的一段开发录制。本项只评分末两点匀速模型，使用原生 2.5 Hz 轨迹。PINNS 使用 10 Hz 五帧观测和 3.0 秒预测，两项均覆盖 0.4 秒观测跨度，采样密度和预测终点不同。

## 数据和计算

数据来自 ETH Zurich Computer Vision Lab 发布的 BIWI Walking Pedestrians 人工轨迹。原数据论文为 Pellegrini 等人的 *You’ll Never Walk Alone: Modeling Social Behavior for Multi-target Tracking*（ICCV 2009）。原包总说明将 `obsmat` 的第 3、5 列定义为地面水平位置，单位米。`seq_eth/info.txt` 给出 2.5 Hz 的标注频率。[ETH 原发布页](https://vision.ee.ethz.ch/datsets.html) · [官方标注包](https://ethz.ch/content/dam/ethz/special-interest/itet/cvl/vision-dam/documents/ewap_dataset_light.tgz) · [原论文](https://doi.org/10.1109/ICCV.2009.5459260)

8,908 条有效标注覆盖 360 人，同一人相邻标注的原始帧号差均为 6。按行人 ID 和时间排序，从每段连续轨迹开头依次取互不重叠的九点块。舍去合计 1,456 个尾点，最终评分 5,796 个未来位置。另用 NumPy 批量计算逐窗预测，两项汇总差异均小于 10⁻¹² 米。

从仓库根目录运行以下命令即可用 Python 3 标准库重建同目录 JSON。

```sh
python3 reviews/eth_development/score_eth_development_cv.py
```

脚本在内存校验官方压缩包的 SHA-256，仅读取总 README、ETH University 的序列说明及轨迹文件。公开归档保留复现代码、文件摘要和汇总误差，Hotel 留出片段保持封存。
