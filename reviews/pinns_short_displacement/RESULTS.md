# 五帧累计路长不足半米的首标注窗口

研究者在读取本次分组误差前登记了阈值、主组和模型比较。所有模型训练和预测均来自此前的四地点留出实验，原队列的整体结果在登记前已经公开。[登记方案](PROTOCOL.md)和[冻结的分组人数、窗数和地点](history_support.json)记录了这一顺序。

按每窗五次实测地面位置之间的四段长度之和分组，1,482 个窗中有 323 个同时满足首次采样标注和累计路长严格小于 0.5 m，涉及 46 个片段。美国、意大利、日本和泰国分别为 146/14、24/8、102/11、51/13 窗/片段。其余三组分别为首次标注且路长至少半米的 126 窗、已有连续至少 35 个前序采样且路长不足半米的 798 窗，以及已有前序采样且路长至少半米的 235 窗。没有窗口恰好位于阈值。首次标注仅说明跟踪记录从此开始。

每窗五个种子等权，先求各地点窗口均值，再对四地等权。主组 M0、M1 和末两点匀速的 ADE/FDE 依次为 0.41302/0.93416、0.43880/0.98040 和 0.35448/0.84907 m。M1 减 M0 的 ADE 为 +25.78 mm（95% 区间 +16.43 至 +35.62），FDE 为 +46.24 mm（+28.05 至 +67.02）。在四个地点内分别配对重采样整片段 20,000 次。五个种子的 ADE 差分别为 +39.18、+3.49、+86.56、−9.25、+8.92 mm。推断条件为这四个已研究地点、本组片段和这五份种子结果。[生产入口的聚合结果](short_displacement_scores.json)及[独立数值复算](independent.json)含四组及逐地点、逐种子的结果。

## 复核材料

原 Kaggle 版本 373583293 已执行网络前向并保存每窗的六候选首选、匀速及事后最优候选误差。本目录离线聚合其中原网络首选与匀速误差，未运行模型。原返回包的 SHA-256 为 `b24ac2bb1ad3d6742be2cbe16b927e42ef9a1a54a17ce2c08be3e92fa286b871`，成对误差成员为 `72c9fd39724dc50aefe2b4b766a84a8b70c236e6965c575dc3bf1937b786361e`。[Kaggle 原预测版本](https://www.kaggle.com/code/pengjy25/shortpose-pinns-history-based-mode-ranking?scriptVersionId=373583293)

复核需私有输入 `pinns_matched_histories.npz`、原 `reranked_modes.zip` 及 PINNS 的 `labels/` 与标定目录。历史缓存 SHA-256 为 `caa8f0959f28f2d69a6f16cc9df115ab6222d0b6eb3297edd575615fef81b94c`。在项目根目录运行下列命令，将输入替换为对应本地路径。第一条运行正式汇总入口，第二条用独立的路长、原始时间戳和片段频数矩阵复算。

```bash
python -m scripts.evaluate_pinns_short_displacement METADATA compiled.json --reranked-bundle reranked_modes.zip
python reviews/pinns_short_displacement/verify_aggregation.py pinns_matched_histories.npz reranked_modes.zip METADATA replay.json --reference compiled.json
```

独立检查覆盖四组身份摘要、每组窗数和涉及的片段、地点与种子均值、各地点原始全样本结果及配对区间。检查 370 项数值，最大绝对差为 4.22×10⁻¹⁵。独立记录还按地点和种子列出各臂均值。两个 JSON 均只含汇总；逐人标识、逐窗记录、路径和权重保留在私有输入中。
