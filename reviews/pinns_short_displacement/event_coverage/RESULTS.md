# 五帧初始路长不足半米的起步和停止

固定使用 323 个真实窗口、46 段录像、种子 42 的原始六条预测。模型 M0 只看运动，M1 增加二维姿态与质量门控。原模型按概率选取一条可部署的轨迹。六条中任一条包含目标动作，说明模型曾生成该动作。事件标签由五帧运动历史与预测或真实的三秒未来按原规则计算。

9 个真实起步中，两模型的首选均命中 0 个，在 314 个阴性窗中也都没有首选起步。M0 的六候选在 6 个真实起步中含有起步，M1 在 7 个中含有；二者各有 22 个阴性窗出现起步候选。在两个真窗，两模型的全部候选都没有起步。9 个真起步来自 5 个片段，美、意、日、泰分别有 3、2、2、2 窗。

15 个真实停止中，M0、M1 首选分别命中 2、3 个，在 308 个阴性窗中分别误报 2、1 个。两模型的六候选各命中 8 个真实停止，其中 7 个共同命中；阴性窗中分别有 24、15 个出现停止候选。另有 6 个真实停止两模型均未生成。四地真实停止数为 9、1、5、0，共涉及 9 段录像。

两个独立程序核对了 323 条真轨迹和 3,876 条候选轨迹，另从独立的 Kaggle 返回包逐窗核对了 646 个首选索引。五类动作的计数均按四地和合并窗口保存。该组的首次观测指首次采样标注，9 个起步和 15 个停止只能描述这批已有预测。[登记协议](PROTOCOL.md) · [原始聚合](first_short_event_coverage.json) · [独立计算](independent_counts.json)

## 复算

输入目录含 PINNS 原标签 `pinns_metadata/labels/` 及 `pinns_matched_histories.npz`、`anchored_six_candidates.zip`、`reranked_modes.zip`、`shortpose_heldout_five_seeds.zip` 四份原归档。从仓库根目录运行。两份程序均使用 NumPy，按输入文件与预登记协议的 SHA-256 检查来源。

```sh
study=reviews/pinns_short_displacement/event_coverage
python "$study/check_events.py" . INPUTS /tmp/first_short_events.json
python "$study/independent_replay.py" --test
python "$study/independent_replay.py" --input INPUTS --protocol "$study/PROTOCOL.md" --output /tmp/first_short_independent.json
```
