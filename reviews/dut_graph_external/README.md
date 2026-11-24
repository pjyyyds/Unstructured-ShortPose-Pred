# 大连校园人车邻居图外部验证

## 真实历史输入与实验状态

固定一处校园共享空间 11 片，共 3,066 个目标窗、879 个片内行人身份。图模型、个体参照、空图训练参照各有四套源权重。全部 12 套权重可严格加载。当前实际核对到历史数据、建图、模型结构和合成输入输出。大连预测误差须待完整姿态缓存可用及 Kaggle 完成前向。

作者人车轨迹在 10 m 范围产生 84,334 条候选人边和 1,373 条候选车边。最多12邻居的规则留下34,028条人边和415条车边。原图中2窗为空、2,672窗仅有人、1窗仅有车、391窗人车兼有。2,578窗触发截断，仅4条保留人边没有前一测量；有效保留边的最近间隔均为3个 23.976 Hz 原图帧，约0.125125s。该图使用作者跟踪记录。速度与距离单位为 m/s、m。

若保留最近11位行人并以最近车辆换掉第12位，810个原本看不到半径内车辆的窗改变，车辆入槽从392窗升至1,202窗。受影响者仅在01至05五个片段，分别为17、30、618、64、81窗，后6片为0。原图与干预边的身份及类型均已用两套读取方法对齐。车辆轨迹来源没有改成自动检测，姿态仍从单独的人体检测与姿态缓存得到。

## 私有 Kaggle 运行

开启 Internet，模型前向使用 CPU。将原始输出 pinns_graph_control.zip（SHA-256 前缀 0b3e590b）和完整 dut_shared_space_pose_cache.zip 挂载到 /kaggle/input，或置于当前 /kaggle/working。只运行 kaggle_cell.py 一个单元即可。图工具从作者固定 GitHub 提交取得 DUT 轨迹与比例尺，逐文件验 SHA，不上传源图或 CSV。完整姿态缓存由已冻结的 DUT 检测管线另建；11窗试点包无法满足历史身份校验。缓存来源、每窗帧号、关联状态、全窗分母都会校验。若复用同名已完成输出，输入缓存或代码不一致时会明确报错。

返回包 /kaggle/working/dut_shared_space_graph_transfer.zip 含 aggregate.json、所有五个坐标条件×四源模型的私有六候选、模式概率、误差和哈希。共120套模型输入条件加原轴4套换槽诊断。原轴首选 ADE 的真图减同源 M1 是主分析。角度仅指地面轨迹的坐标表示，另外报告纵轴镜像。该摄像机没有物理转向真值。私有逐窗文件应留在私有工作区。

## 源数据与重放

从项目固定 git 提交 032800e94bb0828c45d3f5195ac5cc17aa93bc4c 使用 src/data、src/models，在 PYTHONPATH 中指向该 checkout。DUT 数据根目录为作者仓库的 data 所在层。命令在本目录运行。前两条允许本地执行；真实前向和未来评分仅在 Kaggle 执行。

```sh
python test_transfer.py
python check_ready.py DUT_DATA_ROOT pinns_graph_control.zip
python verify_return.py dut_shared_space_graph_transfer.zip dut_shared_space_pose_cache.zip pinns_graph_control.zip
```

verify_return.py 默认核对整个返回归档、所有 SHA、来源权重及姿态缓存、逐窗身份和保留车辆范围。仅在 Kaggle 添加 --replay-on-kaggle，才会以存储的六条轨迹和模式概率重新计算首选 ADE/FDE、四源与11片汇总，以及原轴配对区间；没有新拟合。PROTOCOL.md 和两份 ADDENDUM 分别固定主试验、坐标手性和车辆槽位干预。historical_denominators.json、independent_history.json、checked_sources.json 提供独立分母与身份指纹。
