# IMPTC 外部运动预测协议

实验使用德国阿沙芬堡一个路口的四段 IMPTC 作者演示样例。核对原生时钟后共锁定 564 个同人不重叠的窗口，四段分别含 39、159、163 和 203 窗。预先指定四份 PINNS 种子 42 的纯运动模型，各自输出最高概率轨迹。用同一个模型、同一个窗口的真值计算误差，再按模型和片段取均值。主对照从最后两次轨迹记录按真实时间差外推，首选模型与分母在打开误差前确定。该试验覆盖一个外部路口，资料中的目标是经过跟踪器估计的位置。官方公开材料没有单视角人体框与原始图像的可靠对应，因此本实验只检验运动预测。[协议](external_protocol.md) · [样本与时间锁定记录](external_motion_lock.json) · [作者公开数据](https://zenodo.org/records/14811016) · [作者对原图的说明](https://github.com/kav-institute/imptc-dataset/issues/2#issuecomment-2533717069)

数据使用 CC BY-NC 4.0 许可，作者包括 Manuel Hetzel、Hannes Reichert、Günther Reitberger、Konrad Doll 和 Bernhard Sick，详见数据记录的完整署名及版本。这里发布评测代码和聚合校验。另用源数据的正式测试划分评估时，需要独立协议。[原始研究和数据说明](https://arxiv.org/abs/2307.06165)
