# BFMD 数据核查

四段官网男子单打示例均可下载、逐帧解码，马来西亚站样本另有同场可下载的球、球员、球场、球网和击球标注。执行 `python scripts/fetch_bfmd.py` 可获取本次核查的原文件、验证 SHA-256 并生成 `artifacts/previews/bfmd.jpg`。仅查看预览可加 `--preview-only`。依赖 Python、NumPy 和 OpenCV。作者限定非商业学术研究，并禁止再分发。本项目在本地保存数据和预览，比赛画面版权归 BWF TV。[项目与使用条件](https://ning-d.github.io/BFMD-Dataset/) · [数据包](https://drive.google.com/drive/folders/1wQr4DpMbx-e8jFnvOH9WUJXxfXgTC1IJ)

## 取样与数据规模

四个样本都是作者发布的比赛画面，保留了电视转播记分牌。逐帧解码所得分辨率均为 1280×720，帧率均为 30 fps。下表帧号从短片第 0 帧起算。[官网样本](https://ning-d.github.io/BFMD-Dataset/#samples)

| 2025 赛事 | 运动员 | 文件 | 解码帧数 | 预览帧 |
|---|---|---|---:|---:|
| 印尼公开赛决赛 | Anders Antonsen / Chou Tien Chen | demo_kapal.mp4 | 638 | 286 |
| 马来西亚公开赛半决赛 | Kodai Naraoka / Anders Antonsen | demo_petronas.mp4 | 160 | 45 |
| 中国公开赛半决赛 | Chou Tien Chen / Shi Yu Qi | demo_victor.mp4 | 96 | 45 |
| 全英公开赛半决赛 | Lee Chia Hao / Alex Lanier | demo_yonex.mp4 | 256 | 120 |

整场录像需通过作者 CSV 指向的 BWF TV 视频另行获取。本环境尝试读取马来西亚站整场视频时 YouTube 要求登录。上述官网短片可独立获取。[BWF TV 原场次](https://www.youtube.com/watch?v=Bxhh21Rsk0E)

论文统计 12 场男单和 7 场男双，共 1,687 个回合和 16,751 次击球。本地核查的 `Badminton_video_list.csv` 有 19 个视频链接，`match_index.csv` 只含 12 场男单，汇总行为 1,054 个回合。仓库 README 的发布包介绍写作 1,058 个回合，因此统计单打索引时以实际文件为准。男双的公开标注目录以通用击球、触网、落地、回合边界为主，不能假设有全部单打模态。[论文表 1](https://arxiv.org/html/2603.25533v1) · [仓库文档](https://github.com/Ning-D/BFMD/tree/c7c7e8e4c815085dda42746f16e07f1172503312)

## 马来西亚男单半决赛

数据目录为 `data/raw/bfmd/malaysia/`。本地文件名取自发布包目录名，原文件名含 `PETRONAS-Malaysia-Open-2025-Kodai-Naraok-JPN-8-vs.-Anders-Antonsen-DEN-2-SF`。球场、球网、人物和羽毛球坐标均以画面左上为原点，向右、向下为正。时间轴从整场解码第 0 帧开始。[发布格式说明](https://github.com/Ning-D/BFMD/tree/c7c7e8e4c815085dda42746f16e07f1172503312) · [论文方法](https://arxiv.org/html/2603.25533v1)

| 文件 | 本地检查所得内容与来源 |
|---|---|
| metadata.json | 本场顶层为对象，有 70 个回合。游戏开始、结束和击球时刻来自人工时间轴；比分、胜者等衍生字段可能缺失。 |
| court.json | 静态主机位四个球场角点及两根网柱的顶点和底点，均为像素。文件没有说明每个点是如何取得的。 |
| court_perframe.json | 每秒一次的算法球场检测，含四角、拟合得分和质量标志。4,135 个采样中有 2,112 个 valid，本场余下均为 no_court。 |
| shuttle.json | TrackNet 模型结果，36,076 条框记录。x、y 为框左上，单位为图像宽高的百分比，中心像素为 ((x + w/2)W/100, (y + h/2)H/100)。导出文件注明已转为零基帧，无需再减一。无逐帧概率；enabled 是启用标志。 |
| player_bbox.json | 上下两位球员的百分比框轨迹，含插值；论文描述检测和跟踪流程为 YOLOX 与 OC-SORT。导出项 origin=manual 不能证明所有帧均由人工绘框。 |
| pose.json | 70 个回合的模型姿态。键为原视频帧号，框为像素，17 个关键点分别给出归一化 x、y 和关键点分数。 |
| shot_type.json | Label Studio 人工时间轴，包含单帧击球事件、经人工核验的类型、触网、落地和回合区间。单次击球可同时有通用 hit 和具体类型，应按帧关联，避免重复计数。 |
| hit_inferred.json | 1,158 次击球，由时间轴与球员位置派生；球员侧别与姓名是推断结果。 |
| caption.json | 1,158 个击球记录带文字描述。论文说明采用模型起草、人工核验的流程。 |

第一局首回合起止标签为 10317、10477。官网短片有 160 帧，映射为整场区间 [10317, 10477)，记分牌显示 0–0。人工击球帧相对短片起点依次为 7、30、73、109、133，对应发球、挑球、高远球、吊球和下压。第 136 帧有单独的触网标注。160 帧中有 143 帧可读取启用的模型球心，17 帧缺失。

对齐核查还利用了球体相对背景的亮度变化。每隔四帧估计中值背景，去掉片头片尾各十帧，140 个有效预测点周围 9×9 像素取增亮最大的五点均值。视频与标注同步时该值的中位数为 81.2 灰度级，提前一帧和推后一帧分别为 5.0 和 25.2。这个配合回合长度的检查支持上述帧偏移，单个模型预测点仍可能有误差。
