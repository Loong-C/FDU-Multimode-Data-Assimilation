# 第一阶段 · 羽毛球比赛数据

主实验选用 2019 年福州中国公开赛男单决赛，桃田贤斗对周天成。已下载这场比赛公开的全部 8 段录像，从中选出 3 个能看到发球和死球的完整回合。三段均为底线后方主机位，1280 × 720、25 fps，共 55.56 秒。球心来自 TrackNetV3 所用的人工标注底库，逐拍类型与球员来自同场 ShuttleSet 专家标注。[原片与数据包](https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw) · [ShuttleSet 同场标注](https://github.com/wywyWang/CoachAI-Projects/tree/45517f7d4cb936b03f3eabf939cc7959d39226fe/ShuttleSet/set/Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals)

## 实验样本

文件名的三组数字是局号和**得分后**的比分，桃田在前。转播记分牌在回合进行中显示得分前的比分。下表时长包含发球、死球及所保留的片头片尾。表中的“无记录”专指原 CSV 缺行。

| 回合 | 可用帧数 | 时长 | 可见球心 | 标为不可见 | 无记录 | 击球 | 原片及叠加结果 |
|---|---:|---:|---:|---:|---:|---:|---|
| 第一局 3–2 | 358 | 14.32 s | 306 | 52 | 0 | 16 | [比赛](../artifacts/stage1/1_03_02/rally.mp4) · [标注回放](../artifacts/stage1/1_03_02/overlay.mp4) |
| 第二局 2–5 | 475 | 19.00 s | 370 | 97 | 8 | 18 | [比赛](../artifacts/stage1/2_02_05/rally.mp4) · [标注回放](../artifacts/stage1/2_02_05/overlay.mp4) |
| 第三局 18–17 | 556 | 22.24 s | 443 | 113 | 0 | 24 | [比赛](../artifacts/stage1/3_18_17/rally.mp4) · [标注回放](../artifacts/stage1/3_18_17/overlay.mp4) |
| **合计** | **1,389** | **55.56 s** | **1,119** | **262** | **8** | **58** | [数据清单和原文件 SHA-256](../artifacts/stage1/manifest.json) |

第二局 2–5 的原片第 475 帧已经切到近景，导出区间为 [0, 475)。另一个候选回合第二局 14–17 虽然主机位稳定，开头的发球触球难以确认，因此只用于来源预览。8 段原片均无音轨。[8 段原片的切镜和缺帧记录](sources/tracknetv3.md)

### 六段单次击球

每段从本拍触球的估计帧开始，到对方下一次触球的估计帧之前结束。起止两端均结合相邻画面、挥拍动作和球心运动复核过。25 fps 下每帧相隔 40 ms；球拍遮挡和运动模糊仍会留下约一两帧的判读差异。

| 球路 | 回合、拍序 | 击球者 | 本地帧区间 | 短片 |
|---|---|---|---|---|
| 高远球 | 1_03_02，第 7 拍 | 桃田贤斗 | [121, 146) | [25 帧](../artifacts/stage1/1_03_02/shots/07_clear.mp4) |
| 杀球 | 1_03_02，第 8 拍 | 周天成 | [146, 157) | [11 帧](../artifacts/stage1/1_03_02/shots/08_smash.mp4) |
| 发短球 | 2_02_05，第 1 拍 | 桃田贤斗 | [23, 39) | [16 帧](../artifacts/stage1/2_02_05/shots/01_short_service.mp4) |
| 挑球 | 2_02_05，第 4 拍 | 周天成 | [79, 119) | [40 帧](../artifacts/stage1/2_02_05/shots/04_lob.mp4) |
| 网前放小球 | 3_18_17，第 11 拍 | 周天成 | [220, 244) | [24 帧](../artifacts/stage1/3_18_17/shots/11_net_shot.mp4) |
| 吊球 | 3_18_17，第 14 拍 | 桃田贤斗 | [302, 325) | [23 帧](../artifacts/stage1/3_18_17/shots/14_drop.mp4) |

## 来源覆盖

| 来源 | 本地取得的比赛数据 | 几何、人物和击球信息 |
|---|---|---|
| [BFMD](sources/bfmd.md) | 2025 年四项赛事的官方男单短片，720p30。马来西亚站另取得整场标注，并与首回合短片对齐。 | 有场角、网柱、球员框和姿态、人工击球时刻和类型。首回合 160 帧有 143 个模型球心、5 次人工击球。 |
| [MonoTrack](sources/monotrack.md) | 计划链接返回 404。通过论文作者仓库核对其实际使用的 TrackNetV2 底库，获得戴资颖对马林的 3 段女单回合，720p30。 | 底库有 1,795 个人工球心，157 帧不可见。作者增强的逐拍文件未取得，事件表仅含表头。 |
| [TrackNetV3](sources/tracknetv3.md) | 同一底库的另一场比赛，本次获取福州决赛全部 8 个公开回合，720p25。 | 原始逐帧 CSV 提供人工球心和可见性。 |
| [ShuttleSet](https://github.com/wywyWang/CoachAI-Projects/tree/45517f7d4cb936b03f3eabf939cc7959d39226fe/ShuttleSet) | 补充来源。下载同场三局的逐拍表、比赛索引和球场四角表。 | 提供整场击球帧、球种、击球者、击球双方位置及球场四角。球员位置是逐拍快照。 |

BFMD 限非商业学术研究，并限制再分发。MonoTrack 作者代码带 Adobe Research License，TrackNetV3 的 MIT 许可覆盖代码及权重。TrackNetV2 底库的发布页未列出比赛视频的再分发许可。ShuttleSet 仓库根目录附 MIT 许可。视频、原始标注和生成的画面保存在本地，Git 保存获取脚本、对齐配置与验收记录。[BFMD 使用条件](https://ning-d.github.io/BFMD-Dataset/) · [ShuttleSet 许可](https://github.com/wywyWang/CoachAI-Projects/blob/45517f7d4cb936b03f3eabf939cc7959d39226fe/LICENSE)

## 时间和空间对齐

按比赛、局号、双方得分后的比分匹配 ShuttleSet 回合，再用多次连续回球的间隔和球心运动方向转折定位整场帧轴的偏移。三个回合的本地第 0 帧分别约对应 ShuttleSet 整场第 13925、51261、131231 帧。配置保留逐拍复核得到的 16 个局部调整值，其余 42 拍使用该回合的固定偏移。事件表同时保存原始专家帧、按偏移得到的帧、实际采用的帧和定位方法。例如第一局第 7 拍，触球附近恰有数帧球心缺失，依据球员挥拍及球再次出现的位置将击球时间记在第 121 帧。[对齐配置](../configs/fuzhou2019.json)

画面采用左上角像素坐标，向右、向下为正。青色四角和连线取自 ShuttleSet 同场的双打外边线与底线交点；紫色十字表示本项目在第二局 2–5 第 0 帧目视定位的网柱顶点 (368, 323)、(907, 322)。已逐段比对场线、柱帽与标记。黄色圆环及尾迹表示人工球心，遇到缺失就断开。画面顶部显示当前帧和最近一次击球的类型、球员与事件帧。[第 8 拍杀球的叠加样帧](../artifacts/stage1/alignment.jpg)

额外做了球心错帧实验。选用前后均有观测、两帧间总位移至少 16 px 的球心，每隔 20 帧估计中值背景。在相同点集上比较球心周围 5 × 5 像素相对背景的最大增亮，表中为中位数（灰度级）。三段视频均在零偏移处出现明显峰值，支持原始 CSV 与视频帧的配对。这个检验度量整段的配对质量，无法给出单点像素误差或球拍真实接触时刻。

| 回合 | 参与球心数 | 视频提前 1 帧 | 同帧 | 视频推后 1 帧 |
|---|---:|---:|---:|---:|
| 1_03_02 | 173 | 3 | 119 | 5 |
| 2_02_05 | 176 | 3 | 84 | 4 |
| 3_18_17 | 244 | 2 | 103.5 | 4 |

独立验收程序比对原始 CSV、事件映射、四角和网柱的叠加位置，并逐帧解码全部 12 个输出视频。核查包括分辨率、帧数、每帧时间戳及其与原片图像的对应关系，另检查六段短片的起止帧。[本次验收记录](../results/stage1-validation.json)

## 统一文件格式

跨数据集合并时，使用 `match_id, rally_id, frame_index` 作为帧的联合主键。仅使用得分命名的回合号会把不同比赛中同名的回合混在一起。

- `frames.csv` 每帧一行。`frame_index` 为本地零基帧，`timestamp_s` 为相对短片起点的秒数。`source_frame_index` 保留上游帧号，TrackNet 底库用回合轴，BFMD 用整场轴。`x_px,y_px` 表示原图球心。缺失时坐标为空，`is_missing=1`，`missing_reason` 区分人工标为不可见、缺少标签行或无有效预测。
- `confidence` 留空表示上游未报告概率。`observation_origin` 标记 `human_annotation` 或 `model_prediction`，`observation_source` 指向来源。此次未对缺失球心做插值；以后加入补点时应使用单独来源值，例如 `algorithmic_completion`。帧表用 `shot_index` 关联本回合事件表。
- `events.csv` 每次击球一行，含帧、秒、击球方、原始球种、英文球种、事件来源和时间定位方法。福州样本中 A 为桃田，B 为周天成，双方位置列沿用 ShuttleSet 的原始画面像素。BFMD 保留人工类型，击球方留空；MonoTrack 底库的清单将事件覆盖标记为不可用。
- `shots.csv` 记录导出短片、所属回合内的拍序及左闭右开帧区间。`court.json` 记录原图尺寸、几何点及来源。`manifest.json` 关联比赛、回合、帧率、帧轴及缺失统计。

## 复现

以下命令在仓库根目录运行，实测环境为 Python 3.12.9。视频样帧由 OpenCV 直接解码。单拍切片使用 `imageio-ffmpeg` 自带的 FFmpeg。

```powershell
python -m pip install -r requirements-stage1.txt
python scripts/fetch_bfmd.py
python scripts/fetch_monotrack.py
python scripts/fetch_tracknetv3.py
python scripts/fetch_shuttleset.py
python scripts/prepare_stage1.py
python scripts/prepare_supplements.py
python scripts/validate_stage1.py
```

首次获取从发布方下载样本，已缓存文件按 SHA-256 校验。主实验在 `artifacts/stage1/`，补充样本的标准化 CSV 在 `artifacts/stage1_supplements/`，三个数据集的原画面拼图在 `artifacts/previews/`。验证成功时 `artifacts/stage1/validation.json` 的 `passed` 为 `true`。
