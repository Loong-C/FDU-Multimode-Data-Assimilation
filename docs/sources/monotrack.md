# MonoTrack 样本

计划中的 `kyunghyunlim/MonoTrack` 返回 HTTP 404。论文作者 Jui-Hsien Wang 的仓库是 `jhwang7628/monotrack`。论文第 3 节交代了数据来源，作者在 TrackNetV2 的单打比赛上增加了球场四角和击球者标签。作者的可视化笔记实际读取了底库测试集第 3 场的 `1_03_02`。这里下载同一场的三个原始回合，供复查这项工作的输入画面。[论文](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Liu_MonoTrack_Shuttle_Trajectory_Reconstruction_From_Monocular_Badminton_Video_CVPRW_2022_paper.pdf) · [作者仓库](https://github.com/jhwang7628/monotrack/tree/39e9227bd44fa2a579731b1cc97c7f529f391cd6) · [作者可视化笔记](https://github.com/jhwang7628/monotrack/blob/39e9227bd44fa2a579731b1cc97c7f529f391cd6/notebooks/visualization.ipynb)

## 比赛与文件

戴资颖对卡罗琳·马林，世界巡回赛总决赛女单决赛，于 2021 年 1 月 31 日在曼谷举行。画面从底线后方俯看，两位球员、球场边线和球网均在画面内。视频来自原作者托管在 NYCU 的压缩包 `TrackNetV2/Test/match3/video/`，对应的人工二维标注来自同包的 `csv/`。[TrackNetV2 数据说明及下载](https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw) · [BWF 比赛视频](https://www.youtube.com/watch?v=HU5Ra5XSers)

| 回合文件名 | 回合结束比分 | 实际视频帧数 | 可见坐标帧 | 缺失帧 | 时长 |
| --- | --- | ---: | ---: | ---: | ---: |
| `1_02_00` | 第一局 2∶0 | 707 | 640 | 67 | 23.57 秒 |
| `1_03_02` | 第一局 3∶2 | 466 | 437 | 29 | 15.53 秒 |
| `1_05_02` | 第一局 5∶2 | 779 | 718 | 61 | 25.97 秒 |

三个 MP4 都是 1280 × 720、30 帧/秒、MPEG-4 视频，没有音轨。实际逐帧读取共 1,952 帧，与三个 CSV 的行数逐一相同。每段视频和标注都从帧 0 开始，帧号连续。相对本回合起点，帧 `i` 的时间为 `i / 30` 秒。文件名采用赛事胜者、负者在回合结束后的得分顺序，直播记分牌在回合进行中显示得分前的比分。[底库命名约定](https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw)

## 可以读取的观测

| 内容 | 实际检查结果 |
| --- | --- |
| 视频、二维球心 | 每个回合一对 MP4 和 CSV。表头为 `Frame,Visibility,X,Y`。坐标使用原始视频像素，左上为原点，向右和向下递增。 |
| 缺失及置信度 | `Visibility=0` 时两坐标均为 0，共 157 帧。其余 1,795 帧的像素坐标均落在画面内。没有置信度字段，也未补点。 |
| 球场、网、球员 | 此次取得的底库成员不含相应标注。论文另加四个球场角点，代码可从视频检测球场和估计运动员姿态。本次没有取得这三个回合的作者增强标签或原生网柱关键点。 |
| 击球时刻、击球者、击球类型 | 论文描述了击球时刻及击球者。作者代码引用了本地的逐拍 CSV，公开仓库未附这三个回合对应的逐拍文件，底库 CSV 也没有击球类型。 |
| 使用许可 | 作者仓库的 Adobe Research License 对其许可材料限非商业研究和教学。底库数据说明没有列出单独的数据或转播素材再分发许可。 |

[作者数据处理脚本](https://github.com/jhwang7628/monotrack/blob/39e9227bd44fa2a579731b1cc97c7f529f391cd6/setup/setup.py) · [作者仓库的击球标签询问](https://github.com/jhwang7628/monotrack/issues/4) · [Adobe Research License](https://github.com/jhwang7628/monotrack/blob/39e9227bd44fa2a579731b1cc97c7f529f391cd6/LICENSE)

## 下载与抽帧

在项目根目录运行以下命令。脚本按需下载六个成员，核对固定的 SHA-256，生成 `artifacts/previews/monotrack.jpg`。已有官方压缩包时可加 `--archive 文件路径`。样图按从左到右、从上到下的顺序，取自 `1_02_00` 第 300 帧、`1_03_02` 第 100 和 410 帧、`1_05_02` 第 161 帧，保留了完整画幅。

```powershell
python scripts/fetch_monotrack.py
```
