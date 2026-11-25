# 第三阶段 · 摄像机标定与轨迹投影

2019 年福州中国公开赛男单决赛第二局 2–5 回合，主机位画幅为 1280 × 720。用第 0 帧的 13 个关键点估计摄像机，在同帧另外 11 个关键点上测得重投影均方根误差 **1.82 像素**。这组参数覆盖第 0 至 474 帧，共 19.00 秒。原片第 475 帧切到近景。[原片来源](stage1.md) · [原图与投影对照](../artifacts/stage3/reprojection.jpg) · [完整投影回放](../artifacts/stage3/reprojection.mp4)

## 真实机位

世界坐标以网下场地中心为原点，长度单位为米。近侧底线在负 x 方向，画面左侧边线在正 y 方向，z 轴竖直向上。像素原点在左上，u 向右，v 向下。标定假定方形像素、零斜切，主点固定在 (639.5, 359.5)。先用 EPnP 初始化姿态，再最小化关键点的二维像素残差，同时估计焦距、旋转和平移。针孔解的焦距约为 3048 像素，摄像机中心约为 (−33.12, 0.06, 10.24) 米。[摄像机参数与投影矩阵](../results/stage3/camera.json) · [OpenCV 投影模型](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

在拟合前固定点位与分组。地面交点先在原始帧上目视选定邻域，再分别拟合两条白线的中心并取交点。两根网柱的顶端和两个网面下角在放大的原图上标注。拟合使用 11 个地面点与 2 个柱顶。验收使用其余 9 个地面点与 2 个网面下角。下角高度按柱顶 1.55 米减去标准网深 0.76 米取 0.79 米，实际安装高度未经现场测量。地面模型也用名义场地尺寸与宽 4 厘米的实际漆线中心对齐。[标注、分组与画面定位](../configs/stage3-fuzhou.json) · [球场尺寸依据](stage2.md)

| 第 0 帧验收项 | 点数 | 针孔模型 | 增加一阶径向畸变 |
|---|---:|---:|---:|
| 拟合点像素 RMSE | 13 | 1.20 px | 0.89 px |
| 全部留出点像素 RMSE | 11 | **1.82 px** | 1.66 px |
| 地面留出点像素 RMSE | 9 | 1.31 px | 0.90 px |
| 网面下角像素 RMSE | 2 | 3.24 px | 3.38 px |
| 地面留出点反投影平面距离 RMSE | 9 | 3.11 cm | 2.08 cm |

像素 RMSE 以每个点的二维欧氏距离计算。针孔模型留出点的中位误差为 1.02 px，最大误差为 3.26 px。径向模型在归一化相机坐标上拟合出 k₁ = 1.278，改善主要来自地面点，两个网面下角的误差略增。本阶段导出针孔模型供后续使用。表中厘米误差来自像素射线与已知地面 z = 0 的交点，空中球的三维精度尚待带深度真值的数据检验。[逐点残差](../results/stage3/reprojection.csv) · [两种模型的指标](../results/stage3/calibration.json)

用 8 个固定场地标志逐帧比对原始镜头。主机位的 475 帧均匹配全部 8 个标志，近景帧匹配数为零。另在第 55、228、430、474 帧分别重新量取 6 个地面交点，直接复用第 0 帧的摄像机，各帧 RMSE 依次为 **1.12、1.18、1.02、1.04 px**。这些点用于检查同一回合内的机位稳定性。[镜头检查与重复量点记录](../results/stage3/view-check.json)

导出后重新解码原片和叠加回放，将每帧与原片相邻 ±2 帧比较，475 帧均与同编号原帧最接近。首末显示时间为 0.00 和 18.96 秒，相邻时间戳间隔 40 毫秒。[19 秒重投影回放](../artifacts/stage3/reprojection.mp4)

## 稳定性与正向生成

只使用地面交点时，一般三维投影矩阵会出现多解。合成对照中的两台摄像机在地面点上投影重合，离地后最大分歧达到 64.14 px。加入两根柱顶后，实测训练点的世界坐标仿射秩由 2 增至 3，归一化 DLT 系统中 s₁/s₁₁ 为 14.48，s₁₁/s₁₂ 为 103.71。[合成对照](../results/stage3/validation.json) · [实测奇异值](../results/stage3/calibration.json)

给每个训练点的 u、v 独立加入标准差 1 px 的高斯噪声，固定种子 301，重复拟合 50 次。相对原解，摄像机中心移动量的第 95 百分位为 0.47 米。对分布于场地的 9 个水平位置，0、1.55、3、6 米高度的投影位移第 95 百分位依次为 1.31、1.35、1.55、2.30 px。这些值衡量标注扰动的影响，真实机位的焦距和位置缺少独立测量。[扰动实验](../results/stage3/calibration.json)

第二阶段的 5 条模拟球路共生成 644 个带时刻的像素坐标，在选定的虚拟机位中全部入画，与 OpenCV 参考投影的最大偏差小于 4 × 10⁻¹³ px。也可以将同一正向生成器换成比赛机位。下方命令的高远球示例覆盖 2.18 秒，221 个模拟采样中有 99 个在比赛画面内。CSV 保留时间、世界位置、速度、像素、相机深度和是否入画，可直接按时间列与视频对齐。[虚拟机位图](../results/stage3/virtual-projection.png) · [五条球路的逐点数据](../results/stage3/virtual-trajectories.csv)

## 复现

在仓库根目录运行下列命令。前三项实验使用已提交的标注和第二阶段的初始状态。

```powershell
python -m pip install -r requirements-stage3.txt
python -m badminton.forward
python scripts/validate_stage3.py
python -m badminton.calibrate
```

以下步骤核对原片并导出重投影回放。已有第一阶段原片时可直接从机位检查命令开始。比赛画面的 JPG 与 MP4 写入本地 `artifacts/stage3/`。

```powershell
python scripts/fetch_tracknetv3.py
python scripts/inspect_stage3_view.py --out results/stage3/view-check.json
python -m badminton.render_calibration
```

给定初始位置（米）、速度（米/秒）和阻力系数（每米），用比赛摄像机生成二维球路。若要按 25 fps 取样，可设置 `--dt 0.04`，用 CSV 中的 `t_s` 识别额外插入的轨迹顶点或着地时刻。

```powershell
python -m badminton.forward --camera results/stage3/camera.json --position -5.3 -1.7 2.6 --velocity 33 8 25 --k 0.217 --out artifacts/stage3/real-camera-clear
```

给定任意有限针孔投影矩阵，可调用 `project_points(P, xyz)` 和 `pixel_rays(P, uv)`。读取比赛参数时，下面的接口返回米制世界坐标中的相机中心和单位方向，射线上的点满足 X = center + λ × direction，其中 λ > 0。采用径向模型时，`camera.rays` 会先校正畸变。[摄像机实现](../badminton/camera.py)

```python
import json
from pathlib import Path
from badminton.camera import Camera

camera = Camera.from_dict(json.loads(Path("results/stage3/camera.json").read_text()))
uv = camera.project([0.0, 3.05, 1.55])
center, direction = camera.rays(uv)
```
