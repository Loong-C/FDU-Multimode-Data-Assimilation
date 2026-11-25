# 第六阶段 · 真实比赛中的单次击球重建

2019 年福州公开赛男单决赛的三个回合提供了 9 次单拍自由飞行，涵盖高远球、挑球、吊球、杀球、网前球和发短球。每拍从击球帧开始，到对手下一次触球之前结束。输出包括各原始帧的三维位置、速度、像素投影，以及这些投影在比赛录像中的回放。[原片与标签来源](stage1.md) · [逐拍清单和人工复核记录](../results/stage6/shots.json) · [比赛叠加回放](../artifacts/stage6/reprojection.mp4)

## 样本和相机

第一局 3–2 的两拍用于选择阻力模型。第二局 2–5 和第三局 18–17 共七拍用作最终测试。分组以完整回合为单位，同一拍的起点为帧偏移 0，偏移为 2、6、10……的可见球心留作同机位检验。所有选拍和帧掩码在拟合前确定。每拍回看两端的挥拍动作及每个可见球心的局部原图，逐拍记录球拖影、球网或衣服遮挡以及触拍时刻的估读范围。无法辨认球心的帧保留缺测。[选拍配置](../configs/stage6-fuzhou.json) · [数据准备](../badminton/real_shots.py)

| 样本 | 击球数 | 原片帧数 | 拟合球心 | 留出球心 | 无球心 |
|---|---:|---:|---:|---:|---:|
| 开发，两拍来自第一局 | 2 | 36 | 25 | 9 | 2 |
| 测试，两段独立回合 | 7 | 182 | 106 | 36 | 40 |
| 合计 | 9 | 218 | 131 | 45 | 42 |

三段录像均为 1280 × 720、25 fps。逐段重新测量场线交点和球网标记，各使用 13 个控制点拟合相机，并用 11 个保留点检查标定。三个回合的保留点重投影 RMSE 依次为 1.79、1.83、1.80 px。第二局视频最后一帧切走，已排除。球员脚下的像素与地面平面的交点提供触球水平位置的软约束，两个坐标的标准差各设为 1 米，以容许跨步、起跳和球拍伸展。[相机参数](../results/stage6/cameras.json) · [独立场地点和镜头稳定性](../results/stage6/camera-checks.json)

## 福州七个测试球的结果

在开发集上，固定阻力和逐球估计阻力的逐拍平均留出 RMSE 分别为 2.04 和 2.52 px，因此用固定的二次阻力系数 0.217 m⁻¹ 完成全部测试。二维误差按原画幅中预测与人工球心的欧氏距离计算。七拍的 36 个留出球心合并后 RMSE 为 **4.54 px**，逐拍 RMSE 中位数为 **2.28 px**。已将七拍的全部留出球心删除后重新拟合，得到相同初态与阻力，并逐帧对照原录像、独立数值积分与 OpenCV 投影。[模型与分组](../results/stage6/study.json) · [各拍估计值](../results/stage6/fits.csv) · [独立验收](../results/stage6/validation.json)

| 局与拍号 | 球种 | 留出球心 | 留出 RMSE（px） | 三维物理检查 |
|---|---|---:|---:|---|
| 第二局 #1 | 发短球 | 4 | 1.36 | 通过 |
| 第二局 #4 | 挑球 | 6 | 6.36 | 离拍位置在对手半场，轨迹没有过网 |
| 第三局 #11 | 网前球 | 6 | 2.23 | 通过 |
| 第三局 #13 | 挑球 | 5 | 8.78 | 离拍点贴地且在对手半场，轨迹没有过网 |
| 第三局 #14 | 吊球 | 6 | 2.28 | 过网处低于网沿约 0.47 m，接球点贴地 |
| 第三局 #17 | 高远球 | 6 | 1.61 | 通过 |
| 第三局 #18 | 杀球 | 3 | 3.11 | 过网处低于网沿约 0.08 m，接球点贴地 |

物理检查要求球从击球人一侧飞到接球人一侧，在两根网柱之间越过网沿，并将任一端触球高度低于 0.15 m 或高于 4 m 的解标出。开发集两拍和测试集三拍通过。通过说明重建与这些约束一致，真实福州录像中的三维深度仍缺少同步侧视角直接测量。回放用黄色线和十字表示重投影，绿圈表示参与拟合的人工球心，紫圈表示同机位留出球心。42 个缺测帧仅显示模型预测。三维图以圆点标出离拍，以橙色标出物理检查失败的球路。[九拍三维轨迹图](../results/stage6/real-trajectories.png) · [逐帧三维位置与速度](../results/stage6/trajectories.csv) · [比赛原帧与局部放大](../artifacts/stage6/reprojection.jpg)

## 误差来自哪里

每拍用主模型估计的初态生成 5 份模拟观测，保留相机、采样时间及观测掩码，重新添加每坐标 2.5 px 噪声与 1 m 球员位置误差。35 份测试模拟的留出像素合并 RMSE 为 **3.59 px**，福州实测为 **4.54 px**。在这个生成模型成立的对照实验中，三维位置 RMSE 中位数为 0.224 m。真实比赛还包含运动拖影、标注偏差、时刻误差和模型失配，单凭这组数据不能将误差差额归到某一个因素。[配对模拟明细](../results/stage6/matched-simulation.csv)

下表在同一拍上一次只改变一个条件，每次都重新拟合。数值是七个测试球在共有原始帧上的轨迹相对主结果的三维 RMS 位移，再取逐拍中位数。它衡量模型对输入的敏感性。[逐次重拟合](../results/stage6/sensitivity.csv)

| 改动 | 三维轨迹变化 |
|---|---:|
| 标定训练点各坐标加入 1 px 噪声，重新标定 | 0.031 m |
| 每三个内部拟合球心去掉一个 | 0.124 m |
| 拟合球心各坐标加入 2.5 px 噪声，三组独立扰动 | 0.18–0.32 m |
| 逐球联合估计阻力 | 0.373 m |
| 去掉两端运动员位置的软约束 | 0.396 m |
| 击球时刻提前或推迟一帧（40 ms），保持下一触球时刻 | 0.445 / 0.516 m |
| 保留重力，去掉空气阻力 | 5.20 m |

两记挑球分别缺少 19/40、15/37 帧球心，大段缺测集中在高弧段。吊球的像素 RMSE 仅 2.28 px，却比网沿低约半米。对这几拍，留出帧可以检验同一画面中的预测，过网高度、离拍速度和实际三维深度仍需其他角度约束。

作为阻力敏感性实验，允许逐拍估计阻力后，两记挑球都能越网，留出误差分别降到 4.50 和 3.99 px，但推得的 k 分别为 0.143 和 0.066 m⁻¹。第三局这一拍的三维路径随之改变了 2.52 m。深度和阻力可以相互补偿，表中的正式测试始终沿用开发集选定的固定 k。

在主模型下分别用四条链检查估计区间，9 拍中有 7 拍通过收敛诊断。第三局 #17 高远球的估计初速为 77.70 m/s，条件 90% 后验区间为 64.07–114.09 m/s。这些区间假设相机、击球帧与 k 已知。第一局 #7 与第三局 #13 的链未达到诊断门槛，区间留空。[各拍区间与收敛指标](../results/stage6/posterior.csv)

## 在真实第二机位上检验

阳明交大（NYCU）公开的羽球轨迹数据包含同步的左右原始录像、各自的逐帧球心与相机标定。从带 Vicon 文件的子集预先选取 001、002、003 三段不同的实验室飞行。沿用福州开发集选定的固定阻力 MAP，仅用左机位估计初态，未提供运动员脚点。使用作者发布的同步帧号和相机参数将同一条轨迹投到右机位，在原始 1440 × 1080 畸变画面上与右侧人工球心比较。三段均以作者的剪辑起点为时间零，实际拍摄为 120 fps。[作者数据与说明](https://drive.google.com/drive/folders/1KaPYvxVkpDieZUxfACg5bTdc8senjDDC) · [作者项目](https://gitlab.nol.cs.nycu.edu.tw/gxxygxxy123/trajectory-modeling/-/blob/0fa1da548cd0c41feb9007d7196119c363770199/README.md) · [评测脚本](../scripts/nycu_stereo6.py)

| 实拍编号 | 右机位有效帧 | 左机位拟合 RMSE（px） | 独立右机位 RMSE（px） | 与双目参考的三维 RMSE（m） |
|---|---:|---:|---:|---:|
| 001 | 98 | 1.38 | 18.86 | 0.211 |
| 002 | 88 | 1.02 | 13.16 | 0.181 |
| 003 | 84 | 1.14 | 17.32 | 0.182 |
| **合并全部帧** | **270** | **1.20** | **16.70** | **0.193** |

270 个右机位球心全部纳入评分，预测均位于画幅内。两台相机相距 1.15 m，射线夹角约 10°–19°，各帧均可计算三角参考。假设球心各坐标误差为 2.5 px、相对时刻标准差为 2.41 ms，一阶传播给出的逐段参考位置 RMS 不确定度为 0.059–0.063 m。这是给定标定和噪声假设的误差预算，未包含标定系统偏差。三角参考和重建共用了左侧球心。独立证据是封存到拟合完成后的右侧球心。[实拍结果与误差分解](../results/stage6/nycu-stereo.json) · [双视角原帧](../artifacts/stage6/nycu/independent-view.jpg) · [双视角四分之一速回放](../artifacts/stage6/nycu/stereo-replay.mp4)

独立验收逐帧解码两侧 540 张原图，检查视频时间戳和标注帧号，以 DOP853 重算动力学，再用 OpenCV 原生畸变模型复算右图误差。独立复算也得到 16.70 px。用齐次三角测量替换两条视线的中点法时，三段参考位置之间的 RMS 差分别为 3.75、1.09、1.60 mm。[实拍独立验收记录](../results/stage6/nycu-validation.json)

这三次短距离实验飞行可以直接检验换视角后的误差。其动作、深度范围和相机布置与福州转播不同。作者用轨迹的空间近邻裁剪并对齐 Vicon 时间原点，本次未找到独立的相机至 Vicon 对时记录，因此只报告同一时间轴的双机比较。自录比赛场景的 A/B 验收仍待录像。[作者的 Vicon 比对脚本](https://gitlab.nol.cs.nycu.edu.tw/gxxygxxy123/trajectory-modeling/-/blob/0fa1da548cd0c41feb9007d7196119c363770199/an_task/3d_camera.py)

异步设备另用有三维真值的模拟检验。A 为 25 fps，B 为 59.94 fps，通过三个独立同步事件拟合时钟、另外两个检查时钟，检查 RMSE 为 0.78 ms。固定阻力 MAP 在 B 上的 RMSE 为 2.40 px。基线三角测量有 64 帧可用，相对生成真值的三维 RMSE 为 0.028 m；其他 18 帧缺少临近的 A 观测，仍计入 B 的二维评分。单独改变 B 的标签位置后，A 估计的初态与阻力保持一致。[模拟与独立性测试](../results/stage6/stereo-validation.json)

## 复现

从仓库根目录运行。已有第一阶段三个福州回合及相应标注时，可以从 `badminton.real_shots` 开始。原片、包含发布方球心坐标的逐帧表和带原片的图像保存在本地，Git 保存生成程序、各拍输入的来源、估计值和验收记录。

```powershell
python -m pip install -r requirements-stage3.txt
python scripts/fetch_tracknetv3.py
python scripts/fetch_shuttleset.py
python scripts/prepare_stage1.py
python -m badminton.real_shots
python -m badminton.real_study
python -m badminton.render_real
python scripts/validate_stage6.py
python scripts/validate_stereo6.py
python scripts/fetch_nycu6.py
python scripts/nycu_stereo6.py
python -m badminton.render_nycu
python scripts/validate_nycu6.py
```

要检查原始标注，可执行 `python -m badminton.real_shots --review-dir artifacts/stage6/review/source-crops`，每拍各输出一张逐帧球心放大图和一张前后两端的触拍连拍图。[本地核对图](../artifacts/stage6/review/source-crops)

## 双机位实拍的接入方法

1. 固定两台相机，从不同角度同时拍到同一块场地与球路。保留原始分辨率和逐帧时间戳，每个镜头标明场地同一端和同一侧。记录两机都可见的同步事件，例如开拍和结束各一次闪光，中间再留一次独立闪光作检查。可变帧率视频使用显示时间戳，球心缺测时保留原来的时间轴。
2. 在两幅原图上分别标注场线交点及已知高度的网柱顶点，另留独立地面点与高处点验收。可复制第三阶段的标定配置，填写各自的图像尺寸、帧区间、非共面控制点及其他帧的独立检查，再分别执行 `python -m badminton.calibrate --config capture/landmarks-a.json --out capture/a` 和 B 相机对应命令。两次输出的 `camera.json` 使用同一个米制场地坐标系。[标定配置示例](../configs/stage3-fuzhou.json)
3. 每个机位各写一个 `t_s,u_px,v_px` 球心 CSV，保留该相机的原始时钟，缺测球心留空。在 `sync.csv` 写入 `t_a_s,t_b_s,split`，用 `fit` 事件估计相对时间偏移和时钟漂移，用 `check` 事件衡量对时误差。每拍在 A 时钟上给出离拍和下次触拍或落地时刻。按完整拍摄回合划分开发与测试，并在查看测试 B 球心前确定模型、标定和时间参数。
4. 执行 `python -m badminton.stereo --config capture/shot.json --out artifacts/stage6/stereo`。输入文件路径相对于该配置文件。结果中的 `stereo.json` 汇总 B 机位评分、拟合误差、同步质量、有效样本数与剔除原因，`stereo-points.csv` 给出每个 B 时间戳的预测和可用的三角参考。幅外、提前触地、弱视差或两帧间隔过长均需结合样本数解读。[双机位入口](../badminton/stereo.py)

以下配置选用与福州正式测试一致的固定阻力 MAP 模型。也支持联合估计阻力的 `joint_drag` 和第四阶段的 `fixed_drag_baseline`。两个 MAP 模型均接受 A 侧独立取得的可选脚点 `player_xy_m`、`receiver_xy_m`。时间、噪声尺度和文件名需替换为当次拍摄的值。同步标准差应结合独立同步事件残差和标注时间分辨率估计。对于已有可信公共时钟和逐帧对应关系的录像，可以指定 `clock_source=published_shared` 并省略 `anchors`。

```json
{
  "camera_a": "a/camera.json",
  "camera_b": "b/camera.json",
  "observations_a": "shuttle-a.csv",
  "observations_b": "shuttle-b.csv",
  "model": "fixed_drag_map",
  "anchors": "sync.csv",
  "launch_a_s": 8.292,
  "stop_a_s": 9.772,
  "sigma_a_px": 2.5,
  "sigma_b_px": 2.5,
  "sync_sigma_s": 0.005,
  "max_gap_a_s": 0.065
}
```
