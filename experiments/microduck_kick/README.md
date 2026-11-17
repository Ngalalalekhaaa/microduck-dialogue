# HD1910 固定球踢击仿真

在当前本地 HD1910 CPU MuJoCo + BAM 配置中，测试官方左右脚踢球 ONNX，并切回已训练的 `hd1910_walk_1849.onnx` 零速度站立。仅执行仿真，不调用实体机器人接口。

## 已完成的结果

正式结果在 `outputs/grid01/summary.json`。左右脚各测试 9 个球位：相对官方球心位置，在前后和左右方向各取 `−1.5、0、+1.5 cm`。每个球位还有一份不踢、只站立的对照。

| 项目 | 左脚 | 右脚 |
|---|---:|---:|
| 踢中并恢复站立 | 9/9 | 9/9 |
| 站立对照通过 | 9/9 | 9/9 |
| 全程最大躯干倾角 | 6.50° | 9.01° |
| 踢球窗口及延迟结束时，球前进距离 | 37.1–54.7 cm | 41.6–51.3 cm |
| 原始关节目标超出 XML 范围的最大量 | 1.281 rad | 0.574 rad |

所有正式案例准备阶段均没有机器人碰球，也没有水平球位漂移。球只在场景 reset 时设置一次，整个准备、踢击和恢复过程都不再改变球的位置。接触记录确认踢击由指定脚造成，恢复阶段没有继续推球。

**结论范围：当前名义 HD1910 仿真中可以执行这两个踢球动作。** 这些是确定性网格案例，不是实机成功率，也没有验证不同电压、地面、负载或动作延迟。几何仍为本地训练资产，电机采用现有公开 HD1910 M6 参数，不是完整实测标定的数字孪生。

原始动作目标存在明显超限，仿真没有裁剪或平滑这些目标；实际关节受仿真关节约束，测得的最大软限位越界约为左侧 0.039 rad、右侧 0.011 rad。目标角与实际关节角不能混为一谈。现有结果不能证明这些原始目标能直接用于实体舵机，也不能把裁剪目标后仍然有效视作已验证。

## 查看录像

- 左脚：`outputs/grid01/left_kick.mp4`，关键帧 `left_kick.jpg`。
- 右脚：`outputs/grid01/right_kick.mp4`，关键帧 `right_kick.jpg`。

视频来自正式矩阵中的 `case_008_left` 和 `case_026_right`。左侧是脚部近景，右侧固定远景覆盖球的完整运动。视频回放保存的物理状态，不重新运行控制或摆球。

每次测试是 2 秒站立准备、0.5 秒官方踢球策略、3 秒原步态零速度恢复。25 fps 视频约 5.52 秒。球直径 7 cm、质量 15 g，采用上游球体碰撞和摩擦参数；恢复期间会继续滚动，录像中的数米滚动距离不代表真实场地射程。

## 复现

在当前工作区执行：

```bash
cd /home/luckysir/Strands_Robots
bash experiments/microduck_kick/run.sh --controls
```

重复正式矩阵：

```bash
bash experiments/microduck_kick/run.sh --grid --controls
```

输出目录自动生成，不会覆盖旧结果。也可通过 `--out` 指定新的目录；`--foot left` 或 `--foot right` 选择一侧。退出码 0 表示全部所选踢球与对照通过，2 表示至少一例未通过。

模型已经下载到仓库的 `.cache/microduck_kick_research_20261114/`，运行前会核对官方左右脚权重 SHA256。这个实验不联网安装、不修改权重、不改动用户训练环境。依赖原有 Python 环境，可用 `MICRODUCK_ROOT`、`MICRODUCK_PYTHON` 覆盖位置。

对一次新运行生成录像（把参数替换成该运行的实际目录）：

```bash
env -u PYTHONPATH /home/luckysir/microduck/microduck_rl/.venv/bin/python \
  experiments/microduck_kick/recording.py \
  --scene /实际输出目录/scene.xml \
  --case /实际输出目录/case_000_left \
  --out /实际输出目录/left_kick.mp4
```

## 执行与验收

- 官方模型来源：`pollen-robotics/microduck-policies`，固定版本 `236d1835b03ac7b45d962efbf2fe3a6940206112`。`manifest.json` 指定每个踢球技能窗口 0.5 秒，原始适用电机为 XL330。
- 底层策略 50 Hz，物理 200 Hz，HD1910 BAM 固定 20 ms 延迟；两次模型切换都保留上一帧原始动作与控制器延迟队列。
- 踢球模型输入 61 维，13 维命令全部为零，输出 14 维关节动作。它不接收球位置、颜色、图像或语言。
- 球心初始位置为机器人起始坐标前方 9 cm、左/右侧 4.2 cm，再叠加网格偏移。不是自主接近测试。
- 每 5 ms 检查机器人与球的有效接触力；地面接触不当作脚部踢击。
- 成功需要准备干净、指定脚在踢击及实际延迟窗口接触球、窗口内球前进至少 3 cm、最终前进至少 10 cm，无其他身体/后续阶段推球，无跌倒，且最后 0.5 秒站稳。
- 站稳标准为倾角小于 12°、躯干高度大于 9 cm、水平速度小于 0.05 m/s、角速度模小于 0.4 rad/s。
- 站立对照要求全程无机器人碰球，球最大水平漂移小于 1 cm。

`summary.json` 保存逐案例结果、权重及源码哈希、MuJoCo/ONNXRuntime 版本和物理参数；`contacts.json` 保存接触证据。`trajectory.npz` 包含 50 Hz 完整机器人与球状态、策略阶段、原始动作和关节目标，便于后续审计或回放。

19 个评测测试覆盖侧推误判、恢复阶段推球、准备阶段污染、错脚接触、动态延迟窗口和恢复未完成等反例：

```bash
env -u PYTHONPATH /home/luckysir/microduck/microduck_rl/.venv/bin/python \
  -m pytest -c /dev/null -p no:cacheprovider -q experiments/microduck_kick
```

下一阶段可将这两个动作作为离散技能，接上多色球检测、目标绑定、脚前对齐和踢后检查。现有导航相机有脚前近场盲区，且导航目标点是前方 20 cm；不能直接把导航的“到达”判定当作踢球触发条件。
