# Microduck 视觉闭环仿真

这份样例实现：相机看到红色地面标记 → 估计它相对机器人的位置 → 选择一个短移动动作 → 站稳并重新观察 → 到位后保持站立。

使用你本地的 `hd1910_walk_1849.onnx`、HD1910 BAM 电机模型与 Strands `Policy` 接口。当前决策器是可检查的规则程序，**没有调用 VLM，也没有实现布料抓取、活动嘴部或真机控制**。物理循环独立于 Strands 内置 `Robot.run`。

**直接运行**

在终端执行：

```bash
cd /home/luckysir/Strands_Robots
bash experiments/microduck_vision/run.sh --target 0.4 0.1 --video
```

这表示把红色标记放在初始坐标系前方 40 厘米、左侧 10 厘米。机器人调整到“标记位于自身前方 20 厘米、横向居中”的位置。脚本不需要桌面窗口，运行结束会打印输出目录；用视频播放器打开目录中的 `demo.mp4`。

脚本默认复用 `/home/luckysir/microduck/microduck_rl/.venv/bin/python`。不安装或修改训练环境，不修改模型权重。录像需要已有的 `ffmpeg`。移动了本地工作区时，可以设置：

```bash
MICRODUCK_ROOT=/你的/microduck目录 \
MICRODUCK_PYTHON=/你的/已有环境/bin/python \
bash experiments/microduck_vision/run.sh --target 0.4 -0.1 --video
```

环境需能导入 `numpy`、`mujoco`、`onnxruntime`、`bam`、`PIL`，且本地工作区保留 `deploy_hd1910`、`microduck_rl_hd1910` 和 `scripts/render_hd1910_controller.py`。

**建议依次操作**

1. 运行上面的默认案例，看视频中左侧的外部视角和右侧的机器人相机视角。底部会显示正在执行的技能。
2. 改变目标位置，例如 `--target 0.35 -0.12`。第二个坐标为正表示左侧，为负表示右侧。首次建议在已测试范围内尝试：前方 0.17～0.55 米、左右 ±0.12 米。
3. 跑完整位置矩阵，并查看成功数、最终误差和逐次决策记录：

```bash
bash experiments/microduck_vision/run.sh --batch 20
```

4. 验证看不到目标时不会继续移动：

```bash
bash experiments/microduck_vision/run.sh --hide-target
```

隐藏目标案例正常结果为 `target_lost`、`actions: 0`，这是预期的负例通过，不是导航成功。

默认最多执行 30 个移动动作、运行 60 秒仿真时间。可用 `--max-actions 1` 或 `--max-seconds 3` 检查退出边界。返回码 0 表示本次检查通过，2 表示导航或负例检查失败，3 表示仿真检查通过但录像失败。

可用 `--out /新的/目录` 指定结果位置；已有目录会被拒绝，避免覆盖。默认输出在本目录的 `outputs/` 下，每次创建新目录。

**输出文件**

| 文件 | 内容 |
|---|---|
| `summary.json` | 全部案例结果、模型哈希、仿真参数、相机设置、代码哈希 |
| `case_00/result.json` | 到位状态、真实位置误差、动作次数和视觉定位误差 |
| `case_00/events.jsonl` | 每次视觉观测、所选动作、停止确认、仿真时间 |
| `case_00/camera_*.png` | 决策时实际使用的相机图像 |
| `case_00/camera_*_depth.npz` | 与图像同一时刻的光轴深度，单位米 |
| `case_00/trajectory.npz` | 50 Hz 关节轨迹、动作标签和倾角 |
| `demo.mp4`、`demo.jpg` | 加 `--video` 后生成的原速录像和关键帧拼图 |
| `first_person_start.png`、`first_person_end.png` | 录像对应的机载相机起止画面 |

**闭环如何工作**

- `vision.py` 从 RGB 图像提取最大的红色连通区域，用深度和相机标定反投影成机器人坐标。它不读取目标的仿真坐标。
- `planner.py` 根据前后、左右误差选择 `move_forward`、`move_backward`、`move_left`、`move_right` 或 `stand`。
- `runtime.py` 把移动动作映射为 ±0.2 m/s 指令，驱动训练好的 ONNX 策略。动作脉冲之后继续执行 1 秒零速度站立控制，再观察。
- `policy.py` 用 Strands `Policy` 接口构建 61 维观测，输出 14 个关节目标。策略 50 Hz，物理 200 Hz，保持原 BAM 控制，不加动作裁剪或平滑。
- `demo.py` 要求连续两次视觉观测到位且机身基本静止，再宣布完成。目标真实坐标只用于场景布置和独立验收。

默认验收要求目标在机器人自身坐标系中满足 `|x−0.20|≤0.035 m`、`|y|≤0.035 m`。视觉决策采用更紧的 ±0.025 m 停止范围，预留 1 厘米定位余量。这不是指定全局朝向的任务，也不是嘴尖已对准抓取点的证明。

相机是虚拟 RGB-D 相机，安装在躯干坐标 `[0.14, 0, 0.16] m`，向前下方俯视，垂直视场角 85°。它使用理想深度和标定，**不等同于实际 IMX219 单目相机**。接单目相机时需要另做标定和距离估计；不能直接把图像像素当成厘米。

仿真按仿真时间锁步运行；视觉与规划计算时物理时间暂停。它不是真机实时调度器。未来接较慢的 VLM 或真机时，要把底层平衡控制与模型推理分开调度。

**接 VLM 的位置**

保留执行层，新增实现 `Planner` 接口的适配器：

```python
def decide(self, observation, *, rgb=None, instruction=None) -> SkillCommand:
    # 把图像、任务、已测得的位置交给你的 VLM。
    # 解析成 SkillCommand；未知技能和非法时长由 validate_command 拒绝。
    ...
```

`demo.py` 已把真实相机 RGB 和固定任务文本传入 `decide`；当前 `MarkerPlanner` 只使用位置观测，不理解文本。接入 VLM 后也应一次返回一个短动作，然后重新看图。先用这份规则基线验证运动和视觉，再比较 VLM 决策的成功率。

**本地验证**

```bash
env -u PYTHONPATH /home/luckysir/microduck/microduck_rl/.venv/bin/python \
  -m pytest -c /dev/null -p no:cacheprovider -q experiments/microduck_vision
```

这会检查相机几何、目标丢失、离散命令约束、控制脉冲和停止行为。完整物理验收用 `--batch 20`。

已执行的 20 个固定位置案例全部通过，目标隐藏案例没有产生移动指令。该结果仅代表本地名义参数、无噪声、无遮挡、单个红色标记场景；没有覆盖真实舵机误差、负载、多目标或布料形变。继续做叼取时，应先标定嘴部可达位置，再接张嘴、弯腰、合嘴与抓取成功检测。
