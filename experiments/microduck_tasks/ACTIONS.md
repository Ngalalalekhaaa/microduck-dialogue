# Microduck action playground — 2026-11-15

本机 CPU MuJoCo + HD1910 BAM 仿真。新增动作执行层已可运行，旧的冻结 VLM + 11 类选择头尚未扩类。
新动作记录是 `microduck-action-demo-1`，`training_ready=false`，不会混入旧训练数据。

玩具范围更新：默认 `--profile toy` 仅开放自身动作和踢球；轻跳暂排除，默认 showcase 去掉轻跳。
本文保留此前研究成果和测试记录，其他交互只在 `--profile research` 开放。
完整聊天/组合任务的协议与采集设计见 [TOY_DIALOGUE.md](TOY_DIALOGUE.md)，尚未接入这个动作播放器。

## 直接观看

在 `/home/luckysir/Strands_Robots` 执行：

```bash
# 当前自体动作连续演示：观察、点头、坐站、单脚平衡、舞步、前滚翻
./experiments/microduck_tasks/actions.sh --showcase

# 小球从低底座被夹起，保持、前进，再低头放到青色目标垫
./experiments/microduck_tasks/actions.sh --profile research --action pick_place

# 推动紫色方块
./experiments/microduck_tasks/actions.sh --profile research --action push_block

# 全部命令和实验状态
./experiments/microduck_tasks/actions.sh --profile research --list

# 实时中文交互：help 查看命令，停下中断任务，exit 退出
./experiments/microduck_tasks/actions.sh --interactive
```

窗口左侧是跟随鸭子的外部视角，右侧是前视和额外虚拟近场相机。
批量动作完成后窗口保持打开；关闭窗口后写出轨迹。没有桌面环境时加 `--no-viewer`，
加 `--video` 会从真实录制的物理状态生成 MP4。每次默认生成新的输出目录，不覆盖数据。

已经生成的录像：

```bash
xdg-open experiments/microduck_tasks/outputs/action_pick_place_20261115/demo.mp4
xdg-open experiments/microduck_tasks/outputs/action_showcase_final_20261115/demo.mp4
```

指定动作可重复传入 `--action`，全程是同一个连续世界。例如：

```bash
./experiments/microduck_tasks/actions.sh --action sit_stand --action groove --action caught_dancing
./experiments/microduck_tasks/actions.sh --profile research --action pick_up --action carry --action place
./experiments/microduck_tasks/actions.sh --action '踢红球'
```

`--preset open/pickup/push` 只改变初始物体摆位，保留三色球、小球、方块和目标垫。
默认按任务自动选择；研究交互模式默认 `pickup`，玩具交互模式默认 `open`。叼取演示要求初始对准，若先走远或翻滚，
不能再把一次固定动作当作自主寻物，需重新启动 `--profile research --action pick_place`。
`carry/place/release/throw` 都要求已经夹持；检查失败会返回失败原因。

## 实际接入的动作

| 类别 | 动作 ID / 中文命令 | 结果和范围 |
|---|---|---|
| 基本运动 | `stand`、`move_forward/backward/left/right`、`turn_left/right` | 复用原行走策略和航向控制；六种移动/转向各 2/2 |
| 观察 | `look_around` 左右看、`nod` 点头 | 原行走权重的头部指令；分别 2/2，实际关节变化验收 |
| 姿态 | `sit_stand` 坐站、`ground_pick` 俯身、`body_sway` 轻晃身体 | 分别 2/2；俯身本身不等于叼取 |
| 杂技 | `roll` 前滚翻、`flamingo_left/right` 抬左/右脚、`happy_hop` 轻跳 | 分别 2/2；检查倒置/脚部离地及恢复站立，轻跳有前移 |
| 步态 | `moonwalk` 后退滑步 | 2/2；证实后退，不把“月球步风格”当作已量化结论 |
| 舞步 | `groove` 节奏舞步、`caught_dancing` 害羞舞步 | 分别 2/2；后者是打包进 ONNX 的行走策略+指令时间线，不是新训练的整套舞蹈网络 |
| 鸭嘴 | `open_beak/close_beak` 张嘴/闭嘴 | 独立有限力矩舵机，不占用策略的 14 个动作输出 |
| 叼取 | `pick_up/carry/place/release/pick_place` | 固定停靠位夹取、保持、携带和放置；5 个邻近摆位的完整搬运全部通过 |
| 推物 | `push_block` 推方块 | 鸭子与方块接触、方块前移约 18 cm，恢复站立 |
| 踢球 | `kick_left/right`，或“踢红球/踢黄球/踢蓝球” | 前者是脚部动作预览；后者调用已统一的视觉规则选择器并验收目标接触和位移；新场景红球 1/1 |

第一组验收 [38 个任务](outputs/action_validation_20261115/summary.json) 全部通过：16 种动作各两次，
第二次先执行观察动作改变进入状态；5 次叼取搬运；1 次推方块。
[补充舞步测试](outputs/action_dance_trials_20261115/summary.json) 共 6 次、5 次通过，失败来自下表旋转舞步。
固定站姿附近的两次检查不足以证明随机环境鲁棒性；所有失败和原始轨迹均保留。

叼取用直径 **24 mm、质量 3 g** 的自由小球，放在 **14 mm 高底座**上，初始距机身约 65 mm。
五次摆位是中心及前后/左右各 2 mm；这是预对准的物理交互演示，尚未实现自主视觉寻物、停靠和从平地拾取。
中心场景球最高约 21.4 cm，搬运后落点约 `(0.310, 0.017, 0.012) m`，在目标垫内。
夹持检查要求持续一秒离地、靠近嘴部、有嘴部接触且没有外部支撑；释放需真正分离、落地。
物体没有 weld、跟随写位姿、运动中重置、外部托举力或关闭重力。
嘴部夹持垫最大记录穿透约 1.78 mm、合计法向力峰值约 2.11 N；舵机力矩限制 ±0.15 Nm，
目标变化率上限 2 rad/s，后者不等于实际关节速度的硬限制。

[实时窗口检查](outputs/action_live_smoke_20261115/check.json) 和
[实时叼取搬运](outputs/action_realtime_pick_place_20261115/summary.json) 也已通过；
MP4 是录制状态回放，不会重新模拟或改写动作结果。

## 实验项与暂不能直接运行的模型

| 模型 / 动作 | 当前处理 |
|---|---|
| `hop`、`run` | HD1910 初测摔倒；需要 `--experimental` 才能预览 |
| `pirouette` | 两次测试一成功一摔倒；实验项 |
| `headstand` | 只有整套头倒立中的 fold 阶段，缺少其余阶段和接触切换；当前未完成恢复 |
| `official_roll` | 官方 1 秒切换时机在当前电机模型下未恢复；默认使用已测本地 `roll` |
| `official_sit`、`velstand` | 当前指令下站稳但没有完成所命令的坐下/行走，不算成功 |
| `bow` | 当前幅度很小，未通过有意义的低头幅度检查 |
| `pick_throw/throw` | 能释放甩出，但当前横向偏移大，不算定向投掷成功 |
| `roller/roller_crouch` | 已下载；需要轮脚结构及轮滑动力学，不在普通脚场景冒充运行 |
| `basketball` | 已下载；需要篮球支撑场景及 256 维 LSTM 状态，未接入当前地面场景 |

例如查看不稳定的旋转候选：

```bash
./experiments/microduck_tasks/actions.sh --profile research --experimental --action pirouette
```

当前通过的是“策略+显式关节限幅+恢复流程”的仿真组合。原始网络目标角仍可能越界：
两轮测试最大越界量约为坐站 0.255 rad、俯身 0.127 rad、滚翻 1.418 rad、单脚站立 3.094 rad、
轻跳 3.438 rad。日志保留原始输出、裁剪次数及最终目标，不能把这些成绩当作未裁剪网络或真机成绩。
滚翻等动作只在其有限执行阶段放宽姿态条件，普通移动的跌倒检查仍保留。

## 权重与来源

检索记录包含 Hub 返回的 1000 个 Microduck 仓库条目，并用模型卡、manifest 和实际 ONNX 接口筛选。
其中有大量重复训练版本、专用赛道和不同机械结构；没有把仓库名出现过就当作可执行技能。
目前下载 **23 份公开 ONNX，19.07 MiB**，再复用已有的三个本地 HD1910 姿态权重。
实际 URL、修订、SHA-256、大小和许可证见 [action_weights.json](action_weights.json)。

| 来源 | 内容 |
|---|---|
| [Pollen 官方权重](https://huggingface.co/pollen-robotics/microduck-policies) | 行走、站立、velstand、坐站、俯身、左右踢球、滚翻、轮滑、轮滑下蹲；Apache-2.0 |
| [RemiFabre/flamingo-cycle](https://huggingface.co/RemiFabre/microduck-flamingo-cycle) | 左右单脚平衡；Apache-2.0 |
| [joanfox/happy-hop](https://huggingface.co/joanfox/microduck-happy-hop) | 轻跳；Apache-2.0 |
| [witcheer/hop](https://huggingface.co/witcheer/microduck-hop) | 另一份跳跃候选；未声明许可证 |
| [HannesVonEssen/running](https://huggingface.co/HannesVonEssen/microduck-running) | 跑步；Apache-2.0 |
| [HannesVonEssen/basketball](https://huggingface.co/HannesVonEssen/microduck-basketball) | 在篮球上平衡；Apache-2.0 |
| [q2p/beak-throw](https://huggingface.co/q2p/microduck-beak-throw) | 甩球及配套站立，两份权重；Apache-2.0 |
| [fffiloni/polite-bow](https://huggingface.co/fffiloni/microduck-polite-bow-b1d864) / [moonwalk](https://huggingface.co/fffiloni/microduck-moonwalk-backward-55e6af) | 鞠躬、后退滑步；未声明许可证 |
| [ZachGarner/headstand-fold](https://huggingface.co/ZachGarner/microduck-headstand-fold) | 头倒立的 fold 阶段；未声明许可证 |
| [kramp/groove](https://huggingface.co/kramp/microduck-groove) | 节奏屈膝舞；未声明许可证，模型卡说明训练分支未公开 |
| [q2p/dizzy-duck](https://huggingface.co/q2p/microduck-dizzy-duck) | 旋转舞步；未声明许可证 |
| [microduckapp/caught-dancing](https://huggingface.co/microduckapp/microduck-caught-dancing) | ONNX 内置动作时间线；未声明许可证 |

“公开下载”不等于每份权重都具备明确开源许可。权重只放本机 `.cache/microduck_actions`，没有加入包或发布到外部服务。
可用 `./experiments/microduck_tasks/actions.sh --download` 按固定修订和校验和重新下载；不下载大训练 checkpoint。

嘴部碰撞垫尺寸参考 [llama/microduck-beak-throw](https://github.com/llama/microduck-beak-throw)
的 `410ed2249d62cd7aa4c642cd2c4e9b56f9a0cbfd` 修订（Apache-2.0），
可动下喙建模亦参考 [KeplerC/microduck-with-rss](https://github.com/KeplerC/microduck-with-rss)。
本地改动包括 HD1910 接口、独立舵机力矩限制、质量分配、显式碰撞掩码和物理验收。
既有机器人网格仍引用本机原资产路径，保留其原授权条件。

## 验证与数据边界

```bash
MUJOCO_GL=egl /home/luckysir/microduck/microduck_rl/.venv/bin/python \
  -m experiments.microduck_tasks.validate_actions --output /path/to/new_action_validation
MUJOCO_GL=egl /home/luckysir/microduck/microduck_rl/.venv/bin/python \
  -m pytest -q experiments/microduck_tasks
```

每次执行保存 `scene.xml`、50 Hz `trajectory.npz`、`action_metrics.json`、事件和结果摘要。
旧 11 类训练头的输入合同未修改，已对齐的 102 条数据仍有效。新动作要进入训练还需扩展技能类别、
统一新任务的感知输入和语义返回值，再采集不同场景的示范；当前手工预对准轨迹不能冒充自主寻物数据。

不匹配的旧采集、导出和特征目录已删除约 298.9 MiB，保留新对齐数据、权重和评估报告，
详见 [清理审计](outputs/cleanup_mismatched_inputs_20261115.json)。
