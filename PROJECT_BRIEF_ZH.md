# MicroDuck 中文对话与任务执行项目

截至 2026 年 11 月 17 日，本项目已经在 MuJoCo 中跑通“中文指令 → 大模型任务规划 → 技能执行 → 执行反馈”的完整仿真链路。
用户可以输入“先点头，再去踢红球”，在原生仿真窗口中观察动作，并在右侧聊天框继续对话。
当前是研究原型，尚未证明真机上的可靠性。

本仓库是独立发布的项目快照，包含实验源码、合成数据和精选结果。
基础机器人库复用 [Strands Robots](https://github.com/strands-labs/robots)；机器人的模型与运动资源来自
[microduck-project](https://github.com/Ngalalalekhaaa/microduck-project)。这里的主要研究工作是上层对话、任务规划、执行器和感知支路，底层库并非本项目原创。

## 机器人怎样理解和执行指令

系统分成两个层次：

1. **语言模型**接收中文、结构化观测、机器人状态和历史，生成 `decision + goals + reply`，表示是否执行、执行哪些目标、怎样回应用户。
2. **宿主程序与执行器**检查计划和状态，把目标放入队列，调用已有行走、姿态和踢球策略；依据执行观测验收，再把结果交回语言模型。

例如“先点头，再去踢红球”应产生依次点头、踢指定红球的目标，并回复一句确认。
找球、接近和出脚时机由执行器根据当前观测判断；指定时长的移动按计划中的时长执行。语言模型不逐帧输出关节控制。
支持 13 种自身动作以及移动、转向、接近球和踢球；不在技能库中的任务，例如踢方块，应澄清。
最多保留 15 个未完成目标。普通“暂停、继续、停”有宿主控制通道，暂停会等到稳定执行边界。

**当前 Q1R 不接收原始相机图片。** 本轮微调学习的是文字和结构化输入到计划与回复的映射。
默认执行器使用模拟 RGB-D／IMU；独立 RGB＋测距支路尚未接入默认聊天入口。

## 做过哪些实验

先建立规则教师和可执行的技能链路，再对比直接提示词、冻结大模型后训练小型技能网络等方案。
小型网络在训练集上能拟合，但多步指令的开发集泛化不足，因此继续探索语言模型 LoRA 微调。

| 阶段 | 目的与变化 |
| --- | --- |
| Q0 | 用小规模样本验证 LoRA 训练链路和结构化输出协议能否学会 |
| Q1 | 学习指令、状态和历史到 `decision + goals` 的任务计划 |
| Q1R（当前） | 在 Q1 上继续训练，扩展表达、状态、组合任务和聊天，同时学习 `reply` |

后续进行了真实模型无窗口仿真测试、逐例错误检查、桌面交互和 ONNX 导出测速。
当前高层默认仍使用 **PyTorch**；ONNX 高层桌面集成保留为未完成实验。
底层运动技能原本使用的 ONNX 策略不受这一选择影响。

## 当前微调的方法与参数

| 项目 | 配置 |
| --- | --- |
| 基础模型 | Qwen3.5-2B |
| 微调方法 | QLoRA，基础权重采用 4-bit NF4 |
| 更新范围 | 语言层注意力与 MLP 的 LoRA，共 **8,409,600 个参数** |
| LoRA 参数 | rank=8，alpha=16，dropout=0.05 |
| 优化器／峰值学习率 | AdamW／`5e-5` |
| 批量 | 每次 1 条，梯度累积 16 次后更新 |
| 数据 | **8,196 条合成训练样本、343 条验证样本，覆盖 42 类情境** |
| 训练过程 | 完成 2 轮、1,026 次参数更新 |
| 当前选用 | 第 1 轮结束的 `checkpoint_000513` |

主干权重、视觉编码器、词嵌入、输出头和底层运动控制器保持冻结。
根据事先设定的验证规则，优先选择错误启动更少的检查点，因此最终选择第 513 步，并非训练中断。
数据是合成监督样本，不是 8,196 段真实人机对话；同一句话配合不同状态和历史可以成为不同样本。

可查阅 [训练配置](published/q1r_20261116/train/training_config.json)、
[参数与冻结范围](published/q1r_20261116/train/model_preflight.json)、
[训练初始化](published/q1r_20261116/train/initialization.json)、
[学习曲线](published/q1r_20261116/train/learning_curve.json)、
[训练结果](published/q1r_20261116/train/results.json) 和
[数据审计](published/q1r_20261116/data_final/audit.json)。

## 效果与尚未解决的问题

在相同的 **310 条可比开发输入**上，完整计划正确率从 Q1 的 **167/310（53.9%）**提升到
Q1R 的 **268/310（86.5%）**。常见动作、时间单位、多动作组合和用户改口有明显改善。
这是定向开发集结果，不是盲测，也不能直接解释为真实用户的成功率。
证据见 [Q1/Q1R 对照](published/q1r_20261116/post_training/comparison.json)。

选中的 Q1R 模型直接控制 MuJoCo 时，7 个有限动作会话中的 **11 个目标全部通过验收**，其中包括 4 次踢球。
踢球验收包含实际接触、球位移和恢复站立。聊天、澄清、持续运动控制会话不计入这个分母，
宿主处理的停止等命令也不计作模型自主控制能力。
证据见 [闭环摘要](published/q1r_20261116/evaluation_20261117/closed_loop/summary.json)。

当前仍有以下限制：

- 状态约束和队列容量边界理解不稳定；结构校验能拦住部分错误，但结构合法的错误计划仍可能通过。
- 回复内容不一定与正确计划一致，出现过当前动作结束就把下一个尚未完成的动作说成已完成。
- 方位问答和开放聊天覆盖有限，训练数据存在位置、反馈类型等偏差。
- 上述成绩来自有限仿真和开发测试，没有证明真实硬件、陌生场景或任意说法下的稳定性。

详见 [综合评估](published/q1r_20261116/evaluation_20261117/assessment.json) 和
[数据分布诊断](published/q1r_20261116/evaluation_20261117/data_distribution_review.json)。

## 独立的 RGB＋测距踢球实验

另一路实验模拟头部 RGB 相机、位于相机左侧的 8×8 ToF 测距、IMU 和关节编码器。
通过内外参标定动态投影测区，结合球轮廓、测距和关节正运动学，估计球相对脚的位置，
完成小步接近、停稳复测和踢球。控制输入不包含目标世界坐标、身体真值姿态或基座真值速度；仿真真值用于独立评分。

最新 v6 的固定场景、噪声丢测和扩展位置开发回归合计 **16/16 通过**。
这是小规模受限场景结果，不能推出任意情况下 100% 踢中；该支路尚未并入默认 Q1R 桌面入口。
方法见 [RGB/ToF 实验说明](experiments/microduck_tasks/RGB_TOF_SIM_ZH.md)，
结果见 [发布摘要](published/rgb_tof_20261116/FOOT_REACH_SUMMARY.json)。

## 仓库内容与复现条件

| 位置 | 内容 |
| --- | --- |
| [experiments/microduck_tasks/](experiments/microduck_tasks/) | 数据生成、训练、评估、任务执行器、桌面聊天与感知实验源码 |
| [published/q1r_20261116/data_final/](published/q1r_20261116/data_final/) | 最终合成训练／验证数据及审计清单 |
| [published/q1r_20261116/](published/q1r_20261116/) | 训练配置、检查点选择记录、开发评估与闭环摘要 |
| [实验详细说明](experiments/microduck_tasks/PROJECT_BRIEF_ZH.md) | 本机实验过程与历史命令，含未随仓库发布的本机路径 |
| [分阶段实验记录](experiments/microduck_tasks/STAGED_EXPERIMENTS_ZH.md) | 早期基线、Q0 与 Q1 的探索过程 |
| [原完整实验方案](reference/MicroDuck_完整实验方案20261116.md)／[配套包](reference/MicroDuck_experiment_kit.zip) | 参考设计，所列步骤并非全部已完成 |
| [strands_robots/](strands_robots/) | 复用的上游机器人库 |

**本次不上传模型权重、优化器训练状态、缓存、录屏和个人聊天记录。** 新机器克隆后不能直接一条命令运行 Q1R。
需要先准备兼容的 CUDA/PyTorch、MuJoCo 和 PyQt5 环境，下载基座与机器人资源，
并准备经过验证的 Q1R adapter（含 `identity.json`，父目录有匹配的 `results.json`）。
仅下载发布目录里的 JSON 元数据不能代替 adapter 权重。

通过 `MICRODUCK_ROOT` 指定机器人资源仓库位置；匹配的踢球权重还需放到
[bootstrap.py](experiments/microduck_tasks/bootstrap.py) 约定的缓存目录。
[setup_skill_head.py](experiments/microduck_tasks/setup_skill_head.py) 支持
`--shared-python /path/to/cuda/python --download`；QLoRA 还需要代码校验的 PEFT 版本。
资源与依赖齐全后，在仓库根目录使用实际路径启动：

```bash
export MICRODUCK_ROOT=/path/to/microduck-project
/path/to/prepared/python -m experiments.microduck_tasks.toy_desktop \
  --backend q1r \
  --model-dir /path/to/Qwen3.5-2B \
  --checkpoint /path/to/q1r/train/checkpoint_000513
```

上游 [LICENSE](LICENSE)、[NOTICE](NOTICE)、[README_UPSTREAM.md](README_UPSTREAM.md) 和
[来源版本记录](SOURCE_SNAPSHOT.json) 随仓库保留。外部机器人资源和模型权重遵守各自来源的许可证。
