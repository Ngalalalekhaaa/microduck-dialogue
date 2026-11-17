# 本机技能级 VLA 改进与实测记录

2026-11-15。本次实现、标定和验收均在本机 HD1910 MuJoCo 仿真完成，使用 RTX 4050 Laptop 6 GB；没有连接实体机器人。目标是形成“图像、语言与执行反馈 → 技能计划 → 闭环技能 → 已有行走/踢球策略”的分层训练流程。

目前已完成：运动补偿、视觉导航教师、v2 通用实体与技能序列协议、异步执行器、独立数据导出、训练编码和适配器服务接口，以及 Qwen3.5-2B 本机推理。**v2 生成式 LoRA 尚未训练；30/30 踢球成绩属于规则教师，不属于 VLM。** 随后完成了“冻结 Qwen3.5、只训练技能选择头”的首轮实验：头训练约 1 秒，离线测试准确率 89.2%，新场景闭环踢球 2/9，同场景规则教师 9/9。完整方法、局限和产物见 [README.md](README.md#方案一冻结-qwen35-的技能选择头)。

## 1. 直线运动：命令方向确实不等于实际方向

原 `skills.py` 的平移命令是 `(±0.2, 0, 0)`、`(0, ±0.2, 0)`，由已有行走 ONNX 策略转成关节动作。零目标角速度没有保证实际零偏航；启动姿态、步态相位和接触变化都会改变漂移。固定 yaw 补偿单独使用，在不同初态下也不稳定，因此最终实现是**逐方向实测前馈 + 50 Hz 航向反馈 + 必要的正交线速度补偿**。

| 技能 | 实测前馈 `(vx, vy, wz)`，m/s、rad/s |
|---|---|
| 前进 | `(0.20, 0.00, -0.10)` |
| 后退 | `(-0.20, 0.02, 0.00)` |
| 左移 | `(0.00, 0.20, 0.28)` |
| 右移 | `(-0.02, -0.20, 0.02)` |

控制器用本体姿态和陀螺仪修正剩余偏航，同方向连续动作块保持同一航向参考；转向、站立和改变移动方向会重新建立参考。导航技能可以用当前相机目标方位更新航向参考。控制器不读取仿真世界 XY；世界 XY 只用于离线标定和评估。校准文件绑定行走权重 SHA256，并明确标记 `hardware=false`。

留出初态的 8 秒测试如下。这里比较的是最大侧向偏离，不是最终位置恰好回到直线的误差：

| 方向 | 原最大侧偏 | 改进后最大侧偏 | 原最终偏航 | 改进后最终偏航 |
|---|---:|---:|---:|---:|
| 前进 | 40.3 cm | 6.2 cm | 31.0° | 1.9° |
| 后退 | 15.5 cm | 2.5 cm | 6.3° | -0.2° |
| 左移 | 79.8 cm | 8.4 cm | -55.1° | 0.4° |
| 右移 | 44.6 cm | 8.6 cm | -41.5° | -8.7° |

还验证了 3 秒和 12 秒；完整结果在 [motion_feedback_20261115/calibration.json](outputs/motion_feedback_20261115/calibration.json)。这支持“显著降低漂移”，**不支持保证绝对笔直**，尤其右移仍有剩余误差。IMU 航向保持不能独立观测侧滑；接近目标时还需要相机闭环。真实地面、执行器或行走模型变化后必须重新标定，不能直接把这组仿真数值当实体机校准。

代码：[motion.py](motion.py)、[motion_calibration.json](motion_calibration.json)、[calibrate_motion.py](calibrate_motion.py)、[validate_motion.py](validate_motion.py)。

## 2. 结构化输入输出：任务通用，执行能力明确

原 v1 的 `type/skill/seconds/target_color/reply` 便于当前小任务微调，但颜色被提升为顶层任务字段、每次仅一个动作、缺少承诺中的动作与执行结果，限制扩展。旧协议和已有适配器保留；新协议版本为 `microduck-plans-2`。

查阅的一手项目与论文，并不是都在输出 JSON 技能名：

| 参考 | 输入和输出的实际层级 | 对本项目的启发 |
|---|---|---|
| [Isaac GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T) | 视觉、语言和本体状态，配合动作生成模块 | 保留高层理解与低层连续控制的分工；其动作模块不能直接替代鸭子的既有技能 |
| [π0.5](https://arxiv.org/abs/2504.16054) | 图像、指令、物体检测、语义子任务与低层动作的混合训练 | 用通用任务/对象描述和语义技能边界，避免把颜色当任务本身 |
| [InternNav](https://github.com/InternRobotics/InternNav) | 视觉语言导航与连续导航策略 | 分离目标识别、局部导航和运动执行，不能把导航规划等同关节动作 |
| [LeRobot 异步推理](https://huggingface.co/docs/lerobot/main/en/async) | 后台推理与动作块执行协同 | 输出有界序列，并使模型看到当前已经承诺的动作 |
| [Physical Intelligence RTC](https://www.pi.website/research/real_time_chunking) | 执行前一个连续动作块时生成后续动作 | 借鉴续接时必须考虑已执行/承诺前缀的原则；本实现不是论文的扩散/流模型 RTC 算法 |

本项目的输入是两路当前 RGB、自然语言原始任务，以及以下结构化传感状态和上下文：

| 部分 | 字段 |
|---|---|
| `state` | `schema_version`, `observation_id`, `task_elapsed_s`, `yaw`, `tilt_deg`, `stationary`, `entities`, `available_skills` |
| 每个实体 | `id`, `category`, `attributes`, `position_body_m`, `confidence`, `source`, `camera` |
| `context` | `task_id`, `committed_steps`, `recent_outcomes` |

`position_body_m` 是当前 RGBD 估计的本体坐标，不是世界真值。颜色写成 `attributes: [{"name":"color","value":"red"}]`；同一接口也能表达材质等属性。目标可以为 `null`，问答和普通移动不需要虚构目标。

输出示例：

```json
{
  "schema_version": "microduck-plans-2",
  "status": "continue",
  "target": {
    "id": "ball:red",
    "category": "ball",
    "attributes": [{"name": "color", "value": "red"}]
  },
  "steps": [
    {"skill": "approach_target", "seconds": 1.4},
    {"skill": "stand", "seconds": 0.4},
    {"skill": "kick_left", "seconds": 0.5}
  ],
  "reply": "接近目标，站稳后尝试左脚踢球。"
}
```

这只是格式示例；脚的选择取决于当前观测，执行器仍逐步检查可达性。`status` 为 `continue/answer/clarify/finish`。未知字段、重复 JSON 键、NaN/Inf、非法技能、时长、非末尾踢球等均拒绝。`finish` 只表示模型声称完成，不能替代物理验收。

**通用接口不等于已经具备通用感知或新技能。** 当前 `entities()` 仍接三色球 RGBD 检测器；`approach_target` 与踢球执行仍只支持已实现的球目标。新增物体类别需要对应传感器适配、稳定实体跟踪与技能实现。现在颜色 ID 也只适合每种颜色一个球的实验，多个同色物体需要独立跟踪 ID。旧 v1 数据/适配器不会被静默解释为 v2。

代码：[plan_protocol.py](plan_protocol.py)、[plan_backend.py](plan_backend.py)、[plan_tasks.py](plan_tasks.py)。

## 3. 动作连续性：短序列、后台续接、本地视觉检查

一次最多 4 步、总计划时域最多 3 秒；`approach_target` 单步最多 2 秒，平移原语最多 0.6 秒。该总时域不包含踢球后执行器强制的 3 秒站立恢复。后台只运行一个推理请求，执行器在当前块执行时请求后续计划，把已承诺步骤放入 `committed_steps`。

普通平移按最多 0.2 秒分段重新观察；视觉导航采用自己的短步长和站稳阶段。取消、目标丢失、前缀中断或代次不匹配都会使续接失效。来源观测超过 6 秒的计划拒绝提交；单次控制取图超过 0.3 秒也拒绝继续。模型直接移动命令还有单次预算，避免“前进一点”被连续重复。新版明确依赖 RGBD 实体，不接受 `--vlm-rgb-only`。

普通动作取消了每步强制站立 1 秒，踢球恢复保留。没有有效新计划时保持站立，不无限延长旧动作。推理耗时超过剩余执行时域时仍会有等待，异步本身不保证无间断；必须结合模型延迟和闭环技能时长评估。

单元测试覆盖推理/执行重叠、取消后迟到结果、目标丢失、执行前踢球检查、单次移动预算，以及训练与服务端图像顺序/提示词一致性。

## 4. 本机模型选择：已经实跑新模型，但不能把更新等同更可靠

检查现有权重和服务后，原模型实际上是 **Qwen3-VL-2B-Instruct**，不是 Qwen2-VL-2B。其技术报告属于 2025 年的 Qwen3-VL 系列：[官方模型页](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)、[技术报告](https://arxiv.org/abs/2511.21631)。

| 候选 | 本次结论 |
|---|---|
| 现有 Qwen3-VL-2B + 已训练适配器 | 保留 v1 工作基线；新 v2 数据和训练代码也先沿用这个本机已验证的训练架构 |
| [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 已下载固定版本的 [Unsloth GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)，实际在本机加载 Q4_K_M 语言模型 + F16 视觉投影，并运行 v2 请求 |
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 更小的后续候选，尚未在此任务实测；现在不为追求体积直接牺牲任务效果 |
| [LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M) | 轻量多语言视觉候选，官方提供端侧部署格式；任务表现未实测，许可证为 LFM1.0 |

Qwen3.5-2B 的本机启动后 GPU 总占用曾约 3.5 GB，包含桌面，**不是模型独占峰值**。单独运行时能容纳在本机显存中；同时运行两套 VLM 服务实际发生过显存不足，应只启动一套。下载的两个文件合计约 1.95 GB，启动脚本校验固定 revision、大小和 SHA256。

三条交互烟雾测试中，Qwen3.5 能回答并发出移动/转向 JSON；7 次决策延迟中位约 0.51 秒。但当时前进请求曾被重复执行，此后补了执行器的单次预算。随机 seed 0 红球测试仍 `verification_failed`，没有完成有效踢击。因此这里只能认定**加载、双图输入、结构化输出和执行链路已跑通**，不能认定新模型机器人能力优于现有微调模型。记录保存在 [qwen35_v2_smoke3_20261115](outputs/qwen35_v2_smoke3_20261115) 和 [qwen35_v2_kick_20261115](outputs/qwen35_v2_kick_20261115)，失败记录未删除。

目前训练器仍使用 `Qwen3VLForConditionalGeneration`，不能拿已有 Qwen3-VL LoRA 直接加载到 Qwen3.5，也没有实现 Qwen3.5 微调/ONNX 导出。后续替换应以相同留出任务的成功率、路径、决策次数、P50/P95 延迟和峰值显存来判断。

按本次补充要求，不开展 Radxa 部署。[Rockchip 官方 RKNN-LLM](https://github.com/airockchip/rknn-llm) 已提供相关小视觉模型支持，但 RK3588 需要对应 RKNN/RKLLM 转换和板端测量；本机 CUDA 或 GGUF 跑通不能证明板端延迟。

## 5. 高质量教师与数据：先朝向目标，再视觉闭环接近

新教师先搜索/转向，让目标方位接近相机中心；在前进过程中不断用当前目标方位修正航向。它对可见的其他球做局部避让，目标丢失则搜索，不继续盲走。进入脚边后站稳，用新观测检查踢球范围；达到可踢条件就踢，避免原来围绕名义落脚点反复横移。它是**无地图的局部视觉伺服导航**，没有宣称全局最短路径或一般障碍环境规划能力。

固定复位姿态的左右脚网格共 50 个点全部产生有效踢击，但扩大范围直接应用于行走后姿态曾出现漏踢。因此最终采用较保守的前向距离 `0.075..0.105 m`，每只脚朝自身一侧的本体横向范围 `-0.003..0.055 m`，并保留站稳、脚选择锁定和重新观察。该横向范围比原来的精细点对齐宽；它不是根据实际足端运动学标定的全姿态工作空间。仍可能需要更多步态相位/地面变化测试。

| 同种子 0、1、2 × 三色球，9 个任务 | 原教师 + 原命令 | 新教师 + 补偿 |
|---|---:|---:|
| 有效踢球 | 9/9 | 9/9 |
| 平均路径 | 1.490 m | 0.798 m（减少 46.4%） |
| 平均仿真耗时 | 38.751 s | 10.424 s（减少 73.1%） |
| 平均决策次数 | 55.00 | 18.67 |
| 平均微调决策 | 15.11 | 0 |
| 站立命令时长 | 29.60 s | 4.44 s |

最终版本在种子 `0,1,2,6,9,10,11,12,13,14` × 三色球 **30/30** 成功；其中 `9..14` 是最终调整后未用过的 18 个留出任务。验收仍要求目标脚接触、球位移至少 0.12 m、无提前推球/误碰其他球、无跌倒并恢复站立。旧版验收标准没有放宽。

证据：[原教师 9 个任务](outputs/teacher_legacy_20261115/results.json)、[最终教师 30 个任务](outputs/teacher_stance_margin_20261115/results.json)、[固定姿态踢球范围](outputs/kick_reach_20261115/reach.json)。调参过程的超时/漏踢也保留在 `teacher_efficient_*`、`teacher_final_20261115` 等目录；30/30 是有限场景结果，不代表总体成功概率为 100%。比较包含导航、运动补偿和恢复调度的共同变化，没有把全部增益归因于单个模块。

新采样器固定种子 3100/3101/3102 分别用于 train/val/test，每个种子三色任务，加入观察问答和六种移动/转向。导出时只接收独立验收通过的规则示范，不把 VLM 预测当专家标签，不把真值/未来帧放入输入。导航过程压成 `approach_target` 语义步骤，另生成带已承诺前缀的续接样本。相同场景的相邻帧不会跨数据集拆分。

历史上生成过 train 140、val 127、test 121，共 **388 条**流程样本。随后发现它们的决策边界和动作历史与部署不一致，已按用户要求删除，不能继续使用本节历史采样逻辑训练新头。当前保留的是共用语义执行器重新采集的 102 条样本，见 [README 的接口对齐说明](README.md#训练输入与运行输入的对齐修正) 和 [新清单](outputs/aligned_plan_dataset_20261115/manifest.json)。

## 复现与使用

在仓库根目录执行。所有 `--output` 必须是尚不存在的新目录。

直接运行高效教师，默认已启用补偿：

```bash
./experiments/microduck_tasks/run.sh --teacher efficient --seed 9 --command "踢红球"
```

复现旧行为需同时指定 `--teacher legacy --nominal-motion`。原 v1 批量采集器已显式使用这两个选项，避免默认值改变导致旧标签与恢复语义混用。

启动新模型试验服务（文件已下载，首次安装到另一台电脑需先加 `--download`）：

```bash
/home/luckysir/microduck/microduck_rl/.venv/bin/python \
  -m experiments.microduck_tasks.setup_qwen35 --serve
```

在另一终端运行，结束后在服务终端 Ctrl+C 释放显存：

```bash
./experiments/microduck_tasks/run.sh \
  --backend vlm-v2 \
  --vlm-endpoint http://127.0.0.1:8084/v1/chat/completions \
  --vlm-model microduck-qwen35 --interactive --seed 0
```

重新采集小样本并导出：

```bash
/home/luckysir/microduck/microduck_rl/.venv/bin/python \
  -m experiments.microduck_tasks.collect_plan_teacher --output /path/to/new_samples
/home/luckysir/microduck/microduck_rl/.venv/bin/python \
  -m experiments.microduck_tasks.plan_training_data \
  --input /path/to/new_samples --output /path/to/new_dataset
```

首次验证结束时磁盘约余 4.5 GiB，低于现有训练脚本的 8 GiB 空间要求，因此当时没有启动新版训练。随后按用户要求删除了昨天 300 个旧教师踢球任务及 1 个中断任务的 RGB、深度和轨迹大文件，释放约 10.04 GiB，剩余空间约 14.49 GiB；结果摘要、旧版独立导出数据、权重和今天的数据均保留。详见 [数据清理记录](outputs/cleanup_legacy_navigation_20261115.json)。之后用户进一步要求删除 `.cache/microduck_onnx` 的合并权重、转换产物与专用环境，又释放约 11.49 GiB，剩余约 25.98 GiB；Qwen3.5、Qwen3-VL 基座和训练适配器保留，旧 ONNX 服务需重新导出才能使用，详见 [ONNX 清理记录](outputs/cleanup_onnx_cache_20261115/cleanup.json)。空间条件已满足，尚未启动新训练；v2 训练入口为：

```bash
.cache/microduck_train/venv/bin/python -m experiments.microduck_tasks.train_vlm \
  --dataset experiments/microduck_tasks/outputs/aligned_plan_dataset_20261115 \
  --output /path/to/new_v2_training --max-steps 100
```

这条命令沿用 Qwen3-VL-2B 基座，不能把旧 v1 适配器作为 v2 成品。训练完成后的服务入口为 `serve_qlora.sh --protocol v2 --adapter /path/to/new_v2_training/final_adapter --port 8085`，客户端使用 `--backend vlm-v2 --vlm-model microduck-plan-qlora` 和对应 8085 endpoint。服务会校验 v2 提示词与基座指纹；原 ONNX 服务仍属于 v1。

验证：本实验 437 项测试通过；72 个顶层实验 Python 文件通过 Ruff 检查与格式检查；v1 提示词文件与修改前逐字一致。当前仿真环境未安装 pytest-timeout 插件，测试产生一条已有的 `Unknown config option: timeout` 警告。递归扫描历史 `reports/` 时仍有原报告生成脚本的 Ruff 问题，未顺带修改这些旧报告；本次没有宣称通过整个仓库的完整 CI、LeRobot 集成测试或严格文档构建。
