"""Human-readable action registry, separate from the existing 11-class VLA head."""

LABELS = {
    "stand": "站立",
    "move_forward": "前进",
    "move_backward": "后退",
    "move_left": "左移",
    "move_right": "右移",
    "turn_left": "左转",
    "turn_right": "右转",
    "kick_left": "左脚踢球",
    "kick_right": "右脚踢球",
    "sit_stand": "坐下再站起",
    "ground_pick": "俯身再起",
    "roll": "前滚翻",
    "flamingo_left": "抬左脚",
    "flamingo_right": "抬右脚",
    "happy_hop": "轻跳",
    "moonwalk": "后退滑步",
    "look_around": "左右看",
    "nod": "点头",
    "body_sway": "轻晃身体",
    "open_beak": "张嘴",
    "close_beak": "闭嘴",
    "pick_up": "叼起来",
    "carry": "叼着前进",
    "release": "松嘴",
    "place": "低头放下",
    "pick_place": "叼起搬运再放下",
    "push_block": "推方块",
    "pick_throw": "叼起并甩出",
    "throw": "甩出",
    "hop": "大跳候选",
    "run": "跑步候选",
    "bow": "鞠躬候选",
    "headstand": "头倒立候选",
    "beak_throw": "空嘴甩球动作",
    "official_sit": "官方坐站候选",
    "official_pick": "官方俯身候选",
    "official_roll": "官方滚翻候选",
    "official_walk": "官方行走候选",
    "official_stand": "官方站立",
    "velstand": "官方走站候选",
    "groove": "节奏舞步",
    "pirouette": "旋转舞步",
    "caught_dancing": "害羞舞步",
}
EXPERIMENTAL = {
    "pirouette": "两次 HD1910 测试中一次摔倒，当前不稳定",
    "hop": "HD1910 初测摔倒，原始目标角大幅越界",
    "run": "HD1910 初测摔倒，需要电机/时序适配",
    "headstand": "只下载了整套中的 fold 阶段；当前无法完成整套和恢复",
    "official_roll": "官方 1 秒切换时尚未恢复，本地 HD1910 roll 已另接入",
    "official_sit": "当前 HD1910 上站稳但没有完成坐下",
    "official_walk": "原版 XL330 策略，当前电机效果待验证",
    "velstand": "当前测试 0.2m/s 指令几乎不前进",
    "bow": "动作幅度很小，模型卡未声明许可证，语义效果待复核",
    "pick_throw": "能甩出，当前 HD1910 横向偏移大，不保证目标方向",
    "throw": "能甩出，当前 HD1910 横向偏移大，不保证目标方向",
    "beak_throw": "只播放身体动作，不含物体夹持和释放验收",
}
OBJECT_TASKS = {"pick_up", "carry", "release", "place", "pick_place", "throw", "pick_throw", "push_block"}
ALIASES = {label: key for key, label in LABELS.items()}
ALIASES.update(
    {
        "坐站": "sit_stand",
        "低头触地再起身": "ground_pick",
        "翻滚": "roll",
        "跳一下": "happy_hop",
        "叼起": "pick_up",
        "搬运": "pick_place",
        "放下": "place",
        "月球步": "moonwalk",
    }
)
SHOWCASE = (
    "look_around",
    "nod",
    "sit_stand",
    "flamingo_left",
    "flamingo_right",
    "groove",
    "caught_dancing",
    "happy_hop",
    "roll",
)
SCENE_REQUIREMENTS = {
    "roller / roller_crouch": "已下载；需要换轮脚模型和轮滑动力学，普通脚场景拒绝加载",
    "basketball": "已下载；LSTM 多输入，需要篮球支撑场景与对应循环状态适配",
}


def print_catalog():
    for key, label in LABELS.items():
        status = f"实验：{EXPERIMENTAL[key]}" if key in EXPERIMENTAL else "仿真动作/任务"
        print(f"{key:20s} {label:12s} {status}")
    for key, reason in SCENE_REQUIREMENTS.items():
        print(f"{key}: {reason}")
    print("支持‘踢红球 / 踢黄球 / 踢蓝球’，沿用已统一的 RGB-D 规则选择器。")
    print("新动作演示不会自动导出到旧的 11 类技能头训练集。详细实测及模型来源见 ACTIONS.md。")
