"""Conservative toy candidates; availability is not a hardware stability claim."""

from .action_catalog import LABELS

VERSION = "microduck-toy-candidates-1"
# Keep a single semantic spelling: sitting-and-standing is not persistent sitting.
SELF_ACTIONS = (
    "nod",
    "look_around",
    "sit_stand",
    "ground_pick",
    "roll",
    "flamingo_left",
    "flamingo_right",
    "moonwalk",
    "body_sway",
    "open_beak",
    "close_beak",
    "groove",
    "caught_dancing",
)
BASE_ACTIONS = ("stand", "move_forward", "move_backward", "move_left", "move_right", "turn_left", "turn_right")
CANDIDATES = BASE_ACTIONS + SELF_ACTIONS
SHOWCASE = ("look_around", "nod", "sit_stand", "flamingo_left", "flamingo_right", "groove", "caught_dancing", "roll")
INTERACTIONS = ("kick",)
DEFERRED = {
    "happy_hop": "轻跳有明显前移；连续动作测试前移约 0.74 m，暂不放进桌面玩具候选",
    "sit": "现有 sit_stand 会自动站起；保持坐姿及坐姿下的停止/恢复尚未验收",
    "navigate": "到球旁边的独立停靠距离尚未实现；不能以进入踢球区域代替",
    "turn_angle": "现有左右转是按时间执行；指定角度需要累计实际航向并验收",
    "continuous_move": "需要持续任务、短段续租和中断控制；桌边检测尚未接入",
}


def manifest():
    return {
        "schema_version": VERSION,
        "scope": "HD1910 local MuJoCo candidates with clipping and recovery; not hardware certification",
        "self_actions": [{"action": name, "label": LABELS[name]} for name in CANDIDATES],
        "interactions": list(INTERACTIONS),
        "deferred": DEFERRED,
        "dialogue_executor_ready": False,
        "evidence": [
            "outputs/action_validation_20261115/summary.json",
            "outputs/action_dance_trials_20261115/summary.json",
            "outputs/action_showcase_final_20261115/summary.json",
            "outputs/action_beak_push_final_20261115/summary.json",
            "outputs/action_kick_red_20261115/summary.json",
        ],
    }


def print_catalog():
    print("玩具候选库：本机仿真初测通过的自身动作；交互仅踢球。")
    for name in CANDIDATES:
        print(f"{name:20s} {LABELS[name]}")
    print("踢红球 / 踢黄球 / 踢蓝球：视觉搜索、接近、踢球、独立结果验收。")
    print("滚翻、滑步等需要足够空间；当前演示是地面仿真，不含真实桌边检测。")
    print("坐下保持、转一整圈、连续前进和完整聊天任务协议见 TOY_DIALOGUE.md，尚未接入此播放器。")
