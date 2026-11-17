"""Hand-authored evaluation expectations, never inputs to the learned selector."""

from .toy_fixtures import kick, move, navigate, perform, posture, turn


def _step(text, decision="respond", goals=(), *, wait="done", seconds=0.3):
    return dict(text=text, expected=dict(decision=decision, goals=list(goals)), wait=wait, seconds=seconds)


def cases():
    plans = [
        ("greeting", "hello", "respond", []),
        ("visual_chat", "你现在能看到什么", "respond", []),
        ("unsupported", "做一下后空翻", "decline", []),
        ("ambiguous_target", "踢一下球", "clarify", []),
        ("single_nod", "点点头", "replace", [perform("nod")]),
        ("mouth_pair", "请张开嘴，然后合上嘴", "replace", [perform("open_beak"), perform("close_beak")]),
        (
            "posture_triple",
            "请先点头，随后坐下，最后站起来",
            "replace",
            [perform("nod"), posture("sitting"), posture("standing")],
        ),
        (
            "six_step_queue",
            "先点头，再张嘴，再闭嘴，再坐下，再站起来，再左右看",
            "replace",
            [
                perform("nod"),
                perform("open_beak"),
                perform("close_beak"),
                posture("sitting"),
                posture("standing"),
                perform("look_around"),
            ],
        ),
        ("kick_blue", "请踢一下蓝球", "replace", [kick("blue")]),
        ("approach_kick", "先走到黄球旁边，再踢黄球", "replace", [navigate("yellow"), kick("yellow")]),
        (
            "two_balls_turn",
            "先踢蓝球，再踢红球，再原地转一圈",
            "replace",
            [kick("blue"), kick("red"), turn()],
        ),
        (
            "turn_walk",
            "先向左转九十度，再向前走0.6秒",
            "replace",
            [turn(angle=90), move(seconds=0.6)],
        ),
        ("roll", "做一下前滚翻", "replace", [perform("roll")]),
    ]
    result = [dict(id=name, steps=[_step(text, decision, goals)]) for name, text, decision, goals in plans]
    result.append(
        dict(
            id="continuous_dialogue_control",
            steps=[
                _step("一直往前走不要停", "replace", [move(continuous=True)], wait="advance"),
                _step("你现在能看到什么", wait="advance", seconds=0.1),
                _step("暂停", "pause"),
                _step("你好"),
                _step("继续", "resume", wait="advance"),
                _step("停下", "cancel"),
            ],
        )
    )
    return result
