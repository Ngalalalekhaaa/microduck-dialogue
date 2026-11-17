"""Independent colloquial, default-unit and mixed-script action expressions."""

from .q1r_data_core import base_context
from .toy_fixtures import move, navigate, perform, posture, turn

CASES = (
    ("往后稍稍挪一下", [move("backward")]),
    ("向前挪一点", [move("forward")]),
    ("向左移一下身子", [move("left")]),
    ("右边挪一下就好", [move("right")]),
    ("后退一下吧", [move("backward")]),
    ("请往前走一点点", [move("forward")]),
    ("稍微往左转一下", [turn("left", 30)]),
    ("向右转一下就好", [turn("right", 30)]),
    ("左转吧", [turn("left", 30)]),
    ("右转一点", [turn("right", 30)]),
    ("hou tui san miao", [move("backward", 3)]),
    ("wang hou zou yi miao", [move("backward", 1)]),
    ("qian jin liang miao", [move("forward", 2)]),
    ("dian yi xia tou", [perform("nod")]),
    ("zou jin hong qiu pang bian", [navigate("red")]),
    ("xiang zuo zhuan jiu shi du", [turn("left", 90)]),
    ("后退３ｓ", [move("backward", 3)]),
    ("向前走２秒", [move("forward", 2)]),
    ("左转９０度", [turn("left", 90)]),
    ("请右移０．５秒", [move("right", 0.5)]),
    ("往left走2秒", [move("left", 2)]),
    ("请向right移动3秒", [move("right", 3)]),
    ("backward，走两秒", [move("backward", 2)]),
    ("turn left 九十度", [turn("left", 90)]),
    ("move back for 3 seconds", [move("backward", 3)]),
    ("please nod once", [perform("nod")]),
    ("turn right 90 degrees", [turn("right", 90)]),
    ("sit down and stay seated", [posture("sitting")]),
    ("站起来，别做坐下再起身那一套", [posture("standing")]),
    ("只需要起身站好，不要坐站循环", [posture("standing")]),
    ("保持坐着，不要又站起来", [posture("sitting")]),
    ("请蹲好休息，先不要起立", [posture("sitting")]),
    ("只靠近蓝球，这次别踢它", [navigate("blue")]),
    ("到黄球边上就停住，省掉踢球", [navigate("yellow")]),
    ("走近红球，停在旁边，不要碰球", [navigate("red")]),
    ("请朝蓝球走过去，到了别踢", [navigate("blue")]),
    ("轻轻往身后挪一下即可", [move("backward")]),
    ("朝右侧稍转一点身", [turn("right", 30)]),
    ("hou tui ban miao", [move("backward", 0.5)]),
    ("往后走１．５秒", [move("backward", 1.5)]),
    ("请移动到yellow球附近，do not kick", [navigate("yellow")]),
    ("stand up, 不要坐回去", [posture("standing")]),
    ("恢复直立就好，不需要先坐再站", [posture("standing")]),
    ("挪到蓝球边上停好，踢的动作省略", [navigate("blue")]),
)

TURN_EXTRA = (
    ("顺时针转九十度", [turn("right", 90)]),
    ("逆时针旋转半圈", [turn("left", 180)]),
    ("向右转一百八十度", [turn("right", 180)]),
    ("往左转四分之一圈", [turn("left", 90)]),
    ("顺时针转一整圈", [turn("right", 360)]),
    ("逆时针转三十度", [turn("left", 30)]),
    ("朝右旋转45°", [turn("right", 45)]),
    ("向左转二百七十度", [turn("left", 270)]),
    ("顺着钟表指针方向转四分之一圈", [turn("right", 90)]),
    ("逆时针转过四十五度", [turn("left", 45)]),
)


def generate_colloquial(builder):
    for p, (text, goals) in enumerate(CASES):
        split = "val" if p >= 36 else "train"
        for i in range(3 if split == "val" else 10):
            pose = "sitting" if i % 3 == 0 else "standing"
            req = base_context(text, i, posture_name=pose)
            builder.add(
                f"colloquial_{p}",
                split,
                "colloquial_units",
                req,
                "replace",
                goals,
                rationale="独立手写口语/拼音/全角/中英混合；未给时长的轻移默认0.32秒，未给角度的明确左右转默认30度；局部否定不撤销肯定任务。",
            )
    for p, (text, goals) in enumerate(TURN_EXTRA):
        split = "val" if p >= 8 else "train"
        for i in range(3 if split == "val" else 10):
            req = base_context(text, i, posture_name="sitting" if i % 3 == 0 else "standing")
            builder.add(
                f"colloquial_rotation_{p}",
                split,
                "turn_units",
                req,
                "replace",
                goals,
                rationale="顺时针对应右转，逆时针对应左转；圈分数与中文数词精确换算为角度。",
            )
