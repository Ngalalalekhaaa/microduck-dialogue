"""Conservative rule teacher for real dialogue rollouts, not a generative VLM.

Only request observations are used. Unsupported clauses reject the entire queue;
accepted replies acknowledge future execution and never manufacture completion.
"""

import hashlib
import re

from .action_catalog import LABELS
from .toy_catalog import SELF_ACTIONS
from .toy_inputs import validate_input
from .toy_protocol import VERSION, validate_output

COLORS = {"红": "red", "黄": "yellow", "蓝": "blue"}
COLOR_NAMES = {value: key for key, value in COLORS.items()}
NUMBERS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
ALIASES = {LABELS[action]: action for action in SELF_ACTIONS}
ALIASES.update(
    {
        "点个头": "nod",
        "点点头": "nod",
        "翻滚": "roll",
        "向前翻滚": "roll",
        "左右看看": "look_around",
        "环顾四周": "look_around",
        "看看四周": "look_around",
        "抬起左脚": "flamingo_left",
        "抬起右脚": "flamingo_right",
        "用左脚站立": "flamingo_right",
        "用左脚单脚站立": "flamingo_right",
        "左脚单脚站立": "flamingo_right",
        "用右脚站立": "flamingo_left",
        "用右脚单脚站立": "flamingo_left",
        "右脚单脚站立": "flamingo_left",
        "月球步": "moonwalk",
        "月球漫步": "moonwalk",
        "左右摇摆": "body_sway",
        "晃晃身体": "body_sway",
        "张开嘴": "open_beak",
        "合上嘴": "close_beak",
        "跳舞": "groove",
        "跳个舞": "groove",
        "跳一段舞": "groove",
        "坐站": "sit_stand",
        "坐站动作": "sit_stand",
        "俯身复原": "ground_pick",
        "点一下头": "nod",
        "点头示意": "nod",
        "点头打个招呼": "nod",
        "看看左右": "look_around",
        "向两边看看": "look_around",
        "转头环视": "look_around",
        "左右转头": "look_around",
        "坐站循环": "sit_stand",
        "坐站练习": "sit_stand",
        "坐站演示": "sit_stand",
        "完成坐站动作": "sit_stand",
        "俯身动作": "ground_pick",
        "低头触地动作": "ground_pick",
        "俯身触地复原": "ground_pick",
        "俯身演示": "ground_pick",
        "前翻滚": "roll",
        "向前滚翻": "roll",
        "往前滚翻": "roll",
        "左脚抬起来": "flamingo_left",
        "把左脚抬起": "flamingo_left",
        "抬高左脚": "flamingo_left",
        "右脚抬起来": "flamingo_right",
        "把右脚抬起": "flamingo_right",
        "抬高右脚": "flamingo_right",
        "后退滑步动作": "moonwalk",
        "滑个太空步": "moonwalk",
        "太空步": "moonwalk",
        "摇摇身体": "body_sway",
        "轻轻晃动身体": "body_sway",
        "身体左右晃动": "body_sway",
        "张开鸭嘴": "open_beak",
        "把嘴张开": "open_beak",
        "嘴巴张开": "open_beak",
        "张大嘴巴": "open_beak",
        "合上鸭嘴": "close_beak",
        "把嘴合上": "close_beak",
        "嘴巴合上": "close_beak",
        "闭上嘴巴": "close_beak",
        "来段节奏舞": "groove",
        "跳节奏舞": "groove",
        "跳个害羞舞": "caught_dancing",
        "跳害羞舞": "caught_dancing",
        "害羞舞": "caught_dancing",
        "跳一段害羞舞": "caught_dancing",
        "表演害羞舞步": "caught_dancing",
    }
)
VISUAL = r"看到|看见|看得到|看到了|视野|画面|眼前|面前有什么|有几个球|球在哪|球在你哪|哪个球.*近"
BLOCKED = r"后空翻|倒立|跳跃|跳一下|跑步|飞起来|叼|搬运|推方块|扔|投篮|抓取"
NONAUTHORIZE = r"[‘’“”\"']|如果|假如|要是|昨天|刚才|上次|曾经|是否应该"


def _utterance(text):
    return re.sub(r"\s+", "", text).strip("。！？!?")


def is_cancel(text):
    """Recognize only direct stop requests before image capture or inference."""
    text = _utterance(text)
    if re.search(NONAUTHORIZE, text):
        return False
    return bool(
        re.fullmatch(
            r"(?:小鸭[，,]?|请你|请|先|现在|马上|立即|快|赶紧)*"
            r"(?:停|停下|停下来|停一下|停一停|停住|停止(?:当前|全部|所有)?任务|停止(?:执行|一下)?|站住|"
            r"取消(?:(?:当前|全部|所有)?任务)?|(?:别|不要)(?:再)?(?:动|走|做|执行|继续)(?:下去)?|"
            r"不用(?:再)?(?:做|执行|继续)|不做|算了|够了)(?:吧|来|了|啊|呀|啦)?",
            text,
        )
        or re.fullmatch(r"先停下[，,]?(?:再|然后)(?:描述|说说)(?:一下)?(?:新)?画面", text)
    )


def is_pause(text):
    """Pause preserves the queue; negations, quotations and hypotheticals do not."""
    text = _utterance(text)
    return not re.search(NONAUTHORIZE, text) and bool(re.fullmatch(r"(?:请|先)?暂停(?:一下|任务)?", text))


def is_resume(text):
    text = _utterance(text)
    return not re.search(NONAUTHORIZE, text) and bool(re.fullmatch(r"(?:请)?(?:继续|恢复)(?:执行|任务|吧)?", text))


class Unclear(ValueError):
    """An utterance cannot be safely reduced to the supported grammar."""


def _response(reply, decision="respond", goals=()):
    return {"schema_version": VERSION, "reply": reply, "decision": decision, "goals": list(goals)}


def _number(text):
    if text in NUMBERS:
        return NUMBERS[text]
    if text in ("三十", "九十", "一百八十", "三百六十"):
        return {"三十": 30, "九十": 90, "一百八十": 180, "三百六十": 360}[text]
    return float(text)


def _target(color):
    return {"category": "ball", "attributes": [{"name": "color", "value": color}], "entity_id": None}


def _trim(text):
    text = re.sub(r"\s+", "", text).strip("。！？!?，,；;")
    text = re.sub(r"^(?:请|麻烦|帮我|给我|替我|你|先|最后|依次|做一下|做一个|做个|做|来个)+", "", text)
    return re.sub(r"(?:一下|一遍|一次|吧|好吗|好不好)$", "", text)


def _color(text, request):
    matches = [color for name, color in COLORS.items() if name in text]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise Unclear("每一步请指定一个球；多个球请用‘先……再……’排出顺序。")
    if re.search(r"它|这个|那个", text):
        for item in reversed(request["history"]):
            if item["role"] != "user":
                continue
            mentioned = [color for name, color in COLORS.items() if name in item["text"]]
            if len(mentioned) == 1:
                return mentioned[0]
            if mentioned:
                break
        visible = {
            attribute["value"]
            for entity in request["observation"]["entities"]
            if entity["category"] == "ball" and entity["confidence"] >= 0.5
            for attribute in entity["attributes"]
            if attribute["name"] == "color" and attribute["value"] in COLOR_NAMES
        }
        if len(visible) == 1:
            return next(iter(visible))
    raise Unclear("你指的是哪个球？请明确说红球、黄球或蓝球。")


def _parse_clause(text, request):
    text = _trim(text)
    # This toy's verified low posture is shared by sitting and crouching requests.
    if text in ("坐下", "坐好", "坐着", "保持坐姿", "蹲下", "蹲好", "蹲着", "保持蹲姿"):
        return {"type": "posture", "pose": "sitting"}
    if text in ("站起来", "站起", "起立", "站好", "站立"):
        return {"type": "posture", "pose": "standing"}
    repeat = re.search(r"([一二两三四五六七八九十\d]+)(?:次|遍|下)$", text)
    count = int(_number(repeat[1])) if repeat else 1
    action = ALIASES.get(text[: repeat.start()] if repeat else text)
    if action:
        if count > 3:
            raise Unclear("一次最多安排同一自身动作三遍，请减少次数。")
        return {"type": "perform", "action": action, "repeat": count}
    ball = r"(?:红|黄|蓝)(?:色)?(?:的)?(?:小)?球|它|那个球|这个球"
    if re.fullmatch(rf"(?:踢|踢一下|踢一脚)(?:{ball})(?:一下|一脚|踢走|踢开)?", text) or re.fullmatch(
        rf"把(?:{ball})(?:踢|踢走|踢开|踢一下|踢一脚)", text
    ):
        return {"type": "kick", "target": _target(_color(text, request))}
    if re.fullmatch(rf"(?:走到|靠近|去到|移动到|接近)(?:{ball})(?:旁边|附近|边上|跟前)?", text):
        return {"type": "navigate", "target": _target(_color(text, request)), "standoff_m": 0.25}
    turn = re.fullmatch(
        r"(?:原地)?(?:(?:向|往|朝)?([左右])(?:边)?转|转向([左右])|转)(?:动)?"
        r"(?:(一圈|半圈)|([\d.一二两三四五六七八九十百]+)度)?",
        text,
    )
    if turn:
        direction = "right" if (turn[1] or turn[2]) == "右" else "left"
        angle = 360 if turn[3] == "一圈" else 180 if turn[3] == "半圈" else _number(turn[4]) if turn[4] else 30
        return {"type": "turn", "direction": direction, "angle_deg": angle}
    continuous = bool(re.search(r"一直|不停|不要停|别停|持续", text))
    movement = re.sub(r"一直|不要停|不停|别停|持续", "", text)
    duration = re.search(r"([\d.一二两三四五六七八九十]+)秒$", movement)
    seconds = _number(duration[1]) if duration else 0.32
    movement = movement[: duration.start()] if duration else movement
    movement = re.sub(r"(?:一点|一小步|一下)$", "", movement)
    for direction, pattern in (
        ("forward", r"前进|(?:向|往|朝)?前走|向前|往前"),
        ("backward", r"后退|退后|(?:向|往|朝)?后走|向后|往后"),
        ("left", r"左移|向左|往左|(?:向|往|朝)?左(?:边|侧)?(?:走|横移|移动|挪)"),
        ("right", r"右移|向右|往右|(?:向|往|朝)?右(?:边|侧)?(?:走|横移|移动|挪)"),
    ):
        if re.fullmatch(pattern, movement):
            if continuous and duration:
                raise Unclear("这次按指定秒数走，还是一直走到你喊停？请选一种。")
            return {
                "type": "move",
                "direction": direction,
                "limit": {"kind": "until_cancelled"} if continuous else {"kind": "duration", "seconds": seconds},
            }
    raise Unclear(f"我还不能明确理解‘{text}’。请用具体动作和目标说明；这组动作暂不启动。")


def _visual_reply(request, text):
    obs = request["observation"]
    if obs["age_s"] > 0.5:
        return "当前观察已经过期，不能据此回答现在看到什么；需要刷新相机画面。"
    objects = []
    for entity in obs["entities"]:
        attrs = {item["name"]: item["value"] for item in entity["attributes"]}
        if entity["category"] == "ball" and attrs.get("color") in COLOR_NAMES and entity["confidence"] >= 0.5:
            objects.append((COLOR_NAMES[attrs["color"]], entity["position_body_m"]))
    mentioned = [name for name in COLORS if name in text]
    if mentioned:
        objects = [(name, pos) for name, pos in objects if name in mentioned]
    if not objects:
        target = "、".join(name + "球" for name in mentioned) if mentioned else "红球、黄球或蓝球"
        return f"当前观察没有可靠检测到{target}；这不表示它们不存在。规则识别目前只覆盖这三种球。"
    if re.search(r"几个|多少", text):
        return f"当前观察记录中检测到 {len(objects)} 个目标球；这里只统计可靠识别的红、黄、蓝球。"
    if "近" in text and re.search(r"哪个|哪一个", text):
        closest = min(objects, key=lambda item: item[1][0] ** 2 + item[1][1] ** 2)
        return f"在当前可靠检测到的球中，{closest[0]}球离我最近。"
    descriptions = []
    for color, (x, y, _z) in objects:
        direction = "前方" if x >= 0 else "后方"
        direction = ("左" if y > 0.03 else "右" if y < -0.03 else "正") + direction
        descriptions.append(f"{color}球在{direction}")
    return "按我自己的朝向，当前检测到" + "，".join(descriptions) + "。当前规则只可靠识别红、黄、蓝球。"


def _goal_label(goal):
    kind = goal["type"]
    if kind == "perform":
        return LABELS[goal["action"]] + (f"{goal['repeat']}次" if goal["repeat"] > 1 else "")
    if kind in ("kick", "navigate"):
        attrs = {item["name"]: item["value"] for item in goal["target"]["attributes"]}
        color = COLOR_NAMES.get(attrs.get("color"), "指定的")
        return ("踢" if kind == "kick" else "走到") + color + "球" + ("旁边并停下" if kind == "navigate" else "")
    if kind == "posture":
        return "坐下并保持" if goal["pose"] == "sitting" else "站起来"
    if kind == "turn":
        return f"向{'左' if goal['direction'] == 'left' else '右'}转{goal['angle_deg']:g}度"
    direction = {"forward": "前", "backward": "后", "left": "左", "right": "右"}[goal["direction"]]
    limit = goal["limit"]
    return f"向{direction}走" + ("直到你喊停" if limit["kind"] == "until_cancelled" else f"{limit['seconds']:g}秒")


def _capability_reply(request):
    caps = request["capabilities"]
    names = [LABELS[action] for action in caps["self_actions"]]
    for kind, label in (
        ("move", "短时移动"),
        ("turn", "按角度转身"),
        ("navigate", "走到指定颜色的球旁"),
        ("kick", "踢指定颜色的球"),
    ):
        if kind in caps["goal_types"]:
            names.append(label)
    if caps["sit_hold"]:
        names.append("坐下并保持")
    return "当前可以" + "、".join(names) + "。也可以聊天，或根据当前检测描述红、黄、蓝球；其他开放问答仍需要语言模型。"


class ToyTeacher:
    """Deterministic small-sample teacher; no model inference or fixture lookup."""

    name = "toy_rule_teacher"
    metadata = {
        "backend": "toy_rule_teacher",
        "model_inference": False,
        "visual_scope": "observed_rgbd_red_yellow_blue_balls",
    }

    def decide(self, request):
        validate_input(request)
        output = self._decide(request)
        try:
            return validate_output(output, request)
        except ValueError as exc:
            reason = str(exc)
            if "unbounded" in reason or "until_cancelled" in reason:
                reply = "持续任务后面的动作无法执行；请先停止或替换持续任务，再安排后续动作。"
            elif "standing transition" in reason:
                reply = "当前姿态不适合直接执行；请先明确安排站起来，再做后续动作。"
            elif "Multiple objects" in reason:
                reply = "有多个球符合条件，请进一步说明目标；这组动作暂不启动。"
            else:
                reply = "当前能力或状态不支持这整组任务；我不会先执行其中一部分。请查看可用动作或简化指令。"
            return validate_output(_response(reply, "decline"), request)

    def _decide(self, request):
        if request["event"]["kind"] == "feedback":
            result = request["execution"]["last_result"]
            label = _goal_label(result["goal"])
            if result["status"] == "succeeded" and result["verified"]:
                return _response(f"{label}已经通过执行验收。")
            return _response(f"{label}未完成：{result['reason']}。")
        text = _utterance(request["event"]["text"])
        state = request["execution"]["state"]
        if re.search(NONAUTHORIZE, text):
            return _response("我理解你在提及或讨论动作；这不会启动任务。要现在执行，请直接说明。")
        if is_cancel(text):
            if re.search(VISUAL, text):
                return _response("收到，先取消任务并停止运动；停稳后的描述需要重新采样新画面。", "cancel")
            return _response("收到，正在取消任务并停止运动。", "cancel")
        if is_pause(text):
            return (
                _response("收到，当前动作完成并恢复稳定后暂停，保留剩余任务；正在进行的推理会继续。", "pause")
                if state == "running"
                else _response(
                    "当前任务已暂停，等待继续。" if state == "paused" else "当前没有正在执行的任务可以暂停。"
                )
            )
        if is_resume(text):
            return (
                _response("收到，从尚未完成的目标继续，不重复已经完成的动作。", "resume")
                if state == "paused"
                else _response("收到，取消等待暂停，当前任务继续执行。")
                if state == "running"
                else _response("当前没有暂停的任务可以恢复；请说明新的任务。", "clarify")
            )
        if re.search(VISUAL, text):
            return _response(_visual_reply(request, text))
        if re.search(r"叫什么|你是谁|名字", text):
            return _response(f"你好，我叫{request['identity']['name']}，是陪你互动的小鸭机器人。")
        if re.fullmatch(r"你好|嗨|哈喽|早上好|晚上好|下午好|小鸭你好|hello|hi|hey", text, re.IGNORECASE):
            return _response(f"你好呀，我是{request['identity']['name']}！")
        if re.fullmatch(r"谢谢(?:你)?|辛苦了|做得好|多谢|感谢你|谢谢小鸭|thanks|thankyou", text, re.IGNORECASE):
            return _response("不客气，想聊什么或做什么都可以告诉我。")
        if re.fullmatch(r"再见|拜拜|下次见|回头见|bye|goodbye", text, re.IGNORECASE):
            return _response("再见！如果还在执行任务，我会按原计划继续；要取消请明确说停下。")
        if re.search(r"会做什么|都会什么|有哪些动作|有什么本领|能做什么|有什么技能|支持哪些动作|会不会|会.+吗", text):
            return _response(_capability_reply(request))
        if re.search(BLOCKED, text):
            return _response("这组指令包含当前没有开放的动作；交互只支持踢球，我不会启动其中一部分。", "decline")
        polite = r"^(?:你)?(?:能不能|可不可以|可以)(?:请|帮我|给我|替我)?"
        if re.match(polite, text):
            text = re.sub(polite, "", text)
            text = re.sub(r"(?:吗|么)$", "", text)
        elif re.search(r"(?<!好)吗$|么$", text):
            return _response("你是在询问这个动作，还是希望我现在执行？请明确说明。", "clarify")
        positive = re.sub(r"不要停|别停|不停", "", text)
        if re.search(r"不要|不能|不许|别|禁止|不想|或者|还是|能否|怎么|如何|为什么", positive):
            return _response("我不会把否定、讨论或未确定的选项当作动作授权；请明确现在要做什么。", "clarify")
        append = bool(re.match(r"^(?:追加|接着|然后|再)", text)) and state in ("running", "paused")
        text = re.sub(r"^(?:追加|接着|然后|再)", "", text)
        text = text.replace("俯身再起", "俯身复原").replace("低头触地再起身", "俯身复原")
        clauses = re.split(r"[，,；;、]|然后|接着|随后|接下来|完成后|之后|再", text)
        try:
            goals = [_parse_clause(clause, request) for clause in clauses if clause]
        except (Unclear, ValueError) as exc:
            return _response(
                str(exc) if isinstance(exc, Unclear) else "数值没有明确匹配支持的范围，请重新说明。", "clarify"
            )
        if not goals:
            return _response("请明确你想聊的话题或要执行的动作。", "clarify")
        acknowledgements = ("好的，我会", "收到，我会", "明白了，我会")
        choice = int.from_bytes(hashlib.sha256(request["event"]["text"].encode()).digest()[:2], "big") % 3
        reply = (
            acknowledgements[choice]
            + ("追加" if append else "依次执行" if len(goals) > 1 else "执行")
            + "："
            + "，然后".join(_goal_label(g) for g in goals)
            + "。"
        )
        return _response(reply, "append" if append else "replace", goals)
