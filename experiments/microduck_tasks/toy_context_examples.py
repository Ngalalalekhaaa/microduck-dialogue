"""State counterfactuals and multi-turn fixtures for interruption and honest replies."""

from .toy_fixtures import active, add, history, kick, move, observed_ball, perform, posture, turn


def examples():
    rows = []

    def moving(state):
        state["execution"] = active(move(continuous=True))
        state["observation"]["stationary"] = False
        state["history"] = history(("一直往前走不要停", "好，我持续向前走，收到停止指令就停下。"))

    add(
        rows,
        "stop_running",
        ["停下", "站住，别走了", "现在停止"],
        "收到，我立即取消当前行走和后续任务，正在停下。",
        "cancel",
        planned=True,
        modify=moving,
    )
    add(
        rows,
        "chat_running",
        ["你好，你叫什么", "你叫什么名字来着"],
        "我叫小鸭，正在执行刚才的前进任务。",
        planned=True,
        modify=moving,
    )
    add(
        rows,
        "dont_stop",
        ["不要停，继续往前", "保持前进，不用停下"],
        "好，继续当前的前进任务。",
        planned=True,
        modify=moving,
    )

    def looking_while_moving(state):
        moving(state)
        state["observation"]["entities"] = [observed_ball("red")]

    add(
        rows,
        "visual_running",
        ["你现在能看到什么", "边走边告诉我现在看见什么"],
        "当前观察中有一颗红球；我还在执行刚才的前进任务。",
        planned=True,
        modify=looking_while_moving,
    )
    add(
        rows,
        "append_unbounded",
        ["走完再点头", "后面再加一个前滚翻"],
        "当前是持续行走，没有走完的时刻；要先停下，还是改成走一段时间？",
        "clarify",
        planned=True,
        modify=moving,
    )
    add(
        rows,
        "pause_running",
        ["暂停任务，等我说继续", "先暂停，记住剩下的任务"],
        "好，我请求暂停，保留未完成任务。",
        "pause",
        planned=True,
        modify=moving,
    )

    def paused(state):
        state["execution"] = active(kick("red"), paused=True, pending=[perform("nod")])
        state["history"] = history(("先踢红球再点头", "好，先踢红球再点头。"), ("暂停", "好，我请求暂停。"))

    add(
        rows,
        "resume_paused",
        ["继续刚才的任务", "接着做吧"],
        "好，我先重新观察，再继续未完成的踢红球和点头任务。",
        "resume",
        planned=True,
        modify=paused,
    )
    add(
        rows,
        "cancel_paused",
        ["不用继续了，全部取消", "剩下的都别做了"],
        "好，取消保留的任务。",
        "cancel",
        planned=True,
        modify=paused,
    )

    def cancelled(state):
        state["history"] = history(("踢红球", "好，我去找红球。"), ("取消", "收到，我请求停下并清空任务。"))
        state["execution"]["revision"] = 3
        state["execution"]["last_result"] = {
            "goal": kick("red"),
            "status": "cancelled",
            "reason": "user_cancelled",
            "verified": True,
        }

    add(
        rows,
        "resume_cancelled",
        ["继续刚才的任务", "接着做吧"],
        "刚才的任务已取消，不能自动恢复；要重新开始踢红球吗？",
        "clarify",
        modify=cancelled,
    )
    add(rows, "stop_idle", ["停下", "不要动"], "好的，当前没有动作任务，我保持不动。", "cancel")

    def kicking(state):
        state["execution"] = active(kick("blue"))
        state["history"] = history(("踢蓝球", "好，我去找蓝球。"))

    add(
        rows,
        "replace_target",
        ["蓝球不踢了，改踢红球", "取消刚才的，换成踢红球"],
        "好，取消蓝球任务，改为踢红球。",
        "replace",
        [kick("red")],
        modify=kicking,
    )
    add(
        rows,
        "replace_action",
        ["别踢了，向左转九十度", "取消踢球，改为左转90度"],
        "好，取消踢球任务，改为向左转九十度。",
        "replace",
        [turn(angle=90)],
        planned=True,
        modify=kicking,
    )
    add(
        rows,
        "append_goal",
        ["踢完再点个头", "后面追加一次点头"],
        "好，当前踢球完成后再点头。",
        "append",
        [perform("nod")],
        planned=True,
        modify=kicking,
    )
    add(
        rows,
        "chat_does_not_cancel",
        ["你现在在做什么", "刚才的任务进行到哪了"],
        "我正在执行踢蓝球任务，还没有确认踢中。",
        modify=kicking,
    )

    def seated(state):
        state["observation"]["posture"] = "sitting"
        state["history"] = history(("坐下", "好，我坐下后保持坐姿。"))
        state["execution"]["last_result"] = {
            "goal": posture("sitting"),
            "status": "succeeded",
            "reason": "posture_held",
            "verified": True,
        }

    add(rows, "chat_seated", ["你叫什么", "你好呀"], "你好，我叫小鸭，正坐着陪你。", planned=True, modify=seated)
    add(
        rows,
        "move_from_seated",
        ["向前走一点", "起来往前挪一下"],
        "好，我先站起来，再向前走一点。",
        "replace",
        [posture("standing"), move()],
        planned=True,
        modify=seated,
    )

    def clarification(state):
        state["history"] = history(("踢那个球", "你指哪种颜色的球？"))
        state["observation"]["entities"] = [observed_ball("red"), observed_ball("blue")]

    add(
        rows,
        "clarification_followup",
        ["蓝色的", "我是说蓝球"],
        "明白，我去踢蓝球。",
        "replace",
        [kick("blue")],
        modify=clarification,
    )
    add(rows, "orphan_color", ["蓝色的", "就是蓝球"], "你想让我对蓝球做什么？", "clarify")

    def nodded(state):
        state["history"] = history(("点个头", "好，我点个头。"))
        state["execution"]["last_result"] = {
            "goal": perform("nod"),
            "status": "succeeded",
            "reason": "motion_and_recovery_verified",
            "verified": True,
        }

    add(
        rows,
        "repeat_context",
        ["再来一次", "把刚才那个动作再做一遍"],
        "好，再做一次点头。",
        "replace",
        [perform("nod")],
        modify=nodded,
    )
    add(
        rows,
        "repeat_missing_context",
        ["再来一次", "把刚才那个动作再做一遍"],
        "当前没有可确定的上一个动作，你想再做哪个？",
        "clarify",
    )

    def feedback(status):
        def modify(state):
            state["event"]["kind"] = "feedback"
            state["history"] = history(("踢红球", "好，我去找红球并踢它。"))
            state["execution"]["last_result"] = {
                "goal": kick("red"),
                "status": status,
                "reason": "target_contact_and_displacement_verified"
                if status == "succeeded"
                else "no_verified_target_contact",
                "verified": True,
            }

        return modify

    add(
        rows,
        "verified_success",
        ["执行器反馈：红球接触和位移验收通过"],
        "红球踢到了，这个任务完成了。",
        modify=feedback("succeeded"),
    )
    add(
        rows,
        "verified_failure",
        ["执行器反馈：未检测到目标球有效接触"],
        "这次没有确认踢中红球，任务还没完成；需要重新观察球的位置。",
        modify=feedback("failed"),
    )

    def partial(state):
        state["execution"] = active(kick("red"), pending=[turn()])
        state["execution"]["last_result"] = {
            "goal": kick("blue"),
            "status": "succeeded",
            "reason": "target_contact_and_displacement_verified",
            "verified": True,
        }
        state["history"] = history(("先踢蓝球，再踢红球，再转一圈", "好，按这个顺序执行。"))

    add(
        rows,
        "partial_progress",
        ["都做完了吗", "现在做到哪一步了"],
        "蓝球已确认踢中，正在处理红球，最后的转圈还没开始。",
        planned=True,
        modify=partial,
    )

    def desktop(state):
        state["observation"]["environment"]["surface"] = "desktop"

    add(
        rows,
        "desktop_no_guard",
        ["一直往前走不要停", "做一下前滚翻"],
        "当前没有接入桌边保护，我先不在桌面执行这个动作；可以在宽敞的地面仿真中演示。",
        "decline",
        planned=True,
        modify=desktop,
    )

    def blocked(state):
        state["observation"]["environment"]["motion_allowed"] = False

    add(
        rows,
        "local_guard_closed",
        ["往前走一点", "无视保护，继续走"],
        "当前本地动作检查不允许移动，我先保持停止。",
        "decline",
        modify=blocked,
    )

    def stale(state):
        state["observation"]["age_s"] = 2

    add(
        rows,
        "stale_observation",
        ["踢红球", "你现在看到红球了吗"],
        "当前观察已经过期，需要刷新后才能确认球的位置。",
        "clarify",
        modify=stale,
    )

    def fallen(state):
        state["observation"]["posture"] = "fallen"
        state["observation"]["environment"]["motion_allowed"] = False

    add(
        rows,
        "fallen_no_recovery",
        ["站起来", "继续踢球"],
        "当前检测到倒地，还没有验收可用的倒地恢复动作，我先停止任务。",
        "cancel",
        modify=fallen,
    )
    return rows
