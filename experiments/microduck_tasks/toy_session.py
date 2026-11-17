"""Actual dialogue and semantic queues with cancellation generations and RGB logs."""

import copy
import threading
import time

from .toy_goal_runner import ToyGoalRunner
from .toy_live_inputs import request
from .toy_protocol import VERSION, validate_output
from .toy_recording import ToyRecorder
from .toy_teacher import ToyTeacher, is_cancel, is_pause, is_resume


class ToySession:
    def __init__(self, runtime, *, seed=0, split_group="manual", backend=None):
        self.runtime, self.seed = runtime, seed
        self.backend = backend or ToyTeacher()
        self.recorder = ToyRecorder(
            runtime.output,
            episode_id=runtime.output.name,
            seed=seed,
            split_group=split_group,
            controller_source=getattr(self.backend, "name", "toy_rule_teacher"),
            training_ready=getattr(self.backend, "training_ready", True),
        )
        if not self.recorder.metadata["training_ready"]:
            self.recorder.metadata["controller_metadata"] = copy.deepcopy(self.backend.metadata)
            self.recorder.save_metadata()
        self.dialogue_lock, self.state_lock, self.user_lock = threading.RLock(), threading.RLock(), threading.Lock()
        self._cancel_lock = threading.Lock()
        self._cancel_epoch, self._cancel_latched = 0, False
        self._pause_requested = threading.Event()
        self.execution = dict(state="idle", revision=0, active=None, pending=[], last_result=None)
        self.history, self.results = [], []
        self.worker = None
        self.task_seq = self.turn_seq = self.attempt_seq = self.goal_index = 0
        self.task_id = "task_000"
        self.instruction = "和小鸭聊天，或安排一个动作任务"
        self.last_reply = ""
        self.task_instruction = ""
        self.worker_error = None
        self.interrupted_result = None
        self.runner = ToyGoalRunner(self)

    @property
    def cancel_epoch(self):
        with self._cancel_lock:
            return self._cancel_epoch

    @staticmethod
    def _cancel_response(kind="user"):
        return dict(
            schema_version=VERSION,
            reply="收到，正在取消任务并停止运动。",
            decision="cancel" if kind == "user" else "respond",
            goals=[],
        )

    def request_cancel(self):
        """Latch cancellation without waiting for inference, motion, or session/world locks."""
        with self._cancel_lock:
            self._cancel_epoch += 1
            self._cancel_latched = True
            self._pause_requested.clear()
            self.runtime.cancel.set()
            self.last_reply = self._cancel_response()["reply"]
            return self._cancel_epoch

    @property
    def pause_requested(self):
        return self._pause_requested.is_set()

    def request_pause(self, *, expected_cancel_epoch=None):
        """Request a safe boundary without cancelling motion or in-flight inference."""
        with self._cancel_lock:
            current = expected_cancel_epoch is None or expected_cancel_epoch == self._cancel_epoch
            if current and not self._cancel_latched:
                self._pause_requested.set()
                self.last_reply = "收到，等待当前动作完成并恢复稳定后暂停；正在进行的推理会继续。"
            return self._cancel_epoch

    def request_resume(self, *, expected_cancel_epoch=None):
        """Withdraw a pending pause; only a validated resume can start a paused worker."""
        with self._cancel_lock:
            if expected_cancel_epoch is None or expected_cancel_epoch == self._cancel_epoch:
                self._pause_requested.clear()
            return self._cancel_epoch

    def _clock(self):
        return self.runtime.snapshot()["time_s"]

    def _dialogue(
        self, text, *, kind="user", result_ref=None, cancel_epoch=None, force_cancel=False, host_control=None
    ):
        if cancel_epoch is None:
            cancel_epoch = self.cancel_epoch
        if getattr(self.backend, "requires_lockstep", False):
            # Callers own dialogue_lock; keep the same state -> world ordering as goal feedback.
            with self.state_lock, self.runtime.lock:
                return self._captured_dialogue(
                    text,
                    kind=kind,
                    result_ref=result_ref,
                    cancel_epoch=cancel_epoch,
                    force_cancel=force_cancel,
                    host_control=host_control,
                )
        return self._captured_dialogue(
            text,
            kind=kind,
            result_ref=result_ref,
            cancel_epoch=cancel_epoch,
            force_cancel=force_cancel,
            host_control=host_control,
        )

    def _captured_dialogue(self, text, *, kind, result_ref, cancel_epoch, force_cancel, host_control):
        observation, images, _ = self.runtime.observe()
        with self.state_lock:
            state = request(
                self.runtime, observation, text=text, kind=kind, history=self.history, execution=self.execution
            )
            task_id, revision = self.task_id, self.execution["revision"]
        image_decider = getattr(self.backend, "decide_with_images", None)
        host_cancel = force_cancel or cancel_epoch != self.cancel_epoch
        backend_called = False
        if host_cancel:
            predicted = self._cancel_response(kind)
        elif host_control is not None:
            predicted = ToyTeacher().decide(state)
        else:
            try:
                backend_called = True
                predicted = image_decider(state, images) if image_decider is not None else self.backend.decide(state)
            except Exception:
                if cancel_epoch == self.cancel_epoch:
                    raise
                predicted = self._cancel_response(kind)
        if cancel_epoch != self.cancel_epoch:
            host_cancel = True
            predicted = self._cancel_response(kind)
        response = validate_output(predicted, state)
        if kind == "feedback" and response["decision"] != "respond":
            raise ValueError("Trusted feedback may report results but cannot silently start a new task.")
        self.turn_seq += 1
        media, hashes = self.recorder.images(observation, images)
        trace = getattr(self.backend, "last_trace", None)
        if host_cancel:
            superseded = copy.deepcopy(trace) if backend_called and trace and trace.get("model_called") else None
            trace = dict(
                decision_source="host_stop" if force_cancel else "host_cancel_superseded",
                model_called=superseded is not None,
                raw_prediction=None,
                response=copy.deepcopy(response),
                reply_source="template",
            )
            if superseded is not None:
                trace["superseded_model_trace"] = superseded
            if hasattr(self.backend, "last_trace"):
                self.backend.last_trace = copy.deepcopy(trace)
        elif host_control is not None:
            trace = dict(decision_source=f"host_{host_control}", model_called=False, response=copy.deepcopy(response))
            if hasattr(self.backend, "last_trace"):
                self.backend.last_trace = copy.deepcopy(trace)
        row = self.recorder.append(
            "dialogue",
            turn_id=f"turn_{self.turn_seq:05d}",
            task_id=task_id,
            revision=revision,
            request=state,
            response=response,
            media=media,
            media_sha256=hashes,
            observation_id=observation["frame_id"],
            observation_time_s=observation["time_s"],
            time_s=observation["time_s"] + state["observation"]["age_s"],
            result_ref=result_ref,
            instruction=text if kind == "user" else self.instruction,
            **({"model_trace": copy.deepcopy(trace)} if trace is not None else {}),
        )
        with self._cancel_lock:
            if cancel_epoch == self._cancel_epoch:
                if kind == "user":
                    self.history.append(dict(role="user", text=text))
                    self.instruction = text
                self.history.append(dict(role="assistant", text=response["reply"]))
                self.history = self.history[-12:]
                self.last_reply = response["reply"]
        if cancel_epoch == self.cancel_epoch:
            print(f"小鸭 > {self.last_reply}", flush=True)
        return row

    def _interrupt(self):
        with self.state_lock:
            self.execution["revision"] += 1
            worker = self.worker
            self.interrupted_result = None
        self.runtime.request_stop()
        deadline = time.monotonic() + 20
        while worker is not None and worker.is_alive():
            self.runtime.request_stop()
            worker.join(0.02)
            if time.monotonic() > deadline:
                raise TimeoutError("Cancelled worker did not exit; no subsequent task may start.")
        self.worker = None
        if self.interrupted_result is not None:
            with self.state_lock:
                self.execution["last_result"] = self._result_state(self.interrupted_result)

    def _start(self, cancel_epoch):
        with self.state_lock:
            revision, task_id = self.execution["revision"], self.task_id
            worker = threading.Thread(
                target=self._guarded_work, args=(revision, task_id, cancel_epoch), name="toy-goal-queue", daemon=True
            )
            with self._cancel_lock:
                if cancel_epoch != self._cancel_epoch:
                    return False
                self.runtime.cancel.clear()
                self._cancel_latched = False
                self.worker_error = None
                self.worker = worker
                worker.start()
                return True

    def _clear_cancelled_queue(self):
        self._interrupt()
        with self.state_lock:
            self.execution.update(state="idle", active=None, pending=[])
            if self.interrupted_result is not None:
                self.execution["last_result"] = self._result_state(self.interrupted_result)

    def run(self, text, *, wait=True, expected_cancel_epoch=None, controls_signalled=False):
        """Answer one utterance; a running task does not block unrelated conversation."""
        if expected_cancel_epoch is not None and (type(expected_cancel_epoch) is not int or expected_cancel_epoch < 0):
            raise ValueError("expected_cancel_epoch must be a nonnegative integer or None.")
        immediate_cancel = is_cancel(text)
        immediate_pause, immediate_resume = is_pause(text), is_resume(text)
        if expected_cancel_epoch is not None and expected_cancel_epoch != self.cancel_epoch:
            return self._cancel_response()
        if immediate_cancel:
            with self._cancel_lock:
                already_latched = expected_cancel_epoch == self._cancel_epoch and self._cancel_latched
            cancel_epoch = expected_cancel_epoch if already_latched else self.request_cancel()
        else:
            cancel_epoch = self.cancel_epoch if expected_cancel_epoch is None else expected_cancel_epoch
            if immediate_pause and not controls_signalled:
                self.request_pause(expected_cancel_epoch=cancel_epoch)
            elif immediate_resume and not controls_signalled:
                self.request_resume(expected_cancel_epoch=cancel_epoch)
        with self.user_lock:
            if cancel_epoch != self.cancel_epoch:
                return self._cancel_response()
            with self._cancel_lock:
                pending_cancel = self._cancel_latched
            if pending_cancel and not immediate_cancel:
                self._clear_cancelled_queue()
                with self._cancel_lock:
                    if cancel_epoch == self._cancel_epoch:
                        self._cancel_latched = False
            if immediate_cancel:
                self._clear_cancelled_queue()
            with self.dialogue_lock:
                row = self._dialogue(
                    text,
                    cancel_epoch=cancel_epoch,
                    force_cancel=immediate_cancel,
                    host_control="pause" if immediate_pause else "resume" if immediate_resume else None,
                )
                response = row["response"]
                decision = response["decision"]
                if decision == "append":
                    with self.state_lock:
                        pending = self.execution["pending"] + copy.deepcopy(response["goals"])
                        with self._cancel_lock:
                            if cancel_epoch == self._cancel_epoch:
                                self.execution["pending"] = pending
            if cancel_epoch != self.cancel_epoch:
                self._clear_cancelled_queue()
                return self._cancel_response()
            decision = response["decision"]
            if decision == "replace":
                self._interrupt()
                goals = copy.deepcopy(response["goals"])
                pending = goals[1:]
                with self.state_lock, self._cancel_lock:
                    if cancel_epoch == self._cancel_epoch:
                        self.task_seq += 1
                        self.task_id = f"task_{self.task_seq:03d}"
                        self.task_instruction = text
                        self.goal_index = 0
                        self.execution.update(
                            state="running",
                            active=goals[0],
                            pending=pending,
                        )
                if not self._start(cancel_epoch):
                    self._clear_cancelled_queue()
                    return self._cancel_response()
            elif decision in ("cancel", "pause"):
                if decision == "pause":
                    if not immediate_pause:
                        self.request_pause(expected_cancel_epoch=cancel_epoch)
                elif not immediate_cancel:
                    self.request_cancel()
                    self._clear_cancelled_queue()
                self.recorder.append(
                    "queue_control",
                    task_id=self.task_id,
                    revision=self.execution["revision"],
                    decision=decision,
                    time_s=self._clock(),
                )
            elif decision == "resume":
                if not immediate_resume:
                    self.request_resume(expected_cancel_epoch=cancel_epoch)
                with self.state_lock, self._cancel_lock:
                    if cancel_epoch == self._cancel_epoch:
                        self.execution["revision"] += 1
                        self.execution["state"] = "running"
                if not self._start(cancel_epoch):
                    self._clear_cancelled_queue()
                    return self._cancel_response()
            elif immediate_pause and self.execution["active"] is None:
                self.request_resume(expected_cancel_epoch=cancel_epoch)
        if wait and (decision == "pause" or immediate_pause):
            self.wait()
        if wait and decision in ("replace", "append", "resume"):
            with self.state_lock:
                goals = [self.execution["active"], *self.execution["pending"]]
                continuous = any(g and g["type"] == "move" and g["limit"]["kind"] == "until_cancelled" for g in goals)
            if not continuous:
                self.wait()
        return response

    def _guarded_work(self, revision, task_id, cancel_epoch):
        try:
            self._work(revision, task_id, cancel_epoch)
        except Exception as exc:
            self.runtime.request_stop()
            with self.state_lock:
                self.worker_error = exc
                self.execution.update(state="failed", active=None, pending=[])
                self._pause_requested.clear()
            print(f"任务记录失败：{type(exc).__name__}: {exc}", flush=True)

    @staticmethod
    def _result_state(row):
        return {key: copy.deepcopy(row[key]) for key in ("goal", "status", "reason", "verified")}

    def _record_pause_boundary(self, cancel_epoch, metrics):
        self.recorder.append(
            "queue_control",
            task_id=self.task_id,
            revision=self.execution["revision"],
            decision="pause",
            phase="safe_boundary_reached",
            metrics=metrics,
            time_s=self._clock(),
        )
        with self._cancel_lock:
            if cancel_epoch == self._cancel_epoch:
                self.last_reply = "已在稳定的动作边界暂停，剩余任务已保留；说继续后接着执行。"

    def _pause_before_goal(self, cancel_epoch):
        result = self.runner.settle_for_pause()
        with self.dialogue_lock, self.state_lock:
            with self._cancel_lock:
                if cancel_epoch != self._cancel_epoch or self.runtime.cancel.is_set():
                    self.execution.update(state="idle", active=None, pending=[])
                    return True
                if not result["success"]:
                    self.execution.update(state="failed", active=None, pending=[])
                    self._pause_requested.clear()
                    self.last_reply = "无法确认暂停时的稳定状态，任务已终止；请检查机器人状态。"
                elif self._pause_requested.is_set():
                    self.execution["state"] = "paused"
                else:
                    return False
            if not result["success"]:
                self.runtime.request_stop()
                self.recorder.append("pause_failed", reason=result["reason"], metrics=result.get("metrics", {}))
            else:
                self._record_pause_boundary(cancel_epoch, result.get("metrics", {}))
            return True

    def _work(self, revision, task_id, cancel_epoch):
        while True:
            with self.dialogue_lock, self.state_lock:
                if cancel_epoch != self.cancel_epoch:
                    self.execution.update(state="idle", active=None, pending=[])
                    return
                if revision != self.execution["revision"] or self.execution["state"] != "running":
                    return
                goal, index = copy.deepcopy(self.execution["active"]), self.goal_index
                if goal is None:
                    return
                pause_before_goal = self.pause_requested
                if not pause_before_goal:
                    self.recorder.append(
                        "goal_start",
                        task_id=task_id,
                        goal_index=index,
                        goal=goal,
                        revision=revision,
                        time_s=self._clock(),
                        instruction=self.task_instruction,
                    )
            if pause_before_goal:
                if self._pause_before_goal(cancel_epoch):
                    return
                continue
            try:
                result = self.runner.execute(goal, task_id, index)
                if result["success"] and self.pause_requested:
                    settled = self.runner.settle_for_pause()
                    result["metrics"]["pause_recovery"] = settled
                    if not settled["success"]:
                        result.update(success=False, reason=settled["reason"])
            except Exception as exc:
                self.runtime.request_stop()
                result = dict(
                    success=False, reason="execution_exception", metrics=dict(error=f"{type(exc).__name__}: {exc}")
                )
            with self.dialogue_lock, self.state_lock:
                with self._cancel_lock:
                    host_cancelled = cancel_epoch != self._cancel_epoch
                    interrupted = revision != self.execution["revision"] or host_cancelled
                    local_stop = self.runtime.cancel.is_set()
                    success = bool(result["success"]) and not (interrupted or local_stop)
                    safely_paused = bool(result.get("paused")) and not (interrupted or local_stop)
                row = self.recorder.append(
                    "goal_result",
                    task_id=task_id,
                    goal_index=index,
                    goal=goal,
                    revision=revision,
                    success=success,
                    status="cancelled"
                    if interrupted
                    else "blocked"
                    if safely_paused
                    else "succeeded"
                    if success
                    else "failed",
                    reason="user_or_local_stop" if interrupted else result["reason"],
                    verified=True,
                    metrics=result.get("metrics", {}),
                    time_s=self._clock(),
                    instruction=self.task_instruction,
                )
                self.results.append(row)
                if interrupted:
                    self.interrupted_result = row
                    if host_cancelled:
                        self.execution.update(
                            state="idle", active=None, pending=[], last_result=self._result_state(row)
                        )
                    return
                self.execution["last_result"] = self._result_state(row)
                if safely_paused:
                    self.execution["state"] = "paused" if self.pause_requested else "running"
                elif success and self.execution["pending"]:
                    self.execution["active"] = self.execution["pending"].pop(0)
                    self.goal_index += 1
                    if self.pause_requested and result.get("metrics", {}).get("pause_recovery", {}).get("success"):
                        self.execution["state"] = "paused"
                else:
                    self.execution.update(state="idle" if success else "failed", active=None, pending=[])
                    self._pause_requested.clear()
                self._dialogue(
                    f"执行器反馈：{row['status']}；{row['reason']}",
                    kind="feedback",
                    result_ref=row["event_id"],
                    cancel_epoch=cancel_epoch,
                )
                if cancel_epoch != self.cancel_epoch:
                    self.execution.update(state="idle", active=None, pending=[])
                    return
                if self.execution["state"] == "paused":
                    self._record_pause_boundary(cancel_epoch, result.get("metrics", {}))
                    return
                if self.execution["state"] != "running":
                    return

    def wait(self, timeout=600):
        worker = self.worker
        if worker is not None:
            worker.join(timeout)
            if worker.is_alive():
                self._interrupt()
                raise TimeoutError("Task exceeded the pilot time budget.")
        if self.worker_error is not None:
            raise RuntimeError("Goal worker failed; inspect its original exception.") from self.worker_error

    def close(self):
        self._interrupt()
        with self.state_lock:
            self.execution.update(state="idle", active=None, pending=[])
