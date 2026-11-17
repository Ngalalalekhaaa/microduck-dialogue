"""Dialogue input queue whose stop signal bypasses inference and stale inputs."""

import queue
import threading

from .toy_teacher import is_cancel, is_pause, is_resume


class ToyInputDispatcher:
    def __init__(self, session, shutdown, manual_override, errors):
        self.session, self.shutdown = session, shutdown
        self.manual_override, self.errors = manual_override, errors
        self.pending = queue.Queue()
        self.stop_pending = threading.Event()
        self._submit_lock = threading.Lock()

    def submit(self, command, *, scripted=False):
        if not scripted:
            self.manual_override.set()
        completed, box = threading.Event(), []
        stopping = is_cancel(command)
        # Only admission bookkeeping is serialized, never inference or world access.
        with self._submit_lock:
            epoch = self.session.request_cancel() if stopping else self.session.cancel_epoch
            if stopping:
                self.stop_pending.set()
            elif is_pause(command):
                self.session.request_pause(expected_cancel_epoch=epoch)
            elif is_resume(command):
                self.session.request_resume(expected_cancel_epoch=epoch)
            self.pending.put((epoch, command, stopping, completed, box))
        return completed, box

    def dispatch(self):
        while True:
            item = self.pending.get()
            if item is None:
                return
            epoch, command, stopping, completed, box = item
            try:
                if epoch != self.session.cancel_epoch or (self.shutdown.is_set() and not stopping):
                    continue
                print(f"你 > {command}", flush=True)
                box.append(self.session.run(command, wait=False, expected_cancel_epoch=epoch, controls_signalled=True))
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
                self.session.request_cancel()
                self.shutdown.set()
            finally:
                with self._submit_lock:
                    if stopping and epoch == self.session.cancel_epoch:
                        self.stop_pending.clear()
                completed.set()

    def close(self):
        self.pending.put(None)
