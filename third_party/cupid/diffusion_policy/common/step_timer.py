import time
from contextlib import contextmanager


class StepTimer:
    """Accumulates named wall-clock timings for one training step.

    Usage::

        step_timer = StepTimer()

        # inside the training loop:
        step_timer.reset()
        with step_timer.time("data_transfer"):
            batch = dict_apply(batch, lambda x: x.to(device))
        with step_timer.time("forward_backward"):
            loss = model.compute_loss(batch)
            loss.backward()
        step_log.update(step_timer.to_log_dict())   # adds "timer/..." keys
    """

    def __init__(self):
        self._timings: dict = {}

    @contextmanager
    def time(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._timings[name] = time.perf_counter() - start

    def to_log_dict(self, prefix: str = "timer") -> dict:
        """Return timings as ``{prefix/name: seconds}`` ready for wandb."""
        return {f"{prefix}/{k}": v for k, v in self._timings.items()}

    def reset(self):
        self._timings.clear()
