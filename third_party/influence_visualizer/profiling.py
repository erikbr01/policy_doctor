"""Profiling utilities for performance monitoring in the influence visualizer."""

import functools
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List

import streamlit as st


class PerformanceProfiler:
    """Context manager and decorator for profiling code performance."""

    def __init__(self):
        """Initialize profiler with empty metrics."""
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.stack: List[tuple] = []  # Stack of (name, start_time, parent_indent)

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and print results."""
        return False

    def start(self, name: str) -> None:
        """Start profiling a named section."""
        indent = len(self.stack)
        self.stack.append((name, time.perf_counter(), indent))

    def end(self, name: str) -> float:
        """End profiling a named section and return elapsed time."""
        if not self.stack or self.stack[-1][0] != name:
            raise ValueError(f"Profiler stack mismatch: expected {name}")

        section_name, start_time, indent = self.stack.pop()
        elapsed = time.perf_counter() - start_time
        self.metrics[name].append(elapsed)
        self.call_counts[name] += 1

        return elapsed

    def context(self, name: str):
        """Return a context manager for profiling a section."""
        return _ProfileContext(self, name)

    def print_summary(self) -> str:
        """Return a formatted summary of profiling results."""
        if not self.metrics:
            return "No profiling data collected."

        lines = ["\n" + "=" * 80]
        lines.append("PERFORMANCE PROFILE SUMMARY")
        lines.append("=" * 80)

        # Sort by total time (descending)
        items = [
            (
                name,
                sum(times),
                len(times),
                min(times),
                max(times),
                sum(times) / len(times),
            )
            for name, times in self.metrics.items()
        ]
        items.sort(key=lambda x: x[1], reverse=True)

        lines.append(
            f"{'Name':<50} {'Total (ms)':<12} {'Calls':<8} {'Min':<10} {'Max':<10} {'Avg':<10}"
        )
        lines.append("-" * 80)

        for name, total, count, min_time, max_time, avg_time in items:
            lines.append(
                f"{name:<50} {total * 1000:<12.2f} {count:<8} {min_time * 1000:<10.2f} {max_time * 1000:<10.2f} {avg_time * 1000:<10.2f}"
            )

        lines.append("=" * 80)
        return "\n".join(lines)

    def print_to_streamlit(self, title_suffix: str = "") -> None:
        """Print profiling summary to Streamlit sidebar.

        Use title_suffix=' (latest)' when called from a fragment so the sidebar
        updates after fragment-only runs (e.g. Learning tab "Show ranking charts").
        """
        title = "⏱️ Performance Metrics" + title_suffix
        with st.sidebar.expander(title, expanded=False):
            if not self.metrics:
                st.caption("No data yet. Interact with the app to collect metrics.")
                return
            st.text(self.print_summary())

    def reset(self) -> None:
        """Reset all collected metrics."""
        self.metrics.clear()
        self.call_counts.clear()
        self.stack.clear()


class _ProfileContext:
    """Context manager for profiling a code section."""

    def __init__(self, profiler: PerformanceProfiler, name: str):
        """Initialize context."""
        self.profiler = profiler
        self.name = name
        self.start_time = None

    def __enter__(self):
        """Enter context and start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and record elapsed time."""
        elapsed = time.perf_counter() - self.start_time
        self.profiler.metrics[self.name].append(elapsed)
        self.profiler.call_counts[self.name] += 1
        return False


def profile_function(profiler: PerformanceProfiler) -> Callable:
    """Decorator to profile a function's execution time.

    Usage:
        profiler = PerformanceProfiler()

        @profile_function(profiler)
        def my_function():
            return 42

        result = my_function()
        print(profiler.print_summary())
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with profiler.context(f"{func.__module__}.{func.__name__}"):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Global profiler instance
_global_profiler: PerformanceProfiler = None


def get_profiler() -> PerformanceProfiler:
    """Get or create the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def profile(name: str):
    """Context manager using global profiler.

    Usage:
        with profile("my_operation"):
            do_something()

        get_profiler().print_to_streamlit()
    """
    return get_profiler().context(name)
