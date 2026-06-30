from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


@dataclass(slots=True)
class _FunctionStats:
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0


class FunctionTimingCollector:
    def __init__(self) -> None:
        self._stats_by_name: dict[str, _FunctionStats] = {}
        self._lock = Lock()

    def record(self, name: str, elapsed_ms: float) -> None:
        if elapsed_ms < 0:
            raise ValueError("elapsed_ms must be non-negative.")
        with self._lock:
            stats = self._stats_by_name.get(name)
            if stats is None:
                stats = _FunctionStats()
                self._stats_by_name[name] = stats
            stats.count += 1
            stats.total_ms += elapsed_ms
            if elapsed_ms > stats.max_ms:
                stats.max_ms = elapsed_ms

    def snapshot_all(self) -> list[dict[str, float | int | str]]:
        with self._lock:
            snapshots = [
                {
                    "name": name,
                    "count": stats.count,
                    "total_ms": stats.total_ms,
                    "avg_ms": (stats.total_ms / stats.count) if stats.count > 0 else 0.0,
                    "max_ms": stats.max_ms,
                }
                for name, stats in self._stats_by_name.items()
            ]
        return sorted(
            snapshots,
            key=lambda item: (float(item["avg_ms"]), float(item["total_ms"]), str(item["name"])),
            reverse=True,
        )

    def snapshot_top_n(self, limit: int = 5) -> list[dict[str, float | int | str]]:
        if limit <= 0:
            return []
        return self.snapshot_all()[:limit]


def format_function_timing_top(entries: list[dict[str, float | int | str]]) -> str:
    if not entries:
        return "n/a"
    return " | ".join(
        f"{entry['name']} avg={float(entry['avg_ms']):.2f}ms n={int(entry['count'])}"
        for entry in entries
    )
