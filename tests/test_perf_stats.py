from __future__ import annotations

from threading import Thread

from vr_sbs_converter.perf_stats import FunctionTimingCollector


def test_function_timing_collector_reports_top_n_by_average_time() -> None:
    collector = FunctionTimingCollector()
    collector.record("decode", 1.0)
    collector.record("decode", 3.0)  # avg 2.0
    collector.record("depth", 10.0)
    collector.record("depth", 14.0)  # avg 12.0
    collector.record("encode", 4.0)  # avg 4.0

    top = collector.snapshot_top_n(2)

    assert [item["name"] for item in top] == ["depth", "encode"]
    assert top[0]["avg_ms"] == 12.0
    assert top[0]["count"] == 2
    assert top[0]["total_ms"] == 24.0
    assert top[0]["max_ms"] == 14.0


def test_function_timing_collector_snapshot_all_includes_all_metrics() -> None:
    collector = FunctionTimingCollector()
    collector.record("stereo", 5.0)
    collector.record("stereo", 7.0)

    snapshot = collector.snapshot_all()

    assert len(snapshot) == 1
    assert snapshot[0] == {
        "name": "stereo",
        "count": 2,
        "total_ms": 12.0,
        "avg_ms": 6.0,
        "max_ms": 7.0,
    }


def test_function_timing_collector_is_thread_safe_for_concurrent_updates() -> None:
    collector = FunctionTimingCollector()
    thread_count = 8
    updates_per_thread = 250

    def worker() -> None:
        for _ in range(updates_per_thread):
            collector.record("depth", 1.0)

    threads = [Thread(target=worker) for _ in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    snapshot = collector.snapshot_all()
    assert snapshot[0]["name"] == "depth"
    assert snapshot[0]["count"] == thread_count * updates_per_thread
    assert snapshot[0]["total_ms"] == float(thread_count * updates_per_thread)
    assert snapshot[0]["avg_ms"] == 1.0
