from __future__ import annotations

import threading
from threading import Barrier, Thread
from queue import Queue

import pytest

from vr_sbs_converter.pipeline import ConversionCancelledError
from vr_sbs_converter.pipeline_parallel import (
    END_OF_STREAM_FRAME_INDEX,
    FramePacket,
    ParallelQueueConfig,
    OrderedPacketBuffer,
    ParallelFailureState,
    create_bounded_queues,
    create_cancel_event,
    create_end_of_stream_packet,
    encode_coordinator_worker,
    forward_sentinel,
    is_cancel_requested,
    request_cancel,
    run_parallel_conversion_configured,
)


def test_ordered_packet_buffer_reorders_frames() -> None:
    buffer = OrderedPacketBuffer(start_index=0)

    frame_one = FramePacket(frame_index=1, frame_payload="frame-1")
    frame_zero = FramePacket(frame_index=0, frame_payload="frame-0")
    frame_two = FramePacket(frame_index=2, frame_payload="frame-2")

    assert buffer.push(frame_one) == []
    assert [packet.frame_index for packet in buffer.push(frame_zero)] == [0, 1]
    assert [packet.frame_index for packet in buffer.push(frame_two)] == [2]


def test_end_of_stream_sentinel_is_explicit_and_forwarded_deterministically() -> None:
    queue: Queue[FramePacket] = Queue()
    sentinel = create_end_of_stream_packet(stage="depth")

    assert sentinel.is_end_of_stream is True
    assert sentinel.frame_index == END_OF_STREAM_FRAME_INDEX
    assert sentinel.stage == "depth"

    assert forward_sentinel(sentinel, queue) is True
    assert queue.get_nowait() is sentinel

    with pytest.raises(ValueError, match="end-of-stream sentinel"):
        forward_sentinel(FramePacket(frame_index=0, frame_payload="not-sentinel"), queue)


def test_parallel_failure_state_tracks_first_failure_and_requests_cancel() -> None:
    cancel_event = create_cancel_event()
    failure_state = ParallelFailureState(cancel_event=cancel_event)

    first_error = RuntimeError("decode failed")
    second_error = ValueError("encode failed")

    assert failure_state.record(first_error, stage="decode", frame_index=4) is True
    assert failure_state.record(second_error, stage="encode", frame_index=8) is False
    assert is_cancel_requested(cancel_event) is True

    failure = failure_state.get_failure()
    assert failure is not None
    assert failure.exception is first_error
    assert failure.stage == "decode"
    assert failure.frame_index == 4

    with pytest.raises(RuntimeError, match="decode failed"):
        failure_state.raise_if_failed()


def test_parallel_failure_state_record_keeps_only_first_failure_concurrently() -> None:
    cancel_event = create_cancel_event()
    failure_state = ParallelFailureState(cancel_event=cancel_event)
    worker_count = 8
    start_barrier = Barrier(worker_count)
    errors = [RuntimeError(f"worker-{index}-failed") for index in range(worker_count)]
    winners: list[bool | None] = [None] * worker_count

    def attempt_record(index: int) -> None:
        start_barrier.wait()
        winners[index] = failure_state.record(
            errors[index],
            stage=f"worker-{index}",
            frame_index=index,
        )

    threads = [Thread(target=attempt_record, args=(index,)) for index in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert winners.count(True) == 1
    assert winners.count(False) == worker_count - 1
    assert is_cancel_requested(cancel_event) is True

    winning_index = winners.index(True)
    failure = failure_state.get_failure()
    assert failure is not None
    assert failure.exception is errors[winning_index]
    assert failure.stage == f"worker-{winning_index}"
    assert failure.frame_index == winning_index


def test_ordered_packet_buffer_rejects_duplicate_frame_indices() -> None:
    buffer = OrderedPacketBuffer(start_index=0)
    buffer.push(FramePacket(frame_index=1, frame_payload="first"))

    with pytest.raises(ValueError, match="Duplicate frame index"):
        buffer.push(FramePacket(frame_index=1, frame_payload="duplicate"))


def test_parallel_queue_config_rejects_non_integer_sizes() -> None:
    with pytest.raises(ValueError, match="integers"):
        ParallelQueueConfig(decode_queue_size=1.5)
    with pytest.raises(ValueError, match="integers"):
        ParallelQueueConfig(encode_queue_size=True)


def test_forward_sentinel_rejects_invalid_frame_index() -> None:
    queue: Queue[FramePacket] = Queue()
    invalid_sentinel = create_end_of_stream_packet(stage="stereo")
    invalid_sentinel.frame_index = 0

    with pytest.raises(ValueError, match="END_OF_STREAM_FRAME_INDEX"):
        forward_sentinel(invalid_sentinel, queue)


def test_forward_sentinel_bounded_queue_is_fail_fast_or_cooperative() -> None:
    queue: Queue[FramePacket] = Queue(maxsize=1)
    queue.put_nowait(FramePacket(frame_index=0, frame_payload="already-full"))
    sentinel = create_end_of_stream_packet(stage="encode")

    with pytest.raises(RuntimeError, match="output queue is full"):
        forward_sentinel(sentinel, queue)

    cancel_event = create_cancel_event()
    request_cancel(cancel_event)

    assert forward_sentinel(sentinel, queue, cancel_event=cancel_event) is False


def test_encode_coordinator_reorders_out_of_order_packets_before_writing() -> None:
    queues = create_bounded_queues(ParallelQueueConfig(encode_queue_size=4))
    failure_state = ParallelFailureState()
    writes: list[tuple[int, str]] = []

    queues.encode.put_nowait(FramePacket(frame_index=1, sbs_payload="frame-1"))
    queues.encode.put_nowait(FramePacket(frame_index=0, sbs_payload="frame-0"))
    queues.encode.put_nowait(create_end_of_stream_packet(stage="encode"))

    frames_written = encode_coordinator_worker(
        input_queue=queues.encode,
        write_frame=lambda frame_index, payload: writes.append((frame_index, payload)),
        failure_state=failure_state,
        cancel_event=failure_state.cancel_event,
        total_frames=2,
    )

    assert frames_written == 2
    assert writes == [(0, "frame-0"), (1, "frame-1")]
    assert failure_state.get_failure() is None


def test_encode_coordinator_buffers_sparse_out_of_order_arrivals_and_reports_progress() -> None:
    queues = create_bounded_queues(ParallelQueueConfig(encode_queue_size=8))
    failure_state = ParallelFailureState()
    writes: list[tuple[int, str]] = []
    progress_events: list[dict[str, object]] = []

    class _Callbacks:
        on_start = None
        on_complete = None
        should_cancel = None

        @staticmethod
        def on_progress(payload: dict[str, object]) -> None:
            progress_events.append(payload)

    queues.encode.put_nowait(FramePacket(frame_index=2, sbs_payload="frame-2"))
    queues.encode.put_nowait(FramePacket(frame_index=0, sbs_payload="frame-0"))
    queues.encode.put_nowait(FramePacket(frame_index=3, sbs_payload="frame-3"))
    queues.encode.put_nowait(FramePacket(frame_index=1, sbs_payload="frame-1"))
    queues.encode.put_nowait(create_end_of_stream_packet(stage="encode"))

    frames_written = encode_coordinator_worker(
        input_queue=queues.encode,
        write_frame=lambda frame_index, payload: writes.append((frame_index, payload)),
        failure_state=failure_state,
        cancel_event=failure_state.cancel_event,
        callbacks=_Callbacks(),
        total_frames=4,
    )

    assert frames_written == 4
    assert writes == [
        (0, "frame-0"),
        (1, "frame-1"),
        (2, "frame-2"),
        (3, "frame-3"),
    ]
    assert [event["frame_index"] for event in progress_events] == [1, 2, 3, 4]
    assert [event["stage"] for event in progress_events] == ["converting"] * 4
    assert [event["percent"] for event in progress_events] == pytest.approx([25.0, 50.0, 75.0, 100.0])
    assert failure_state.get_failure() is None


def test_parallel_pipeline_worker_failure_requests_cancel_and_stops_downstream() -> None:
    cancel_event = create_cancel_event()
    writes: list[tuple[int, str]] = []
    frames = iter(["frame-0", "frame-1", None])

    def read_frame() -> str | None:
        return next(frames)

    def estimate_depth(frame: str) -> str:
        raise RuntimeError(f"depth failed on {frame}")

    with pytest.raises(RuntimeError, match="depth failed on frame-0"):
        run_parallel_conversion_configured(
            read_frame=read_frame,
            estimate_depth=estimate_depth,
            synthesize_stereo=lambda frame, depth: (frame, depth),
            compose_sbs=lambda stereo_payload: f"sbs:{stereo_payload[0]}:{stereo_payload[1]}",
            write_frame=lambda frame_index, payload: writes.append((frame_index, payload)),
            cancel_event=cancel_event,
        )

    assert is_cancel_requested(cancel_event) is True
    assert writes == []


def test_parallel_pipeline_sentinel_drain_joins_workers_cleanly() -> None:
    frames = iter(["frame-0", "frame-1", "frame-2", None])
    writes: list[tuple[int, str]] = []

    result = run_parallel_conversion_configured(
        read_frame=lambda: next(frames),
        estimate_depth=lambda frame: f"depth:{frame}",
        synthesize_stereo=lambda frame, depth: (f"left:{frame}", f"right:{depth}"),
        compose_sbs=lambda stereo_payload: f"sbs:{stereo_payload[0]}|{stereo_payload[1]}",
        write_frame=lambda frame_index, payload: writes.append((frame_index, payload)),
        total_frames=3,
    )

    assert writes == [
        (0, "sbs:left:frame-0|right:depth:frame-0"),
        (1, "sbs:left:frame-1|right:depth:frame-1"),
        (2, "sbs:left:frame-2|right:depth:frame-2"),
    ]
    assert result["frames_written"] == 3
    assert result["all_workers_joined"] is True
    assert result["failure"] is None
    assert result["cancel_requested"] is False


def test_parallel_pipeline_external_cancel_does_not_emit_completion() -> None:
    cancel_event = create_cancel_event()
    request_cancel(cancel_event)
    completion_events: list[dict[str, object]] = []

    class _Callbacks:
        on_start = None
        on_progress = None
        should_cancel = None

        @staticmethod
        def on_complete(payload: dict[str, object]) -> None:
            completion_events.append(payload)

    with pytest.raises(ConversionCancelledError, match="cancelled"):
        run_parallel_conversion_configured(
            read_frame=lambda: None,
            estimate_depth=lambda frame: frame,
            synthesize_stereo=lambda frame, depth: (frame, depth),
            compose_sbs=lambda stereo_payload: stereo_payload,
            write_frame=lambda _frame_index, _payload: None,
            callbacks=_Callbacks(),
            cancel_event=cancel_event,
        )

    assert completion_events == []


def test_parallel_pipeline_should_cancel_requests_cancel_and_does_not_emit_completion() -> None:
    completion_events: list[dict[str, object]] = []
    callback_thread_ids: list[int] = []
    main_thread_id = threading.get_ident()
    frames = iter(["frame-0", "frame-1", None])

    class _Callbacks:
        on_start = None
        on_progress = None

        @staticmethod
        def should_cancel() -> bool:
            callback_thread_ids.append(threading.get_ident())
            return True

        @staticmethod
        def on_complete(payload: dict[str, object]) -> None:
            completion_events.append(payload)

    with pytest.raises(ConversionCancelledError, match="cancelled"):
        run_parallel_conversion_configured(
            read_frame=lambda: next(frames),
            estimate_depth=lambda frame: f"depth:{frame}",
            synthesize_stereo=lambda frame, depth: (f"left:{frame}", f"right:{depth}"),
            compose_sbs=lambda stereo_payload: f"sbs:{stereo_payload[0]}|{stereo_payload[1]}",
            write_frame=lambda _frame_index, _payload: None,
            callbacks=_Callbacks(),
            total_frames=2,
        )

    assert callback_thread_ids
    assert set(callback_thread_ids) == {main_thread_id}
    assert completion_events == []


def test_parallel_pipeline_does_not_recheck_cancel_after_workers_finish(monkeypatch) -> None:
    frames = iter([None])
    writes: list[tuple[int, str]] = []

    monkeypatch.setattr(
        "vr_sbs_converter.pipeline_parallel.encode_coordinator_worker",
        lambda **_kwargs: 0,
    )

    def _should_cancel() -> bool:
        return threading.current_thread() is threading.main_thread()

    class _Callbacks:
        on_start = None
        on_progress = None
        on_complete = None
        should_cancel = staticmethod(_should_cancel)

    result = run_parallel_conversion_configured(
        read_frame=lambda: next(frames),
        estimate_depth=lambda frame: f"depth:{frame}",
        synthesize_stereo=lambda frame, depth: (f"left:{frame}", f"right:{depth}"),
        compose_sbs=lambda stereo_payload: f"sbs:{stereo_payload[0]}|{stereo_payload[1]}",
        write_frame=lambda frame_index, payload: writes.append((frame_index, payload)),
        callbacks=_Callbacks(),
        total_frames=1,
    )

    assert writes == []
    assert result["frames_written"] == 0
    assert result["cancel_requested"] is False
