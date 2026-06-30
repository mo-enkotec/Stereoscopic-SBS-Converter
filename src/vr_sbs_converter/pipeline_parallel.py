from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Any, Callable, Protocol

END_OF_STREAM_FRAME_INDEX = -1
_QUEUE_TIMEOUT_SECONDS = 0.05


@dataclass(slots=True)
class FramePacket:
    """Transport packet exchanged between parallel pipeline stages.

    The payload fields are intentionally optional so later tasks can progressively
    enrich packets as they move decode -> depth -> stereo -> encode.
    """

    frame_index: int
    frame_payload: Any | None = None
    depth_payload: Any | None = None
    stereo_payload: Any | None = None
    sbs_payload: Any | None = None
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    stage: str | None = None
    is_end_of_stream: bool = False

    def __post_init__(self) -> None:
        if self.is_end_of_stream:
            if self.frame_index != END_OF_STREAM_FRAME_INDEX:
                raise ValueError("End-of-stream packets must use END_OF_STREAM_FRAME_INDEX.")
            return
        if self.frame_index < 0:
            raise ValueError("Frame packets must use a non-negative frame index.")


@dataclass(frozen=True, slots=True)
class ParallelQueueConfig:
    """Bounded queue topology for the default multi-worker pipeline."""

    decode_queue_size: int = 4
    depth_queue_size: int = 4
    stereo_queue_size: int = 4
    encode_queue_size: int = 4

    def __post_init__(self) -> None:
        sizes = {
            "decode_queue_size": self.decode_queue_size,
            "depth_queue_size": self.depth_queue_size,
            "stereo_queue_size": self.stereo_queue_size,
            "encode_queue_size": self.encode_queue_size,
        }
        for name, size in sizes.items():
            if not isinstance(size, int) or isinstance(size, bool):
                raise ValueError(f"{name} must be positive integers.")
        if any(size <= 0 for size in sizes.values()):
            raise ValueError("Parallel queue sizes must be positive integers.")


DEFAULT_QUEUE_CONFIG = ParallelQueueConfig()


def create_end_of_stream_packet(stage: str) -> FramePacket:
    """Create an explicit sentinel packet marking stage completion."""

    return FramePacket(
        frame_index=END_OF_STREAM_FRAME_INDEX,
        stage=stage,
        is_end_of_stream=True,
    )


def forward_sentinel(
    packet: FramePacket,
    output_queue: Queue[FramePacket],
    *,
    cancel_event: Event | None = None,
) -> bool:
    """Forward an EOS sentinel to the next stage queue.

    Enforces that only sentinel packets can be forwarded through this helper.
    Returns True when forwarded. Returns False when cancellation was already
    requested and forwarding should stop cooperatively.
    """

    if not packet.is_end_of_stream:
        raise ValueError("forward_sentinel only accepts an end-of-stream sentinel.")
    if packet.frame_index != END_OF_STREAM_FRAME_INDEX:
        raise ValueError("Sentinel packets must use END_OF_STREAM_FRAME_INDEX.")
    if cancel_event is not None and is_cancel_requested(cancel_event):
        return False

    try:
        output_queue.put_nowait(packet)
    except Full as exc:
        if cancel_event is not None and is_cancel_requested(cancel_event):
            return False
        raise RuntimeError("Cannot forward sentinel because output queue is full.") from exc
    return True


def create_cancel_event() -> Event:
    """Create a cancellation event shared by pipeline workers."""

    return Event()


def request_cancel(cancel_event: Event) -> None:
    """Request cooperative cancellation across all workers."""

    cancel_event.set()


def is_cancel_requested(cancel_event: Event) -> bool:
    """Return whether cancellation has been requested."""

    return cancel_event.is_set()


@dataclass(frozen=True, slots=True)
class ParallelFailure:
    """Details for the first failure observed by any worker."""

    exception: BaseException
    stage: str
    frame_index: int | None = None


class ParallelFailureState:
    """Thread-safe first-failure container with cancellation propagation."""

    def __init__(self, cancel_event: Event | None = None) -> None:
        self._cancel_event = cancel_event or create_cancel_event()
        self._failure: ParallelFailure | None = None
        self._lock = Lock()

    @property
    def cancel_event(self) -> Event:
        return self._cancel_event

    def record(self, exception: BaseException, stage: str, frame_index: int | None = None) -> bool:
        """Record the first failure and trigger cancellation.

        Returns True when this call stored the first failure, False when a
        previous failure already exists.
        """

        with self._lock:
            if self._failure is not None:
                return False
            self._failure = ParallelFailure(
                exception=exception,
                stage=stage,
                frame_index=frame_index,
            )
            request_cancel(self._cancel_event)
            return True

    def get_failure(self) -> ParallelFailure | None:
        with self._lock:
            return self._failure

    def raise_if_failed(self) -> None:
        failure = self.get_failure()
        if failure is not None:
            raise failure.exception


class OrderedPacketBuffer:
    """Collect out-of-order packets and release them in ascending frame order."""

    def __init__(self, start_index: int = 0) -> None:
        if start_index < 0:
            raise ValueError("OrderedPacketBuffer start index must be non-negative.")
        self._next_index = start_index
        self._pending_by_index: dict[int, FramePacket] = {}

    @property
    def next_index(self) -> int:
        return self._next_index

    def push(self, packet: FramePacket) -> list[FramePacket]:
        """Add a packet and return all now-ready packets in deterministic order."""

        if packet.is_end_of_stream:
            raise ValueError("OrderedPacketBuffer only accepts non-sentinel frame packets.")
        if packet.frame_index < self._next_index:
            raise ValueError(f"Duplicate frame index received: {packet.frame_index}")
        if packet.frame_index in self._pending_by_index:
            raise ValueError(f"Duplicate frame index received: {packet.frame_index}")
        self._pending_by_index[packet.frame_index] = packet

        ready: list[FramePacket] = []
        while self._next_index in self._pending_by_index:
            ready.append(self._pending_by_index.pop(self._next_index))
            self._next_index += 1
        return ready


@dataclass(slots=True)
class ParallelStageQueues:
    decode: Queue[FramePacket]
    depth: Queue[FramePacket]
    stereo: Queue[FramePacket]
    encode: Queue[FramePacket]


def create_bounded_queues(config: ParallelQueueConfig = DEFAULT_QUEUE_CONFIG) -> ParallelStageQueues:
    """Create queue topology for decode -> depth -> stereo -> encode stages."""

    return ParallelStageQueues(
        decode=Queue(maxsize=config.decode_queue_size),
        depth=Queue(maxsize=config.depth_queue_size),
        stereo=Queue(maxsize=config.stereo_queue_size),
        encode=Queue(maxsize=config.encode_queue_size),
    )


class ParallelCallbacks(Protocol):
    on_start: Callable[[dict[str, Any]], None] | None
    on_progress: Callable[[dict[str, Any]], None] | None
    on_complete: Callable[[dict[str, Any]], None] | None
    should_cancel: Callable[[], bool] | None


def _put_with_backpressure(
    output_queue: Queue[FramePacket],
    packet: FramePacket,
    *,
    cancel_event: Event,
    should_cancel: Callable[[], bool] | None = None,
) -> bool:
    while True:
        if should_cancel is not None and should_cancel():
            request_cancel(cancel_event)
            return False
        if is_cancel_requested(cancel_event):
            return False
        try:
            output_queue.put(packet, timeout=_QUEUE_TIMEOUT_SECONDS)
            return True
        except Full:
            continue


def _get_with_backpressure(
    input_queue: Queue[FramePacket],
    *,
    cancel_event: Event,
    should_cancel: Callable[[], bool] | None = None,
) -> FramePacket | None:
    while True:
        if should_cancel is not None and should_cancel():
            request_cancel(cancel_event)
            return None
        if is_cancel_requested(cancel_event):
            return None
        try:
            return input_queue.get(timeout=_QUEUE_TIMEOUT_SECONDS)
        except Empty:
            continue


def decode_worker(
    *,
    read_frame: Callable[[], Any | None],
    output_queue: Queue[FramePacket],
    failure_state: ParallelFailureState,
    cancel_event: Event,
    should_cancel: Callable[[], bool] | None = None,
) -> int:
    frame_index = 0
    try:
        while True:
            if should_cancel is not None and should_cancel():
                request_cancel(cancel_event)
                return frame_index
            if is_cancel_requested(cancel_event):
                return frame_index
            decode_started = perf_counter()
            frame_payload = read_frame()
            decode_elapsed_ms = (perf_counter() - decode_started) * 1000.0
            if frame_payload is None:
                sentinel = create_end_of_stream_packet(stage="decode")
                _put_with_backpressure(
                    output_queue,
                    sentinel,
                    cancel_event=cancel_event,
                    should_cancel=should_cancel,
                )
                return frame_index
            packet = FramePacket(
                frame_index=frame_index,
                frame_payload=frame_payload,
                stage_timings_ms={"decode": decode_elapsed_ms},
                stage="decode",
            )
            if not _put_with_backpressure(
                output_queue,
                packet,
                cancel_event=cancel_event,
                should_cancel=should_cancel,
            ):
                return frame_index
            frame_index += 1
    except BaseException as exc:
        failure_state.record(exc, stage="decode", frame_index=frame_index)
        return frame_index


def depth_worker(
    *,
    input_queue: Queue[FramePacket],
    output_queue: Queue[FramePacket],
    estimate_depth: Callable[[Any], Any],
    failure_state: ParallelFailureState,
    cancel_event: Event,
    should_cancel: Callable[[], bool] | None = None,
) -> int:
    processed = 0
    try:
        while True:
            packet = _get_with_backpressure(
                input_queue,
                cancel_event=cancel_event,
                should_cancel=should_cancel,
            )
            if packet is None:
                return processed
            if packet.is_end_of_stream:
                _put_with_backpressure(
                    output_queue,
                    create_end_of_stream_packet(stage="depth"),
                    cancel_event=cancel_event,
                    should_cancel=should_cancel,
                )
                return processed
            depth_started = perf_counter()
            depth_payload = estimate_depth(packet.frame_payload)
            depth_elapsed_ms = (perf_counter() - depth_started) * 1000.0
            stage_timings_ms = dict(packet.stage_timings_ms)
            stage_timings_ms["depth"] = depth_elapsed_ms
            output_packet = FramePacket(
                frame_index=packet.frame_index,
                frame_payload=packet.frame_payload,
                depth_payload=depth_payload,
                stage_timings_ms=stage_timings_ms,
                stage="depth",
            )
            if not _put_with_backpressure(
                output_queue,
                output_packet,
                cancel_event=cancel_event,
                should_cancel=should_cancel,
            ):
                return processed
            processed += 1
    except BaseException as exc:
        frame_index = packet.frame_index if "packet" in locals() and not packet.is_end_of_stream else None
        failure_state.record(exc, stage="depth", frame_index=frame_index)
        return processed


def stereo_worker(
    *,
    input_queue: Queue[FramePacket],
    output_queue: Queue[FramePacket],
    synthesize_stereo: Callable[[Any, Any], Any],
    failure_state: ParallelFailureState,
    cancel_event: Event,
    should_cancel: Callable[[], bool] | None = None,
) -> int:
    processed = 0
    try:
        while True:
            packet = _get_with_backpressure(
                input_queue,
                cancel_event=cancel_event,
                should_cancel=should_cancel,
            )
            if packet is None:
                return processed
            if packet.is_end_of_stream:
                _put_with_backpressure(
                    output_queue,
                    create_end_of_stream_packet(stage="stereo"),
                    cancel_event=cancel_event,
                    should_cancel=should_cancel,
                )
                return processed
            stereo_started = perf_counter()
            stereo_payload = synthesize_stereo(packet.frame_payload, packet.depth_payload)
            stereo_elapsed_ms = (perf_counter() - stereo_started) * 1000.0
            stage_timings_ms = dict(packet.stage_timings_ms)
            stage_timings_ms["stereo"] = stereo_elapsed_ms
            output_packet = FramePacket(
                frame_index=packet.frame_index,
                frame_payload=packet.frame_payload,
                depth_payload=packet.depth_payload,
                stereo_payload=stereo_payload,
                stage_timings_ms=stage_timings_ms,
                stage="stereo",
            )
            if not _put_with_backpressure(
                output_queue,
                output_packet,
                cancel_event=cancel_event,
                should_cancel=should_cancel,
            ):
                return processed
            processed += 1
    except BaseException as exc:
        frame_index = packet.frame_index if "packet" in locals() and not packet.is_end_of_stream else None
        failure_state.record(exc, stage="stereo", frame_index=frame_index)
        return processed


def encode_worker(
    *,
    input_queue: Queue[FramePacket],
    output_queue: Queue[FramePacket],
    compose_sbs: Callable[[Any], Any],
    failure_state: ParallelFailureState,
    cancel_event: Event,
    should_cancel: Callable[[], bool] | None = None,
) -> int:
    processed = 0
    try:
        while True:
            packet = _get_with_backpressure(
                input_queue,
                cancel_event=cancel_event,
                should_cancel=should_cancel,
            )
            if packet is None:
                return processed
            if packet.is_end_of_stream:
                _put_with_backpressure(
                    output_queue,
                    create_end_of_stream_packet(stage="encode"),
                    cancel_event=cancel_event,
                    should_cancel=should_cancel,
                )
                return processed
            encode_started = perf_counter()
            sbs_payload = compose_sbs(packet.stereo_payload)
            encode_elapsed_ms = (perf_counter() - encode_started) * 1000.0
            stage_timings_ms = dict(packet.stage_timings_ms)
            stage_timings_ms["encode"] = encode_elapsed_ms
            output_packet = FramePacket(
                frame_index=packet.frame_index,
                frame_payload=packet.frame_payload,
                depth_payload=packet.depth_payload,
                stereo_payload=packet.stereo_payload,
                sbs_payload=sbs_payload,
                stage_timings_ms=stage_timings_ms,
                stage="encode",
            )
            if not _put_with_backpressure(
                output_queue,
                output_packet,
                cancel_event=cancel_event,
                should_cancel=should_cancel,
            ):
                return processed
            processed += 1
    except BaseException as exc:
        frame_index = packet.frame_index if "packet" in locals() and not packet.is_end_of_stream else None
        failure_state.record(exc, stage="encode", frame_index=frame_index)
        return processed


def encode_coordinator_worker(
    *,
    input_queue: Queue[FramePacket],
    write_frame: Callable[[int, Any], None],
    failure_state: ParallelFailureState,
    cancel_event: Event,
    callbacks: ParallelCallbacks | None = None,
    total_frames: int | None = None,
) -> int:
    ordered_buffer = OrderedPacketBuffer(start_index=0)
    frames_written = 0
    try:
        while True:
            should_cancel = callbacks.should_cancel if callbacks is not None else None
            packet = _get_with_backpressure(
                input_queue,
                cancel_event=cancel_event,
                should_cancel=should_cancel,
            )
            if packet is None:
                return frames_written
            if packet.is_end_of_stream:
                return frames_written
            for ready_packet in ordered_buffer.push(packet):
                write_frame(ready_packet.frame_index, ready_packet.sbs_payload)
                frames_written += 1
                if callbacks is not None and callbacks.on_progress is not None:
                    percent = 0.0
                    if total_frames is not None and total_frames > 0:
                        percent = (frames_written / total_frames) * 100.0
                    callbacks.on_progress(
                        {
                            "frame_index": frames_written,
                            "total_frames": total_frames,
                            "percent": percent,
                            "stage": "converting",
                        }
                    )
    except BaseException as exc:
        frame_index = packet.frame_index if "packet" in locals() and not packet.is_end_of_stream else None
        failure_state.record(exc, stage="coordinator", frame_index=frame_index)
        return frames_written


def _create_cancelled_error() -> RuntimeError:
    try:
        from .pipeline import ConversionCancelledError

        return ConversionCancelledError("Conversion cancelled by user.")
    except Exception:
        return RuntimeError("Conversion cancelled by user.")


def run_parallel_conversion_configured(
    *,
    read_frame: Callable[[], Any | None],
    estimate_depth: Callable[[Any], Any],
    synthesize_stereo: Callable[[Any, Any], Any],
    compose_sbs: Callable[[Any], Any],
    write_frame: Callable[[int, Any], None],
    queue_config: ParallelQueueConfig = DEFAULT_QUEUE_CONFIG,
    callbacks: ParallelCallbacks | None = None,
    total_frames: int | None = None,
    cancel_event: Event | None = None,
) -> dict[str, Any]:
    """Run a deterministic parallel conversion flow with injected stage callables."""

    shared_cancel_event = cancel_event or create_cancel_event()
    failure_state = ParallelFailureState(cancel_event=shared_cancel_event)
    queues = create_bounded_queues(queue_config)

    if callbacks is not None and callbacks.on_start is not None:
        callbacks.on_start(
            {
                "total_frames": total_frames,
                "mode": "parallel",
            }
        )

    coordinator_result: dict[str, int] = {"frames_written": 0}

    worker_threads = [
        Thread(
            target=decode_worker,
            kwargs={
                "read_frame": read_frame,
                "output_queue": queues.decode,
                "failure_state": failure_state,
                "cancel_event": shared_cancel_event,
                "should_cancel": None,
            },
            name="parallel-decode-worker",
        ),
        Thread(
            target=depth_worker,
            kwargs={
                "input_queue": queues.decode,
                "output_queue": queues.depth,
                "estimate_depth": estimate_depth,
                "failure_state": failure_state,
                "cancel_event": shared_cancel_event,
                "should_cancel": None,
            },
            name="parallel-depth-worker",
        ),
        Thread(
            target=stereo_worker,
            kwargs={
                "input_queue": queues.depth,
                "output_queue": queues.stereo,
                "synthesize_stereo": synthesize_stereo,
                "failure_state": failure_state,
                "cancel_event": shared_cancel_event,
                "should_cancel": None,
            },
            name="parallel-stereo-worker",
        ),
        Thread(
            target=encode_worker,
            kwargs={
                "input_queue": queues.stereo,
                "output_queue": queues.encode,
                "compose_sbs": compose_sbs,
                "failure_state": failure_state,
                "cancel_event": shared_cancel_event,
                "should_cancel": None,
            },
            name="parallel-encode-worker",
        ),
    ]

    for thread in worker_threads:
        thread.start()

    coordinator_result["frames_written"] = encode_coordinator_worker(
        input_queue=queues.encode,
        write_frame=write_frame,
        failure_state=failure_state,
        cancel_event=shared_cancel_event,
        callbacks=callbacks,
        total_frames=total_frames,
    )

    for thread in worker_threads:
        thread.join(timeout=5.0)

    all_workers_joined = all(not thread.is_alive() for thread in worker_threads)
    if not all_workers_joined:
        request_cancel(shared_cancel_event)
        for thread in worker_threads:
            thread.join(timeout=5.0)
        all_workers_joined = all(not thread.is_alive() for thread in worker_threads)
        if not all_workers_joined:
            failure_state.record(RuntimeError("Parallel workers did not terminate cleanly."), stage="join")

    failure = failure_state.get_failure()
    if failure is not None:
        raise failure.exception
    if is_cancel_requested(shared_cancel_event):
        raise _create_cancelled_error()

    if callbacks is not None and callbacks.on_complete is not None:
        callbacks.on_complete(
            {
                "frames_processed": coordinator_result["frames_written"],
                "effective_fps": 0.0,
                "mode": "parallel",
            }
        )

    return {
        "frames_written": coordinator_result["frames_written"],
        "all_workers_joined": all_workers_joined,
        "failure": failure,
        "cancel_requested": is_cancel_requested(shared_cancel_event),
    }
