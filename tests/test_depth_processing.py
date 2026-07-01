import cv2
import inspect
import numpy as np

from vr_sbs_converter.depth import condition_depth_for_stereo


def _import_torch_or_skip():
    try:
        import torch

        return torch
    except Exception:
        import pytest

        pytest.skip("torch unavailable")


def _install_fake_midas_modules(monkeypatch, compile_fn=None, *, has_compile: bool = True):
    import sys
    import types

    class FakeTensor:
        def view(self, *_args):
            return self

        def to(self, *_args, **_kwargs):
            return self

    class FakeModel:
        def __init__(self) -> None:
            self.to_device = None
            self.eval_called = False

        def to(self, device):
            self.to_device = device
            return self

        def half(self):
            return self

        def eval(self):
            self.eval_called = True
            return self

    class FakeProcessor:
        image_mean = [0.5, 0.5, 0.5]
        image_std = [0.5, 0.5, 0.5]
        size = {"height": 384, "width": 384}

    model = FakeModel()

    torch_module = types.ModuleType("torch")
    torch_module.float32 = object()
    torch_module.tensor = lambda *_args, **_kwargs: FakeTensor()
    torch_module.cuda = types.SimpleNamespace(is_available=lambda: True)
    torch_module.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(benchmark=False)
    )
    if has_compile:
        torch_module.compile = compile_fn or (lambda compiled_model, **_kwargs: compiled_model)

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoImageProcessor = types.SimpleNamespace(
        from_pretrained=lambda _model_name: FakeProcessor()
    )
    transformers_module.DPTForDepthEstimation = types.SimpleNamespace(
        from_pretrained=lambda _model_name: model
    )

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    return model


def test_edge_aware_depth_filter_preserves_hard_boundary() -> None:
    height, width = 64, 96
    depth = np.full((height, width), 0.2, dtype=np.float32)
    depth[:, width // 2 :] = 0.8

    rng = np.random.default_rng(7)
    noise = (rng.normal(0, 0.03, size=(height, width))).astype(np.float32)
    noisy_depth = np.clip(depth + noise, 0, 1)

    guide = np.zeros((height, width, 3), dtype=np.uint8)
    guide[:, : width // 2] = (30, 30, 30)
    guide[:, width // 2 :] = (220, 220, 220)

    naive = cv2.GaussianBlur(noisy_depth, (0, 0), sigmaX=1.2, sigmaY=1.2)
    conditioned = condition_depth_for_stereo(noisy_depth, guide, edge_protect_strength=0.9)

    left_slice = slice(width // 2 - 6, width // 2 - 1)
    right_slice = slice(width // 2 + 1, width // 2 + 6)
    naive_contrast = float(naive[:, right_slice].mean() - naive[:, left_slice].mean())
    conditioned_contrast = float(
        conditioned[:, right_slice].mean() - conditioned[:, left_slice].mean()
    )
    assert conditioned_contrast > naive_contrast + 0.04


def test_conditioned_depth_stays_normalized() -> None:
    depth = np.linspace(0, 1, 120, dtype=np.float32).reshape(10, 12)
    guide = np.full((10, 12, 3), 127, dtype=np.uint8)
    conditioned = condition_depth_for_stereo(depth, guide, edge_protect_strength=0.75)
    assert float(conditioned.min()) >= 0.0
    assert float(conditioned.max()) <= 1.0


def test_condition_depth_does_not_require_bilateral_filter(monkeypatch) -> None:
    depth = np.linspace(0, 1, 80, dtype=np.float32).reshape(8, 10)
    guide = np.full((8, 10, 3), 127, dtype=np.uint8)

    def _fail_bilateral(*args, **kwargs):
        raise AssertionError("bilateralFilter should not be called in optimized path")

    monkeypatch.setattr(cv2, "bilateralFilter", _fail_bilateral)
    conditioned = condition_depth_for_stereo(depth, guide, edge_protect_strength=0.75)
    assert conditioned.shape == depth.shape


def test_midas_torch_preprocess_normalizes_with_mean_and_std_on_target_size() -> None:
    import pytest

    torch = None
    try:
        import torch as _torch
        torch = _torch
    except Exception:
        pytest.skip("torch unavailable")

    from vr_sbs_converter.depth import _midas_torch_preprocess

    rgb = np.full((10, 20, 3), 128, dtype=np.uint8)
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)

    out = _midas_torch_preprocess(
        rgb,
        torch=torch,
        device=torch.device("cpu"),
        mean=mean,
        std=std,
        target_size=(6, 12),
        dtype=torch.float32,
    )

    assert tuple(out.shape) == (1, 3, 6, 12)
    assert out.dtype == torch.float32
    # (128/255 - 0.5) / 0.5 ≈ 0.00392
    assert float(out.mean().item()) == pytest.approx(0.00392, abs=1e-3)


def test_midas_torch_preprocess_supports_fp16_output() -> None:
    import pytest

    torch = None
    try:
        import torch as _torch
        torch = _torch
    except Exception:
        pytest.skip("torch unavailable")

    from vr_sbs_converter.depth import _midas_torch_preprocess

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)

    out = _midas_torch_preprocess(
        rgb,
        torch=torch,
        device=torch.device("cpu"),
        mean=mean,
        std=std,
        target_size=(4, 4),
        dtype=torch.float16,
    )

    assert out.dtype == torch.float16


def test_midas_torch_preprocess_with_pinned_buffer_matches_unpinned() -> None:
    from vr_sbs_converter.depth import _midas_torch_preprocess

    torch = _import_torch_or_skip()
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, size=(8, 12, 3), dtype=np.uint8)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    pinned_buffer = torch.empty(rgb.shape, dtype=torch.uint8)

    unpinned = _midas_torch_preprocess(
        rgb,
        torch=torch,
        device=torch.device("cpu"),
        mean=mean,
        std=std,
        target_size=(6, 10),
        dtype=torch.float32,
    )
    staged = _midas_torch_preprocess(
        rgb,
        torch=torch,
        device=torch.device("cpu"),
        mean=mean,
        std=std,
        target_size=(6, 10),
        dtype=torch.float32,
        pinned_buffer=pinned_buffer,
    )

    assert torch.allclose(staged, unpinned, atol=1e-5)


def test_midas_torch_preprocess_pinned_buffer_shape_mismatch_raises_or_falls_back() -> None:
    import pytest

    from vr_sbs_converter.depth import _midas_torch_preprocess

    torch = _import_torch_or_skip()
    rgb = np.zeros((8, 12, 3), dtype=np.uint8)
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)
    wrong_shape_buffer = torch.empty((8, 11, 3), dtype=torch.uint8)

    with pytest.raises(ValueError, match="pinned_buffer shape .* must match rgb shape"):
        _midas_torch_preprocess(
            rgb,
            torch=torch,
            device=torch.device("cpu"),
            mean=mean,
            std=std,
            target_size=(8, 12),
            pinned_buffer=wrong_shape_buffer,
        )


def test_get_pinned_rgb_buffer_caches_by_shape(monkeypatch) -> None:
    from vr_sbs_converter.depth import MidasDepthEstimator

    calls = []

    class FakeTorch:
        uint8 = object()

        @staticmethod
        def empty(shape, *, dtype, pin_memory):
            calls.append((shape, dtype, pin_memory))
            return {"shape": shape}

    estimator = MidasDepthEstimator(device="cuda")
    estimator._torch = FakeTorch()
    monkeypatch.setattr(estimator, "_resolve_device", lambda: "cuda")

    first = estimator._get_pinned_rgb_buffer((10, 20, 3))
    second = estimator._get_pinned_rgb_buffer((10, 20, 3))

    assert second is first
    assert calls == [((10, 20, 3), FakeTorch.uint8, True)]


def test_get_pinned_rgb_buffer_creates_separate_buffers_for_different_shapes(monkeypatch) -> None:
    from vr_sbs_converter.depth import MidasDepthEstimator

    calls = []

    class FakeTorch:
        uint8 = object()

        @staticmethod
        def empty(shape, *, dtype, pin_memory):
            calls.append((shape, dtype, pin_memory))
            return {"shape": shape}

    estimator = MidasDepthEstimator(device="cuda")
    estimator._torch = FakeTorch()
    monkeypatch.setattr(estimator, "_resolve_device", lambda: "cuda")

    first = estimator._get_pinned_rgb_buffer((10, 20, 3))
    second = estimator._get_pinned_rgb_buffer((12, 20, 3))

    assert second is not first
    assert calls == [
        ((10, 20, 3), FakeTorch.uint8, True),
        ((12, 20, 3), FakeTorch.uint8, True),
    ]


def test_midas_load_compiles_model_when_enabled_on_cuda(monkeypatch) -> None:
    from vr_sbs_converter import depth as depth_module
    from vr_sbs_converter.depth import MidasDepthEstimator

    monkeypatch.setattr(depth_module, "_triton_available", lambda: True)
    compile_calls = []
    compiled_model = object()

    def fake_compile(model, **kwargs):
        compile_calls.append((model, kwargs))
        return compiled_model

    original_model = _install_fake_midas_modules(monkeypatch, fake_compile)
    estimator = MidasDepthEstimator(device="cuda", depth_compile=True)

    estimator._load()

    assert compile_calls == [
        (
            original_model,
            {"mode": "reduce-overhead", "fullgraph": False, "dynamic": False},
        )
    ]
    assert estimator._model is compiled_model
    assert estimator._compiled is True


def test_midas_load_does_not_compile_model_on_cpu(monkeypatch) -> None:
    from vr_sbs_converter.depth import MidasDepthEstimator

    compile_calls = []
    original_model = _install_fake_midas_modules(
        monkeypatch,
        lambda model, **kwargs: compile_calls.append((model, kwargs)) or object(),
    )
    estimator = MidasDepthEstimator(device="cpu", depth_compile=True)

    estimator._load()

    assert compile_calls == []
    assert estimator._model is original_model
    assert estimator._compiled is False


def test_midas_load_warns_and_keeps_original_model_when_compile_fails(monkeypatch) -> None:
    import pytest

    from vr_sbs_converter import depth as depth_module
    from vr_sbs_converter.depth import MidasDepthEstimator

    monkeypatch.setattr(depth_module, "_triton_available", lambda: True)

    def failing_compile(_model, **_kwargs):
        raise RuntimeError("compile unavailable")

    original_model = _install_fake_midas_modules(monkeypatch, failing_compile)
    estimator = MidasDepthEstimator(device="cuda", depth_compile=True)

    with pytest.warns(RuntimeWarning, match="torch.compile failed.*uncompiled MiDaS"):
        estimator._load()

    assert estimator._model is original_model
    assert estimator._compiled is False


def test_midas_load_warns_and_skips_compile_when_triton_missing(monkeypatch) -> None:
    import pytest

    from vr_sbs_converter import depth as depth_module
    from vr_sbs_converter.depth import MidasDepthEstimator

    monkeypatch.setattr(depth_module, "_triton_available", lambda: False)
    compile_calls: list = []

    def unexpected_compile(_model, **_kwargs):
        compile_calls.append(_model)
        return object()

    original_model = _install_fake_midas_modules(monkeypatch, unexpected_compile)
    estimator = MidasDepthEstimator(device="cuda", depth_compile=True)

    with pytest.warns(RuntimeWarning, match="triton"):
        estimator._load()

    assert compile_calls == []
    assert estimator._model is original_model
    assert estimator._compiled is False


def test_midas_load_skips_compile_when_torch_compile_unavailable(monkeypatch) -> None:
    import warnings

    from vr_sbs_converter.depth import MidasDepthEstimator

    original_model = _install_fake_midas_modules(monkeypatch, has_compile=False)
    estimator = MidasDepthEstimator(device="cuda", depth_compile=True)

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        estimator._load()

    assert emitted == []
    assert estimator._model is original_model
    assert estimator._compiled is False


def test_midas_predict_warns_restores_and_retries_when_compiled_forward_fails() -> None:
    import pytest

    from vr_sbs_converter.depth import MidasDepthEstimator

    class OriginalModel:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, *, pixel_values):
            self.calls += 1
            return type("Output", (), {"predicted_depth": "depth"})()

    class FailingCompiledModel:
        def __call__(self, *, pixel_values):
            raise RuntimeError("compiled graph failed")

    original_model = OriginalModel()
    compiled_model = FailingCompiledModel()
    estimator = MidasDepthEstimator(device="cuda", depth_compile=True)
    estimator._model = compiled_model
    estimator._uncompiled_model = original_model
    estimator._compiled = True

    with pytest.warns(RuntimeWarning, match="Compiled MiDaS forward failed.*uncompiled MiDaS"):
        predicted_depth = estimator._predict_depth(pixel_values=object())

    assert predicted_depth == "depth"
    assert estimator._model is original_model
    assert estimator._uncompiled_model is None
    assert estimator._compiled is False
    assert original_model.calls == 1


def test_condition_depth_torch_matches_numpy_within_tolerance() -> None:
    from vr_sbs_converter.depth import _condition_depth_for_stereo_torch

    torch = _import_torch_or_skip()
    height, width = 16, 24
    depth = np.linspace(0.0, 1.0, height * width, dtype=np.float32).reshape(height, width)
    frame_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    frame_bgr[:, : width // 2] = (20, 40, 60)
    frame_bgr[:, width // 2 :] = (220, 200, 180)

    expected = condition_depth_for_stereo(depth, frame_bgr, edge_protect_strength=0.75)
    actual = _condition_depth_for_stereo_torch(
        torch.from_numpy(depth),
        torch.from_numpy(frame_bgr),
        0.75,
        torch=torch,
    )

    difference = torch.abs(actual - torch.from_numpy(expected))
    assert float(difference.mean().item()) <= 0.05
    assert float(difference.max().item()) <= 0.20


def test_condition_depth_torch_returns_normalized_range() -> None:
    from vr_sbs_converter.depth import _condition_depth_for_stereo_torch

    torch = _import_torch_or_skip()
    depth = torch.linspace(5.0, 9.0, 120, dtype=torch.float32).reshape(10, 12)
    frame_bgr = torch.full((10, 12, 3), 127, dtype=torch.uint8)

    conditioned = _condition_depth_for_stereo_torch(
        depth,
        frame_bgr,
        0.75,
        torch=torch,
    )

    assert float(conditioned.min().item()) >= 0.0
    assert float(conditioned.max().item()) <= 1.0


def test_condition_depth_torch_skips_edge_protect_when_strength_zero() -> None:
    from vr_sbs_converter.depth import _condition_depth_for_stereo_torch

    torch = _import_torch_or_skip()
    depth = torch.tensor([[2.0, 4.0], [6.0, 10.0]], dtype=torch.float32)
    frame_bgr = torch.zeros((2, 2, 3), dtype=torch.uint8)

    conditioned = _condition_depth_for_stereo_torch(
        depth,
        frame_bgr,
        0.0,
        torch=torch,
    )

    expected = (depth - depth.min()) / (depth.max() - depth.min())
    assert torch.allclose(conditioned, expected, atol=1e-5)


def test_condition_depth_torch_handles_zero_spread() -> None:
    from vr_sbs_converter.depth import _condition_depth_for_stereo_torch

    torch = _import_torch_or_skip()
    depth = torch.full((6, 8), 3.0, dtype=torch.float32)
    frame_bgr = torch.full((6, 8, 3), 64, dtype=torch.uint8)

    conditioned = _condition_depth_for_stereo_torch(
        depth,
        frame_bgr,
        0.75,
        torch=torch,
    )

    assert torch.isfinite(conditioned).all()
    assert torch.count_nonzero(conditioned).item() == 0


def test_condition_depth_torch_avoids_cuda_scalar_synchronization() -> None:
    import vr_sbs_converter.depth as depth_module

    source = inspect.getsource(depth_module)

    assert "bool(" not in source


def test_extract_processor_size_from_plain_dict() -> None:
    from vr_sbs_converter.depth import _extract_processor_size

    assert _extract_processor_size({"height": 384, "width": 256}) == (384, 256)
    assert _extract_processor_size({"height": 512}) == (512, 512)
    assert _extract_processor_size({"shortest_edge": 320}) == (320, 320)


def test_extract_processor_size_from_none_uses_default() -> None:
    from vr_sbs_converter.depth import _extract_processor_size

    assert _extract_processor_size(None) == (384, 384)
    assert _extract_processor_size(None, default=256) == (256, 256)


def test_extract_processor_size_from_int() -> None:
    from vr_sbs_converter.depth import _extract_processor_size

    assert _extract_processor_size(384) == (384, 384)
    assert _extract_processor_size(224) == (224, 224)


def test_extract_processor_size_from_size_dict_object() -> None:
    """HuggingFace transformers 4.x may return a SizeDict dataclass; make sure we handle it."""
    from vr_sbs_converter.depth import _extract_processor_size

    class _FakeSizeDict:
        def __init__(self, height=None, width=None, shortest_edge=None):
            self.height = height
            self.width = width
            self.shortest_edge = shortest_edge

    assert _extract_processor_size(_FakeSizeDict(height=384, width=384)) == (384, 384)
    assert _extract_processor_size(_FakeSizeDict(shortest_edge=320)) == (320, 320)
    assert _extract_processor_size(_FakeSizeDict(height=480, width=640)) == (480, 640)


def test_extract_processor_size_ignores_non_int_values() -> None:
    """SizeDict attributes that aren't int-coercible fall back to default."""
    from vr_sbs_converter.depth import _extract_processor_size

    class _FakeSizeDict:
        height = {"nested": "value"}
        width = None
        shortest_edge = None

    assert _extract_processor_size(_FakeSizeDict(), default=384) == (384, 384)


def test_triton_available_returns_bool_without_raising() -> None:
    from vr_sbs_converter.depth import _triton_available

    result = _triton_available()
    assert isinstance(result, bool)


def test_autocast_ctx_prefers_new_api_when_available() -> None:
    torch = _import_torch_or_skip()
    from vr_sbs_converter.depth import _autocast_ctx

    calls: list[dict] = []

    class _FakeCtx:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _new_autocast(**kwargs):
        calls.append(kwargs)
        return _FakeCtx()

    class _FakeAmp:
        autocast = staticmethod(_new_autocast)

    class _FakeTorch:
        amp = _FakeAmp()

        class cuda:
            class amp:
                @staticmethod
                def autocast(**kwargs):
                    raise AssertionError("should not fall back to torch.cuda.amp.autocast")

    ctx = _autocast_ctx(_FakeTorch, device_type="cuda", enabled=True)
    with ctx:
        pass

    assert calls == [{"device_type": "cuda", "enabled": True}]


def test_autocast_ctx_falls_back_to_cuda_amp_when_new_api_missing() -> None:
    torch = _import_torch_or_skip()
    from vr_sbs_converter.depth import _autocast_ctx

    calls: list[dict] = []

    class _FakeCtx:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _FakeCudaAmp:
        @staticmethod
        def autocast(**kwargs):
            calls.append(kwargs)
            return _FakeCtx()

    class _FakeCuda:
        amp = _FakeCudaAmp()

    class _FakeTorch:
        # No .amp attribute → forces fallback
        cuda = _FakeCuda()

    ctx = _autocast_ctx(_FakeTorch, device_type="cuda", enabled=False)
    with ctx:
        pass

    assert calls == [{"enabled": False}]
