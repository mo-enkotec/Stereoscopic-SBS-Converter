from vr_sbs_converter.compatibility import VideoStreamInfo, evaluate_player_compatibility


def test_warns_for_444_profile_and_pix_fmt() -> None:
    info = VideoStreamInfo(
        codec_name="h264",
        profile="High 4:4:4 Predictive",
        pix_fmt="yuv444p",
        width=3840,
        height=1080,
    )
    warnings = evaluate_player_compatibility(info)
    combined = " ".join(warnings).lower()
    assert "yuv420p" in combined
    assert "4:4:4" in combined


def test_warns_for_extreme_full_sbs_width() -> None:
    info = VideoStreamInfo(
        codec_name="h264",
        profile="High",
        pix_fmt="yuv420p",
        width=7680,
        height=2160,
    )
    warnings = evaluate_player_compatibility(info)
    combined = " ".join(warnings).lower()
    assert "width" in combined
    assert "decoder limit" in combined


def test_compatible_stream_has_no_warnings() -> None:
    info = VideoStreamInfo(
        codec_name="h264",
        profile="High",
        pix_fmt="yuv420p",
        width=3840,
        height=1080,
    )
    assert evaluate_player_compatibility(info) == []
