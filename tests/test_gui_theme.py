from vr_sbs_converter.gui.theme import build_dark_stylesheet


def test_dark_stylesheet_contains_core_colors() -> None:
    stylesheet = build_dark_stylesheet()
    assert "QWidget" in stylesheet
    assert "#121212" in stylesheet
    assert "#1E1E1E" in stylesheet
