from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if "--gui" in sys.argv:
    args = [arg for arg in sys.argv[1:] if arg != "--gui"]
    try:
        from vr_sbs_converter.gui.app import launch_gui
    except ImportError as exc:
        message = (
            "GUI dependencies are not installed. Install requirements and run again.\n"
            f"Details: {exc}\n"
        )
        raise SystemExit(message)
    raise SystemExit(launch_gui(args))

from vr_sbs_converter.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
