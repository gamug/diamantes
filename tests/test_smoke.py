"""Minimal smoke test so the suite is non-empty until the module is built out.

The FTI source tree was removed from ``main`` pending a restructure to the
tutor's specification; this keeps ``pytest`` (and CI) green in the meantime and
checks the course data dictionary and the project skeleton are in place.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_project_skeleton_present() -> None:
    """The data dictionary and the top-level project folders must ship with the repo."""
    assert (REPO_ROOT / "data" / "01_raw" / "datos_diamantes_Info.txt").is_file()
    assert (REPO_ROOT / "conf").is_dir()
    assert (REPO_ROOT / "notebooks").is_dir()
