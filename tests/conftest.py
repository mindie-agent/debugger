"""Pytest environment isolation fixture for mindie-debugger tests.

[Role]
  Isolates HOME, XDG, and MINDIE_* directories in tmp_path so tests never
  touch host directories.

[Invariants]
  * Restores all environment variables after each test.

[Timeouts & Locks]
  In-memory monkeypatch only.

[Side-Effect Policy]
  Writes only under pytest's tmp_path.
"""
import pytest


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    home = tmp_path.resolve() / "home"
    for key, value in (
        ("HOME", home),
        ("USERPROFILE", home),
        ("XDG_CONFIG_HOME", tmp_path.resolve() / "xdg-config"),
        ("XDG_DATA_HOME", tmp_path.resolve() / "xdg-data"),
        ("XDG_STATE_HOME", tmp_path.resolve() / "xdg-state"),
        ("MINDIE_CONFIG_HOME", tmp_path.resolve() / "mindie-config"),
        ("MINDIE_DATA_HOME", tmp_path.resolve() / "mindie-data"),
        ("MINDIE_DIAGNOSTICS_ROOT", tmp_path.resolve() / "mindie-diagnostics"),
    ):
        monkeypatch.setenv(key, str(value))
