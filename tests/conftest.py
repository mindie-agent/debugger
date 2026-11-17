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
    root = tmp_path.resolve()
    home = root / "home"
    tmp_dir = root / "tmp"
    home.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for key, value in (
        ("HOME", home),
        ("USERPROFILE", home),
        ("APPDATA", root / "appdata"),
        ("LOCALAPPDATA", root / "localappdata"),
        ("XDG_CONFIG_HOME", root / "xdg-config"),
        ("XDG_DATA_HOME", root / "xdg-data"),
        ("XDG_CACHE_HOME", root / "xdg-cache"),
        ("XDG_STATE_HOME", root / "xdg-state"),
        ("MINDIE_CONFIG_HOME", root / "mindie-config"),
        ("MINDIE_DATA_HOME", root / "mindie-data"),
        ("MINDIE_DIAGNOSTICS_ROOT", root / "mindie-diagnostics"),
        ("MINDIE_DIAGNOSTICS_CONFIG", root / "mindie-diagnostics" / "config.json"),
        ("REMOTE_DEV_AUTH_DIR", root / "remote-dev-auth"),
        ("REMOTE_DEV_CREDENTIAL_FILE", root / "remote-dev-auth" / "credentials.json"),
        ("UV_CACHE_DIR", root / "uv-cache"),
        ("TMPDIR", tmp_dir),
        ("TEMP", tmp_dir),
        ("TMP", tmp_dir),
    ):
        monkeypatch.setenv(key, str(value))
