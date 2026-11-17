# Contributing to `mindie-debugger`

`mindie-debugger` is a zero-mandatory-dependency Python toolkit (`requires-python >= 3.11`, optional `torch` / `torch_npu` loaded lazily only on precision and ACLGraph execution paths).

## 1. Flat Sibling & Worktree Layout

Develop inside a linked Git worktree under `../.worktrees/` while keeping primary checkouts clean:

```bash
git -C ../debugger fetch origin
git -C ../debugger worktree add -b fix/<topic> ../.worktrees/debugger-<topic> origin/main
cd ../.worktrees/debugger-<topic>
```

## 2. Local Setup & Test Isolation

All tests automatically isolate `HOME`, `USERPROFILE`, `APPDATA`, `LOCALAPPDATA`, `XDG_*`, `MINDIE_*`, `REMOTE_DEV_*`, `UV_CACHE_DIR`, and `TMPDIR` via [`tests/conftest.py`](tests/conftest.py).

```bash
# Run Tier 0 + Tier 1 tests (with CPU PyTorch)
uv run --with torch pytest -q -m "not live_npu"

# Run Tier 2 live NPU verification when connected to standing A2/A3 hosts
uv run --with torch pytest -q -m live_npu
```

## 3. Installed-Wheel Smoke Verification

Always verify the built wheel from a directory outside the repository checkout so missing package files or implicit `PYTHONPATH=.` assumptions are caught immediately:

```bash
uv build
tmpdir="$(mktemp -d)"
uv venv "$tmpdir/venv"
uv pip install --python "$tmpdir/venv/bin/python" --no-deps dist/*.whl
(
  cd "$tmpdir"
  HOME="$tmpdir/home" MINDIE_DATA_HOME="$tmpdir/data" \
    "$tmpdir/venv/bin/mindie-debugger" triage --text "[ERROR] RUNTIME ErrCode=507015"
  test ! -e "$tmpdir/data"
)
rm -rf "$tmpdir"
```

## 4. Pull Request Checklist (`device-tool` Profile)

Every PR must use [`.github/pull_request_template.md`](.github/pull_request_template.md), keep Unix `LF` (`\n`) line endings (`python3 .github/scripts/check_lf_eol.py`), and provide concrete evidence (or `not-applicable: docs-only` for documentation-only changes) for all four `device-tool` gates:

- `DEV-ZERO-CTX`: Zero accidental `torch_npu` import or `Phy-ID 0` HBM allocation on CPU/control paths.
- `DEV-READONLY-PATH`: Read-only `triage`/`decode-code` behavior, zero `rmtree` on user paths, and Receipt Honesty on auxiliary warnings.
- `DEV-TOPOLOGY`: Accurate storage-alias, RNG/generator, and multi-rank/device topology handling.
- `DEV-BOUNDED-MEM`: Bounded log/trace I/O (`<= 512 MiB`) and `< 800` lines per module.
