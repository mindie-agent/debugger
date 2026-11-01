"""Module execution entry point (`python -m mindie_debugger`).

[Role]
  Dispatches `python -m mindie_debugger` to `mindie_debugger.cli.main`.

[Invariants]
  * Returns the integer exit code from `cli.main()`.

[Timeouts & Locks]
  Delegated to individual CLI subcommands.

[Side-Effect Policy]
  Exits with the subcommand's exit status.
"""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
