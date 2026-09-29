"""Repository-local wrapper for ``python -m madl.cli``."""

from madl.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
