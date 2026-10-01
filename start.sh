#!/usr/bin/env bash
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

# A working project environment is sufficient even if the system Python is old.
for candidate in "$PWD/.venv/bin/python" python3.12 python3.11 python3.10 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 && \
    "$candidate" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)' 2>/dev/null; then
    exec "$candidate" scripts/bootstrap.py "$@"
  fi
done

echo "Token Atlas needs Python 3.10, 3.11, or 3.12."
echo "请先安装 Python 3.12，然后重新运行。下载 / Download: https://www.python.org/downloads/"
exit 1
