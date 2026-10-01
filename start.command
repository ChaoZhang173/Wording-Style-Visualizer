#!/usr/bin/env bash
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)" || exit 1
bash start.sh "$@"
result=$?
if [ "$result" -ne 0 ]; then
  echo ""
  read -r -p "启动未完成。按回车关闭 / Press Enter to close. " _unused
fi
exit "$result"
