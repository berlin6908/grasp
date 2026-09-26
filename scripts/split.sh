#!/usr/bin/env bash
set -euo pipefail

benchmark=${1:?Usage: split.sh cwq|webqsp|2wiki [split arguments]}
shift
case "$benchmark" in
  cwq) module=cwq ;;
  webqsp) module=webqsp ;;
  2wiki) module=two_wiki ;;
  *) printf 'Unknown benchmark: %s\n' "$benchmark" >&2; exit 2 ;;
esac
exec "${PYTHON:-python}" -m "dataset_split.$module" "$@"
