#!/usr/bin/env bash
# Restore tracked Python patches after a clean sglang-kt install. Compiled
# extensions are deliberately excluded because they are machine-specific.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO/.venv}"
BACKUP="${1:-}"

if [ -z "$BACKUP" ] || [ ! -d "$BACKUP" ]; then
  echo "usage: $0 <overlay-backup-directory>" >&2
  exit 2
fi
while IFS= read -r -d '' source_file; do
  relative="${source_file#"$BACKUP/"}"
  target="$VENV/$relative"
  mkdir -p "$(dirname "$target")"
  cp -f "$source_file" "$target"
done < <(find "$BACKUP" -type f -name '*.py' -print0)
echo "Applied Python overlays to $VENV; kept the locally built kt-kernel extension."
