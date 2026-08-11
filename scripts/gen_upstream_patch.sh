#!/usr/bin/env bash
# Regenerate patches/sglang-vs-upstream.patch: every vendored SGLang/kt_kernel
# file we ship in .venv/, diffed against the pristine upstream source that
# kt-kernel pins at ktransformers/third_party/sglang.
#
# The venv copies are what actually runs; the submodule is what upstream wrote.
# Diffing the two is the only honest answer to "what did you change?".
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SG="ktransformers/third_party/sglang/python/sglang"
KT="ktransformers/kt-kernel/python"
OUT="patches/sglang-vs-upstream.patch"

if [ ! -d "$SG" ]; then
    echo "error: $SG missing. Clone ktransformers and init its submodules:" >&2
    echo "  git -C ktransformers submodule update --init third_party/sglang" >&2
    exit 1
fi

mkdir -p patches
{
    echo "# Vendored SGLang / kt_kernel overlays vs pristine upstream."
    echo "# upstream sglang: $(git -C ktransformers/third_party/sglang rev-parse --short HEAD)"
    echo "# generated: $(date -u +%Y-%m-%dT%H:%M:%SZ) by scripts/gen_upstream_patch.sh"
    echo
} > "$OUT"

for f in $(git ls-files .venv); do
    rel="${f#.venv/lib/python3.12/site-packages/}"
    case "$rel" in
        sglang/*)    up="$SG/${rel#sglang/}" ;;
        kt_kernel/*) up="$KT/${rel#kt_kernel/}" ;;
        *)           echo "skip (unknown package): $rel" >&2; continue ;;
    esac
    if [ ! -f "$up" ]; then
        echo "skip (no upstream match): $rel" >&2
        continue
    fi
    # --no-index diffs two paths regardless of tracking; exit 1 just means
    # "files differ", which is the normal case here.
    git diff --no-index --src-prefix=upstream/ --dst-prefix=ours/ "$up" "$f" >> "$OUT" || true
done

echo "wrote $OUT ($(wc -l < "$OUT") lines)"
