# patches/

Machine-generated diffs of every line this project changed in someone else's code.
Written so the archive answers "what did you actually modify?" without anyone
having to read 95 commits. Narrative explanation is in [`../CHANGES.md`](../CHANGES.md).

| File | Against | Size |
|---|---|---|
| `sglang-vs-upstream.patch` | `kvcache-ai/ktransformers` submodule `third_party/sglang @ 51032b712` | 16 files, ~6.7k lines |
| `ktransformers-vs-upstream.patch` | `kvcache-ai/ktransformers @ 6c9c9560` | 17 files, ~5.1k lines |

Regenerate the first with `scripts/gen_upstream_patch.sh` (needs the ktransformers
checkout and its `third_party/sglang` submodule). The second is
`git -C ktransformers diff 6c9c9560 49a9bd8 -- kt-kernel`.

These are **snapshots for reading, not for applying**. To actually run the system,
use the vendored files under `.venv/` (which git tracks by force-add) and the pinned
ktransformers commit — see `PINS.txt`.
