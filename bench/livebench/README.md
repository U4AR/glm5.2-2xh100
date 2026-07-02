# LiveBench reasoning benchmark

One command to score the running GLM-5.2 server on the full **LiveBench
reasoning** suite (200 questions), one question at a time. Defaults to the
fast **top-2** expert tier (served as `GLM5.2-top2`; see [`run_fast.sh`](../../run_fast.sh)).

The suite is 100 `zebra_puzzle` + 50 `spatial` + 50 `web_of_lies_v2`. Each task
has its own answer format, and `run_livebench.py` grades each accordingly (zebra
gets partial credit per slot; spatial and web-of-lies are all-or-nothing exact
match with number-word normalization).

## Start the benchmark

1. **Start the server** (if it isn't already):

   ```bash
   ./run_fast.sh            # top-2 + MTP, served as GLM5.2 with -topN tiers
   ```

   Wait until it logs `The server is fired up and ready to roll!` (weight load
   is a few minutes). Check with `curl -s localhost:8000/v1/models`.

2. **Run the benchmark** — one portable command:

   ```bash
   ./bench/run_benchmark.sh
   ```

   The dataset is pulled from the HuggingFace hub on first run (needs network
   the first time; then it's cached), so this works on any fresh box — nothing
   to copy over.

## Options (all env-overridable)

```bash
MODEL=GLM5.2-top8 ./bench/run_benchmark.sh   # baseline tier, for A/B vs top-2
LIMIT=20 ./bench/run_benchmark.sh            # quick 20-question smoke test
TASK=zebra_puzzle ./bench/run_benchmark.sh   # a single task
BASE=http://otherhost:8000/v1 ./bench/run_benchmark.sh
```

`MODEL`, `BASE`, `TASK`, `LIMIT`, `CATEGORY`, `MAX_TOKENS`, `OUT`. The tier
suffix `-topN` (N=0..8) picks the per-request expert count live — see
[`../../BLOG_INTELLIGENCE_TIER.md`](../../BLOG_INTELLIGENCE_TIER.md).

## Output

Per-question lines (score, tok/s, ground truth vs. prediction) stream as it
runs, then a per-task + overall summary:

```
task              score  avg tok/s    n
zebra_puzzle      ..... %      ....  100
spatial           ..... %      ....   50
web_of_lies_v2    ..... %      ....   50
----------------------------------------
OVERALL           ..... %      ....  200
```

Full per-question records are written to `livebench_results.json` (override
with `OUT=`).

Call the driver directly for more control:

```bash
python bench/livebench/run_livebench.py --help
```
