#!/usr/bin/env python3
"""Is a storage-sourced expert stream viable? Machine-independent test.

The question is never "how fast is this SSD" in isolation. A storage->VRAM
expert path has to beat two bars at once, and both are ratios, so the verdict
ports to any box:

  BAR 1 -- SCHEDULABILITY. Per decode layer the model needs `fetch_per_layer`
  non-resident experts within one layer's wall time. If the source cannot
  deliver that, duty > 100% and no amount of overlap or queueing helps; the
  fetch is simply behind and stays behind.

  BAR 2 -- UNIT ECONOMICS. Moving an expert's weights to the GPU has to cost
  LESS than letting the CPU compute that expert in place. The CPU reads the
  same bytes from its own DRAM, so this reduces to a bandwidth ratio:
  link_bw vs dram_bw. Where DRAM is faster -- true of every PCIe machine --
  move-once-use-once loses, and only reuse (caching) can pay it back.

  reuse_needed = dram_bw / link_bw

  i.e. how many times you must USE a moved expert before the move breaks even.
  Prefetch gets reuse ~= 1.4 (measured, this model). A resident cache gets
  hundreds. That gap, not the SSD's speed, is the whole story.

Reads with O_DIRECT so the page cache is neither polluted nor credited -- on a
box whose free RAM is already thin, a cached read would report DRAM speed and
fake a pass. See the reclaim-trap note in memory.
"""
import argparse, ctypes, json, mmap, os, statistics, sys, time

O_DIRECT = getattr(os, "O_DIRECT", 0o40000)


def direct_read_gbs(path, total_mb=512, block_mb=8, offset_mb=0):
    """Sequential O_DIRECT read throughput, GB/s. Returns None if unreadable."""
    bs = block_mb * 1024 * 1024
    # O_DIRECT needs the destination aligned to the logical block size.
    buf = mmap.mmap(-1, bs)
    try:
        fd = os.open(path, os.O_RDONLY | O_DIRECT)
    except OSError as e:
        return None, f"open failed: {e}"
    try:
        os.lseek(fd, offset_mb * 1024 * 1024, os.SEEK_SET)
        n_blocks = max(total_mb // block_mb, 1)
        got = 0
        t0 = time.perf_counter()
        for _ in range(n_blocks):
            r = os.readv(fd, [buf])
            if r <= 0:
                break
            got += r
        dt = time.perf_counter() - t0
    finally:
        os.close(fd)
        buf.close()
    if got == 0 or dt <= 0:
        return None, "no bytes read"
    return got / dt / 1e9, f"{got/1e6:.0f} MB in {dt*1000:.0f} ms"


def gds_status():
    """Is real GPUDirect Storage available, or only the DRAM-bouncing fallback?"""
    st = {}
    st["nvidia_fs_loaded"] = os.path.exists("/proc/driver/nvidia-fs")
    libs = []
    for root in ("/usr/local/cuda/lib64", "/usr/local/cuda-12.9/lib64"):
        p = os.path.join(root, "libcufile.so")
        if os.path.exists(p):
            libs.append(p)
    st["libcufile"] = libs
    # cuFile without nvidia-fs runs "compatibility mode": POSIX read into a host
    # bounce buffer, then H2D. That traverses host DRAM, which is precisely the
    # resource a storage path exists to avoid -- so it is NOT a partial win, it
    # is the same design we already measured at 0.79x with a slower source.
    st["real_p2p"] = bool(st["nvidia_fs_loaded"] and libs)
    st["mode"] = "GDS p2p" if st["real_p2p"] else (
        "cuFile COMPAT (bounces through host DRAM)" if libs else "unavailable")
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="", help="comma-separated files to read")
    ap.add_argument("--total-mb", type=int, default=512)
    ap.add_argument("--block-mb", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=3)
    # Model/workload shape. Defaults are the measured GLM-5.2 decode numbers,
    # but every one is an override so the verdict can be recomputed for another
    # model or another machine without touching the code.
    ap.add_argument("--expert-mb", type=float, default=9.7,
                    help="bytes moved per expert per card")
    ap.add_argument("--fetch-per-layer", type=float, default=1.71,
                    help="non-resident experts needed per layer-call")
    ap.add_argument("--layers", type=int, default=75)
    ap.add_argument("--step-ms", type=float, default=66.6)
    ap.add_argument("--cards", type=int, default=2)
    ap.add_argument("--dram-gbs", type=float, default=130.0,
                    help="per-socket DRAM read bw actually achieved by the CPU experts")
    ap.add_argument("--cpu-ms-per-expert", type=float, default=0.074,
                    help="what the CPU costs to compute one expert in place")
    ap.add_argument("--out", default="bench/profile_out/storage_feasibility.json")
    a = ap.parse_args()

    res = {"gds": gds_status(), "sources": {}, "requirement": {}, "verdict": {}}

    layer_ms = a.step_ms / a.layers
    per_layer_mb = a.fetch_per_layer * a.expert_mb
    need_gbs_per_card = (per_layer_mb / 1e3) / (layer_ms / 1e3)
    need_gbs_total = need_gbs_per_card * a.cards
    res["requirement"] = {
        "layer_ms": layer_ms,
        "mb_per_layer_per_card": per_layer_mb,
        "gbs_per_card": need_gbs_per_card,
        "gbs_total_all_cards": need_gbs_total,
        "note": "BAR 1: source must sustain this or duty > 100% and it never catches up",
    }

    for p in [x for x in a.paths.split(",") if x.strip()]:
        p = p.strip()
        runs, note = [], ""
        for i in range(a.repeat):
            gbs, note = direct_read_gbs(p, a.total_mb, a.block_mb,
                                        offset_mb=i * a.total_mb)
            if gbs is None:
                break
            runs.append(gbs)
        if not runs:
            res["sources"][p] = {"error": note}
            continue
        gbs = statistics.median(runs)
        ms_per_expert = (a.expert_mb / 1e3) / gbs * 1e3
        res["sources"][p] = {
            "gbs_median": gbs,
            "gbs_runs": runs,
            "ms_per_expert": ms_per_expert,
            # BAR 1
            "duty_pct_of_layer": 100.0 * (a.fetch_per_layer * ms_per_expert) / layer_ms,
            # BAR 2
            "vs_cpu_compute": ms_per_expert / a.cpu_ms_per_expert,
            "reuse_needed_to_break_even": ms_per_expert / a.cpu_ms_per_expert,
            "note": note,
        }

    # The portable form of BAR 2: it is a bandwidth ratio, nothing else.
    res["verdict"]["reuse_rule"] = {
        "explain": ("A moved expert must be USED this many times before the move "
                    "beats computing it in place. Prefetch achieves ~1.4; a "
                    "resident cache achieves hundreds."),
        "formula": "reuse_needed = dram_gbs / source_gbs",
        "dram_gbs": a.dram_gbs,
        "per_source": {p: (a.dram_gbs / v["gbs_median"])
                       for p, v in res["sources"].items() if "gbs_median" in v},
    }

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)

    print(f"GDS mode: {res['gds']['mode']}")
    print(f"BAR 1 requirement: {need_gbs_per_card:.1f} GB/s per card "
          f"({need_gbs_total:.1f} GB/s across {a.cards} cards)")
    print(f"  = {a.fetch_per_layer:.2f} experts x {a.expert_mb:.1f} MB "
          f"per {layer_ms:.2f} ms layer")
    print()
    hdr = f"{'source':<34}{'GB/s':>8}{'ms/expert':>11}{'duty%':>8}{'vs CPU':>9}{'reuse':>8}"
    print(hdr); print("-" * len(hdr))
    for p, v in res["sources"].items():
        if "error" in v:
            print(f"{p:<34}{'--':>8}  {v['error']}")
            continue
        print(f"{p:<34}{v['gbs_median']:>8.2f}{v['ms_per_expert']:>11.3f}"
              f"{v['duty_pct_of_layer']:>8.0f}{v['vs_cpu_compute']:>8.1f}x"
              f"{a.dram_gbs/v['gbs_median']:>7.0f}x")
    print()
    print("duty%  > 100 -> BAR 1 FAIL: cannot keep up, no scheduling fixes it")
    print("vs CPU > 1   -> BAR 2 FAIL for prefetch: costs more than computing in place")
    print("reuse        -> times a moved expert must be reused to break even")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
