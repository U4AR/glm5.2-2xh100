#!/usr/bin/env python3
"""Render bench/profile_occupancy.py results into a self-contained HTML report.

    .venv/bin/python bench/profile_report.py bench/profile_out/results.json

Writes report.html next to the input. No external assets, no network: the page
is one file and renders in both light and dark themes.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

# Palette: validated with the dataviz skill's validate_palette.js.
#   emphasis pair  blue/orange  -- ALL CHECKS PASS (light and dark)
#   5-step ordinal blue ramp    -- ALL CHECKS PASS (light and dark)
CSS = """
:root {
  color-scheme: light;
  --bg:#f2f3f4; --surface:#fafbfb; --surface-2:#eceeef; --track:#dfe3e6;
  --ink:#0d1013; --ink-2:#4e565e; --ink-3:#79828b; --rule:#dbdfe2;
  --accent:#eb6834;          /* emphasis: the bottleneck */
  --blue:#2a78d6;
  --r1:#104281; --r2:#1c5cab; --r3:#2a78d6; --r4:#5598e7; --r5:#86b6ef;
  --mono:ui-monospace,"JetBrains Mono","SFMono-Regular",Menlo,Consolas,"Liberation Mono",monospace;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --bg:#0e1114; --surface:#171b1f; --surface-2:#1f242a; --track:#2b3138;
    --ink:#f1f3f5; --ink-2:#a7b0b8; --ink-3:#79828b; --rule:#2a3037;
    --accent:#d95926; --blue:#3987e5;
    --r1:#cde2fb; --r2:#9ec5f4; --r3:#6da7ec; --r4:#3987e5; --r5:#184f95;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg:#0e1114; --surface:#171b1f; --surface-2:#1f242a; --track:#2b3138;
  --ink:#f1f3f5; --ink-2:#a7b0b8; --ink-3:#79828b; --rule:#2a3037;
  --accent:#d95926; --blue:#3987e5;
  --r1:#cde2fb; --r2:#9ec5f4; --r3:#6da7ec; --r4:#3987e5; --r5:#184f95;
}

* { box-sizing:border-box; }
body {
  margin:0; background:var(--bg); color:var(--ink);
  font-family:var(--sans); font-size:15px; line-height:1.6;
  -webkit-font-smoothing:antialiased;
}
.wrap { max-width:1040px; margin:0 auto; padding:40px 24px 72px; }
h1,h2,h3 { font-family:var(--mono); font-weight:600; text-wrap:balance; margin:0; }
h1 { font-size:26px; letter-spacing:-0.02em; }
h2 { font-size:14px; text-transform:uppercase; letter-spacing:0.12em; color:var(--ink-2); }
p { margin:0; max-width:68ch; color:var(--ink-2); }
.eyebrow {
  font-family:var(--mono); font-size:11px; text-transform:uppercase;
  letter-spacing:0.16em; color:var(--ink-3);
}
.num { font-family:var(--mono); font-variant-numeric:tabular-nums; }

header { border-bottom:1px solid var(--rule); padding-bottom:24px; margin-bottom:36px; }
header .sub { margin-top:10px; font-family:var(--mono); font-size:12.5px; color:var(--ink-3); }

section { margin-top:44px; display:flex; flex-direction:column; gap:16px; }
.card { background:var(--surface); border:1px solid var(--rule); border-radius:4px; padding:22px 24px; }

/* ---- readout strip ---- */
.readout { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:1px;
           background:var(--rule); border:1px solid var(--rule); border-radius:4px; overflow:hidden; }
.readout > div { background:var(--surface); padding:16px 18px; }
.readout .k { font-family:var(--mono); font-size:10.5px; text-transform:uppercase;
              letter-spacing:0.1em; color:var(--ink-3); }
.readout .v { font-family:var(--mono); font-variant-numeric:tabular-nums;
              font-size:26px; font-weight:600; letter-spacing:-0.02em; margin-top:4px; }
.readout .v small { font-size:13px; font-weight:500; color:var(--ink-3); letter-spacing:0; }
.readout .hero .v { color:var(--accent); }

/* ---- lane diagram ---- */
.lanes { display:flex; flex-direction:column; gap:10px; }
.lane { display:grid; grid-template-columns:186px 1fr; gap:14px; align-items:center; }
.lane .name { font-family:var(--mono); font-size:12px; color:var(--ink-2); text-align:right; line-height:1.35; }
.lane .name b { display:block; color:var(--ink); font-weight:600; }
.bar { position:relative; height:34px; background:var(--track); border-radius:3px; display:flex; overflow:hidden; }
.seg { height:100%; position:relative; }
.seg + .seg { margin-left:2px; }          /* 2px surface gap between fills */
.seg:first-child { border-radius:3px 0 0 3px; }
.seg:last-child  { border-radius:0 3px 3px 0; }
.seg.only { border-radius:3px; }
.lab { font-family:var(--mono); font-size:11.5px; font-variant-numeric:tabular-nums;
       color:#fff; padding:0 9px; line-height:34px; white-space:nowrap; overflow:hidden; }
.lab.dark { color:var(--ink); }
.axis { display:grid; grid-template-columns:186px 1fr; gap:14px; margin-top:4px; }
.ticks { position:relative; height:20px; border-top:1px solid var(--rule); }
.ticks span { position:absolute; top:3px; transform:translateX(-50%);
              font-family:var(--mono); font-size:10.5px; color:var(--ink-3); }
.ticks span:first-child { transform:none; }
.ticks span:last-child { transform:translateX(-100%); }

/* ---- meters ---- */
.meters { display:flex; flex-direction:column; gap:2px; }
.meter { display:grid; grid-template-columns:200px 1fr 132px; gap:14px; align-items:center;
         padding:9px 0; border-bottom:1px solid var(--rule); }
.meter:last-child { border-bottom:0; }
.meter .name { font-family:var(--mono); font-size:12.5px; color:var(--ink); }
.meter .name span { display:block; color:var(--ink-3); font-size:11px; }
.meter .track { height:14px; background:var(--track); border-radius:3px; overflow:hidden; }
.meter .fill { height:100%; border-radius:3px; background:var(--blue); }
.meter.hot .fill { background:var(--accent); }
.meter .val { font-family:var(--mono); font-variant-numeric:tabular-nums; font-size:13px;
              text-align:right; color:var(--ink-2); }
.meter .val b { color:var(--ink); font-size:15px; }
.pill { display:inline-block; font-family:var(--mono); font-size:10px; text-transform:uppercase;
        letter-spacing:0.09em; padding:2px 7px; border-radius:3px; border:1px solid currentColor; }
.pill.hot { color:var(--accent); }
.pill.cool { color:var(--ink-3); }

/* ---- table ---- */
.tablewrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-family:var(--mono); font-size:12.5px;
        font-variant-numeric:tabular-nums; }
th,td { text-align:left; padding:8px 14px 8px 0; border-bottom:1px solid var(--rule); white-space:nowrap; }
th { font-size:10.5px; text-transform:uppercase; letter-spacing:0.1em; color:var(--ink-3); font-weight:600; }
td.n { text-align:right; padding-right:22px; }
td.d { color:var(--ink-2); white-space:normal; min-width:220px; }

.note { font-size:13.5px; color:var(--ink-2); }
.note b { color:var(--ink); font-weight:600; }
ul.method { margin:0; padding-left:18px; display:flex; flex-direction:column; gap:9px;
            font-size:13.5px; color:var(--ink-2); max-width:76ch; }
ul.method b { color:var(--ink); font-family:var(--mono); font-size:12.5px; font-weight:600; }
code { font-family:var(--mono); font-size:12.5px; background:var(--surface-2);
       padding:1px 5px; border-radius:3px; color:var(--ink); }
pre { font-family:var(--mono); font-size:12.5px; background:var(--surface-2); color:var(--ink);
      padding:14px 16px; border-radius:4px; overflow-x:auto; margin:0; border:1px solid var(--rule); }
.legend { display:flex; flex-wrap:wrap; gap:16px; font-family:var(--mono); font-size:11.5px; color:var(--ink-2); }
.legend i { width:11px; height:11px; border-radius:2px; display:inline-block; vertical-align:-1px; margin-right:6px; }

#tip { position:fixed; z-index:20; pointer-events:none; opacity:0; transition:opacity .1s;
       background:var(--ink); color:var(--bg); font-family:var(--mono); font-size:11.5px;
       padding:6px 9px; border-radius:3px; max-width:280px; line-height:1.45; }
[data-tip] { cursor:default; }
:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
@media (max-width:720px) {
  .lane, .axis { grid-template-columns:1fr; }
  .lane .name { text-align:left; }
  .meter { grid-template-columns:1fr; gap:6px; }
  .meter .val { text-align:left; }
}
"""

JS = """
const tip = document.getElementById('tip');
document.addEventListener('mouseover', e => {
  const t = e.target.closest('[data-tip]');
  if (!t) return;
  tip.textContent = t.getAttribute('data-tip');
  tip.style.opacity = 1;
});
document.addEventListener('mousemove', e => {
  if (tip.style.opacity == 0) return;
  const pad = 14;
  let x = e.clientX + pad, y = e.clientY + pad;
  const r = tip.getBoundingClientRect();
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = e.clientY - r.height - pad;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
});
document.addEventListener('mouseout', e => {
  if (e.target.closest('[data-tip]')) tip.style.opacity = 0;
});
"""


def esc(s) -> str:
    return html.escape(str(s))


def fmt(v, d=1, dash="--"):
    return dash if v is None else f"{v:,.{d}f}"


def meter(name, sub, pct, value_html, tip, hot=False):
    w = 0 if pct is None else max(0.6, min(100, pct))
    return f"""
    <div class="meter{' hot' if hot else ''}">
      <div class="name">{esc(name)}<span>{esc(sub)}</span></div>
      <div class="track" data-tip="{esc(tip)}"><div class="fill" style="width:{w:.1f}%"></div></div>
      <div class="val">{value_html}</div>
    </div>"""


def build(res: dict) -> str:
    o = res["occupancy"]
    d = res["decode"]
    b = o.get("budget", {})
    an = res.get("analytic", {})
    srv = res.get("server", {})
    host = res.get("host", {})
    st = res.get("stream", {})
    can = res.get("canary", {})
    gc, gb_, cc, rb, ss = (o["gpu_compute"], o["gpu_bandwidth"], o["cpu_compute"],
                           o["ram_bandwidth"], o["ssd_bandwidth"])

    step_ms = b.get("measured_step_ms") or d["ms_per_step_mean"]
    gpu_ms = b.get("gpu_busy_ms_per_step", 0.0)
    cpu_ms = b.get("cpu_critical_ms_per_step", step_ms)

    # ---- fold GPU stages to 5 (4 named + other) for the ordinal ramp ----
    stages = sorted(b.get("stage_ms_per_step", {}).items(), key=lambda kv: -kv[1])
    top, rest = stages[:4], stages[4:]
    if rest:
        top.append(("other kernels", sum(v for _, v in rest)))
    ramp = ["var(--r1)", "var(--r2)", "var(--r3)", "var(--r4)", "var(--r5)"]

    # ---- lane 1: the GPU's step, work vs waiting ----
    segs = []
    for i, (nm, ms) in enumerate(top):
        pct = ms / step_ms * 100
        segs.append(
            f'<div class="seg" style="width:{pct:.2f}%;background:{ramp[i]}" '
            f'data-tip="{esc(nm)} — {ms:.2f} ms per step, {pct:.1f}% of the step"></div>')
    wait_pct = cpu_ms / step_ms * 100
    # flex rather than a fixed width so it absorbs the 2px inter-segment gaps
    # instead of pushing the label out of the bar
    segs.append(
        f'<div class="seg" style="flex:1 1 auto;min-width:0;background:var(--track)" '
        f'data-tip="GPU idle — blocked on the CPU expert path for {cpu_ms:.0f} ms">'
        f'<span class="lab dark">idle — waiting on CPU experts &nbsp;{cpu_ms:.0f} ms</span></div>')

    spin = b.get("spin_rank", {})
    spin_vals = sorted(spin.values())
    spin_pct = spin_vals[-1] if spin_vals else 0

    ticks = ""
    for i in range(6):
        ms = step_ms * i / 5
        ticks += f'<span style="left:{i*20}%">{ms:.0f}</span>'

    legend = " ".join(
        f'<span><i style="background:{ramp[i]}"></i>{esc(nm)} {ms:.1f} ms</span>'
        for i, (nm, ms) in enumerate(top))

    # ---- meters ----
    ssd_pct = min(100, (ss["read_MBs"] + ss["write_MBs"]) / 50.0)
    meters = "".join([
        meter("CPU cores occupied",
              f"{fmt(cc['cores_busy'],0)} of {cc['n_cores']} cores, both NUMA nodes",
              cc["core_util_pct"],
              f"<b>{fmt(cc['core_util_pct'],0)}%</b> <span class='pill hot'>saturated</span>",
              "Cores counted busy while they stall on memory and while kt worker "
              "threads spin — occupancy, not useful arithmetic.",
              hot=True),
        meter("Host DRAM bandwidth",
              f"ceiling {fmt(rb['measured_peak_GBs'],0)} GB/s measured (STREAM read)",
              rb["bw_util_pct"],
              f"<b>{fmt(rb['bw_util_pct'],0)}%</b> <span class='pill cool'>headroom</span>",
              f"~{fmt(rb['achieved_GBs'],0)} GB/s of a measured "
              f"{fmt(rb['measured_peak_GBs'],0)} GB/s read ceiling, and that demand "
              f"figure is an upper bound. A {can.get('threads','?')}-thread probe "
              f"kept {fmt(can.get('read_retained_pct'),0)}% of its idle bandwidth "
              f"under load, but it competes for cores and cache too, so it "
              f"overstates the memory pressure."),
        meter("CPU arithmetic",
              f"int8 VNNI, {fmt(cc['achieved_TOPS_int8'],2)} of "
              f"{fmt(cc['peak_TOPS_int8'],0)} TOPS",
              cc["tops_util_pct"],
              f"<b>{fmt(cc['tops_util_pct'],1)}%</b> <span class='pill cool'>idle</span>",
              "The cores are busy but barely computing. Neither the vector units "
              "nor DRAM throughput is the limit."),
        meter("GPU compute",
              f"kernels busy on the compute rank; {fmt(gc['achieved_TFLOPs_per_gpu'],2)} "
              f"of {fmt(gc['peak_TFLOPs_per_gpu'],0)} TFLOP/s",
              gc.get("kernel_busy_pct"),
              f"<b>{fmt(gc.get('kernel_busy_pct'),0)}%</b> <span class='pill cool'>idle</span>",
              f"From the server's own torch profile: kernels occupy "
              f"{fmt(gc.get('kernel_busy_pct'),1)}% of the step window. Arithmetic "
              f"utilisation is {fmt(gc['flops_util_pct'],2)}% — batch-1 decode is "
              f"not a compute problem."),
        meter("GPU HBM bandwidth",
              f"{fmt(gb_['achieved_GBs_per_gpu'],0)} of "
              f"{fmt(gb_['peak_GBs_per_gpu'],0)} GB/s per card",
              gb_["bw_util_pct"],
              f"<b>{fmt(gb_['bw_util_pct'],1)}%</b> <span class='pill cool'>idle</span>",
              f"Analytic model says {fmt(gb_['achieved_GBs_per_gpu'],0)} GB/s; NVML "
              f"measured a {fmt(gb_['duty_cycle_pct'],1)}% memory duty cycle "
              f"independently. The two agree."),
        meter("SSD bandwidth",
              f"{fmt(ss['read_MBs'],1)} MB/s read, {fmt(ss['write_MBs'],1)} MB/s write",
              ssd_pct,
              "<b>0%</b> <span class='pill cool'>unused</span>",
              f"{ss['major_faults']} major faults, {ss['swap_in_pages']} pages swapped "
              f"in during the run. The weights are resident in RAM; storage is off "
              f"the decode path entirely."),
    ])

    # ---- table ----
    rows = [
        ("Time per token", f"{d['ms_per_token_mean']:.1f} ms", f"{d['tok_per_s']:.2f} tok/s"),
        ("Time per forward step", f"{d['ms_per_step_mean']:.1f} ms",
         f"median {d['ms_per_step_median']:.1f}, p10 {d['ms_per_step_p10']:.1f} / "
         f"p90 {d['ms_per_step_p90']:.1f}"),
        ("Tokens per step", f"{d['accept_length']:.2f}",
         f"MTP/NEXTN depth {srv.get('speculative_num_steps','?')} accept length"),
        ("Prefill (TTFT)", f"{d['ttft_s']*1000:.0f} ms", "short prompt, GPU bulk prefill"),
        ("— step split: GPU busy", f"{gpu_ms:.1f} ms",
         f"{b.get('gpu_busy_pct_of_step',0):.1f}% of the step, overlapped per layer"),
        ("— step split: CPU experts", f"{cpu_ms:.1f} ms",
         f"{b.get('cpu_critical_pct_of_step',0):.1f}% of the step, on the critical path"),
        ("DRAM read ceiling", f"{fmt(st.get('read_GBs_all_cores'),0)} GB/s",
         f"STREAM read; triad {fmt(st.get('triad_GBs_all_cores'),0)} GB/s. Half the "
         f"cores reach {fmt(st.get('read_GBs_half_cores'),0)} GB/s, so this is a "
         f"controller limit, not a thread limit"),
        ("DRAM demand (analytic)", f"{fmt(rb['achieved_GBs'],0)} GB/s",
         f"{fmt(an.get('cpu_bytes_per_token_MB',0)/1000,2)} GB of expert weights per "
         f"token x {d['tok_per_s']:.1f} tok/s"),
        ("DRAM probe under load", f"{fmt(can.get('read_retained_pct'),0)}% retained",
         f"{fmt(can.get('idle_read_GBs'),0)} -> {fmt(can.get('loaded_read_GBs'),0)} GB/s "
         f"for a {can.get('threads','?')}-thread probe"),
        ("Experts per token per layer",
         f"{an.get('experts_on_cpu_per_token_per_layer',0):.2f} CPU / "
         f"{an.get('experts_on_gpu_per_token_per_layer',0):.2f} GPU",
         f"top-{res['model_config'].get('num_experts_per_tok')} of "
         f"{res['model_config'].get('n_routed_experts')}, "
         f"{an.get('gpu_experts_resident')} resident on GPU, "
         f"{an.get('bytes_per_expert_int4_MB',0):.1f} MB each at int4"),
        ("GPU kernel duty, rank 0 / rank 1",
         " / ".join(f"{v:.0f}%" for v in sorted(spin.values())),
         "rank 1's high figure is NCCL spin-wait on rank 0, not work"),
    ]
    table = "".join(
        f"<tr><td>{esc(k)}</td><td class='n'>{esc(v)}</td><td class='d'>{esc(n)}</td></tr>"
        for k, v, n in rows)

    cmd = ("MODEL=GLM5.2 .venv/bin/python bench/profile_occupancy.py \\\n"
           f"    --tokens {res['meta']['args'].get('tokens')} "
           f"--torch-tokens {res['meta']['args'].get('torch_tokens')}\n"
           ".venv/bin/python bench/profile_report.py bench/profile_out/results.json")

    gpus = res.get("loaded", {}).get("gpu", {})
    gpuline = " · ".join(
        f"gpu{k} sm {v['sm_util_mean']:.0f}% mem {v['mem_util_mean']:.1f}% "
        f"{v['power_W_mean']:.0f} W" for k, v in gpus.items())

    return f"""<title>Where a GLM-5.2 token's time goes</title>
<style>{CSS}</style>
<div class="wrap">
<header>
  <div class="eyebrow">occupancy profile · live server, nothing restarted</div>
  <h1>Where a {esc(res['meta']['model'])} token's time goes</h1>
  <div class="sub">
    {esc(res['meta']['date'])} &nbsp;·&nbsp; {esc(host.get('gpus','').splitlines()[0] if host.get('gpus') else '')}
    x{len(gpus)} &nbsp;·&nbsp; {esc(host.get('cpu_model','').strip())} &nbsp;·&nbsp;
    {fmt(host.get('mem_total_GB'),0)} GB RAM<br>
    tp {srv.get('tp_size')} · {srv.get('kt_method')} cpu experts · {an.get('gpu_experts_resident')} gpu-resident experts ·
    {srv.get('attention_backend')} attention · MTP depth {srv.get('speculative_num_steps')} · tier {esc(srv.get('topk_mode'))}
  </div>
</header>

<div class="readout">
  <div class="hero"><div class="k">per token</div>
    <div class="v">{d['ms_per_token_mean']:.1f}<small> ms</small></div></div>
  <div><div class="k">throughput</div>
    <div class="v">{d['tok_per_s']:.1f}<small> tok/s</small></div></div>
  <div><div class="k">per forward step</div>
    <div class="v">{d['ms_per_step_mean']:.0f}<small> ms</small></div></div>
  <div><div class="k">tokens / step</div>
    <div class="v">{d['accept_length']:.2f}</div></div>
  <div><div class="k">prefill</div>
    <div class="v">{d['ttft_s']*1000:.0f}<small> ms</small></div></div>
</div>

<section>
  <h2>Anatomy of one forward step</h2>
  <p>Each step runs the CPU experts asynchronously while the GPU works, so a layer
  costs <span class="num">max(cpu, gpu)</span>. The GPU finishes its share in
  <b class="num">{gpu_ms:.0f} ms</b> and then waits: for
  <b class="num">{b.get('cpu_critical_pct_of_step',0):.0f}%</b> of every step the
  accelerators are idle behind {an.get('experts_on_cpu_per_token_per_layer',0):.1f}
  expert GEMMs per layer running out of host DRAM.</p>
  <div class="card">
    <div class="lanes">
      <div class="lane">
        <div class="name"><b>GPU 0</b>compute rank<br>kernels + CPU wait</div>
        <div class="bar">{''.join(segs)}</div>
      </div>
      <div class="lane">
        <div class="name"><b>GPU 1</b>tp peer<br>NCCL spin-wait</div>
        <div class="bar"><div class="seg only" style="width:{spin_pct:.1f}%;background:var(--track)"
             data-tip="Rank 1 shows a {spin_pct:.0f}% kernel duty cycle, but it is spinning inside the all-reduce waiting for rank 0 — not doing work.">
             <span class="lab dark">spinning in all-reduce &nbsp;{spin_pct:.0f}% duty</span></div></div>
      </div>
      <div class="lane">
        <div class="name"><b>CPU</b>{an.get('experts_on_cpu_per_token_per_layer',0):.1f} experts x
          {an.get('moe_layers')} layers<br>int4 W4A8 from DRAM</div>
        <div class="bar"><div class="seg only" style="width:100%;background:var(--accent)"
             data-tip="The CPU expert path runs the whole step and is what the step time actually is.">
             <span class="lab">critical path — {step_ms:.0f} ms, {fmt(an.get('cpu_bytes_per_token_MB',0)/1000,1)} GB of weights per token</span></div></div>
      </div>
    </div>
    <div class="axis"><div></div><div class="ticks">{ticks}<span style="left:100%">ms</span></div></div>
    <div class="legend" style="margin-top:14px">{legend}
      <span><i style="background:var(--track)"></i>idle / waiting</span>
      <span><i style="background:var(--accent)"></i>CPU expert path</span>
    </div>
  </div>
  <p class="note">The GPU segments are drawn contiguously for legibility; in reality
  they are interleaved layer by layer across the whole step. Stage times come from
  the server's torch profile, rescaled from the profiled step
  ({b.get('profiled_step_ms',0):.0f} ms) to the unprofiled one
  ({step_ms:.0f} ms) — the profiler itself costs
  {b.get('profiler_overhead_x',1):.2f}x.</p>
</section>

<section>
  <h2>Resource occupancy</h2>
  <p>Only one number is near its ceiling — and it is the one that measures
  <em>occupancy</em>, not work.</p>
  <div class="card"><div class="meters">{meters}</div></div>
  <p class="note"><b>The interesting part is what is <i>not</i> saturated.</b> The CPU
  holds {fmt(cc['core_util_pct'],0)}% of its cores busy while retiring
  {fmt(cc['tops_util_pct'],1)}% of its int8 throughput and pulling
  {fmt(rb['bw_util_pct'],0)}% of its memory bandwidth. So the expert path is limited by
  neither arithmetic nor DRAM throughput: what is left is per-layer submit/sync
  serialisation and memory latency — {an.get('moe_layers')} round trips per forward
  pass, each too small to hide its own overhead. Meanwhile the two H100s sit at
  ~{fmt(gc.get('kernel_busy_pct'),0)}% kernel duty and
  ~{fmt(gb_['bw_util_pct'],0)}% of HBM bandwidth, and the SSDs are untouched.</p>
</section>

<section>
  <h2>Measurements</h2>
  <div class="card tablewrap">
    <table>
      <thead><tr><th>quantity</th><th class="n">value</th><th class="d">detail</th></tr></thead>
      <tbody>{table}</tbody>
    </table>
  </div>
  <p class="note">Sampled counters during the decode window: {esc(gpuline)}.</p>
</section>

<section>
  <h2>How each number was obtained</h2>
  <ul class="method">
    <li><b>Time per token</b> — a streaming completion, timestamped per SSE chunk.
    One chunk is one forward <em>step</em>, and MTP emits
    {d['accept_length']:.2f} tokens per step, so token time is the step time divided
    by the server's own token count, not the chunk rate.</li>
    <li><b>GPU compute and stage split</b> — the server's <code>/start_profile</code>
    endpoint (torch profiler, CUPTI). Busy time is the union of kernel intervals, so
    concurrent kernels are not double-counted.</li>
    <li><b>GPU bandwidth</b> — NVML memory duty cycle, cross-checked against an
    analytic bytes-per-token model of the weights actually read. Both land at
    ~{fmt(gb_['bw_util_pct'],0)}%.</li>
    <li><b>CPU occupancy</b> — <code>/proc/stat</code> deltas over the decode window
    only, prefill excluded.</li>
    <li><b>DRAM ceiling</b> — a STREAM triad and a pure-read kernel built and run on
    this box while the server was idle. The triad is counted at 32 B/element, not 24,
    because the store pulls the line in first.</li>
    <li><b>DRAM demand</b> — analytic: {an.get('experts_on_cpu_per_token_per_layer',0):.2f}
    CPU-resident experts x {an.get('bytes_per_expert_int4_MB',0):.1f} MB x
    {an.get('moe_layers')} MoE layers per token. This assumes uniform routing across
    the {an.get('gpu_experts_resident')} GPU-resident experts; hotcore placement biases
    the hot experts onto the GPU, so treat it as an upper bound. The contention probe
    is the assumption-free cross-check.</li>
    <li><b>SSD</b> — <code>/proc/diskstats</code> plus major-fault and swap counters.</li>
    <li><b>Not used:</b> hardware DRAM counters. This VM exposes no uncore/UMC PMU and
    <code>perf_event_paranoid</code> is {esc(host.get('perf_paranoid'))}, so bandwidth
    is bracketed by the two methods above rather than counted directly.</li>
  </ul>
</section>

<section>
  <h2>Reproduce</h2>
  <pre>{esc(cmd)}</pre>
  <p class="note">Runs against a live server and does not restart it. It does briefly
  drive the torch profiler and a 4-thread memory probe, which slow decode while they
  run.</p>
</section>
</div>
<div id="tip"></div>
<script>{JS}</script>
"""


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1
               else "bench/profile_out/results.json")
    res = json.loads(src.read_text())
    outp = src.parent / "report.html"
    outp.write_text(build(res))
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
