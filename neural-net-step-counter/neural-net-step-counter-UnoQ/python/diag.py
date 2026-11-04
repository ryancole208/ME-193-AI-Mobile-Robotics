"""diag: a live snapshot of everything that can go wrong, as text and as a dict.

Printed every 2 s when the app is started in diag mode
(`echo diag > RUN_MODE`, then restart the app), and always available as
GET /api/diag. Counting keeps running in diag mode.

  ankle   packet rate, gaps (seq / source / resync), loss, MTU, battery
  hip     sample rate, late samples, skipped slots, ADC errors, clipping %,
          calibrated |a| (should be ~1.00 g when still)
  sync    per source: slope (ppm drift), offset, fit residual (ms)
  nn      inference ms per member (budget: 100 ms for the whole ensemble)
  p       the latest final p_right / p_left
  gest    the last gesture verdicts with reasons
"""

from __future__ import annotations

import numpy as np


def _sync(s) -> dict:
    return {"slope": round(s.a, 7), "ppm": round((s.a - 1) * 1e6, 1), "offset_s": None if s.b is None else round(s.b, 4),
            "residual_ms": None if s.residual_ms is None else round(s.residual_ms, 2), "fitted": s.fitted,
            "buckets": s.n_buckets}


def build_diag(e, now: float) -> dict:
    a, h = e.ankle, e.hip
    hip_g = e.fusion.hip.val[-100:]
    amag = float(np.linalg.norm(hip_g, axis=1).mean()) if len(hip_g) else None
    k0, ch, p, _ = e.grid.rows(e.grid.end - 1000, e.grid.end)
    present = {"ankle": round(float(ch[:, 10].mean()), 3) if len(ch) else None,
               "hip": round(float(ch[:, 11].mean()), 3) if len(ch) else None}
    p_last = [None, None]
    if e.infer is not None and e.infer.final_k is not None and e.grid.base is not None:
        i = e.infer.final_k - 1 - e.grid.base
        if 0 <= i < len(e.grid.p):
            p_last = [round(float(v), 3) for v in e.grid.p[i]]
    st = a.stats
    data = {
        "ankle": {"state": a.link_state(now), "relay": a.state, "rate_hz": round(a.rate.rate(now), 1),
                  "packets": st.packets, "seq_gaps": st.seq_gaps, "source_gaps": st.source_gaps,
                  "resyncs": st.resyncs, "loss_pct": round(st.loss_pct, 2), "mtu": a.mtu,
                  "n_per_packet": sorted(a.n_values), "battery_mv": a.battery_mV, "bad_packets": a.packet_errors},
        "hip": {"ok": h.ok(now), "reason": h.reason, "rate_hz": round(h.rate.rate(now), 1),
                "late": h.late, "skipped": h.skipped, "adc_errors": h.adc_errors, "adc_bits": h.adc_bits,
                "gaps": h.stats.gaps, "clipped_pct": round(h.clipped_pct, 2), "rail_pct": round(h.rail_pct, 2),
                "amag_g": None if amag is None else round(amag, 3), "cal": h.cal.get("method"),
                "last_raw": None if h.last_raw is None else h.last_raw.tolist()},
        "sync": {"ankle": _sync(e.ank_sync), "hip": _sync(e.hip_sync)},
        "present_10s": present,
        "nn": {"model": e.model.version if e.model else None,
               "member_ms": [round(x, 2) for x in e.model.last_ms] if e.model else [],
               "last_ms": round(e.infer.last_ms, 1) if e.infer else None,
               "max_ms": round(e.infer.max_ms, 1) if e.infer else None,
               "p_right": p_last[0], "p_left": p_last[1]},
        "gestures": list(e.gesture_log),
        "grid_rows": e.fusion.rows_built,
    }
    d = data
    lines = [
        f"  ankle  {d['ankle']['state']:9s} {d['ankle']['rate_hz']:5.1f} Hz  pkts {st.packets}  gaps seq/src/resync "
        f"{st.seq_gaps}/{st.source_gaps}/{st.resyncs}  loss {st.loss_pct:.2f}%  MTU {a.mtu}  batt {a.battery_mV} mV",
        f"  hip    {'OK' if d['hip']['ok'] else 'BAD':9s} {d['hip']['rate_hz']:5.1f} Hz  late {h.late}  skipped {h.skipped}"
        f"  adc err {h.adc_errors}  clipped {h.clipped_pct:.2f}%  |a| {d['hip']['amag_g']} g  ({h.reason})",
        f"  sync   {e.ank_sync.describe()}",
        f"         {e.hip_sync.describe()}",
        f"  grid   present (10 s): ankle {present['ankle']}, hip {present['hip']}",
        f"  nn     {d['nn']['model'] or 'none (baseline)'}  ms/member {d['nn']['member_ms']}  last {d['nn']['last_ms']}"
        f"  max {d['nn']['max_ms']}  p_right {p_last[0]}  p_left {p_last[1]}",
    ] + [f"  gest   {g}" for g in d["gestures"][-3:]]
    return {"data": data, "text": "\n".join(lines)}
