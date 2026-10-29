"""stepnet: one CLI for the laptop side of the neural-net step counter.

    python stepnet.py pull                     download finished sessions from the UNO Q
    python stepnet.py truth <id> <steps> [--distance-in X] [--calib-walk] [--notes "..."]
    python stepnet.py label [--include-review] count-constrained auto-labelling
    python stepnet.py train [--include-review] train + calibrate the 5-member ensemble
    python stepnet.py export                   models/model.npz, verified numpy == PyTorch
    python stepnet.py upload                   send model.npz to the UNO Q
    python stepnet.py dash [--demo]            live web dashboard on http://localhost:8080 (--demo: no hardware)
    python stepnet.py diag                     connectivity, token, RTT, model self-checks

Workflow: record mode -> record -> pull -> label -> train -> export -> upload -> live mode.
torch is only imported by train/export (and optionally diag), so pull/dash work without it.
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import data  # noqa: E402
from unoq import UnoQError, cfg_client  # noqa: E402

SESSION_ID = re.compile(r"^\d{8}-\d{6}$")
# meta.json first so a session folder is identifiable even if the pull stops early.
FILES = ["meta.json", "fused.npy", "events.csv", "gestures.csv", "raw_ankle.bin", "raw_hip.bin"]
REQUIRED_FILES = {"meta.json", "fused.npy"}


def load_config(path=None):
    path = Path(path) if path else HERE / "config.yaml"
    cfg = yaml.safe_load(path.read_text()) or {}
    cfg.setdefault("paths", {})
    defaults = {"data": "data", "sessions": "data/sessions", "live_logs": "data/live_logs",
                "models": "models", "reports": "reports"}
    for k, v in defaults.items():
        p = Path(cfg["paths"].get(k, v))
        cfg["paths"][k] = p if p.is_absolute() else HERE / p
    # Keep the real token out of git if you like: STEPNET_TOKEN overrides config.yaml.
    cfg["token"] = os.environ.get("STEPNET_TOKEN", cfg.get("token", ""))
    return cfg


# ---------------------------------------------------------------- pull / truth

def cmd_pull(cfg, args):
    c = cfg_client(cfg)
    sessions_dir = cfg["paths"]["sessions"]
    sessions_dir.mkdir(parents=True, exist_ok=True)
    listing = c.files().get("sessions", [])
    new = 0
    for info in listing:
        sid = str(info.get("session_id", ""))
        if not SESSION_ID.match(sid):          # never let a server string become a path like ../../x
            print(f"  skipping unexpected session id {sid!r}")
            continue
        if info.get("recording"):
            print(f"  {sid}: still recording, skipped")
            continue
        d = sessions_dir / sid
        if (d / ".pulled").exists():
            # Already have it. Re-fetch only meta.json so truth entered later on the
            # dashboard reaches the local copy.
            (d / "meta.json.part").unlink(missing_ok=True)
            c.download(f"/api/files/{sid}/meta.json", d / "meta.json")
            continue
        d.mkdir(parents=True, exist_ok=True)
        files = [f for f in FILES if not (args.no_raw and f.startswith("raw_"))]
        print(f"  {sid}: downloading ({info.get('bytes', 0) / 1e6:.1f} MB on the UNO Q)")
        complete = True
        for f in files:
            if (d / f).exists() and f != "meta.json":
                continue                         # finished in an earlier, interrupted pull
            try:
                found = c.download(f"/api/files/{sid}/{f}", d / f)
            except UnoQError as e:
                print(f"    {f}: {e}")
                complete = False
                break
            if not found:
                print(f"    {f}: not on the UNO Q" + (" (REQUIRED)" if f in REQUIRED_FILES else ""))
                complete = complete and f not in REQUIRED_FILES
        if complete:
            (d / ".pulled").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            new += 1
    print(f"  pulled {new} new session(s) into {sessions_dir}")


def cmd_truth(cfg, args):
    c = cfg_client(cfg)
    res = c.truth(args.session_id, args.steps, args.distance_in, args.calib_walk, args.notes)
    print(f"  UNO Q: {json.dumps(res)}")
    meta_path = cfg["paths"]["sessions"] / args.session_id / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["truth"] = {"steps": args.steps, "distance_in": args.distance_in,
                         "is_calib_walk": bool(args.calib_walk), "notes": args.notes or ""}
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"  updated {meta_path}")
    else:
        print("  (session not pulled yet; the truth will arrive with `pull`)")


# ---------------------------------------------------------------- label / train / export / upload

def cmd_label(cfg, args):
    import labeling
    sessions = data.list_sessions(cfg["paths"]["sessions"])
    if not sessions:
        raise SystemExit("No sessions in data/sessions: run `pull` first.")
    rows = labeling.run(sessions, cfg["paths"]["reports"], args.include_review)
    review = [sid for sid, lab in rows if lab["needs_review"]]
    print(f"  labelled {len(rows)} session(s); {len(review)} need review" + (f": {', '.join(review)}" if review else ""))


def cmd_train(cfg, args):
    import train
    train.run_train(cfg, args.include_review, args.epochs, args.steps, args.seed)


def cmd_export(cfg, args):
    import train
    train.export(cfg)


def cmd_upload(cfg, args):
    path = Path(args.file) if args.file else cfg["paths"]["models"] / "model.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found: run `export` first.")
    c = cfg_client(cfg, timeout=60, retries=1)
    print(f"  uploading {path} ({path.stat().st_size / 1024:.0f} KB)...")
    res = c.post_bytes("/api/model", path.read_bytes())
    for name, val in (res.get("checks") or {}).items():
        print(f"    {name}: {val}")
    print(f"  ok = {res.get('ok')}   active_version = {res.get('active_version')}")
    if not res.get("ok"):
        if res.get("error"):
            print(f"  error: {res['error']}")
        sys.exit(1)


def cmd_dash(cfg, args):
    try:
        from dashboard.server import run
    except ModuleNotFoundError as e:
        raise SystemExit(f"  {e} in {sys.executable}.\n  Likely fix: activate the venv first "
                         "(.\\my_env\\Scripts\\Activate.ps1) or pip install -r requirements.txt")
    run(cfg, demo=args.demo, browser=not args.no_browser)


# ---------------------------------------------------------------- diag

def cmd_diag(cfg, args):
    import requests
    import baseline
    import stepnet_np

    results = []

    def report(name, ok, detail):
        results.append(ok)
        print(f"  [{'OK' if ok else ('FAIL' if ok is False else '--')}] {name}: {detail}")

    print(f"UNO Q at {cfg['unoq_url']}")
    c = cfg_client(cfg, timeout=3.0, retries=0)
    status = None
    try:
        t0 = time.perf_counter()
        status = c.status()
        report("reachable + token accepted", True, f"{(time.perf_counter() - t0) * 1000:.0f} ms")
    except UnoQError as e:
        report("reachable + token accepted", False, str(e))

    if status is not None:
        # A wrong token must be refused, otherwise the API is open to anyone on the Wi-Fi.
        try:
            r = requests.get(cfg["unoq_url"].rstrip("/") + "/api/status", headers={"X-Token": "wrong-token"}, timeout=3)
            report("wrong token refused", r.status_code == 401, f"HTTP {r.status_code}")
        except requests.RequestException as e:
            report("wrong token refused", False, str(e))
        rtts = []
        for _ in range(5):
            t0 = time.perf_counter()
            try:
                c.status()
                rtts.append((time.perf_counter() - t0) * 1000)
            except UnoQError:
                pass
        if rtts:
            report("round-trip time", None, f"min {min(rtts):.0f} / median {statistics.median(rtts):.0f} / "
                                            f"max {max(rtts):.0f} ms over {len(rtts)} requests")
        keys = ["mode", "using", "model_version", "steps", "steps_ci", "ankle_link", "ankle_batt_mv",
                "hip_ok", "infer_ms", "uptime_s", "session_id"]
        print("  status: " + ", ".join(f"{k}={status.get(k)}" for k in keys))

    print("\nLocal")
    report("prep_hash", None, data.PREP_HASH)
    for name, ok, detail in baseline.self_check():
        report(name, ok, detail)

    path = cfg["paths"]["models"] / "model.npz"
    if not path.exists():
        report("model.npz", None, f"{path} not found (run train + export)")
    else:
        m = stepnet_np.load_model(path)
        ok, checks = stepnet_np.check_model(m, data.PREP_HASH)
        for name, good, detail in checks:
            report(name, good, detail)
        report("model version", None, str(m["version"]))
        try:
            from model import from_npz_dict, probs
            members = [from_npz_dict(m, i) for i in range(int(m["n_members"]))]
            diff = float(abs(probs(members, m["ref_x"]) - stepnet_np.ensemble_probs(m, m["ref_x"])).max())
            report("numpy vs PyTorch (rebuilt from npz)", diff < 1e-4, f"max abs diff {diff:.2e}")
        except ImportError:
            report("numpy vs PyTorch (rebuilt from npz)", None, "torch not installed, skipped")

    failed = sum(1 for r in results if r is False)
    print(f"\n{failed} check(s) failed" if failed else "\nall checks passed")
    sys.exit(1 if failed else 0)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to config.yaml (default: next to stepnet.py)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pull", help="download finished sessions")
    p.add_argument("--no-raw", action="store_true", help="skip raw_ankle.bin / raw_hip.bin")
    p.set_defaults(fn=cmd_pull)

    p = sub.add_parser("truth", help="enter the true step count for a session")
    p.add_argument("session_id")
    p.add_argument("steps", type=int)
    p.add_argument("--distance-in", type=float, default=None, help="measured distance in inches")
    p.add_argument("--calib-walk", action="store_true", help="this was a distance calibration walk")
    p.add_argument("--notes", default="")
    p.set_defaults(fn=cmd_truth)

    p = sub.add_parser("label", help="count-constrained auto-labelling")
    p.add_argument("--include-review", action="store_true", help="train on needs_review sessions too")
    p.set_defaults(fn=cmd_label)

    p = sub.add_parser("train", help="train the ensemble, calibrate, write reports")
    p.add_argument("--include-review", action="store_true")
    p.add_argument("--epochs", type=int, default=30, help="max epochs per member (early stopping usually ends sooner)")
    p.add_argument("--steps", type=int, default=50, help="batches per epoch")
    p.add_argument("--seed", type=int, default=0, help="changes the split and the member seeds")
    p.set_defaults(fn=cmd_train)

    sub.add_parser("export", help="write models/model.npz").set_defaults(fn=cmd_export)

    p = sub.add_parser("upload", help="send model.npz to the UNO Q")
    p.add_argument("--file", help="a different .npz to upload")
    p.set_defaults(fn=cmd_upload)

    p = sub.add_parser("dash", help="run the web dashboard")
    p.add_argument("--demo", action="store_true", help="simulated UNO Q: try the page with no hardware")
    p.add_argument("--no-browser", action="store_true", help="don't open the page in a browser")
    p.set_defaults(fn=cmd_dash)
    sub.add_parser("diag", help="connectivity and self-checks").set_defaults(fn=cmd_diag)

    args = ap.parse_args()
    cfg = load_config(args.config)
    try:
        args.fn(cfg, args)
    except UnoQError as e:
        raise SystemExit(f"  {e}")


if __name__ == "__main__":
    main()
