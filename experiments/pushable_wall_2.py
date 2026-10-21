"""Statistics for the controllers on the 'Pushable wall 2' scene.

    python experiments/pushable_wall_2.py --runs 50 --workers 7          # full experiment -> results.csv
    python experiments/pushable_wall_2.py --bench                        # time a few steps per config
    python experiments/pushable_wall_2.py --summary                      # table from results.csv

Everything runs through server.plan_request (the same code path the app uses). The rover physics, lidar scan and
push contact model are Python ports of app.js (lidarScan, resolveContacts); the parameters are read from the
`params` defaults in app.js so they cannot drift, and each config only overrides what it names.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):  # one thread per worker process
    os.environ.setdefault(_v, "1")

import argparse
import csv
import math
import re
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import server  # noqa: E402

RESULTS = os.path.join(HERE, "results.csv")
MAX_STEPS = 400          # 40 s of simulated time
MCTS_EVERY = 5           # default cadence of the *_e5 style configs below; configs set "_every" to override it
GOAL_TOL = 0.4
FIELDS = ["config", "run", "result", "steps", "path_len", "objects_moved", "mean_step_ms", "final_goal_dist"]

CONFIGS = {  # name -> overrides on the app defaults
    "mppi_h15": {"horizon": 15, "use_mcts": False},
    "mppi_h100": {"horizon": 100, "use_mcts": False},
    "h15_waypoint+prior": {"horizon": 15, "use_mcts": True, "mcts_inject": "waypoint+prior"},
    "h15_waypoint": {"horizon": 15, "use_mcts": True, "mcts_inject": "waypoint"},
    "h15_prior": {"horizon": 15, "use_mcts": True, "mcts_inject": "prior"},
    # the same three, with the tree search on every step (the app default); "_every" is not a controller parameter
    "h15_waypoint+prior_e1": {"horizon": 15, "use_mcts": True, "mcts_inject": "waypoint+prior", "_every": 1},
    "h15_waypoint_e1": {"horizon": 15, "use_mcts": True, "mcts_inject": "waypoint", "_every": 1},
    "h15_prior_e1": {"horizon": 15, "use_mcts": True, "mcts_inject": "prior", "_every": 1},
    # all six MCTS configs again with MPPI horizon 25 (the app default)
    "h25_waypoint+prior": {"horizon": 25, "use_mcts": True, "mcts_inject": "waypoint+prior"},
    "h25_waypoint": {"horizon": 25, "use_mcts": True, "mcts_inject": "waypoint"},
    "h25_prior": {"horizon": 25, "use_mcts": True, "mcts_inject": "prior"},
    "h25_waypoint+prior_e1": {"horizon": 25, "use_mcts": True, "mcts_inject": "waypoint+prior", "_every": 1},
    "h25_waypoint_e1": {"horizon": 25, "use_mcts": True, "mcts_inject": "waypoint", "_every": 1},
    "h25_prior_e1": {"horizon": 25, "use_mcts": True, "mcts_inject": "prior", "_every": 1},
}


def app_params():
    """The `params` defaults object from app.js, parsed into a dict."""
    src = open(os.path.join(ROOT, "app.js"), encoding="utf-8").read()
    body = src[src.index("const params = {") + len("const params = {"):]
    body = body[:body.index("};")]
    body = re.sub(r"//.*", "", body)
    out = {}
    for k, v in re.findall(r"(\w+):\s*([^,\n]+?)\s*(?:,|$)", body):
        v = v.strip()
        out[k] = True if v == "true" else False if v == "false" else v.strip("'\"") if v[0] in "'\"" else float(v)
    for k in ("num_samples", "horizon", "mcts_iterations", "mcts_depth", "mcts_rollout_depth", "mcts_num_turns", "num_viz_samples"):
        if k in out:
            out[k] = int(out[k])
    return out


def scene():
    """Pushable wall 2 (mirrors scenes.js)."""
    obs = [{"x": 10., "y": 0.7 + 1.4 * k, "r": 0.7, "movable": k in (2, 4, 6)} for k in range(9)]
    obs += [{"x": x, "y": y, "r": r, "movable": False} for x, y, r in
            [(7.3, 10.1, .8), (7.25, 9.05, .9), (7.3, 7.6, .7), (7.35, 6.2, .8), (7.35, 4.7, .8), (7.0, 3.5, .7), (7.1, 2.05, .9), (16, 4, .8)]]
    obs += [{"x": 14.5, "y": 0.7 + 1.4 * k, "r": 0.7, "movable": False} for k in range(9) if k != 7]
    return {"w": 20., "h": 12., "robotR": .35, "start": [1.5, 6., 0.], "goal": [18.5, 6.], "obstacles": obs}


def lidar_scan(sc, st, p, rng):
    n, R = int(p["lidar_beams"]), p["lidar_range"]
    ang, rg, mv = [], [], []
    for i in range(n):
        a = st[2] - math.pi + i * 2 * math.pi / n
        dx, dy = math.cos(a), math.sin(a)
        t, m = R, 0
        tx = (sc["w"] - st[0]) / dx if dx > 1e-9 else -st[0] / dx if dx < -1e-9 else math.inf
        ty = (sc["h"] - st[1]) / dy if dy > 1e-9 else -st[1] / dy if dy < -1e-9 else math.inf
        t = min(t, tx, ty)
        for o in sc["obstacles"]:
            fx, fy = st[0] - o["x"], st[1] - o["y"]
            b = fx * dx + fy * dy
            disc = b * b - (fx * fx + fy * fy - o["r"] ** 2)
            if disc >= 0:
                t0 = -b - math.sqrt(disc)
                if 0 < t0 < t:
                    t, m = t0, 1 if o["movable"] else 0
        if t < R:
            t = min(max(t + p["lidar_noise"] * rng.normal(), 0.01), R - 0.001)
        ang.append(a); rg.append(t); mv.append(m if t < R else 0)
    return {"angles": ang, "ranges": rg, "movable": mv, "max_range": R}


def resolve_contacts(sc, pos):
    """Quasi-static pushing, as app.js resolveContacts()."""
    R = sc["robotR"]
    mov = [o for o in sc["obstacles"] if o["movable"]]
    fix = [o for o in sc["obstacles"] if not o["movable"]]

    def sep(ax, ay, ar, bx, by, br):
        dx, dy = bx - ax, by - ay
        d = math.hypot(dx, dy)
        ov = ar + br - d
        return None if ov <= 0 else ((dx / d, dy / d, ov) if d > 1e-9 else (1., 0., ov))

    for _ in range(12):
        for m in mov:
            c = sep(pos[0], pos[1], R, m["x"], m["y"], m["r"])
            if c:
                m["x"] += c[0] * c[2]; m["y"] += c[1] * c[2]
            for f in fix:
                c = sep(f["x"], f["y"], f["r"], m["x"], m["y"], m["r"])
                if c:
                    m["x"] += c[0] * c[2]; m["y"] += c[1] * c[2]
            m["x"] = min(max(m["x"], m["r"]), sc["w"] - m["r"]); m["y"] = min(max(m["y"], m["r"]), sc["h"] - m["r"])
            for o in mov:
                if o is not m:
                    c = sep(o["x"], o["y"], o["r"], m["x"], m["y"], m["r"])
                    if c:
                        m["x"] += c[0] * c[2] / 2; m["y"] += c[1] * c[2] / 2; o["x"] -= c[0] * c[2] / 2; o["y"] -= c[1] * c[2] / 2
    for m in mov:
        c = sep(m["x"], m["y"], m["r"], pos[0], pos[1], R)
        if c:
            pos[0] += c[0] * c[2]; pos[1] += c[1] * c[2]
    for f in fix:
        c = sep(f["x"], f["y"], f["r"], pos[0], pos[1], R)
        if c:
            pos[0] += c[0] * c[2]; pos[1] += c[1] * c[2]


def run_one(task):
    """One episode. task = (config_name, run_index, max_steps). Returns a result row."""
    name, run, max_steps = task
    p = app_params(); p.update({k: v for k, v in CONFIGS[name].items() if k != "_every"})
    every = CONFIGS[name].get("_every", MCTS_EVERY)
    sc = scene()
    start_pos = {id(o): (o["x"], o["y"]) for o in sc["obstacles"] if o["movable"]}
    seed0 = 1_000_003 * (run + 1)            # same seeds for every config -> paired comparison
    rng = np.random.default_rng(seed0)       # lidar range noise
    s = list(sc["start"]); U = None; wp = None
    res, path, step_ms = "timeout", 0.0, []
    for i in range(max_steps):
        lidar = lidar_scan(sc, s, p, rng)
        t0 = time.perf_counter()
        r = server.plan_request({"state": s, "scene": sc, "params": p, "U": U, "seed": seed0 + i, "waypoint": wp,
                                 "mcts": bool(p["use_mcts"]) and i % every == 0, "lidar": lidar})
        step_ms.append((time.perf_counter() - t0) * 1000)
        U = r["U_next"]
        if p["use_mcts"] and i % every == 0:
            wp = r["waypoint"]
        u = r["u"]
        nxt = [s[0] + u[0] * math.cos(s[2]) * p["dt"], s[1] + u[0] * math.sin(s[2]) * p["dt"], s[2] + u[1] * p["dt"]]
        hit_fixed = float(server.clearance(sc, nxt[0], nxt[1], skip_movable=True)) < 0  # commanded move ran into something fixed
        resolve_contacts(sc, nxt)
        path += math.hypot(nxt[0] - s[0], nxt[1] - s[1])
        s = nxt
        if math.hypot(s[0] - sc["goal"][0], s[1] - sc["goal"][1]) < GOAL_TOL:
            res = "goal"; break
        if hit_fixed:
            res = "collision"; break
    moved = sum(math.hypot(o["x"] - start_pos[id(o)][0], o["y"] - start_pos[id(o)][1]) > 0.05 for o in sc["obstacles"] if o["movable"])
    return {"config": name, "run": run, "result": res, "steps": i + 1, "path_len": round(path, 2), "objects_moved": moved,
            "mean_step_ms": round(float(np.mean(step_ms)), 1), "final_goal_dist": round(math.hypot(s[0] - sc["goal"][0], s[1] - sc["goal"][1]), 2)}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def summary(path=RESULTS):
    rows = list(csv.DictReader(open(path)))
    out = ["| config | runs | goal reached (95% CI) | collision | timeout | steps to goal, mean ± sd (median) | path length m | objects pushed | ms / step |",
           "|---|---|---|---|---|---|---|---|---|"]
    for name in CONFIGS:
        rs = [r for r in rows if r["config"] == name]
        if not rs:
            continue
        n = len(rs); ok = [r for r in rs if r["result"] == "goal"]
        lo, hi = wilson(len(ok), n)
        steps = np.array([int(r["steps"]) for r in ok], float)
        plen = np.array([float(r["path_len"]) for r in ok], float)
        st = "%.0f ± %.0f (%.0f)" % (steps.mean(), steps.std(ddof=1) if len(steps) > 1 else 0, np.median(steps)) if len(ok) else "-"
        out.append("| %s | %d | %d/%d = %.0f%% (%.0f–%.0f%%) | %d | %d | %s | %s | %.1f | %.0f |" % (
            name, n, len(ok), n, 100 * len(ok) / n, 100 * lo, 100 * hi,
            sum(r["result"] == "collision" for r in rs), sum(r["result"] == "timeout" for r in rs), st,
            "%.1f" % plen.mean() if len(ok) else "-", np.mean([int(r["objects_moved"]) for r in rs]), np.mean([float(r["mean_step_ms"]) for r in rs])))
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS), help="default: all; existing rows in results.csv are skipped")
    a = ap.parse_args()
    if a.summary:
        print(summary()); sys.exit()
    if a.bench:
        for name in a.configs:
            t0 = time.time(); r = run_one((name, 0, 30)); print("%-20s %.1f s for 30 steps -> %.0f ms/step" % (name, time.time() - t0, r["mean_step_ms"]), flush=True)
        sys.exit()
    done = set()
    if os.path.exists(RESULTS):
        done = {(r["config"], int(r["run"])) for r in csv.DictReader(open(RESULTS))}   # resume
    tasks = [(c, i, MAX_STEPS) for i in range(a.runs) for c in a.configs if (c, i) not in done]
    new = not os.path.exists(RESULTS)
    print("%d episodes to run (%d already done)" % (len(tasks), len(done)), flush=True)
    t_start = time.time()
    with open(RESULTS, "a", newline="") as f, Pool(a.workers) as pool:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        for k, row in enumerate(pool.imap_unordered(run_one, tasks), 1):
            w.writerow(row); f.flush()
            if k % 10 == 0 or k == len(tasks):
                print("%d/%d done, %.0f s elapsed" % (k, len(tasks), time.time() - t_start), flush=True)
    print(summary())
