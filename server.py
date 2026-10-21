"""MPPI playground backend. Run: python server.py  ->  http://localhost:8000
Serves the static UI and exposes POST /api/plan, which runs one MPPI iteration (numpy).
The server is stateless: the client sends the nominal sequence U and a seed each step,
so scrubbing back in time and resuming is reproducible.

Nomenclature follows tud-amr/mppi-isaac (mppi_torch): K = num_samples, T = horizon,
noise_sigma (covariance), lambda_, u_min / u_max, U, perturbed_action, cost_total,
omega (weights), eta (normaliser), beta (baseline cost), running_cost, terminal_cost.
"""
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from scipy.spatial import cKDTree

import mcts


PUSH_PAD = 0.15  # a movable object counts as "being pushed" when the rover is this close to touching it


def clearance(scene, x, y, skip_movable=False):
    """Signed distance from the rover's body to the nearest wall / obstacle (negative = collision).
    With skip_movable, movable (red) objects are ignored: touching them is pushing, not colliding."""
    c = np.minimum.reduce([x, y, scene["w"] - x, scene["h"] - y]) - scene["robotR"]
    for o in scene["obstacles"]:
        if skip_movable and o.get("movable"):
            continue
        c = np.minimum(c, np.hypot(x - o["x"], y - o["y"]) - o["r"] - scene["robotR"])
    return c


def movable_contact(scene, x, y):
    """1 where the rover is touching (or within PUSH_PAD of) a movable object, else 0."""
    hit = np.zeros(np.shape(x))
    for o in scene["obstacles"]:
        if o.get("movable"):
            hit = np.maximum(hit, (np.hypot(x - o["x"], y - o["y"]) - o["r"] - scene["robotR"] < PUSH_PAD).astype(float))
    return hit


def lidar_points(state, lidar, labels=False):
    """World-frame hit points of a scan, as (fixed_points, movable_points). Beams that returned max range hit
    nothing. With labels (a semantic lidar), hits on movable objects are separated out; otherwise every hit is
    a fixed point."""
    ranges = np.asarray(lidar["ranges"], dtype=float)
    angles = np.asarray(lidar["angles"], dtype=float)
    hit = ranges < lidar["max_range"] - 1e-6
    pts = np.stack([state[0] + ranges[hit] * np.cos(angles[hit]), state[1] + ranges[hit] * np.sin(angles[hit])], axis=1)
    if labels and "movable" in lidar:
        mov = np.asarray(lidar["movable"], dtype=bool)[hit]
        return pts[~mov], pts[mov]
    return pts, np.zeros((0, 2))


def lidar_clearance(scene, state, lidar, labels=False):
    """(clearance, contact) functions built only from what the lidar saw: the hit points of the latest scan.
    Free space beyond a beam's range is simply unknown (= free). Fixed hit points give the clearance; hits on
    movable objects (if the lidar labels them) give `contact`, which is penalised but not a collision."""
    pts, mpts = lidar_points(state, lidar, labels)
    R = scene["robotR"]

    def dist_to(points):
        if len(points) == 0:
            return lambda x, y: np.full(np.shape(x), np.inf)
        tree = cKDTree(points)
        return lambda x, y: tree.query(np.stack([np.ravel(x), np.ravel(y)], axis=1))[0].reshape(np.shape(x))
    d_fixed, d_mov = dist_to(pts), dist_to(mpts)
    return (lambda x, y: d_fixed(x, y) - R), (lambda x, y: (d_mov(x, y) - R < PUSH_PAD).astype(float))


def dynamics(state, u, dt):
    """Unicycle. state = (x, y, theta) columns, u = (v, omega) columns."""
    x, y, th = state
    v, w = u
    return x + v * np.cos(th) * dt, y + v * np.sin(th) * dt, th + w * dt


def running_cost(scene, p, state, u, clr, contact):
    x, y, _ = state
    gx, gy = scene["goal"]
    c = p["w_goal"] * ((x - gx) ** 2 + (y - gy) ** 2) + p["w_effort"] * u[1] ** 2
    d = clr(x, y)
    soft = p["w_obstacle"] * (1 - d / p["margin"]) ** 2
    return c + np.where(d < 0, p["w_collision"], np.where(d < p["margin"], soft, 0.0)) + p.get("w_push", 0.0) * contact(x, y)


def terminal_cost(scene, p, state):
    x, y, _ = state
    gx, gy = scene["goal"]
    return p["w_terminal"] * ((x - gx) ** 2 + (y - gy) ** 2)


def mppi_step(state, scene, p, U, seed, lidar=None, prior=None, goal=None):
    """One MPPI iteration. Returns the command, the updated nominal and visualisation data."""
    K, T, dt, lambda_ = int(p["num_samples"]), int(p["horizon"]), p["dt"], p["lambda_"]
    rng = np.random.default_rng(seed)
    # Arrival speed limit: v <= max(floor, gain * distance_to_goal), measured to the TRUE goal (not a waypoint).
    # At full speed the tightest turn (radius v/omega_max = 0.8 m) is wider than the goal circle, so a rover that
    # arrives slightly off-line cannot turn in and has to loop around.
    gx, gy = goal if goal is not None else scene["goal"]
    gain, v_floor = p.get("goal_speed_gain", 0.0), p.get("goal_speed_floor", 0.2)
    v_cap = lambda x, y: np.clip(gain * np.hypot(x - gx, y - gy), v_floor, p["u_max_v"]) if gain > 0 else p["u_max_v"]

    # noise_sigma is a covariance matrix over (v, omega); the UI exposes the std devs.
    noise_sigma = np.diag([p["noise_sigma_v"] ** 2, p["noise_sigma_w"] ** 2])
    noise_mu = np.zeros(2)
    u_min = np.array([p["u_min_v"], -p["u_max_w"]])
    u_max = np.array([p["u_max_v"], p["u_max_w"]])

    # what the controller believes about obstacles: lidar hit points, or ground truth for comparison
    labels = bool(p.get("use_lidar_labels"))  # semantic lidar: the controller knows which hits are movable
    if p.get("use_lidar") and lidar:
        clr, contact = lidar_clearance(scene, state, lidar, labels)
    else:
        clr = lambda x, y: clearance(scene, x, y, skip_movable=labels)
        contact = (lambda x, y: movable_contact(scene, x, y)) if labels else (lambda x, y: np.zeros(np.shape(x)))

    # nominal sequence U (T x nu), resized to the current horizon
    U = np.zeros((0, 2)) if U is None else np.asarray(U, dtype=float).reshape(-1, 2)
    if len(U) < T:
        pad = U[-1:] if len(U) else np.zeros((1, 2))
        U = np.vstack([U, np.repeat(pad, T - len(U), axis=0)])
    U = U[:T]
    U_in = U.copy()

    # 1. sample perturbations and roll out
    noise = rng.multivariate_normal(noise_mu, noise_sigma, size=(K, T))
    perturbed_action = np.clip(U[None] + noise, u_min, u_max)
    if prior is not None:
        # Prior sample (as in mppi-isaac's `prior`): sample 0 is the given sequence, unperturbed. Its effective
        # noise is prior - U, so the usual update U += sum(omega * noise) weighs it like any other sample.
        pr = np.asarray(prior, dtype=float).reshape(-1, 2)[:T]
        pr = np.vstack([pr, np.repeat(pr[-1:], T - len(pr), axis=0)]) if len(pr) < T else pr
        perturbed_action[0] = np.clip(pr, u_min, u_max)
        noise[0] = perturbed_action[0] - U
    s = tuple(np.full(K, c) for c in state)
    states = np.empty((K, T + 1, 2)); states[:, 0] = state[:2]
    cost_total = np.zeros(K)
    for t in range(T):
        u = (np.minimum(perturbed_action[:, t, 0], v_cap(s[0], s[1])), perturbed_action[:, t, 1])
        s = dynamics(s, u, dt)
        cost_total += running_cost(scene, p, s, u, clr, contact)
        states[:, t + 1, 0], states[:, t + 1, 1] = s[0], s[1]
    cost_total += terminal_cost(scene, p, s)

    # 2. importance weights
    beta = cost_total.min()
    cost_total_non_zero = np.exp(-(cost_total - beta) / lambda_)
    eta = cost_total_non_zero.sum()
    omega = cost_total_non_zero / eta
    ess = 1.0 / np.sum(omega ** 2)

    # 3. update nominal sequence
    U = np.clip(U + np.einsum("k,ktc->tc", omega, noise), u_min, u_max)
    u = U[0].copy()
    u[0] = min(u[0], float(v_cap(state[0], state[1])))  # the applied command obeys the same limit

    # plan = rollout of updated nominal
    s = tuple(np.array([c]) for c in state)
    plan = [(s[0][0], s[1][0])]
    for t in range(T):
        s = dynamics(s, (U[t:t + 1, 0], U[t:t + 1, 1]), dt)
        plan.append((s[0][0], s[1][0]))

    # normalised cost for colouring (0 = best); worst-first so the best draw on top
    num_viz_samples = min(int(p["num_viz_samples"]), K)
    cost_hi = np.quantile(cost_total, 0.8)
    if cost_hi <= beta + 1e-9:
        cost_hi = beta + 1.0
    cost_norm = np.clip((cost_total[:num_viz_samples] - beta) / (cost_hi - beta), 0, 1)
    order = np.argsort(-cost_norm, kind="stable")

    # control-space distribution at the first step, for visualisation: the sampling distribution
    # N(U[0], noise_sigma) versus the omega-weighted covariance of the perturbations
    noise0 = noise[:, 0]
    mean0 = omega @ noise0
    centred = noise0 - mean0
    cov_weighted = (centred * omega[:, None]).T @ centred

    # 4. warm start: shift one step
    U_next = np.vstack([U[1:], U[-1:]])

    return {
        "u": u.tolist(), "U_in": U_in.tolist(), "U_next": U_next.tolist(),
        "viz": states[:num_viz_samples][order].ravel().tolist(), "viz_cost": cost_norm[order].tolist(),
        "num_viz_samples": num_viz_samples, "noise_sigma": noise_sigma.tolist(),
        "u_in0": U_in[0].tolist(), "u_viz0": perturbed_action[:num_viz_samples, 0][order].ravel().tolist(),
        "cov_weighted": cov_weighted.tolist(), "plan": np.array(plan).ravel().tolist(),
        "prior_weight": float(omega[0]) if prior is not None else None,
        "min_cost": float(beta), "mean_cost": float(cost_total.mean()), "ess": float(ess),
    }


def pick_waypoint(tree, lookahead, always=False):
    """Point on the tree's best path about `lookahead` metres ahead of the rover (the path's end if it is
    shorter). By default only used when the path reaches the goal, since otherwise it may end in a dead end;
    with `always` the best path is used regardless, so MPPI never falls back to the true goal."""
    if tree["best_term"] != 2 and not always:
        return None
    path = np.array(tree["best_path"]).reshape(-1, 2)
    if tree["best_term"] == 1 and len(path) > 2:
        path = path[:-1]  # the last node of a path that ended in a collision is inside an obstacle
    if len(path) < 2:
        return None  # the tree found nothing to follow
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    i = int(np.searchsorted(arc, lookahead))
    return path[min(i, len(path) - 1)].tolist()


def plan_request(req):
    """One control step. On MCTS steps the tree search (lidar only) runs first and, if enabled,
    its best path replaces MPPI's nominal sequence U (warm start)."""
    p, U, lidar = req["params"], req.get("U"), req.get("lidar")
    tree, prior = None, None
    waypoint = req.get("waypoint")  # held between MCTS steps by the client
    mode = p.get("mcts_inject")
    if req.get("mcts") and p.get("use_mcts"):
        pts, mpts = lidar_points(req["state"], lidar, bool(p.get("use_lidar_labels"))) if lidar else (np.zeros((0, 2)), np.zeros((0, 2)))
        tree = mcts.search(req["state"], req["scene"], p, pts, int(req["seed"]), mpts)
        if p.get("mcts_warm_start") and mode in ("waypoint", "waypoint+prior"):
            waypoint = pick_waypoint(tree, p.get("mcts_waypoint_dist", 5.0), bool(p.get("mcts_waypoint_always")))  # None -> fall back to the true goal
        if p.get("mcts_warm_start") and mode in ("prior", "waypoint+prior"):
            prior = tree["warm_start_U"]  # MPPI keeps its own nominal; the tree's path competes as one sample
        elif p.get("mcts_warm_start") and mode == "nominal":
            U_tree = np.array(tree["warm_start_U"])
            if p.get("mcts_keep_speed") and U is not None and len(U):  # steering from the tree, speed from MPPI's own nominal
                U_old = np.asarray(U, dtype=float).reshape(-1, 2)
                n = min(len(U_old), len(U_tree))
                U_tree[:n, 0] = U_old[:n, 0]
            U = U_tree.tolist()
    scene = req["scene"]
    # Release the waypoint near the goal: it is a point on the tree's path, not the goal itself, so close in it
    # would pull the rover past or beside the goal. Within the release distance MPPI aims at the true goal.
    release = p.get("mcts_waypoint_release", 0.0)
    gx, gy = scene["goal"]
    if release > 0 and np.hypot(req["state"][0] - gx, req["state"][1] - gy) < release:
        waypoint = None
    if waypoint is not None and p.get("mcts_warm_start") and mode in ("waypoint", "waypoint+prior"):
        scene = {**scene, "goal": waypoint}  # MPPI's goal-distance cost now pulls toward the waypoint
    else:
        waypoint = None
    out = mppi_step(req["state"], scene, p, U, int(req["seed"]), lidar, prior, goal=req["scene"]["goal"])
    out["waypoint"] = waypoint
    if tree:
        out["mcts"] = tree
        out["warm_started"] = bool(p.get("mcts_warm_start"))
    return out


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/api/plan":
            self.send_error(404)
            return
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        out = plan_request(req)
        body = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        # dev server: never let the browser reuse stale app.js / scenes.js
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("MPPI playground on http://localhost:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
