"""UCT Monte-Carlo tree search over the unicycle's motion primitives, built from a lidar scan.

The robot knows obstacles only through the lidar hit points `pts` (world frame). Anything the
lidar did not return a hit for is treated as free. The arena edge is known (map extents).
Hitting an obstacle (within robotR of a hit point) or the arena edge is a terminal state.

Action = hold (v = u_max_v, omega = one of `mcts_num_turns` turn rates spread over +-mcts_turn_max)
for `mcts_action_time` s. Keep mcts_turn_max * mcts_action_time well below pi: at the full rover limit a
single 1 s action turns ~143 deg, so U-turns and loops become one-step moves and the search turns noisy.

Reward (accumulated along the path; a node's value is its expected return-to-go):
  - proximity penalty per action: -mcts_w_obstacle * max over the action of (1 - clearance / mcts_margin)^2,
    where clearance is the distance to the nearest lidar hit point minus robotR (only inside the margin)
  - collision (obstacle or arena edge): -mcts_w_collision, terminal
  - goal reached: +1, terminal
  - optional per-action costs: mcts_w_time (time) and mcts_w_turn * |omega| / mcts_turn_max (turning)
  - non-terminal leaf: 1 - distance_to_goal / initial_distance  (progress, in [0, 1])
"""
import math
import random
import time

import numpy as np
from scipy.spatial import cKDTree

GOAL_TOL = 0.4
PUSH_PAD = 0.15  # same contact distance as server.py
RES = 0.1  # clearance grid resolution (m)


def build_clearance_grid(scene, pts):
    """dist[i][j] = distance from grid cell (i*RES, j*RES) to the nearest lidar hit point."""
    nx, ny = int(scene["w"] / RES) + 1, int(scene["h"] / RES) + 1
    if len(pts) == 0:
        return [[math.inf] * ny for _ in range(nx)]
    gx, gy = np.meshgrid(np.arange(nx) * RES, np.arange(ny) * RES, indexing="ij")
    d, _ = cKDTree(pts).query(np.stack([gx.ravel(), gy.ravel()], axis=1))
    return d.reshape(nx, ny).tolist()


def search(state, scene, p, pts, seed, mpts=None):
    t_start = time.perf_counter()
    rng = random.Random(seed)
    R, W, H = scene["robotR"], scene["w"], scene["h"]
    gx, gy = scene["goal"]
    dist = build_clearance_grid(scene, pts)
    # lidar hits on movable objects: not obstacles, but pushing costs mcts_w_push per action
    dist_mov = build_clearance_grid(scene, mpts) if mpts is not None and len(mpts) else None
    w_push = p.get("mcts_w_push", 0.0)

    n_turns = max(2, int(p["mcts_num_turns"]))
    w_max, v = min(p.get("mcts_turn_max", p["u_max_w"]), p["u_max_w"]), p["u_max_v"]  # tree turn-rate range
    omegas = [-w_max + 2 * w_max * i / (n_turns - 1) for i in range(n_turns)]
    n_sub = max(1, round(p["mcts_action_time"] / p["dt"]))
    h = p["mcts_action_time"] / n_sub
    c_uct, max_depth, rollout_depth = p["mcts_c"], int(p["mcts_depth"]), int(p["mcts_rollout_depth"])
    w_obs, margin, w_col = p["mcts_w_obstacle"], max(p["mcts_margin"], 1e-6), p["mcts_w_collision"]
    w_time, w_turn = p.get("mcts_w_time", 0.0), p.get("mcts_w_turn", 0.0)
    w_dturn = p.get("mcts_w_dturn", 0.0)  # cost on changing the turn rate between consecutive actions (anti-zigzag)
    mid = n_turns // 2  # the straight-ahead action (n_turns is odd); the root's "previous action"
    d0 = max(math.hypot(state[0] - gx, state[1] - gy), 1e-6)

    def clearance(x, y):
        """Distance to the nearest lidar hit minus robotR; negative = collision. Arena edge is terminal too."""
        if x < R or y < R or x > W - R or y > H - R:
            return -1.0
        return dist[int(x / RES + 0.5)][int(y / RES + 0.5)] - R

    def simulate(x, y, th, a, prev):
        """Apply one action. Returns (x, y, th, term, reward); term 0 = alive, 1 = collision, 2 = goal."""
        w = omegas[a]
        step_cost = w_time + w_turn * abs(w) / w_max + w_dturn * abs(w - omegas[prev]) / (2 * w_max)  # time, turning, zigzag
        closest = math.inf
        pushing = False
        for _ in range(n_sub):
            x += v * math.cos(th) * h
            y += v * math.sin(th) * h
            th += w * h
            c = clearance(x, y)
            if c <= 0:
                return x, y, th, 1, -w_col
            if c < closest:
                closest = c
            if dist_mov is not None and not pushing and dist_mov[int(x / RES + 0.5)][int(y / RES + 0.5)] - R < PUSH_PAD:
                pushing = True
                step_cost += w_push
            if math.hypot(x - gx, y - gy) < GOAL_TOL:
                return x, y, th, 2, 1.0 - step_cost - _prox(closest)
        return x, y, th, 0, -step_cost - _prox(closest)

    def _prox(closest):
        return w_obs * (1.0 - closest / margin) ** 2 if closest < margin else 0.0

    def heuristic(x, y):
        return max(0.0, 1.0 - math.hypot(x - gx, y - gy) / d0)

    def rollout(x, y, th, steps, prev):
        """Random actions; returns the return-to-go (rewards collected + progress at the end)."""
        total = 0.0
        for _ in range(steps):
            a = rng.randrange(n_turns)
            x, y, th, term, r = simulate(x, y, th, a, prev)
            prev = a
            total += r
            if term:
                return total
        return total + heuristic(x, y)

    # tree as parallel lists; node 0 is the root. REWARD[n] = reward of the action leading into n.
    X, Y, TH, PARENT, DEPTH, ACT = [state[0]], [state[1]], [state[2]], [-1], [0], [-1]
    N, Wsum, TERM, REWARD = [0], [0.0], [0], [0.0]
    children = [[]]
    untried = [list(range(n_turns))]

    for _ in range(int(p["mcts_iterations"])):
        node, path = 0, [0]
        while True:
            if TERM[node]:
                break
            if DEPTH[node] >= max_depth:
                break
            if untried[node]:  # expansion
                a = untried[node].pop(rng.randrange(len(untried[node])))
                x, y, th, term, r = simulate(X[node], Y[node], TH[node], a, ACT[node] if ACT[node] >= 0 else mid)
                child = len(X)
                X.append(x); Y.append(y); TH.append(th); PARENT.append(node); DEPTH.append(DEPTH[node] + 1); ACT.append(a)
                N.append(0); Wsum.append(0.0); TERM.append(term); REWARD.append(r)
                children.append([]); untried.append(list(range(n_turns)))
                children[node].append(child)
                node = child
                path.append(node)
                break
            log_n = math.log(N[node])  # selection: UCT on action value = edge reward + mean return-to-go
            node = max(children[node], key=lambda c: REWARD[c] + Wsum[c] / N[c] + c_uct * math.sqrt(log_n / N[c]))
            path.append(node)

        # return-to-go from the leaf: terminal nodes already paid their reward on the edge into them
        if TERM[node]:
            g = 0.0
        elif DEPTH[node] >= max_depth:
            g = heuristic(X[node], Y[node])
        else:
            g = rollout(X[node], Y[node], TH[node], rollout_depth, ACT[node] if ACT[node] >= 0 else mid)
        for n in reversed(path):  # backup: each node's G includes the rewards of the edges below it
            N[n] += 1
            Wsum[n] += g
            g += REWARD[n]

    # best path: follow the most visited child
    best, best_nodes, node = [(X[0], Y[0])], [], 0
    while children[node] and not TERM[node]:
        node = max(children[node], key=lambda c: (N[c], REWARD[c] + Wsum[c] / N[c]))  # visits, then value
        best.append((X[node], Y[node]))
        best_nodes.append(node)

    # Warm start for MPPI: the best path's actions as a (v, omega) sequence on MPPI's time grid.
    # A trailing collision action is dropped; short paths are padded with straight-ahead motion.
    acts = [ACT[n] for n in best_nodes if TERM[n] != 1]
    reps = max(1, round(p["mcts_action_time"] / p["dt"]))
    seq = [(v, omegas[a]) for a in acts for _ in range(reps)] or [(v, 0.0)]
    horizon = int(p["horizon"])
    seq += [seq[-1]] * max(0, horizon - len(seq))
    ws = np.array(seq[:horizon], dtype=float)
    k = int(round(p.get("mcts_smooth_time", 0.0) / p["dt"])) | 1  # odd moving-average window on omega (cancels zigzag)
    if k > 1 and len(ws) > 1:
        pad = k // 2
        ws[:, 1] = np.convolve(np.pad(ws[:, 1], pad, mode="edge"), np.ones(k) / k, mode="valid")
    warm_start_U = np.round(ws, 4).tolist()

    r3 = lambda a: np.round(np.asarray(a, dtype=float), 3).tolist()
    return {
        "x": r3(X), "y": r3(Y), "parent": PARENT, "visits": N,
        "value": r3([r + (w / n if n else 0.0) for r, w, n in zip(REWARD, Wsum, N)]), "term": TERM,
        "best_path": r3(np.array(best).ravel()), "best_depth": len(best) - 1, "best_term": TERM[node], "warm_start_U": warm_start_U,
        "iterations": int(p["mcts_iterations"]), "num_nodes": len(X),
        "reached_goal": any(t == 2 for t in TERM), "ms": (time.perf_counter() - t_start) * 1000,
    }
