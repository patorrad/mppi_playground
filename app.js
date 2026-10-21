(function () {
  // Dynamics and collision check mirror server.py (MPPI itself runs in Python).
  function mulberry32(a) {
    return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  // 2D lidar: ray-cast from the rover pose against the circles and the arena walls.
  // Returns absolute beam angles and ranges; ranges == lidar_range means "no return".
  // Range noise is seeded per step so re-planning / scrubbing reproduces the same scan.
  function lidarScan(st, seedIdx) {
    const n = Math.max(2, params.lidar_beams | 0), R = params.lidar_range, fov = params.lidar_fov * Math.PI / 180;
    const full = params.lidar_fov >= 359.9, rnd = mulberry32(seedIdx ^ 0x9e3779b9);
    const gauss = () => Math.sqrt(-2 * Math.log(rnd() || 1e-12)) * Math.cos(2 * Math.PI * rnd());
    const angles = [], ranges = [], movable = []; // movable[i] = 1 if beam i hit a movable object
    for (let i = 0; i < n; i++) {
      const a = st[2] + (full ? -Math.PI + i * 2 * Math.PI / n : -fov / 2 + i * fov / (n - 1));
      const dx = Math.cos(a), dy = Math.sin(a);
      let t = R, mov = 0;
      const tx = dx > 1e-9 ? (scene.w - st[0]) / dx : dx < -1e-9 ? -st[0] / dx : Infinity;
      const ty = dy > 1e-9 ? (scene.h - st[1]) / dy : dy < -1e-9 ? -st[1] / dy : Infinity;
      t = Math.min(t, tx, ty);
      for (const o of scene.obstacles) {
        const fx = st[0] - o.x, fy = st[1] - o.y, b = fx * dx + fy * dy, disc = b * b - (fx * fx + fy * fy - o.r * o.r);
        if (disc >= 0) { const t0 = -b - Math.sqrt(disc); if (t0 > 0 && t0 < t) { t = t0; mov = o.movable ? 1 : 0; } }
      }
      if (t < R) t = Math.min(Math.max(t + params.lidar_noise * gauss(), 0.01), R - 0.001);
      angles.push(a); ranges.push(t); movable.push(t < R ? mov : 0);
    }
    return { angles, ranges, movable, max_range: R };
  }

  const unicycleStep = (s, u, dt) => [s[0] + u[0] * Math.cos(s[2]) * dt, s[1] + u[0] * Math.sin(s[2]) * dt, s[2] + u[1] * dt];
  function clearance(sc, x, y, skipMovable = false) {
    let c = Math.min(x, y, sc.w - x, sc.h - y) - sc.robotR;
    for (const o of sc.obstacles) if (!(skipMovable && o.movable)) c = Math.min(c, Math.hypot(x - o.x, y - o.y) - o.r - sc.robotR);
    return c;
  }
  const $ = id => document.getElementById(id);

  // ---------- parameters ----------
  const params = {
    num_samples: 500, horizon: 25, dt: 0.1, lambda_: 5, noise_sigma_v: 0.8, noise_sigma_w: 1.5, u_min_v: 0, u_max_v: 2, u_max_w: 2.5,
    w_goal: 1, w_terminal: 10, w_obstacle: 20, margin: 0.6, w_collision: 1000, w_effort: 0.05, num_viz_samples: 100,
    lidar_beams: 90, lidar_range: 14, lidar_fov: 360, lidar_noise: 0.02, use_lidar: true,
    use_mcts: true, mcts_iterations: 6000, mcts_c: 0.2, mcts_depth: 15, mcts_rollout_depth: 8, mcts_action_time: 1.0, mcts_num_turns: 5,
    mcts_w_obstacle: 0.02, mcts_margin: 1.0, mcts_w_collision: 2.0,
    mcts_every: 1, mcts_warm_start: true, mcts_turn_max: 1.0, mcts_w_time: 0, mcts_w_turn: 0, mcts_w_dturn: 0.02, mcts_smooth_time: 0.8, mcts_inject: 'waypoint', mcts_waypoint_dist: 6, mcts_waypoint_always: true, mcts_waypoint_release: 2.0, goal_speed_gain: 1.0, goal_speed_floor: 0.2,
    use_lidar_labels: true, w_push: 30, mcts_w_push: 0.1,
  };
  const SPEC = [
    ['Sampling', [
      ['num_samples', 'num_samples', 10, 5000, 10], ['horizon', 'horizon', 5, 100, 1], ['dt', 'dt (s)', 0.02, 0.25, 0.01],
      ['lambda_', 'lambda_', 0.1, 50, 0.1], ['noise_sigma_v', 'noise_sigma v', 0.05, 2, 0.05], ['noise_sigma_w', 'noise_sigma ω', 0.05, 4, 0.05]]],
    ['Rover limits', [
      ['goal_speed_gain', 'goal_speed_gain (1/s)', 0, 3, 0.1], ['goal_speed_floor', 'goal_speed_floor', 0, 1, 0.05],
      ['u_min_v', 'u_min v', -1, 0, 0.1], ['u_max_v', 'u_max v', 0.2, 4, 0.1], ['u_max_w', 'u_max ω', 0.2, 5, 0.1]]],
    ['Cost', [
      ['w_goal', 'w_goal', 0, 5, 0.05], ['w_terminal', 'w_terminal', 0, 50, 0.5], ['w_obstacle', 'w_obstacle', 0, 100, 1],
      ['margin', 'margin', 0.1, 2, 0.05], ['w_collision', 'w_collision', 0, 5000, 50], ['w_effort', 'w_effort', 0, 1, 0.01]]],
    ['Lidar', [
      ['lidar_beams', 'lidar_beams', 8, 720, 1], ['lidar_range', 'lidar_range', 1, 20, 0.5],
      ['lidar_fov', 'lidar_fov (°)', 30, 360, 10], ['lidar_noise', 'lidar_noise', 0, 0.3, 0.01]]],
    ['MCTS', [
      ['mcts_every', 'mcts_every (steps)', 1, 50, 1],
      ['mcts_iterations', 'mcts_iterations', 100, 20000, 100], ['mcts_c', 'mcts_c', 0, 2, 0.05], ['mcts_depth', 'mcts_depth', 3, 40, 1],
      ['mcts_rollout_depth', 'mcts_rollout_depth', 0, 30, 1], ['mcts_action_time', 'mcts_action_time', 0.2, 1.5, 0.1],
      ['mcts_num_turns', 'mcts_num_turns', 3, 9, 2], ['mcts_turn_max', 'mcts_turn_max (rad/s)', 0.2, 5, 0.1],
      ['mcts_w_time', 'mcts_w_time', 0, 0.2, 0.005], ['mcts_w_turn', 'mcts_w_turn', 0, 0.2, 0.005],
      ['mcts_w_dturn', 'mcts_w_dturn', 0, 0.2, 0.005], ['mcts_smooth_time', 'mcts_smooth_time (s)', 0, 3, 0.1],
      ['mcts_waypoint_dist', 'mcts_waypoint_dist (m)', 1, 15, 0.5], ['mcts_waypoint_release', 'mcts_waypoint_release (m)', 0, 6, 0.25], ['mcts_w_obstacle', 'mcts_w_obstacle', 0, 2, 0.05],
      ['mcts_margin', 'mcts_margin', 0.2, 3, 0.1], ['mcts_w_collision', 'mcts_w_collision', 0, 5, 0.1]]],
    ['Movable objects', [['w_push', 'w_push', 0, 200, 5], ['mcts_w_push', 'mcts_w_push', 0, 1, 0.05]]],
    ['Display', [['num_viz_samples', 'num_viz_samples', 0, 300, 10]]],
  ];

  const sliderBox = $('sliders');
  for (const [group, items] of SPEC) {
    const h = document.createElement('h3'); h.textContent = group; sliderBox.appendChild(h);
    for (const [key, label, min, max, step] of items) {
      const row = document.createElement('div'); row.className = 'row';
      row.innerHTML = `<span>${label}</span><input type="range" min="${min}" max="${max}" step="${step}" value="${params[key]}"><span></span>`;
      const input = row.children[1], out = row.children[2];
      const show = () => { out.textContent = +params[key]; };
      input.oninput = () => { params[key] = +input.value; show(); draw(); scheduleReplan(); };
      show(); sliderBox.appendChild(row);
    }
  }

  // ---------- scene + simulation state ----------
  let scene, sim;
  const sceneSel = $('scene');
  Object.keys(window.SCENES).forEach(n => sceneSel.add(new Option(n, n)));

  let nextId = 1;
  function loadScene(name) {
    scene = window.SCENES[name]();
    scene.obstacles.forEach(o => { o.id = nextId++; o.x0 = o.x; o.y0 = o.y; }); // x0,y0 = layout restored on Reset
    resetSim();
  }

  // ---------- movable objects ----------
  // Quasi-static pushing: a movable object overlapped by the rover is moved out along the contact normal. It is in
  // turn pushed out of walls, fixed obstacles and other movable objects. If it cannot move (against a wall, a fixed
  // obstacle or a jammed pile) the rover is pushed back instead, so a blocked object acts like a wall.
  function resolveContacts(pos) {
    const R = scene.robotR, mov = scene.obstacles.filter(o => o.movable), fix = scene.obstacles.filter(o => !o.movable);
    const sep = (ax, ay, ar, bx, by, br) => { // unit normal a->b and overlap, or null
      const dx = bx - ax, dy = by - ay, d = Math.hypot(dx, dy), ov = ar + br - d;
      return ov > 0 ? (d > 1e-9 ? [dx / d, dy / d, ov] : [1, 0, ov]) : null;
    };
    for (let it = 0; it < 12; it++) {
      for (const m of mov) {
        let c = sep(pos[0], pos[1], R, m.x, m.y, m.r);
        if (c) { m.x += c[0] * c[2]; m.y += c[1] * c[2]; }
        for (const f of fix) { c = sep(f.x, f.y, f.r, m.x, m.y, m.r); if (c) { m.x += c[0] * c[2]; m.y += c[1] * c[2]; } }
        m.x = Math.min(Math.max(m.x, m.r), scene.w - m.r); m.y = Math.min(Math.max(m.y, m.r), scene.h - m.r);
        for (const o of mov) if (o !== m) {
          c = sep(o.x, o.y, o.r, m.x, m.y, m.r);
          if (c) { m.x += c[0] * c[2] / 2; m.y += c[1] * c[2] / 2; o.x -= c[0] * c[2] / 2; o.y -= c[1] * c[2] / 2; }
        }
      }
    }
    for (const m of mov) { const c = sep(m.x, m.y, m.r, pos[0], pos[1], R); if (c) { pos[0] += c[0] * c[2]; pos[1] += c[1] * c[2]; } }
    for (const f of fix) { const c = sep(f.x, f.y, f.r, pos[0], pos[1], R); if (c) { pos[0] += c[0] * c[2]; pos[1] += c[1] * c[2]; } }
  }
  // Object positions at every state are kept so scrubbing back also moves the objects back.
  const snapshot = () => Object.fromEntries(scene.obstacles.filter(o => o.movable).map(o => [o.id, [o.x, o.y]]));
  function applySnap(i) {
    const snap = sim.snaps[i];
    if (snap) for (const o of scene.obstacles) if (o.movable && snap[o.id]) { o.x = snap[o.id][0]; o.y = snap[o.id][1]; }
  }

  function resetSim() {
    for (const o of scene.obstacles) if (o.movable) { o.x = o.x0; o.y = o.y0; }
    sim = { snaps: [snapshot()], U: null, states: [scene.start.slice()], plans: [], preview: null, waypoint: null, vi: 0, done: false, status: '' };
    playing = false; updateUI();
  }

  function goalDist(s) { return Math.hypot(s[0] - scene.goal[0], s[1] - scene.goal[1]); }

  // signature of everything a plan depends on besides state/seed; stale plans are re-planned when viewed
  const sig = () => JSON.stringify([params, scene.goal, scene.obstacles]);
  const toPlan = (r, t0, planSig, scan, wpIn) => ({ sig: planSig, scan, waypoint: r.waypoint || null, waypointIn: wpIn || null, mcts: r.mcts || null, warmStarted: !!r.warm_started, priorWeight: r.prior_weight, u: r.u, Uin: r.U_in, viz: Float32Array.from(r.viz), vizT: Float32Array.from(r.viz_cost),
    numVizSamples: r.num_viz_samples, opt: Float32Array.from(r.plan), minCost: r.min_cost, meanCost: r.mean_cost, ess: r.ess,
    ms: performance.now() - t0,
    noiseSigma: r.noise_sigma, covWeighted: r.cov_weighted, uIn0: r.u_in0, uViz0: r.u_viz0 });
  // plan shown for the current view; `preview` holds a re-plan of a state that has no recorded plan yet
  // the tree shown is the most recent one at or before the viewed step (MCTS only runs every mcts_every steps)
  const treePlan = () => { for (let i = Math.min(sim.vi, sim.plans.length - 1); i >= 0; i--) if (sim.plans[i].mcts) return { plan: sim.plans[i], step: i };
    return sim.preview && sim.preview.mcts && sim.vi === sim.states.length - 1 ? { plan: sim.preview, step: sim.vi } : null; };
  const curPlan = () => sim.plans[sim.vi] || (sim.vi === sim.states.length - 1 ? sim.preview : null);

  // Re-run MPPI for the viewed state with the current parameters/scene (paused only).
  // Uses the same warm start and seed as the original step, so only the parameter change differs.
  let replanTimer = null, replanToken = 0;
  function scheduleReplan() {
    if (!$('liveReplan').checked) return;
    clearTimeout(replanTimer); replanTimer = setTimeout(replan, 60);
  }
  async function replan() {
    if (playing || busy || !sim) return;
    const mine = sim, vi = sim.vi, token = ++replanToken, t0 = performance.now(), reqSig = sig();
    const scan = lidarScan(sim.states[vi], (+$('seed').value) * 100003 + vi);
    const old = sim.plans[vi];
    const wpIn = old ? old.waypointIn : (vi === sim.states.length - 1 ? sim.waypoint : null);
    const U = old ? old.Uin : (vi === sim.states.length - 1 ? sim.U : null);
    let r;
    try {
      const res = await fetch('/api/plan', { method: 'POST', body: JSON.stringify({
        state: sim.states[vi], scene, params, lidar: scan, mcts: vi % params.mcts_every === 0, waypoint: wpIn, U, seed: (+$('seed').value) * 100003 + vi }) });
      r = await res.json();
    } catch (e) { return; }
    if (sim !== mine || token !== replanToken || playing || sim.vi !== vi) return;
    const plan = toPlan(r, t0, reqSig, scan, wpIn);
    if (old) sim.plans[vi] = plan; else if (vi === sim.states.length - 1) sim.preview = plan;
    updateUI();
  }

  async function step() {
    if (sim.done || busy) return;
    busy = true; replanToken++;
    const mine = sim;
    if (sim.vi < sim.plans.length - 1) { // resume from a rewound step
      const prev = sim.plans[sim.vi];
      if (prev) { sim.U = prev.Uin; sim.waypoint = prev.waypointIn; }
      sim.states.length = sim.vi + 1; sim.plans.length = sim.vi; sim.snaps.length = sim.vi + 1;
    }
    applySnap(sim.states.length - 1); // objects as they are at the head of the history
    const s = sim.states[sim.states.length - 1];
    const t0 = performance.now(), reqSig = sig();
    const scan = lidarScan(s, (+$('seed').value) * 100003 + sim.plans.length), wpIn = sim.waypoint;
    let r;
    try {
      const res = await fetch('/api/plan', { method: 'POST', body: JSON.stringify({
        state: s, scene, params, lidar: scan, mcts: sim.plans.length % params.mcts_every === 0, waypoint: wpIn, U: sim.U, seed: (+$('seed').value) * 100003 + sim.plans.length }) });
      r = await res.json();
    } catch (e) { busy = false; playing = false; updateUI(); return; }
    busy = false;
    if (sim !== mine) return; // reset while waiting
    const plan = toPlan(r, t0, reqSig, scan, wpIn);
    sim.preview = null;
    sim.U = r.U_next; sim.waypoint = r.waypoint || null; // held until the next MCTS step
    const next = unicycleStep(s, plan.u, params.dt);
    const hitFixed = clearance(scene, next[0], next[1], true) < 0; // the commanded move ran into a fixed obstacle / wall
    resolveContacts(next); // pushes movable objects (and moves the rover back if one is blocked)
    sim.plans.push(plan); sim.states.push(next); sim.snaps.push(snapshot());
    if (goalDist(next) < 0.4) { sim.done = true; sim.status = 'Goal reached'; }
    else if (hitFixed) { sim.done = true; sim.status = 'Collision'; }
    else if (sim.plans.length >= 3000) { sim.done = true; sim.status = 'Timeout'; }
    sim.vi = sim.done ? sim.states.length - 1 : sim.plans.length - 1;
    applySnap(sim.vi);
    if (sim.done) playing = false;
    updateUI();
  }

  function seek(i) {
    sim.vi = Math.max(0, Math.min(sim.states.length - 1, i));
    applySnap(sim.vi);
    playing = false; updateUI();
    const p = curPlan();
    if (p && p.sig !== sig()) scheduleReplan(); // planned with older settings
  }

  // ---------- transport ----------
  let busy = false, playing = false, acc = 0, last = performance.now();
  function frame(now) {
    const dtms = now - last; last = now;
    if (playing) {
      acc += dtms;
      const period = 1000 / +$('speed').value;
      if (acc >= period && !busy) { acc %= period; step(); }
      if (acc > period) acc = 0;
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  $('play').onclick = () => { if (sim.done) return; playing = !playing; acc = 0; updateUI(); };
  $('stepFwd').onclick = () => { playing = false; if (sim.vi < sim.plans.length - 1) seek(sim.vi + 1); else step(); };
  $('stepBack').onclick = () => seek(sim.vi - 1);
  $('reset').onclick = () => resetSim();
  $('timeline').oninput = e => seek(+e.target.value);
  $('scene').onchange = () => loadScene(sceneSel.value);
  $('speed').oninput = () => { $('speedV').textContent = $('speed').value; };
  ['showSamples', 'showPlan', 'showPath', 'showLidar', 'showMcts'].forEach(id => $(id).onchange = draw);
  $('lidarLabels').onchange = () => { params.use_lidar_labels = $('lidarLabels').checked; scheduleReplan(); };
  $('waypointAlways').onchange = () => { params.mcts_waypoint_always = $('waypointAlways').checked; scheduleReplan(); };
  $('injectMode').value = params.mcts_inject;
  $('injectMode').onchange = () => { params.mcts_inject = $('injectMode').value; scheduleReplan(); };
  $('warmStart').onchange = () => { params.mcts_warm_start = $('warmStart').checked; scheduleReplan(); };
  $('useMcts').onchange = () => { params.use_mcts = $('useMcts').checked; scheduleReplan(); };
  $('useLidar').onchange = () => { params.use_lidar = $('useLidar').checked; scheduleReplan(); };
  window.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
    if (e.code === 'Space') { e.preventDefault(); $('play').click(); }
    else if (e.code === 'ArrowRight') $('stepFwd').click();
    else if (e.code === 'ArrowLeft') $('stepBack').click();
  });

  function updateUI() {
    $('play').textContent = playing ? 'Pause' : 'Play';
    const tl = $('timeline'); tl.max = sim.states.length - 1; tl.value = sim.vi;
    const s = sim.states[sim.vi], p = curPlan();
    const stat = [
      ['step', sim.vi], ['t', (sim.vi * params.dt).toFixed(1) + ' s'], ['dist to goal', goalDist(s).toFixed(2) + ' m'],
    ];
    if (p) stat.push(['v', p.u[0].toFixed(2)], ['ω', p.u[1].toFixed(2)], ['min cost', p.minCost.toFixed(1)],
      ['ESS', p.ess.toFixed(0) + ' / ' + params.num_samples], ['compute', p.ms.toFixed(1) + ' ms']);
    if (p && p.priorWeight != null) stat.push(['MCTS prior weight', p.priorWeight.toFixed(3)]);
    const tp = treePlan();
    if (tp) { const m = tp.plan.mcts;
      stat.push(['MCTS from step', tp.step + (tp.plan.warmStarted && tp.step === sim.vi ? ' (warm-started MPPI)' : '')], ['nodes', m.num_nodes],
        ['best depth', m.best_depth + (m.reached_goal ? ' (goal found)' : '')], ['MCTS', m.ms.toFixed(0) + ' ms']); }
    $('stats').innerHTML = stat.map(([k, v]) => `<span style="white-space:nowrap;margin-right:16px">${k} <b>${v}</b></span>`).join(' ') +
      (sim.done ? ` <span id="status">● ${sim.status}</span>` : '');
    draw();
  }

  // ---------- rendering ----------
  const cv = $('cv'), ctx = cv.getContext('2d');
  let scale = 1, ox = 0, oy = 0;
  const X = x => ox + x * scale, Y = y => oy + (scene.h - y) * scale; // y up

  function resize() {
    const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    const w = Math.round(r.width * dpr), h = Math.round(r.height * dpr);
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; } // assigning width/height clears the canvas
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const pad = 16;
    scale = Math.min((r.width - 2 * pad) / scene.w, (r.height - 2 * pad) / scene.h);
    ox = (r.width - scene.w * scale) / 2; oy = (r.height - scene.h * scale) / 2;
    draw();
  }
  new ResizeObserver(resize).observe(cv.parentElement);

  function polyline(arr, off, n) {
    ctx.beginPath();
    for (let i = 0; i < n; i++) { const x = X(arr[off + 2 * i]), y = Y(arr[off + 2 * i + 1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
    ctx.stroke();
  }

  function draw() {
    if (!scene || !sim) return;
    drawCov();
    const r = cv.getBoundingClientRect();
    ctx.clearRect(0, 0, r.width, r.height);
    // world
    ctx.fillStyle = '#10151e'; ctx.fillRect(X(0), Y(scene.h), scene.w * scale, scene.h * scale);
    ctx.strokeStyle = '#1b2230'; ctx.lineWidth = 1;
    for (let x = 0; x <= scene.w; x++) { ctx.beginPath(); ctx.moveTo(X(x), Y(0)); ctx.lineTo(X(x), Y(scene.h)); ctx.stroke(); }
    for (let y = 0; y <= scene.h; y++) { ctx.beginPath(); ctx.moveTo(X(0), Y(y)); ctx.lineTo(X(scene.w), Y(y)); ctx.stroke(); }
    ctx.strokeStyle = '#3a4258'; ctx.strokeRect(X(0), Y(scene.h), scene.w * scale, scene.h * scale);
    // obstacles
    for (const o of scene.obstacles) {
      ctx.beginPath(); ctx.arc(X(o.x), Y(o.y), o.r * scale, 0, 7); ctx.fillStyle = o.movable ? '#e05252' : '#5b6478'; ctx.fill();
      if (o.movable) { ctx.strokeStyle = '#ffa3a3'; ctx.lineWidth = 1.5; ctx.stroke(); continue; } // movable: no safety ring
      ctx.beginPath(); ctx.arc(X(o.x), Y(o.y), (o.r + scene.robotR + params.margin) * scale, 0, 7);
      ctx.strokeStyle = 'rgba(255,255,255,.08)'; ctx.setLineDash([3, 4]); ctx.stroke(); ctx.setLineDash([]);
    }
    // start / goal
    ctx.beginPath(); ctx.arc(X(scene.start[0]), Y(scene.start[1]), 5, 0, 7); ctx.strokeStyle = '#8892a6'; ctx.lineWidth = 2; ctx.stroke();
    const g = scene.goal;
    ctx.beginPath(); ctx.arc(X(g[0]), Y(g[1]), 0.4 * scale, 0, 7); ctx.fillStyle = 'rgba(80,220,120,.25)'; ctx.fill();
    ctx.strokeStyle = '#50dc78'; ctx.stroke();

    const vi = sim.vi, s = sim.states[vi], p = curPlan();
    if (p && p.waypoint) { // MPPI's current target when following the MCTS waypoint
      const wx = X(p.waypoint[0]), wy = Y(p.waypoint[1]);
      ctx.strokeStyle = '#e879f9'; ctx.lineWidth = 2; ctx.beginPath();
      ctx.moveTo(wx, wy - 7); ctx.lineTo(wx + 7, wy); ctx.lineTo(wx, wy + 7); ctx.lineTo(wx - 7, wy); ctx.closePath(); ctx.stroke();
    }
    // driven path
    if ($('showPath').checked) {
      ctx.strokeStyle = '#f4a261'; ctx.lineWidth = 2; ctx.beginPath();
      for (let i = 0; i <= vi; i++) { const q = sim.states[i]; i ? ctx.lineTo(X(q[0]), Y(q[1])) : ctx.moveTo(X(q[0]), Y(q[1])); }
      ctx.stroke();
    }
    // MCTS tree (step 0): edges parent -> child binned by mean value, dead ends red, best path gold
    const tp = treePlan();
    if (tp && $('showMcts').checked) {
      ctx.save(); ctx.globalAlpha = tp.step === vi ? 1 : 0.4; // older tree: faint
      const m = tp.plan.mcts, bins = 5, paths = Array.from({ length: bins }, () => new Path2D());
      const vmin = Math.min(...m.value.slice(1)), vspan = (Math.max(...m.value.slice(1)) - vmin) || 1; // values can be negative
      for (let i = 1; i < m.x.length; i++) {
        const b = Math.min(bins - 1, Math.floor((m.value[i] - vmin) / vspan * bins)), pi = m.parent[i];
        paths[b].moveTo(X(m.x[pi]), Y(m.y[pi])); paths[b].lineTo(X(m.x[i]), Y(m.y[i]));
      }
      ctx.lineWidth = 1;
      paths.forEach((pth, b) => { ctx.strokeStyle = `hsla(${40 + 90 * b / (bins - 1)}, 80%, 60%, ${0.12 + 0.12 * b})`; ctx.stroke(pth); });
      for (let i = 1; i < m.x.length; i++) {
        if (m.term[i] === 1) { ctx.fillStyle = 'rgba(255,70,70,.7)'; ctx.fillRect(X(m.x[i]) - 1.5, Y(m.y[i]) - 1.5, 3, 3); }
        else if (m.term[i] === 2) { ctx.fillStyle = '#50dc78'; ctx.beginPath(); ctx.arc(X(m.x[i]), Y(m.y[i]), 3, 0, 7); ctx.fill(); }
      }
      ctx.strokeStyle = '#ffd166'; ctx.lineWidth = 3; polyline(m.best_path, 0, m.best_path.length / 2);
      ctx.restore();
    }
    // lidar: faint beams, red hit points
    if (p && p.scan && $('showLidar').checked) {
      const sc = p.scan, hx = [], hy = [];
      ctx.lineWidth = 1;
      for (let i = 0; i < sc.ranges.length; i++) {
        const ex = s[0] + sc.ranges[i] * Math.cos(sc.angles[i]), ey = s[1] + sc.ranges[i] * Math.sin(sc.angles[i]);
        const hit = sc.ranges[i] < sc.max_range - 1e-6;
        ctx.strokeStyle = hit ? 'rgba(212,225,87,.18)' : 'rgba(212,225,87,.06)';
        ctx.beginPath(); ctx.moveTo(X(s[0]), Y(s[1])); ctx.lineTo(X(ex), Y(ey)); ctx.stroke();
        if (hit) { hx.push(ex); hy.push(ey); }
      }
      ctx.fillStyle = '#d4e157'; // lidar hits are yellow-green so they are not confused with red movable objects
      for (let i = 0; i < hx.length; i++) { ctx.beginPath(); ctx.arc(X(hx[i]), Y(hy[i]), 2, 0, 7); ctx.fill(); }
    }
    // sampled trajectories
    if (p && $('showSamples').checked) {
      const n = p.viz.length / p.numVizSamples / 2;
      ctx.lineWidth = 1;
      for (let i = 0; i < p.numVizSamples; i++) {
        const t = p.vizT[i];
        ctx.strokeStyle = `hsla(${185 + 140 * t}, 85%, ${65 - 10 * t}%, ${0.55 - 0.4 * t})`;
        polyline(p.viz, i * n * 2, n);
      }
    }
    // plan
    if (p && $('showPlan').checked) { ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2.5; polyline(p.opt, 0, p.opt.length / 2); }
    // rover
    ctx.save(); ctx.translate(X(s[0]), Y(s[1])); ctx.rotate(-s[2]);
    ctx.beginPath(); ctx.arc(0, 0, scene.robotR * scale, 0, 7); ctx.fillStyle = 'rgba(244,162,97,.25)'; ctx.fill();
    ctx.beginPath(); const R = scene.robotR * scale;
    ctx.moveTo(R, 0); ctx.lineTo(-R * 0.8, R * 0.7); ctx.lineTo(-R * 0.8, -R * 0.7); ctx.closePath();
    ctx.fillStyle = '#f4a261'; ctx.fill(); ctx.restore();
  }


  // ---------- control-space covariance panel ----------
  const covCv = $('covCv'), cctx = covCv.getContext('2d');
  const fmt = x => (Math.abs(x) < 1e-3 ? 0 : x).toFixed(3);
  const matHtml = (title, m) => `<div><b>${title}</b><table><tr><td>${fmt(m[0][0])}</td><td>${fmt(m[0][1])}</td></tr>` +
    `<tr><td>${fmt(m[1][0])}</td><td>${fmt(m[1][1])}</td></tr></table></div>`;

  // draw k-sigma ellipse of a symmetric 2x2 covariance centred at (cx, cy) in control space
  function covEllipse(C, cx, cy, k, px, py) {
    const a = C[0][0], b = C[0][1], c = C[1][1];
    const mid = (a + c) / 2, rad = Math.sqrt(((a - c) / 2) ** 2 + b * b);
    const l1 = Math.max(mid + rad, 0), l2 = Math.max(mid - rad, 0), ang = 0.5 * Math.atan2(2 * b, a - c);
    cctx.beginPath();
    for (let i = 0; i <= 64; i++) {
      const t = (i / 64) * 2 * Math.PI, ex = k * Math.sqrt(l1) * Math.cos(t), ey = k * Math.sqrt(l2) * Math.sin(t);
      const x = cx + ex * Math.cos(ang) - ey * Math.sin(ang), y = cy + ex * Math.sin(ang) + ey * Math.cos(ang);
      i ? cctx.lineTo(px(x), py(y)) : cctx.moveTo(px(x), py(y));
    }
    cctx.stroke();
  }

  function drawCov() {
    const w = covCv.clientWidth, h = covCv.clientHeight, dpr = window.devicePixelRatio || 1;
    if (covCv.width !== Math.round(w * dpr)) { covCv.width = Math.round(w * dpr); covCv.height = Math.round(h * dpr); }
    cctx.setTransform(dpr, 0, 0, dpr, 0, 0); cctx.clearRect(0, 0, w, h);
    const p = curPlan(), m = 28;
    // axes span the control limits (x: v, y: omega)
    const vLo = params.u_min_v, vHi = params.u_max_v, wLim = params.u_max_w;
    const px = v => m + (v - vLo) / (vHi - vLo || 1) * (w - m - 8), py = om => h - m + (om + wLim) / (2 * wLim) * -(h - m - 8);
    cctx.fillStyle = '#10151e'; cctx.fillRect(px(vLo), py(wLim), px(vHi) - px(vLo), py(-wLim) - py(wLim));
    cctx.strokeStyle = '#3a4258'; cctx.lineWidth = 1; cctx.strokeRect(px(vLo), py(wLim), px(vHi) - px(vLo), py(-wLim) - py(wLim));
    cctx.beginPath(); cctx.moveTo(px(vLo), py(0)); cctx.lineTo(px(vHi), py(0)); cctx.stroke();
    cctx.fillStyle = '#8892a6'; cctx.font = '11px system-ui';
    cctx.fillText('v', px(vHi) - 8, py(0) - 4); cctx.fillText('ω', px(vLo) + 4, py(wLim) + 12);
    cctx.fillText(`${vLo}`, px(vLo) - 2, h - 12); cctx.fillText(`${vHi}`, px(vHi) - 14, h - 12);
    if (!p) { $('covText').innerHTML = ''; return; }
    // sampled controls at t=0, coloured like the rollouts
    const n = p.numVizSamples;
    for (let i = 0; i < n; i++) {
      const t = p.vizT[i];
      cctx.fillStyle = `hsla(${185 + 140 * t}, 85%, ${65 - 10 * t}%, ${0.7 - 0.45 * t})`;
      cctx.beginPath(); cctx.arc(px(p.uViz0[2 * i]), py(p.uViz0[2 * i + 1]), 2, 0, 7); cctx.fill();
    }
    // sampling distribution N(U[0], noise_sigma): 1 and 2 sigma
    cctx.setLineDash([4, 3]); cctx.strokeStyle = '#8892a6'; cctx.lineWidth = 1;
    covEllipse(p.noiseSigma, p.uIn0[0], p.uIn0[1], 1, px, py); covEllipse(p.noiseSigma, p.uIn0[0], p.uIn0[1], 2, px, py);
    // omega-weighted covariance around the updated command
    cctx.setLineDash([]); cctx.strokeStyle = '#ffffff'; cctx.lineWidth = 2;
    covEllipse(p.covWeighted, p.u[0], p.u[1], 1, px, py);
    cctx.fillStyle = '#f4a261'; cctx.beginPath(); cctx.arc(px(p.uIn0[0]), py(p.uIn0[1]), 3, 0, 7); cctx.fill();
    cctx.fillStyle = '#fff'; cctx.beginPath(); cctx.arc(px(p.u[0]), py(p.u[1]), 3.5, 0, 7); cctx.fill();
    $('covText').innerHTML = matHtml('noise_sigma (sampling)', p.noiseSigma) + matHtml('ω-weighted (after update)', p.covWeighted) +
      '<div style="grid-column:1/3">dashed: sampling 1σ/2σ around nominal U[0] (orange). white: 1σ of the cost-weighted perturbations around the new command.</div>';
  }

  // ---------- editing the scene with the mouse ----------
  let drag = null;
  const toWorld = e => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left - ox) / scale, scene.h - (e.clientY - r.top - oy) / scale]; };
  const obstacleAt = (x, y) => scene.obstacles.findIndex(o => Math.hypot(x - o.x, y - o.y) <= o.r);

  cv.onmousedown = e => {
    const [x, y] = toWorld(e);
    if (Math.hypot(x - scene.goal[0], y - scene.goal[1]) < 0.6) drag = { kind: 'goal' };
    else if (Math.hypot(x - scene.start[0], y - scene.start[1]) < 0.6) drag = { kind: 'start' };
    else { const i = obstacleAt(x, y); if (i >= 0) drag = { kind: 'obs', i, dx: x - scene.obstacles[i].x, dy: y - scene.obstacles[i].y }; }
  };
  window.onmousemove = e => {
    if (!drag) return;
    const [x, y] = toWorld(e);
    if (drag.kind === 'goal') scene.goal = [x, y];
    else if (drag.kind === 'start') { scene.start[0] = x; scene.start[1] = y; }
    else { const o = scene.obstacles[drag.i]; o.x = x - drag.dx; o.y = y - drag.dy; o.x0 = o.x; o.y0 = o.y; }
    draw(); scheduleReplan();
  };
  window.onmouseup = () => {
    if (drag && drag.kind === 'start') resetSim();
    else if (drag && drag.kind === 'obs' && sim.snaps[sim.vi]) Object.assign(sim.snaps[sim.vi], snapshot()); // keep a dragged red object where it was put
    drag = null;
  };
  cv.ondblclick = e => {
    const [x, y] = toWorld(e), i = obstacleAt(x, y);
    if (e.shiftKey) { // shift+double-click: toggle an obstacle between fixed and movable, or add a movable one
      if (i >= 0) scene.obstacles[i].movable = !scene.obstacles[i].movable;
      else scene.obstacles.push({ x, y, r: 0.7, movable: true, id: nextId++, x0: x, y0: y });
      if (sim.snaps[sim.vi]) Object.assign(sim.snaps[sim.vi], snapshot());
    } else if (i >= 0) scene.obstacles.splice(i, 1);
    else scene.obstacles.push({ x, y, r: 0.8, id: nextId++, x0: x, y0: y });
    draw(); scheduleReplan();
  };

  loadScene(sceneSel.value);
  resize();
})();
