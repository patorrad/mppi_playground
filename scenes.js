// Scene presets. World is w x h metres; obstacles are circles.
(function (g) {
  const base = () => ({ w: 20, h: 12, robotR: 0.35, start: [1.5, 6, 0], goal: [18.5, 6], obstacles: [] });

  const scenes = {
    'Scattered': () => {
      const s = base();
      [[5, 4, 1], [6, 8.5, 1.1], [9.5, 6, 1.2], [12, 2.5, 1], [12.5, 9.5, 1.1], [15.5, 5, 0.9], [15, 8.5, 0.8], [8, 1.5, 0.8], [8.5, 10.8, 0.7]]
        .forEach(([x, y, r]) => s.obstacles.push({ x, y, r }));
      return s;
    },
    'Wall with gap': () => {
      const s = base();
      for (let y = 0.5; y <= 5.5; y += 1) s.obstacles.push({ x: 10, y, r: 0.7 });
      for (let y = 8.5; y <= 11.5; y += 1) s.obstacles.push({ x: 10, y, r: 0.7 });
      return s;
    },
    'Concave trap': () => {
      const s = base();
      for (let y = 3; y <= 9; y += 0.8) s.obstacles.push({ x: 12, y, r: 0.6 });
      for (let x = 8; x <= 12; x += 0.8) { s.obstacles.push({ x, y: 3, r: 0.6 }); s.obstacles.push({ x, y: 9, r: 0.6 }); }
      return s;
    },
    // The concave trap plus clutter on the approach and around the goal. Routes around the U stay open
    // (above it and below it), but they are narrower and need threading. Drag obstacles to rearrange.
    'Concave trap + clutter': () => {
      const s = scenes['Concave trap']();
      [[4, 3, 0.8], [4.5, 9, 0.9], [6.5, 5.3, 0.6], [6.5, 8.1, 0.6], [15.5, 3.5, 0.9], [15, 8.3, 0.8], [17, 4.7, 0.7],
        [16.5, 10, 0.7], [14, 1.5, 0.6], [14, 10.8, 0.5]].forEach(([x, y, r]) => s.obstacles.push({ x, y, r }));
      return s;
    },
    // A full-height wall of tangent circles. Red ones are movable: the rover can push them out of the way to open
    // a gap. The gap left by a pushed object is 1.4 m wide (the two fixed neighbours are one object apart).
    'Pushable wall 1': () => {
      const s = base();
      for (let k = 0; k <= 8; k++) s.obstacles.push({ x: 10, y: 0.7 + 1.4 * k, r: 0.7, movable: [2, 4, 6].includes(k) });
      // Fixed obstacles: a ragged column on the approach side (some overlap) that leaves only a squeeze along the bottom,
      // and one block behind the wall. Layout traced from a screenshot of the arrangement the user wanted.
      [[7.3, 10.1, 0.8], [7.25, 9.05, 0.9], [7.3, 7.6, 0.7], [7.35, 6.2, 0.8], [7.35, 4.7, 0.8], [7.0, 3.5, 0.7], [7.1, 2.05, 0.9],
        [16, 4, 0.8]].forEach(([x, y, r]) => s.obstacles.push({ x, y, r }));
      return s;
    },
    // Pushable wall 1 plus a third wall of fixed circles behind it (x = 14.5). It has a single 1.4 m opening near
    // the top, so objects pushed east along y ~ 6 end up jammed against it and the rover must route to the opening.
    'Pushable wall 2': () => {
      const s = scenes['Pushable wall 1']();
      for (let k = 0; k <= 8; k++) if (k !== 7) s.obstacles.push({ x: 14.5, y: 0.7 + 1.4 * k, r: 0.7 });
      return s;
    },
  };

  if (typeof module !== 'undefined') module.exports = scenes; else g.SCENES = scenes;
})(typeof window !== 'undefined' ? window : globalThis);
