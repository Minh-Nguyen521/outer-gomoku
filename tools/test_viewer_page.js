// Smoke test for the page embedded in tools/watch_match_web.py.
//
// Three separate faults in that page failed SILENTLY and each cost a
// debugging round-trip, because a broken viewer and a stalled data feed look
// identical from outside:
//
//   1. a duplicate `const r` -> SyntaxError, whole script dead
//   2. a getElementById on an id that did not exist -> render loop died
//      after one pass, freezing the board mid-match
//   3. a leftover `r` in cx.arc() after renaming it to `rad` -> every stone
//      draw threw, so the grid rendered but no stones ever appeared
//
// `node --check` catches only the first. This drives the real page code with
// real records through a mock DOM and asserts that stones actually get drawn.
//
// Usage:
//   node tools/test_viewer_page.js [live.jsonl]
//
// Exits non-zero on failure so it can gate a commit.

const fs = require('fs');
const path = require('path');

function extractPage() {
  const src = fs.readFileSync(
      path.join(__dirname, 'watch_match_web.py'), 'utf8');
  const m = src.match(/PAGE = r"""([\s\S]*?)"""/);
  if (!m) throw new Error('could not find PAGE in watch_match_web.py');
  return m[1];
}

function sampleRecords() {
  const file = process.argv[2];
  if (file && fs.existsSync(file)) {
    return fs.readFileSync(file, 'utf8').trim().split('\n')
             .filter(Boolean).map(JSON.parse);
  }
  // Self-contained fallback: one short game with a vertical five for White.
  const recs = [{t: 'start', game: 0, black: 'third_party/A/pbrain-a',
                 white: 'third_party/B/pbrain-b', a_color: 'B', opening: [0, 7]}];
  const moves = [['B', 0, 7], ['W', 8, 4], ['B', 6, 6], ['W', 8, 5],
                 ['B', 6, 7], ['W', 8, 6], ['B', 6, 8], ['W', 8, 7],
                 ['B', 5, 5], ['W', 8, 8]];
  moves.forEach(([side, r, c], i) =>
      recs.push({t: 'move', game: 0, n: i, side, r, c, ms: 500}));
  recs.push({t: 'end', game: 0, winner: 'W', result: 'B',
             reason: '5-in-a-row at (8,8)', plies: moves.length - 1});
  return recs;
}

function run() {
  const js = extractPage().split('<script>')[1].split('</script>')[0];
  const records = sampleRecords();

  let arcs = 0, gradients = 0;
  const ctx = {
    fillRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {},
    arc() { arcs++; }, fill() {},
    createRadialGradient() { gradients++; return {addColorStop() {}}; },
    set fillStyle(v) {}, set strokeStyle(v) {}, set lineWidth(v) {},
  };
  const store = {};
  const el = (id) => store[id] || (store[id] = {
    textContent: '', style: {}, innerHTML: '', options: {length: 0}, value: 0,
    getContext: () => ctx, width: 640, height: 640,
    // Play mode attaches a click handler to the canvas and reads its on-screen
    // box. Without these the whole script throws on load.
    addEventListener() {}, removeEventListener() {},
    getBoundingClientRect: () => ({left: 0, top: 0, width: 640, height: 640}),
    onclick: null, onchange: null, dataset: {},
  });

  const listeners = {};
  global.document = {
    getElementById: el, createElement: () => el('_tmp'),
    body: {insertBefore() {}, firstChild: null}, documentElement: {},
    // The page registers a visibilitychange handler to catch up after a
    // browser throttles a hidden tab. Without this stub the whole script
    // throws on load and the test reports a bare crash.
    addEventListener: (ev, fn) => { (listeners[ev] = listeners[ev] || []).push(fn); },
    visibilityState: 'visible',
  };
  global.window = {};
  global.getComputedStyle = () => ({getPropertyValue: () => '#000'});
  global.performance = {now: () => Date.now()};

  let pending = records.slice();
  global.fetch = async () => ({
    json: async () => ({records: pending.splice(0, pending.length), off: 1}),
  });

  const timers = [];
  global.setTimeout = (fn) => { if (timers.length < 500) timers.push(fn); };

  let firstError = null;
  const realLog = console.log;
  global.console = {
    log: realLog,
    error: (...a) => {
      if (!firstError) {
        firstError = a.map((x) => (x && x.stack) ? x.stack : String(x)).join(' ');
      }
    },
  };

  new Function(js)();

  return (async () => {
    for (let i = 0; i < 200 && timers.length; i++) {
      const fn = timers.shift();
      try { await fn(); } catch (e) { if (!firstError) firstError = String(e); }
    }
    const fails = [];
    if (firstError) fails.push('runtime error: ' + firstError.split('\n')[0]);
    // 5 star points are drawn per repaint even with an empty board, so a
    // gradient is the reliable signal that a STONE was drawn.
    if (gradients === 0) fails.push('no stones drawn (0 gradients)');
    if (!store.polls || !/ok$/.test(store.polls.textContent)) {
      fails.push('poll counter never advanced: ' +
                 JSON.stringify(store.polls && store.polls.textContent));
    }
    if (!store.ply || store.ply.textContent === '') {
      fails.push('ply never rendered');
    }

    realLog(`arcs=${arcs} gradients=${gradients} ` +
            `ply=${JSON.stringify(store.ply && store.ply.textContent)} ` +
            `polls=${JSON.stringify(store.polls && store.polls.textContent)}`);
    if (fails.length) {
      realLog('FAIL');
      fails.forEach((f) => realLog('  - ' + f));
      process.exit(1);
    }
    realLog('PASS: page renders stones and polls the feed');
  })();
}

run();
