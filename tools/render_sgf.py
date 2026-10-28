"""Render an SGF game to a scrollable HTML page with playback controls."""
import re, sys, html

sgf_path = sys.argv[1] if len(sys.argv) > 1 else "third_party/ludii/games/game_1.sgf"
out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/game_replay.html"

sgf = open(sgf_path).read()
moves = re.findall(r';([BW])\[([a-o])([a-o])\]', sgf)
result = re.search(r'RE\[(\w+)\]', sgf).group(1)

board_html = []
pos = set()
n = 1
for c, x, y in moves:
    xi, yi = ord(x)-97, ord(y)-97
    board_html.append({'c': c, 'x': xi, 'y': yi, 'n': n})
    n += 1

cells = ""
for i in range(15):
    row = ""
    for j in range(15):
        row += f'<div class="cell" id="c{i}_{j}"></div>'
    cells += f'<div class="row">{row}</div>'

js_moves = str(board_html).replace("'", '"')

html_doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Gomoku Replay</title>
<style>
body {{ font-family: sans-serif; background:#222; color:#eee; display:flex; flex-direction:column; align-items:center; }}
#board {{ display:inline-block; background:#dcb35c; padding:10px; border-radius:6px; margin-top:10px; }}
.row {{ display:flex; }}
.cell {{ width:30px; height:30px; border:1px solid rgba(0,0,0,.35); position:relative; cursor:pointer; }}
.stone {{ position:absolute; inset:2px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:bold; }}
.b {{ background:#111; color:#fff; }}
.w {{ background:#f5f5f5; color:#111; }}
.last {{ outline:2px solid #e33; z-index:2; }}
#controls button {{ font-size:16px; margin:4px; padding:6px 14px; cursor:pointer; }}
#info {{ margin:8px; font-size:18px; }}
#slider {{ width:480px; }}
</style></head>
<body>
<h2>{sgf_path.split('/')[-1]} — result: {result} ({len(moves)} moves)</h2>
<div id="board">{cells}</div>
<div id="info">move 0/{len(moves)}</div>
<input id="slider" type="range" min="0" max="{len(moves)}" value="0">
<div id="controls">
<button onclick="step(-1)">&lt;</button>
<button onclick="step(1)">&gt;</button>
<button onclick="playpause()" id="pp">Play</button>
<button onclick="setMove(0)">Reset</button>
<button onclick="setMove(MOVES.length)">End</button>
</div>
<script>
const MOVES = {js_moves};
let cur = 0, timer = null;
function setMove(n) {{
  cur = Math.max(0, Math.min(MOVES.length, n));
  document.querySelectorAll('.stone').forEach(e => e.remove());
  let last = null;
  for (let i = 0; i < cur; i++) {{
    const m = MOVES[i];
    const cell = document.getElementById('c'+m.y+'_'+m.x);
    if (!cell) continue;
    const s = document.createElement('div');
    s.className = 'stone ' + (m.c==='B'?'b':'w');
    s.textContent = i+1;
    cell.appendChild(s);
    last = s;
  }}
  if (last) last.classList.add('last');
  document.getElementById('info').textContent = 'move '+cur+'/'+MOVES.length;
  document.getElementById('slider').value = cur;
}}
function step(d) {{ setMove(cur+d); }}
function playpause() {{
  if (timer) {{ clearInterval(timer); timer=null; document.getElementById('pp').textContent='Play'; }}
  else {{ document.getElementById('pp').textContent='Pause';
    timer=setInterval(()=>{{ if(cur>=MOVES.length) {{clearInterval(timer);timer=null;document.getElementById('pp').textContent='Play';}} else step(1); }},600); }}
}}
document.getElementById('slider').oninput = e => setMove(+e.target.value);
setMove(0);
</script></body></html>"""

open(out_path,"w").write(html_doc)
print(f"wrote {out_path} ({len(moves)} moves, result {result})")
