#!/usr/bin/env python3
"""Browser GUI for tools/match.py -- watch games render live on a real board.

Companion to tools/watch_match.py, which is the terminal version. Same data
source (the --live-file JSONL), but drawn as an actual goban in a browser:
wood board, stones, last-move marker, the Outer-Open ring shaded, per-side
think times, and a game picker so finished games can be scrubbed through.

Why a local server rather than a standalone .html file: a page opened via
file:// cannot read the growing log, and a hosted page cannot reach your disk
at all. So this serves both the page and the log from localhost.

Usage
-----
    tools/match.py --games 40 --timeout-turn 10000 \\
        --live-file /tmp/live.jsonl --engine-a ... --engine-b ...

    tools/watch_match_web.py /tmp/live.jsonl        # one match
    tools/watch_match_web.py /tmp/                 # follow whichever is newest

Options
-------
    --port N     port to serve on (default 8777)
    --no-open    do not launch a browser automatically

The page polls for new records, so it can be started before, during or after
a match, and it picks up a restarted match automatically.
"""

import argparse
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import webbrowser

PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>Outer-Gomoku live</title>
<style>
  :root{
    --bg:#14161a; --panel:#1d2027; --ink:#e8e6e3; --muted:#8b929e;
    --line:#6b5636; --board:#d9b473; --accent:#e0574a; --edge:#c8a05e;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
  .wrap{display:flex;gap:22px;padding:22px;flex-wrap:wrap;align-items:flex-start}
  .panel{background:var(--panel);border-radius:12px;padding:16px 18px;
         min-width:270px;flex:1 1 270px;max-width:420px}
  h1{font-size:16px;margin:0 0 14px;letter-spacing:.3px}
  canvas{background:var(--board);border-radius:10px;display:block;
         box-shadow:0 8px 28px #0008;max-width:100%;height:auto}
  .row{display:flex;justify-content:space-between;gap:12px;padding:5px 0;
       border-bottom:1px solid #ffffff10}
  .row:last-child{border-bottom:0}
  .k{color:var(--muted)}
  .v{text-align:right;font-variant-numeric:tabular-nums}
  .eng{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;
       word-break:break-all;text-align:right}
  .dot{display:inline-block;width:11px;height:11px;border-radius:50%;
       margin-right:7px;vertical-align:-1px}
  .b{background:#16181d;border:1px solid #000}
  .w{background:#f6f4ef;border:1px solid #bbb}
  select,button{background:#272b33;color:var(--ink);border:1px solid #3a3f4a;
       border-radius:7px;padding:6px 10px;font:inherit}
  .tag{display:inline-block;padding:2px 9px;border-radius:20px;font-size:12px}
  .live{background:#1f6f43}.done{background:#5a4a1f}.idle{background:#3a3f4a}
  .big{font-size:26px;font-variant-numeric:tabular-nums}
</style>
<div style="padding:22px 22px 0;display:flex;gap:10px;align-items:center;flex-wrap:wrap">
  <button id="modewatch">watch</button>
  <button id="modeplay">play vs engine</button>
  <span id="watchbar" style="display:flex;gap:10px;align-items:center">
    <span class="k">match</span>
    <select id="feedpick" style="min-width:240px"></select>
    <button id="autobtn">auto (newest)</button>
  </span>
  <span id="playbar" style="display:none;gap:10px;align-items:center">
    <span class="k">engine</span><select id="engpick" style="min-width:150px"></select>
    <span class="k">you play</span>
    <select id="colpick"><option value="B">black (first)</option>
      <option value="W">white</option></select>
    <span class="k">think</span>
    <select id="thinkpick"><option value="1000">1s</option>
      <option value="3000" selected>3s</option><option value="10000">10s</option></select>
    <button id="newgame">new game</button>
  </span>
</div>
<div class="wrap">
  <div><canvas id="bd" width="640" height="640"></canvas></div>
  <div class="panel">
    <h1>Match <span id="status" class="tag idle">waiting</span></h1>
    <div class="row"><span class="k">game</span>
      <span class="v"><select id="pick"></select></span></div>
    <div class="row"><span class="k">ply</span><span class="v big" id="ply">0</span></div>
    <div class="row"><span class="k"><i class="dot b"></i>black</span>
      <span class="v eng" id="eb">—</span></div>
    <div class="row"><span class="k"><i class="dot w"></i>white</span>
      <span class="v eng" id="ew">—</span></div>
    <div class="row"><span class="k">black think</span><span class="v" id="tb">—</span></div>
    <div class="row"><span class="k">white think</span><span class="v" id="tw">—</span></div>
    <div class="row"><span class="k">result</span><span class="v" id="res">—</span></div>
  </div>
  <div class="panel" id="playpanel" style="display:none">
    <h1>Your game <span id="pstatus" class="tag idle">no game</span></h1>
    <div class="row"><span class="k">engine</span><span class="v" id="peng">—</span></div>
    <div class="row"><span class="k">you are</span><span class="v" id="pyou">—</span></div>
    <div class="row"><span class="k">to move</span><span class="v" id="pturn">—</span></div>
    <div class="row"><span class="k">moves</span><span class="v big" id="pply">0</span></div>
    <div class="row"><span class="k">engine time</span><span class="v" id="ptime">—</span></div>
    <div class="row" style="border:0"><span class="v" id="pmsg"
      style="color:var(--accent);text-align:left"></span></div>
  </div>
  <div class="panel" id="scorepanel">
    <h1>Score</h1>
    <div class="row"><span class="k" id="la">engine A</span><span class="v big" id="sa">0</span></div>
    <div class="row"><span class="k" id="lb">engine B</span><span class="v big" id="sb">0</span></div>
    <div class="row"><span class="k">draws</span><span class="v" id="sd">0</span></div>
    <div class="row"><span class="k">finished</span><span class="v" id="sn">0</span></div>
    <div class="row"><span class="k" id="lwr">winrate</span><span class="v" id="swr">—</span></div>
    <div class="row"><span class="k">polls</span>
      <span class="v" id="polls">—</span></div>
    <div class="row"><span class="k">last move</span>
      <span class="v" id="age">—</span></div>
    <div class="row"><span class="k">replay lag</span>
      <span class="v" id="behind"></span></div>
    <div class="row" style="border:0;padding-top:12px">
      <button id="follow">following live</button></div>
  </div>
</div>
<script>
// Three separate faults in this page failed SILENTLY -- a SyntaxError that
// killed the whole script, a dead render loop, and a stale cached copy. From
// the outside all three looked identical to "the data is not arriving", which
// sent the debugging to the wrong end of the pipe every time. Surface errors
// on the page itself so a broken viewer says so.
window.onerror=function(msg,src,line,col){
  const d=document.createElement('div');
  d.style.cssText='background:#a3312a;color:#fff;padding:10px 14px;'
                 +'font:12px ui-monospace,monospace;white-space:pre-wrap';
  d.textContent='JS error: '+msg+'  (line '+line+':'+col+')';
  document.body.insertBefore(d, document.body.firstChild);
  return false;
};
const N=15, cv=document.getElementById('bd'), cx=cv.getContext('2d');
let games={}, cur=null, off=0, follow=true;
let score={A:0,B:0,draw:0};
let nameA=null, nameB=null;

function isRing(x,y){return x<2||x>=N-2||y<2||y>=N-2}

// Engine "names" arrive as whole command lines -- "third_party/rapfi-4t/
// pbrain-rapfi" or a multi-flag python invocation of the KataGomo bridge.
// Reduce them to the thing a human calls the engine, which is nearly always
// the containing directory rather than the binary (every rapfi variant is
// "pbrain-rapfi"; only the directory says whether it is 4t, bin or classical).
function label(cmd){
  if(!cmd) return '—';
  const s=String(cmd);
  if(/katagomo_bridge/.test(s)){
    const m=s.match(/--model\s+\S*?([^\/\s]+)\.bin\.gz/);
    return m? 'KataGomo ('+m[1]+')' : 'KataGomo';
  }
  const toks=s.split(/\s+/);
  const bin=toks.find(t=>/pbrain|\.exe$/i.test(t));
  if(bin){
    const parts=bin.split('/').filter(Boolean);
    if(parts.length>=2) return parts[parts.length-2];
    return parts[parts.length-1];
  }
  const path=toks.find(t=>t.includes('/'))||toks[0]||s;
  const parts=path.split('/').filter(Boolean);
  return parts[parts.length-1]||s;
}

function draw(g){
  const S=cv.width, pad=S/(N+1), cell=(S-2*pad)/(N-1);
  cx.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--board');
  cx.fillRect(0,0,S,S);
  // Outer-Open ring: the only legal squares for Black's first stone.
  cx.fillStyle='#00000010';
  cx.fillRect(0,0,S,pad+1.5*cell); cx.fillRect(0,S-pad-1.5*cell,S,pad+1.5*cell);
  cx.fillRect(0,0,pad+1.5*cell,S); cx.fillRect(S-pad-1.5*cell,0,pad+1.5*cell,S);
  cx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue('--line');
  cx.lineWidth=1;
  for(let i=0;i<N;i++){
    cx.beginPath();
    cx.moveTo(pad,pad+i*cell); cx.lineTo(S-pad,pad+i*cell);
    cx.moveTo(pad+i*cell,pad); cx.lineTo(pad+i*cell,S-pad);
    cx.stroke();
  }
  cx.fillStyle='#5c4a2e';
  for(const [sx,sy] of [[3,3],[11,3],[3,11],[11,11],[7,7]]){
    cx.beginPath(); cx.arc(pad+sx*cell,pad+sy*cell,3,0,7); cx.fill();
  }
  if(!g) return;
  for(const k in g.board){
    // Keys are "r,c" -- match.py:261 returns (y,x) as (r,c), so r is the ROW.
    // Screen x is the COLUMN. Unpacking these as [x,y] renders the whole board
    // transposed: still a legal-looking position (a five stays a five, and the
    // ring is symmetric), which is exactly why it went unnoticed.
    const [kr,kc]=k.split(',').map(Number), x=kc, y=kr, v=g.board[k];
    const cxp=pad+x*cell, cyp=pad+y*cell, rad=cell*0.44;
    const grd=cx.createRadialGradient(cxp-rad*.35,cyp-rad*.35,rad*.1,cxp,cyp,rad);
    if(v==='B'){grd.addColorStop(0,'#4a4d55');grd.addColorStop(1,'#0d0f13')}
    else{grd.addColorStop(0,'#ffffff');grd.addColorStop(1,'#cfcabd')}
    cx.fillStyle=grd; cx.beginPath(); cx.arc(cxp,cyp,rad,0,7); cx.fill();
    cx.strokeStyle='#00000055'; cx.lineWidth=1; cx.stroke();
  }
  if(g.last){
    const x=g.last[1], y=g.last[0], cxp=pad+x*cell, cyp=pad+y*cell;
    cx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue('--accent');
    cx.lineWidth=3; cx.beginPath(); cx.arc(cxp,cyp,cell*0.2,0,7); cx.stroke();
  }
}

function fmt(t){ if(!t||!t.length) return '—';
  const m=t.reduce((a,b)=>a+b,0)/t.length;
  return (t[t.length-1]/1000).toFixed(2)+'s  (mean '+(m/1000).toFixed(2)+'s, n='+t.length+')'; }

function paint(){
  if(mode==='play') return;   // play mode owns the canvas
  const g=games[cur];
  draw(g);
  document.getElementById('ply').textContent = g? g.plies : 0;
  document.getElementById('eb').textContent  = g? label(g.black) : '—';
  document.getElementById('ew').textContent  = g? label(g.white) : '—';
  document.getElementById('tb').textContent  = g? fmt(g.tB) : '—';
  document.getElementById('tw').textContent  = g? fmt(g.tW) : '—';
  document.getElementById('res').textContent = g&&g.over? (g.over+' — '+g.reason) : 'in progress';
  const st=document.getElementById('status');
  if(g&&g.over){st.className='tag done';st.textContent='finished'}
  else if(g){st.className='tag live';st.textContent='live'}
  const n=score.A+score.B+score.draw;
  document.getElementById('sa').textContent=score.A;
  document.getElementById('sb').textContent=score.B;
  document.getElementById('sd').textContent=score.draw;
  document.getElementById('sn').textContent=n;
  document.getElementById('swr').textContent = n? (((score.A+score.draw/2)/n*100).toFixed(1)+'%') : '—';
  if(nameA) document.getElementById('la').textContent=nameA;
  if(nameB) document.getElementById('lb').textContent=nameB;
  if(nameA) document.getElementById('lwr').textContent=nameA+' winrate';
  const pick=document.getElementById('pick');
  const keys=Object.keys(games).sort((a,b)=>a-b);
  if(pick.options.length!==keys.length){
    pick.innerHTML=keys.map(k=>'<option value="'+k+'">game '+k+'</option>').join('');
  }
  pick.value=cur;
}

document.getElementById('pick').onchange=e=>{cur=+e.target.value;follow=false;
  document.getElementById('follow').textContent='viewing game '+cur;paint()};
document.getElementById('follow').onclick=()=>{follow=true;
  document.getElementById('follow').textContent='following live';};

// Records are applied from a queue rather than all at once. A 1s/move match
// finishes a 30-ply game in ~15s, so applying a whole poll's worth at once
// makes the board jump and the game looks like it only appears when finished.
// One record per tick makes the moves visible; if we fall far behind (a page
// opened mid-match, or a fast match) the queue drains quickly so we stay live.
let queue=[], holdUntil=0;
const MIN_MS=140, END_HOLD_MS=2500, DRAIN_AT=80;

function apply(rec){
  const gi=rec.game;
  if(rec.t==='start'){
    games[gi]={board:{},last:null,plies:0,tB:[],tW:[],
               black:rec.black,white:rec.white,over:null,reason:''};
    // a_color says which colour engine A took this game; colours alternate,
    // so this is the only reliable way to map A/B onto the two engines.
    if(rec.a_color==='B'){nameA=label(rec.black); nameB=label(rec.white);}
    else if(rec.a_color==='W'){nameA=label(rec.white); nameB=label(rec.black);}
    if(follow) cur=gi;
  }else if(rec.t==='move'){
    let g=games[gi]; if(!g) g=games[gi]={board:{},last:null,plies:0,tB:[],tW:[],
                                        black:'?',white:'?',over:null,reason:''};
    g.board[rec.r+','+rec.c]=rec.side; g.last=[rec.r,rec.c]; g.plies=rec.n;
    if(rec.ms) (rec.side==='B'?g.tB:g.tW).push(rec.ms);
    if(follow) cur=gi;
  }else if(rec.t==='end'){
    const g=games[gi]; if(g){g.over=rec.winner;g.reason=rec.reason}
    if(rec.result in score) score[rec.result]++;
    if(follow) cur=gi;
    // Let the finished position sit long enough to read before moving on.
    holdUntil=performance.now()+END_HOLD_MS;
  }
}

let lastApplied=performance.now();

function tick(){
  // Everything is wrapped: a single uncaught exception in here stops the
  // render loop for good, and the page just freezes with no visible error.
  try{ tickBody(); }catch(e){ console.error('tick', e); }
  // A 10s/move match changes the board only 6 times a minute, so a still
  // board is normal and indistinguishable from a dead page. This counter is
  // the heartbeat that tells them apart.
  try{
    const a=document.getElementById('age');
    if(a){
      const s=(performance.now()-lastApplied)/1000;
      a.textContent = s<1 ? 'just now' : s.toFixed(0)+'s ago';
      a.style.color = s>45 ? 'var(--accent)' : '';
    }
  }catch(e){}
  setTimeout(tick, MIN_MS);
}

function tickBody(){
  const now=performance.now();
  if(queue.length && now>=holdUntil){
    // Behind by a lot? Catch up in bulk instead of crawling.
    const n = queue.length>DRAIN_AT ? queue.length-DRAIN_AT : 1;
    for(let i=0;i<n;i++) apply(queue.shift());
    lastApplied=performance.now();
    paint();
    const b=document.getElementById('behind');
    if(b) b.textContent = queue.length? ('+'+queue.length+' queued') : '';
  }
}

let polls=0, lastPollErr='', curName=null;
// ONE poll loop, ever. poll() used to end with setTimeout(poll,300) while the
// visibilitychange handler also called poll() directly -- so every tab switch
// started an additional concurrent loop, and they raced on the shared `off`,
// duplicating and dropping records. A single timer handle, cleared on entry,
// makes re-entry harmless.
let pollTimer=null;
// Bumped whenever the watched feed changes. A fetch issued before the switch
// can land after it; without this its stale `off` and records overwrite the
// reset and the board never recovers.
let pollGen=0;

function schedulePoll(ms){ clearTimeout(pollTimer); pollTimer=setTimeout(poll,ms); }
// '' means auto-follow whichever feed was written most recently. An explicit
// pick sticks even while that match is idle: a 10s/move game writes rarely and
// would otherwise lose the newest-file race to a 2s/move one every few seconds.
let wantFeed='';
// Auto mode must be STICKY. "Whichever file was written most recently" flips
// between feeds every few seconds when two matches run at once, and each flip
// resets the board and refetches from zero -- so the position never advances
// past the opening. Instead: latch one feed, and only move on once it has gone
// quiet for a while.
let autoPick=null, autoIdleSince=0;
const AUTO_IDLE_S=90;

function effFeed(){ return wantFeed || autoPick || ''; }

async function refreshFeeds(){
  try{
    const r=await fetch('/feeds',{cache:'no-store'});
    const d=await r.json();
    // Latch / re-latch the auto choice.
    if(!wantFeed && d.feeds.length){
      // NB: must not be called `cur` -- that is the global holding the current
      // game index, and shadowing it here then assigning to the shadow threw
      // "Assignment to constant variable" on every cycle. The throw was
      // swallowed by the catch below, so auto-follow silently stopped
      // re-latching and the feed list stopped refreshing.
      const latched=d.feeds.find(f=>f.name===autoPick);
      if(!latched || latched.age>AUTO_IDLE_S){
        const live=d.feeds.filter(f=>f.age<=AUTO_IDLE_S);
        const next=(live[0]||d.feeds[0]).name;
        if(next!==autoPick){ autoPick=next; off=0; curName=null; pollGen++;
          games={}; cur=null; queue=[]; score={A:0,B:0,draw:0};
          nameA=null; nameB=null; holdUntil=0; draw(null); }
      }
    }
    const sel=document.getElementById('feedpick');
    if(!sel) return;
    const opts=['<option value="">auto — newest<\/option>']
      .concat(d.feeds.map(f=>{
        const idle = f.age<120 ? 'live' : (f.age/60).toFixed(0)+'m idle';
        return '<option value="'+f.name+'">'+f.name+'  ('+idle+')<\/option>';
      }));
    const joined=opts.join('');
    if(sel.dataset.sig!==joined){ sel.dataset.sig=joined; sel.innerHTML=joined;
      sel.value=wantFeed; }
  }catch(e){}
  setTimeout(refreshFeeds, 3000);
}
async function poll(){
  clearTimeout(pollTimer);
  const myGen=pollGen;
  try{
    const f=effFeed();
    const r=await fetch('/events?from='+off+(f?('&feed='+encodeURIComponent(f)):''),{cache:'no-store'});
    const d=await r.json();
    if(myGen!==pollGen){
      // The feed changed while this fetch was in flight. Drop the response --
      // its `off` and records belong to the previous match. Not an error, so
      // do not surface it in the poll counter.
      schedulePoll(300);
      return;
    }
    // Reset when the feed we are being served actually changes. This must key
    // off the NAME, not a server-side switch counter: the counter is global,
    // so a second tab (or our own dropdown) requesting a different feed bumped
    // it for everyone and wiped every client's board -- which looked like the
    // page had stopped updating.
    if(d.name!==undefined && d.name!==curName){
      curName=d.name;
      games={}; cur=null; queue=[]; score={A:0,B:0,draw:0};
      nameA=null; nameB=null; holdUntil=0;
      const fp=document.getElementById('feedpick');
      if(fp && wantFeed==='') fp.value='';
      draw(null);
    }
    off=d.off;
    queue.push(...d.records);
    polls++; lastPollErr='';
  }catch(e){ lastPollErr=String(e); }
  // A silent catch here hid a dead feed once. Show the poll count so a page
  // that is talking to the server looks different from one that is not.
  const p=document.getElementById('polls');
  if(p) p.textContent = lastPollErr ? ('ERR '+lastPollErr) : (polls+' ok');
  schedulePoll(300);
}

document.getElementById('feedpick').onchange=e=>{
  wantFeed=e.target.value; autoPick=null; off=0; curName=null; pollGen++;
  games={}; cur=null; queue=[]; score={A:0,B:0,draw:0};
  nameA=null; nameB=null; holdUntil=0; draw(null);
};
document.getElementById('autobtn').onclick=()=>{
  wantFeed=''; autoPick=null; off=0; curName=null; pollGen++;
  games={}; cur=null; queue=[]; score={A:0,B:0,draw:0};
  nameA=null; nameB=null; holdUntil=0; draw(null);
  const sel=document.getElementById('feedpick'); if(sel) sel.value='';
};
// Browsers throttle timers in hidden tabs -- Chrome clamps setTimeout to ~1/s
// when backgrounded and ~1/min after a few minutes. The poll loop therefore
// starves while you are on another tab, and the board is stale when you come
// back. Catch up immediately on becoming visible, and drain the whole backlog
// at once rather than replaying it at 140ms a move.
document.addEventListener('visibilitychange',()=>{
  if(document.visibilityState==='visible'){
    holdUntil=0;
    if(queue.length){
      while(queue.length) apply(queue.shift());
      lastApplied=performance.now();
      paint();
    }
    poll();
  }
});
// ---------------- play vs engine ----------------
let mode='watch', pgame=null;

function setMode(m){
  mode=m;
  document.getElementById('watchbar').style.display  = m==='watch'?'flex':'none';
  document.getElementById('playbar').style.display   = m==='play' ?'flex':'none';
  document.getElementById('scorepanel').style.display= m==='watch'?'':'none';
  document.getElementById('playpanel').style.display = m==='play' ?'':'none';
  if(m==='play'){ pgame?drawPlay():draw(null); } else { paint(); }
}
document.getElementById('modewatch').onclick=()=>setMode('watch');
document.getElementById('modeplay').onclick =()=>setMode('play');

function drawPlay(){
  if(!pgame){ draw(null); return; }
  const b={}; let last=null;
  for(const m of pgame.moves){ b[m.r+','+m.c]=m.side; last=[m.r,m.c]; }
  draw({board:b,last:last});
  const st=document.getElementById('pstatus');
  if(pgame.over){ st.className='tag done';
    st.textContent = pgame.over==='draw' ? 'draw'
      : (pgame.over===pgame.human ? 'you win' : 'engine wins'); }
  else { st.className='tag live'; st.textContent='in play'; }
  document.getElementById('peng').textContent  = pgame.engine_label||'—';
  document.getElementById('pyou').textContent  = pgame.human==='B'?'black (first)':'white';
  document.getElementById('pturn').textContent = pgame.over ? '—'
      : (pgame.to_move===pgame.human ? 'you' : 'engine…');
  document.getElementById('pply').textContent  = pgame.moves.length;
  const et=pgame.moves.filter(m=>m.side===pgame.engine&&m.ms);
  document.getElementById('ptime').textContent = et.length
      ? (et[et.length-1].ms/1000).toFixed(2)+'s' : '—';
  document.getElementById('pmsg').textContent =
      pgame.error || (pgame.over ? (pgame.reason||'') : '');
}

// Canvas clicks -> board coords. Uses the same pad/cell geometry as draw(),
// and getBoundingClientRect so it stays correct when CSS scales the canvas.
cv.addEventListener('click', async (ev)=>{
  if(mode!=='play' || !pgame || pgame.over) return;
  if(pgame.to_move!==pgame.human) return;
  const rect=cv.getBoundingClientRect();
  const S=cv.width, pad=S/(N+1), cell=(S-2*pad)/(N-1);
  const px=(ev.clientX-rect.left)*(cv.width/rect.width);
  const py=(ev.clientY-rect.top )*(cv.height/rect.height);
  const c=Math.round((px-pad)/cell), r=Math.round((py-pad)/cell);
  if(c<0||c>=N||r<0||r>=N) return;
  try{
    const q=await fetch('/play/move?r='+r+'&c='+c,{cache:'no-store'});
    pgame=await q.json(); drawPlay();
  }catch(e){}
});

async function loadEngines(){
  try{
    const r=await fetch('/play/engines',{cache:'no-store'});
    const d=await r.json();
    document.getElementById('engpick').innerHTML =
      d.engines.map(e=>'<option>'+e+'<\/option>').join('');
  }catch(e){}
}
document.getElementById('newgame').onclick=async()=>{
  const e=document.getElementById('engpick').value;
  const h=document.getElementById('colpick').value;
  const t=document.getElementById('thinkpick').value;
  document.getElementById('pmsg').textContent='starting engine…';
  try{
    const r=await fetch('/play/new?engine='+encodeURIComponent(e)+'&human='+h+'&think='+t,
                        {cache:'no-store'});
    pgame=await r.json(); drawPlay();
  }catch(err){ document.getElementById('pmsg').textContent=String(err); }
};

draw(null); poll(); tick(); refreshFeeds(); loadEngines(); setMode('watch');

</script>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    live_file = ""      # a file, OR a directory to follow
    # NOTE: clients detect feed changes from the returned `name`, not from
    # server state -- several browser tabs may each be watching a different
    # match through this one server.

    @classmethod
    def list_feeds(cls):
        """Every live_*.jsonl in the followed directory, newest first."""
        base = cls.live_file
        if not os.path.isdir(base):
            return ([{"name": os.path.basename(base)[5:-6], "file": base}]
                    if base else [])
        out = []
        for f in sorted(os.listdir(base)):
            if not (f.startswith("live_") and f.endswith(".jsonl")):
                continue
            full = os.path.join(base, f)
            try:
                if os.path.getsize(full) == 0:
                    continue
                mtime = os.path.getmtime(full)
            except OSError:
                continue
            out.append({"name": f[5:-6], "file": full, "mtime": mtime,
                        "age": int(time.time() - mtime)})
        out.sort(key=lambda d: -d.get("mtime", 0))
        return out

    @classmethod
    def resolve(cls, want=""):
        """If live_file is a directory, follow whichever feed is newest.

        Matches roll over: a lane finishes one game set and starts writing a
        new live_<tag>.jsonl. A viewer pinned to one file then shows a frozen,
        finished match forever, which is indistinguishable from a broken page
        -- it happened three times in one session before this existed.
        """
        target = cls.live_file
        if os.path.isdir(target):
            feeds = cls.list_feeds()
            picked = ""
            if want:
                # An explicit choice sticks even while it is idle -- a 10s/move
                # match writes rarely and would otherwise lose the "newest"
                # race to a fast one every few seconds.
                for d in feeds:
                    if d["name"] == want:
                        picked = d["file"]
                        break
            if not picked:
                picked = feeds[0]["file"] if feeds else ""
            target = picked
        return target

    def log_message(self, *a):
        pass  # keep the console clean; the match is the interesting output

    # One human-vs-engine game at a time. A second "new" replaces the first,
    # which also kills the previous engine process -- otherwise each new game
    # would leak a Wine container.
    session = None
    session_lock = threading.Lock()

    def handle_play(self):
        from urllib.parse import urlparse, parse_qs
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}

        if u.path == "/play/engines":
            import play_session
            _json(self, {"engines": sorted(play_session.ENGINES)})
            return

        if u.path == "/play/new":
            import play_session
            name = q.get("engine", "")
            cmd = play_session.ENGINES.get(name)
            if not cmd:
                _json(self, {"error": f"unknown engine {name!r}"}, 400)
                return
            colour = "W" if q.get("human", "B").upper() == "W" else "B"
            think = max(200, min(30000, int(q.get("think", "3000"))))
            with Handler.session_lock:
                if Handler.session:
                    Handler.session.close()
                    Handler.session = None
                try:
                    sess = play_session.PlaySession(cmd, colour, think, name)
                    st = sess.start()
                    Handler.session = sess
                except Exception as e:
                    _json(self, {"error": f"{type(e).__name__}: {e}"}, 500)
                    return
            _json(self, st)
            return

        if u.path in ("/play/move", "/play/state"):
            with Handler.session_lock:
                sess = Handler.session
            if not sess:
                _json(self, {"error": "no game -- press New game"}, 400)
                return
            if u.path == "/play/state":
                _json(self, sess.state())
                return
            try:
                r, c = int(q.get("r", "-1")), int(q.get("c", "-1"))
            except ValueError:
                _json(self, {"error": "bad coordinates"}, 400)
                return
            _json(self, sess.human_move(r, c))
            return

        _json(self, {"error": "unknown play endpoint"}, 404)

    def do_GET(self):
        if self.path.startswith("/play/"):
            self.handle_play()
            return

        if self.path.startswith("/feeds"):
            body = json.dumps({"feeds": [
                {"name": d["name"], "age": d.get("age", 0)}
                for d in Handler.list_feeds()]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/events"):
            try:
                frm = int(self.path.split("from=")[1].split("&")[0])
            except (IndexError, ValueError):
                frm = 0
            records, off = [], frm
            want = ""
            if "feed=" in self.path:
                want = self.path.split("feed=")[1].split("&")[0]
            path = Handler.resolve(want)
            try:
                if not path:
                    raise FileNotFoundError
                size = os.path.getsize(path)
                # A restarted match truncates the file; rewind rather than
                # reading past the end forever.
                if frm > size:
                    frm = off = 0
                with open(path, "r") as fh:
                    fh.seek(frm)
                    data = fh.read()
                # Trailing partial line: leave it for the next poll.
                cut = data.rfind("\n")
                if cut >= 0:
                    off = frm + len(data[:cut + 1].encode())
                    for line in data[:cut].splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            except FileNotFoundError:
                pass
            body = json.dumps({"records": records, "off": off,
                               "name": os.path.basename(path)[5:-6] if path else ""
                               }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # The page is regenerated whenever this file changes, and the server is
        # restarted often during development. Without no-store a browser serves
        # a cached copy and silently runs old JavaScript, which looks exactly
        # like the viewer being broken.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)


def _json(handler, obj, code=200):
    body = json.dumps(obj).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("live_file")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    Handler.live_file = os.path.abspath(a.live_file)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"watching {Handler.live_file}")
    print(f"open     {url}")
    if not a.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    with Server(("127.0.0.1", a.port), Handler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
