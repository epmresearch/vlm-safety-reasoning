#!/usr/bin/env python3
"""
The HTML template for the offline review app, kept out of build_review.py so that
file stays readable.

DESIGN CONSTRAINTS, all forced by who runs this: a civil-engineering student, on
their own laptop, probably Windows, who should not have to install anything.

  * ONE FILE, opened by double-click. No server, no Python, no npm, no CDN -- a
    `file://` page cannot fetch() a sibling .json (CORS), so the record data is
    INLINED into the HTML as a <script> blob. Images are plain relative <img> srcs,
    which do work under file://.
  * BOXES ARE DRAWN ON A CANVAS from the [0,1] coords, not burnt into the jpg. That
    is what makes per-rule layer toggles and box CORRECTION possible -- and box
    correction is the whole point: rule_2 and rule_3 have no MOCS geometry, so
    without it a good detection with a sloppy box can only be rejected, and those
    are the two scarcest rules in the project.
  * ONE SAVING MECHANISM: a real .json file on the reviewer's disk, written on every
    change via the File System Access API. No browser storage anywhere -- on a
    file:// page it is blocked in some configurations and wiped on close in others,
    so it can appear to work right up until a day of review disappears. There is
    also no Export/Import: they were one-shot snapshots that did not establish
    ongoing saving, which is exactly the trap this design removes.
  * NOBODY CAN START WITHOUT A FILE. A full-screen gate blocks the app until one is
    chosen, so there is no path where someone works for an hour into nothing.
"""

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root{
    --bg:#14161a; --panel:#1c1f26; --panel2:#242832; --line:#333947;
    --fg:#e8eaf0; --dim:#9aa3b2; --accent:#4c8dff;
    --ok:#2ecc71; --no:#ff5252; --maybe:#ffb020;
    --r1:#00E5FF; --r2:#FFEA00; --r3:#FF3D00; --r4:#D500F9; --mocs:#00E676;
  }
  *{box-sizing:border-box}
  /* The page is a flex COLUMN, not a hardcoded calc(100vh - 53px). The toolbar wraps
     to two or three rows on a laptop screen, and any fixed guess at its height pushes
     the bottom of the app off-screen -- invisibly, because of overflow:hidden. */
  html,body{height:100%}
  body{margin:0;font:13px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
       background:var(--bg);color:var(--fg);overflow:hidden;
       display:flex;flex-direction:column}
  button{font:inherit;cursor:pointer;border:1px solid var(--line);background:var(--panel2);
         color:var(--fg);border-radius:6px;padding:4px 8px}
  button:hover{border-color:var(--accent)}
  button.on{background:var(--accent);border-color:var(--accent);color:#fff}
  select,input,textarea{font:inherit;background:var(--panel2);color:var(--fg);
         border:1px solid var(--line);border-radius:6px;padding:4px 7px}
  .bar{display:flex;gap:6px;align-items:center;padding:6px 9px;background:var(--panel);
       border-bottom:1px solid var(--line);flex-wrap:wrap;flex:0 0 auto}
  .grow{flex:1}
  .wrap{display:flex;flex:1 1 auto;min-height:0}
  .left{flex:1 1 auto;min-width:0;min-height:0;display:flex;flex-direction:column;
        background:#0e1013}
  .canvasbox{flex:1 1 auto;min-height:0;position:relative;display:flex;
             align-items:center;justify-content:center;overflow:hidden}
  canvas{max-width:100%;max-height:100%;cursor:crosshair}
  .right{width:380px;flex:0 0 380px;min-height:0;overflow-y:auto;background:var(--panel);
         border-left:1px solid var(--line);padding:10px}
  @media (max-width:1250px){ .right{width:330px;flex:0 0 330px} }
  .sec{margin-bottom:10px;padding-bottom:9px;border-bottom:1px solid var(--line)}
  .sec:last-child{border:0}
  h3{margin:0 0 5px;font-size:10px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim)}
  .row{display:flex;gap:5px;align-items:center;flex-wrap:wrap}
  .chip{display:inline-block;padding:1px 6px;border-radius:99px;font-size:11px;
        background:var(--panel2);border:1px solid var(--line);color:var(--dim)}
  .rule{border:1px solid var(--line);border-radius:8px;padding:6px 7px;margin-bottom:5px;
        background:var(--panel2)}
  .rule.prop{border-left:4px solid var(--line)}
  .rule[data-r="rule_1"].prop{border-left-color:var(--r1)}
  .rule[data-r="rule_2"].prop{border-left-color:var(--r2)}
  .rule[data-r="rule_3"].prop{border-left-color:var(--r3)}
  .rule[data-r="rule_4"].prop{border-left-color:var(--r4)}
  .rule .hd{display:flex;align-items:center;gap:6px;margin-bottom:5px}
  .rule .nm{font-weight:600}
  .reason{color:var(--fg);font-size:13px;margin:4px 0}
  .muted{color:var(--dim);font-size:12px}
  .y{border-color:var(--ok)} .y.on{background:var(--ok);border-color:var(--ok);color:#06210f}
  .n{border-color:var(--no)} .n.on{background:var(--no);border-color:var(--no);color:#2a0606}
  .m{border-color:var(--maybe)} .m.on{background:var(--maybe);border-color:var(--maybe);color:#2a1a00}
  textarea{width:100%;min-height:52px;resize:vertical}
  .big{font-size:14px;padding:6px 11px;font-weight:600}
  .caption{background:var(--panel2);border:1px solid var(--line);border-radius:8px;
           padding:7px;font-size:12.5px;max-height:130px;overflow-y:auto}
  textarea#notes{min-height:40px}
  .badge{background:var(--no);color:#fff;border-radius:99px;padding:1px 7px;font-size:11px}
  #help,#gate{position:fixed;inset:0;background:rgba(0,0,0,.82);display:none;z-index:9;
        align-items:center;justify-content:center}
  #gate{background:rgba(10,11,14,.97);z-index:10}
  #help>div,#gate>div{background:var(--panel);border:1px solid var(--line);border-radius:12px;
            padding:22px;max-width:620px;max-height:84vh;overflow:auto}
  kbd{background:var(--panel2);border:1px solid var(--line);border-radius:4px;
      padding:1px 6px;font:12px ui-monospace,monospace}
  table{border-collapse:collapse;width:100%} td{padding:3px 6px;vertical-align:top}
  .sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;
      vertical-align:-1px}
</style>
</head>
<body>

<div class="bar">
  <select id="fQueue"></select>
  <select id="fRule">
    <option value="">all rules</option>
    <option value="rule_1">rule_1</option><option value="rule_2">rule_2</option>
    <option value="rule_3">rule_3</option><option value="rule_4">rule_4</option>
  </select>
  <select id="fState">
    <option value="">all</option><option value="todo">undecided</option>
    <option value="done">decided</option>
  </select>
  <select id="fSource">
    <option value="">val+test</option><option value="val">val</option><option value="test">test</option>
  </select>
  <input id="fText" placeholder="search" style="width:110px">
  <span class="grow"></span>
  <span id="prog" class="muted"></span>
  <span id="smode" class="chip"></span>
  <button id="bSaveFile" class="big" title="Save to a new file">Save to file…</button>
  <button id="bOpenFile" class="big" title="Open a file you saved earlier">Open…</button>
  <button id="bHelp" title="Help">?</button>
</div>

<div class="wrap">
  <div class="left">
    <div class="canvasbox"><canvas id="cv"></canvas></div>
    <div class="bar" style="border-top:1px solid var(--line);border-bottom:0">
      <span class="muted">show</span>
      <button class="lay on" data-r="rule_1" title="rule_1 boxes"><span class="sw" style="background:var(--r1)"></span>1</button>
      <button class="lay on" data-r="rule_2" title="rule_2 boxes"><span class="sw" style="background:var(--r2)"></span>2</button>
      <button class="lay on" data-r="rule_3" title="rule_3 boxes"><span class="sw" style="background:var(--r3)"></span>3</button>
      <button class="lay on" data-r="rule_4" title="rule_4 boxes"><span class="sw" style="background:var(--r4)"></span>4</button>
      <button class="lay on" data-r="mocs" title="human-annotated MOCS box"><span class="sw" style="background:var(--mocs)"></span>MOCS</button>
      <span class="grow"></span>
      <span class="muted">draw</span>
      <select id="drawRule" title="Pick a rule, then drag on the image to add a box">
        <option value="">off</option>
        <option value="rule_1">rule_1</option><option value="rule_2">rule_2</option>
        <option value="rule_3">rule_3</option><option value="rule_4">rule_4</option>
      </select>
      <button id="bClearBoxes" title="Remove every box you drew on this image">clear mine</button>
    </div>
  </div>

  <div class="right">
    <div class="sec">
      <div class="row" style="justify-content:space-between">
        <strong id="rid">-</strong>
        <span><span class="chip" id="rsrc"></span> <span class="chip" id="rrun"></span></span>
      </div>
      <div class="muted" id="rq" style="margin-top:4px"></div>
    </div>

    <div class="sec" id="secCap">
      <h3>Caption &mdash; is it true of this photo? <kbd>C</kbd></h3>
      <div class="caption" id="cap"></div>
      <div class="row" style="margin-top:6px">
        <span class="muted">accurate?</span>
        <button class="y" data-cap="y">yes</button>
        <button class="n" data-cap="n">no</button>
        <span id="capTodo" class="chip" style="border-color:var(--maybe);color:var(--maybe)">not judged</span>
      </div>
    </div>

    <div class="sec">
      <h3>Rules &mdash; judge all four <span style="color:var(--maybe)">(not just the flagged ones)</span></h3>
      <div id="rules"></div>
    </div>

    <div class="sec">
      <h3>MOCS</h3>
      <div class="muted" id="cats"></div>
      <div class="muted" id="r4hint" style="margin-top:4px"></div>
    </div>

    <div class="sec">
      <h3>Decision</h3>
      <div class="row">
        <button class="dec big y" data-dec="accept">Accept <kbd>A</kbd></button>
        <button class="dec big n" data-dec="reject">Reject <kbd>R</kbd></button>
        <button class="dec big m" data-dec="unsure">Unsure <kbd>U</kbd></button>
      </div>
      <div class="row" style="margin-top:7px">
        <button id="bHard" class="m">flag as hard / needs 2nd opinion <kbd>H</kbd></button>
      </div>
      <textarea id="notes" placeholder="notes (optional)" style="margin-top:7px"></textarea>
    </div>

    <div class="sec">
      <div class="row">
        <button id="bPrev">&larr; prev</button>
        <button id="bNext">next &rarr;</button>
        <button id="bNextTodo" class="big">next undecided <kbd>N</kbd></button>
      </div>
      <div class="row" style="margin-top:7px">
        <span class="muted">jump</span>
        <input id="jump" style="width:90px" placeholder="# or id">
      </div>
    </div>
  </div>
</div>

<div id="gate"><div>
  <h2 style="margin-top:0">Choose where your work is saved</h2>
  <p>Everything you do is written straight to a file on your own computer, as you go.
     Nothing is kept in the browser, so nothing can be lost by closing a tab or clearing
     browsing data &mdash; but you have to pick that file before you start.</p>
  <div class="row" style="margin:18px 0">
    <button id="gSave" class="big">Save to new file&hellip;</button>
    <button id="gOpen" class="big">Open saved file&hellip;</button>
  </div>
  <p class="muted"><strong>First time?</strong> Press <em>Save to new file&hellip;</em> and
     save it as <code>review_results.json</code> somewhere you will remember, such as your
     Documents folder.<br>
     <strong>Coming back?</strong> Press <em>Open saved file&hellip;</em> and pick that same
     file &mdash; your work reappears and keeps saving to it.</p>
  <p id="gateWarn" style="color:#ff8a8a"></p>
</div></div>

<div id="help"><div>
  <h2 style="margin-top:0">How to review</h2>
  <p>Every row is a <strong>model proposal, not a label</strong>. Your decision is what makes it data.</p>
  <h3>Two things that are easy to get wrong</h3>
  <ol>
    <li><strong>On every image you accept, judge all four rules AND the caption</strong> &mdash;
      not only the rules the model flagged. About 1 image in 10 violates rule&nbsp;1 (no hard
      hat / uncovered shoulders or legs), so leaving it unset on a photo that really shows it
      puts a false negative into the strongest rule in the project. A rule you never marked is
      recorded as <em>not violated</em>, and a caption you never marked cannot be used at all.
      The app will stop you if you try to accept with anything unjudged.</li>
    <li><strong>Do not trust the model's boxes.</strong> Where a green <em>MOCS r4</em> box
      exists it came from human annotation &mdash; prefer it. Otherwise, if the finding is
      right but the box is wrong, pick the rule under &ldquo;draw box for&rdquo; and drag a
      new one on the image.</li>
    <li><strong>If you turn a rule ON that the model did not propose, fill in the reason
      box.</strong> It appears inside the rule card as soon as you mark the rule
      &ldquo;yes&rdquo;. One sentence: <em>who or what is at fault, identified by position
      or appearance, and what the breach is</em> &mdash; e.g. &ldquo;The worker on the left
      is on foot without a hard hat.&rdquo; Without it we get a violation with nothing to
      learn from. For a rule the model <em>did</em> propose, leave it blank unless its
      reason is wrong.</li>
  </ol>
  <h3>Colours</h3>
  <table>
    <tr><td><span class="sw" style="background:var(--r1)"></span>rule_1</td><td>basic PPE &mdash; no hard hat, uncovered shoulders/legs</td></tr>
    <tr><td><span class="sw" style="background:var(--r2)"></span>rule_2</td><td>working at height with no safety harness</td></tr>
    <tr><td><span class="sw" style="background:var(--r3)"></span>rule_3</td><td>open excavation / edge with no guard rail or barrier</td></tr>
    <tr><td><span class="sw" style="background:var(--r4)"></span>rule_4</td><td>person inside a machine's operating radius / blind spot</td></tr>
    <tr><td><span class="sw" style="background:var(--mocs)"></span>MOCS r4</td><td><strong>human-annotated</strong> worker+machine box. Trust this over the model</td></tr>
  </table>
  <p class="muted">Dashed boxes are ones you drew. Click a dashed box to delete it.</p>
  <h3>Keyboard</h3>
  <table>
    <tr><td><kbd>1</kbd><kbd>2</kbd><kbd>3</kbd><kbd>4</kbd></td><td>cycle that rule: yes &rarr; no &rarr; unset</td></tr>
    <tr><td><kbd>A</kbd> <kbd>R</kbd> <kbd>U</kbd></td><td>accept / reject / unsure</td></tr>
    <tr><td><kbd>C</kbd></td><td>toggle &ldquo;caption accurate&rdquo;</td></tr>
    <tr><td><kbd>H</kbd></td><td>flag as hard</td></tr>
    <tr><td><kbd>&larr;</kbd> <kbd>&rarr;</kbd></td><td>previous / next image</td></tr>
    <tr><td><kbd>N</kbd></td><td>next undecided</td></tr>
    <tr><td><kbd>E</kbd></td><td>export</td></tr>
  </table>
  <h3>Saving</h3>
  <p>Your work goes <strong>straight into a file on your own computer</strong>, automatically,
     every time you change anything. Nothing is stored in the browser, so closing the tab or
     clearing browsing data cannot lose it.</p>
  <p><strong>Starting out:</strong> press <em>Save to new file&hellip;</em> and save it as
     <code>review_results.json</code> somewhere you will remember.<br>
     <strong>Coming back:</strong> press <em>Open saved file&hellip;</em> and pick that same file.
     Your work reappears and keeps saving to it.</p>
  <p>The chip in the toolbar shows the filename and the time of the last save, so you can always
     see that it is working. If it ever turns red and says
     <span class="chip" style="background:#4d1414;border-color:#ff5252;color:#ffb3b3">NOT SAVING</span>,
     stop and choose the file again.</p>
  <p class="muted">When you finish a session, send that <code>.json</code> file back. There is
     nothing to export &mdash; it is already up to date.</p>
  <p class="muted">Suggested order: <strong>sample</strong> first (it tells us how accurate
     the model is), then <strong>tier1</strong> (the two rarest, most valuable rules).</p>
  <button onclick="document.getElementById('help').style.display='none'">Close</button>
</div></div>

<script>
const DATA = __DATA__;
const META = __META__;
const RULES = ["rule_1","rule_2","rule_3","rule_4"];
const COL = {rule_1:"#00E5FF",rule_2:"#FFEA00",rule_3:"#FF3D00",rule_4:"#D500F9",mocs:"#00E676"};
const LS_KEY = "mocs_review_" + META.corpus_key;

let state = {};          // id -> verdict object
let view = [];           // filtered indices into DATA
let pos = 0;
let layers = {rule_1:1,rule_2:1,rule_3:1,rule_4:1,mocs:1};
const img = new Image();

// ---------------------------------------------------------------- persistence
//
// ONE mechanism: a real file on the reviewer's disk, via the File System Access
// API. They pick it before they start; after that every change is written to it
// automatically. Nothing is kept in the browser.
//
// Why not localStorage as a backup: on a file:// page it is blocked outright in
// some browser configurations and wiped on close in others, which means it can
// look like it is working right up until a day of review disappears. A mechanism
// that fails silently is worse than no mechanism, so there is only the file --
// and the app refuses to let anyone start reviewing until it has one.
let fileHandle = null, saveTimer = null, pending = false, reviewer = "";

function setMode(ok, extra){
  const el = document.getElementById("smode");
  if(ok){ el.textContent = "saving to " + (fileHandle ? fileHandle.name : "file") +
                           (extra ? " · " + extra : "");
          el.style.cssText = "background:#123d22;border-color:#2ecc71;color:#8ef0b0"; }
  else  { el.textContent = "NOT SAVING";
          el.style.cssText = "background:#4d1414;border-color:#ff5252;color:#ffb3b3;font-weight:700"; }
}
function showGate(msg){
  document.getElementById("gateWarn").textContent = msg || "";
  gate.style.display = "flex";
}
function save(){
  if(!fileHandle){ showGate("Your last change was not saved — choose a file."); return; }
  pending = true;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(writeFile, 600);   // debounce: typing a reason fires many events
}
async function writeFile(){
  if(!fileHandle) return;
  try{
    const w = await fileHandle.createWritable();
    await w.write(payloadJSON());
    await w.close();
    pending = false;
    setMode(true, "saved " + new Date().toLocaleTimeString());
  }catch(e){
    console.warn("file write failed", e);
    fileHandle = null; setMode(false);
    showGate("Could not write to that file (" + (e.message || e.name) +
             "). Choose it again, or pick a new one.");
  }
}
function vd(id){ const o = state[id] || (state[id] = {rules:{}, boxes:{}, reasons:{}});
                 o.rules=o.rules||{}; o.boxes=o.boxes||{}; o.reasons=o.reasons||{}; return o; }
function decided(id){ return !!(state[id] && state[id].decision); }

// ---------------------------------------------------------------- filtering
function applyFilters(){
  const q=fQueue.value, r=fRule.value, s=fState.value, src=fSource.value,
        t=fText.value.trim().toLowerCase();
  view = DATA.map((d,i)=>i).filter(i=>{
    const d=DATA[i];
    if(q && !(d.q||[]).includes(q)) return false;
    if(r && !(d.r[r] && d.r[r].p)) return false;
    if(src && d.src!==src) return false;
    if(s==="todo" && decided(d.id)) return false;
    if(s==="done" && !decided(d.id)) return false;
    if(t && !((d.cap||"").toLowerCase().includes(t) || d.id.toLowerCase().includes(t))) return false;
    return true;
  });
  if(pos>=view.length) pos=0;
  render();
}

// ---------------------------------------------------------------- canvas
const cv=document.getElementById("cv"), ctx=cv.getContext("2d");
let tf={s:1,ox:0,oy:0};
function drawCanvas(){
  const d=cur(); if(!d) return;
  const box=cv.parentElement.getBoundingClientRect();
  const iw=img.naturalWidth||d.w, ih=img.naturalHeight||d.h;
  const s=Math.min(box.width/iw, box.height/ih);
  cv.width=Math.max(1,Math.round(iw*s)); cv.height=Math.max(1,Math.round(ih*s));
  tf={s:1,ox:0,oy:0};
  ctx.clearRect(0,0,cv.width,cv.height);
  if(img.complete && img.naturalWidth) ctx.drawImage(img,0,0,cv.width,cv.height);
  else { ctx.fillStyle="#222"; ctx.fillRect(0,0,cv.width,cv.height);
         ctx.fillStyle="#888"; ctx.fillText("image not found: "+d.img,12,22); }
  const lw=Math.max(2,Math.round(Math.min(cv.width,cv.height)/280));
  const fs=Math.max(12,Math.round(Math.min(cv.width,cv.height)/42));
  const placed=[];
  function box1(b,colour,label,dashed){
    const x=b[0]*cv.width,y=b[1]*cv.height,w=(b[2]-b[0])*cv.width,h=(b[3]-b[1])*cv.height;
    ctx.setLineDash(dashed?[9,6]:[]); ctx.lineWidth=dashed?lw+1:lw; ctx.strokeStyle=colour;
    ctx.strokeRect(x,y,w,h); ctx.setLineDash([]);
    ctx.font="600 "+fs+"px system-ui"; const tw=ctx.measureText(label).width+10, th=fs+6;
    let lx=Math.min(Math.max(0,x), cv.width-tw), ly=y-th>=0?y-th:y;
    for(let k=0;k<8;k++){ const c=[lx,ly+k*(th+2),lx+tw,ly+k*(th+2)+th];
      if(!placed.some(p=>c[0]<p[2]&&p[0]<c[2]&&c[1]<p[3]&&p[1]<c[3])){ ly=c[1]; break; } }
    placed.push([lx,ly,lx+tw,ly+th]);
    ctx.fillStyle=colour; ctx.fillRect(lx,ly,tw,th);
    ctx.fillStyle="#000"; ctx.fillText(label,lx+5,ly+th-6);
  }
  for(const r of RULES){
    if(!layers[r]) continue;
    const v=d.r[r]; if(v&&v.p) for(const b of v.boxes||[]) box1(b,COL[r],r,false);
    for(const b of (vd(d.id).boxes[r]||[])) box1(b,COL[r],r+" (mine)",true);
  }
  if(layers.mocs) for(const b of (d.r4||[])) box1(b,COL.mocs,"MOCS r4",false);
  if(drag) { ctx.setLineDash([6,4]); ctx.lineWidth=lw;
    ctx.strokeStyle=COL[drawRule.value]||"#fff";
    ctx.strokeRect(drag.x0,drag.y0,drag.x-drag.x0,drag.y-drag.y0); ctx.setLineDash([]); }
}
let drag=null;
function xy(e){ const r=cv.getBoundingClientRect(); return {x:e.clientX-r.left,y:e.clientY-r.top}; }
cv.addEventListener("mousedown",e=>{
  const d=cur(); if(!d) return; const p=xy(e);
  if(!drawRule.value){                       // click a dashed box to delete it
    const mine=vd(d.id).boxes;
    for(const r of RULES) for(let i=0;i<(mine[r]||[]).length;i++){
      const b=mine[r][i];
      if(p.x>=b[0]*cv.width&&p.x<=b[2]*cv.width&&p.y>=b[1]*cv.height&&p.y<=b[3]*cv.height){
        mine[r].splice(i,1); save(); drawCanvas(); return;
      }
    }
    return;
  }
  drag={x0:p.x,y0:p.y,x:p.x,y:p.y};
});
cv.addEventListener("mousemove",e=>{ if(drag){ const p=xy(e); drag.x=p.x; drag.y=p.y; drawCanvas(); }});
window.addEventListener("mouseup",()=>{
  if(!drag) return;
  const d=cur(), r=drawRule.value;
  const x1=Math.min(drag.x0,drag.x)/cv.width,  y1=Math.min(drag.y0,drag.y)/cv.height;
  const x2=Math.max(drag.x0,drag.x)/cv.width,  y2=Math.max(drag.y0,drag.y)/cv.height;
  drag=null;
  if(d && r && (x2-x1)>0.01 && (y2-y1)>0.01){
    const v=vd(d.id); (v.boxes[r]=v.boxes[r]||[]).push([+x1.toFixed(4),+y1.toFixed(4),
                                                        +x2.toFixed(4),+y2.toFixed(4)]);
    if(!v.rules[r]) v.rules[r]="y";          // drawing a box asserts the rule
    save(); render();
  } else drawCanvas();
});
window.addEventListener("resize",drawCanvas);

// ---------------------------------------------------------------- render
function cur(){ return view.length ? DATA[view[pos]] : null; }
function render(){
  const d=cur();
  prog.textContent = view.length
    ? `${pos+1} / ${view.length}   ·   ${Object.values(state).filter(v=>v.decision).length} decided of ${DATA.length}`
    : "nothing matches these filters";
  if(!d){ ctx.clearRect(0,0,cv.width,cv.height); rid.textContent="-"; rules.innerHTML=""; return; }
  const v=vd(d.id);
  rid.textContent=d.id; rsrc.textContent=d.src; rrun.textContent=d.run;
  rq.textContent="queues: "+(d.q||[]).join(", ");
  cap.textContent=d.cap||"(no caption)";
  cats.textContent="categories: "+((d.cats||[]).join(", ")||"none (test split carries no MOCS annotations)");
  r4hint.textContent=(d.r4&&d.r4.length)
      ? "green box = human-annotated worker+machine region — prefer it for rule_4"
      : "no MOCS geometry for this image";
  document.querySelectorAll("[data-cap]").forEach(b=>b.classList.toggle("on",v.caption_ok===b.dataset.cap));
  capTodo.style.display = v.caption_ok ? "none" : "inline-block";
  document.querySelectorAll(".dec").forEach(b=>b.classList.toggle("on",v.decision===b.dataset.dec));
  bHard.classList.toggle("on",!!v.hard);
  notes.value=v.notes||"";
  rules.innerHTML = RULES.map(r=>{
    const p=d.r[r]&&d.r[r].p, val=v.rules[r]||"";
    const mine=(v.boxes[r]||[]).length;
    return `<div class="rule ${p?'prop':''}" data-r="${r}">
      <div class="hd"><span class="sw" style="background:${COL[r]}"></span>
        <span class="nm">${r}</span>
        <span class="chip">${p?'proposed':'not proposed'}</span>
        <span class="grow" style="flex:1"></span>
        <button class="y rv ${val==='y'?'on':''}" data-r="${r}" data-v="y">yes</button>
        <button class="n rv ${val==='n'?'on':''}" data-r="${r}" data-v="n">no</button>
      </div>
      ${p?`<div class="reason">${esc(d.r[r].reason||"")}</div>
           <div class="muted">${(d.r[r].boxes||[]).length} model box(es)${mine?` · ${mine} of yours`:""}</div>`
         :`<div class="muted">${mine?`${mine} box(es) you drew`:"&mdash;"}</div>`}
      ${val==="y"?`<input class="rsn" data-r="${r}" value="${esc(v.reasons[r]||"")}"
          style="width:100%;margin-top:6px" placeholder="${p
            ? "reason looks wrong? write a better one (optional)"
            : "REASON NEEDED — one sentence: who/what is at fault, and what the breach is"}">`:""}
    </div>`;
  }).join("");
  rules.querySelectorAll(".rv").forEach(b=>b.onclick=()=>{
    const o=vd(d.id); o.rules[b.dataset.r]= o.rules[b.dataset.r]===b.dataset.v?"":b.dataset.v;
    save(); render();
  });
  // oninput only SAVES -- it must not re-render, or the field loses focus mid-word.
  rules.querySelectorAll(".rsn").forEach(el=>el.oninput=()=>{
    vd(d.id).reasons[el.dataset.r]=el.value; save();
  });
  img.onload=drawCanvas; img.onerror=drawCanvas;
  if(img.getAttribute("src")!==d.img){ img.src=d.img; } else drawCanvas();
}
// Escapes " as well as &<> because this also fills a value="..." attribute; a reason
// containing a quote would otherwise break out of it and mangle the rule card.
function esc(s){ return String(s==null?"":s).replace(/[&<>"]/g,
    c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }

// ---------------------------------------------------------------- actions
function setDec(x){
  const d=cur(); if(!d) return; const v=vd(d.id);
  // An ACCEPT with anything left unset is the one silently damaging outcome: a rule
  // nobody looked at is recorded as "not violated" (a false negative in the data),
  // and an unchecked caption cannot be used as a training target at all. Reject needs
  // neither, and if the reviewer follows the instructions this never fires.
  if(x==="accept" && v.decision!==x){
    const miss = RULES.filter(r=>!v.rules[r]);
    if(!v.caption_ok) miss.push("caption");
    if(miss.length && !confirm(
        "Not judged yet: "+miss.join(", ")+".\n\n"+
        "An unjudged rule gets recorded as NOT violated, and an unjudged caption cannot "+
        "be used at all. Press Cancel and mark them first — keys 1-4 for the rules, "+
        "C for the caption.\n\n"+
        "Accept anyway?")) return;
  }
  v.decision = v.decision===x ? "" : x; v.ts=new Date().toISOString(); save();
  if(v.decision) nextTodo(); else render();
}
function step(n){ if(!view.length) return; pos=(pos+n+view.length)%view.length; render(); }
function nextTodo(){ for(let k=1;k<=view.length;k++){ const i=(pos+k)%view.length;
    if(!decided(DATA[view[i]].id)){ pos=i; render(); return; } } step(1); }

document.querySelectorAll(".dec").forEach(b=>b.onclick=()=>setDec(b.dataset.dec));
document.querySelectorAll("[data-cap]").forEach(b=>b.onclick=()=>{
  const v=vd(cur().id); v.caption_ok = v.caption_ok===b.dataset.cap?"":b.dataset.cap; save(); render(); });
bHard.onclick=()=>{ const v=vd(cur().id); v.hard=!v.hard; save(); render(); };
notes.oninput=()=>{ vd(cur().id).notes=notes.value; save(); };
bPrev.onclick=()=>step(-1); bNext.onclick=()=>step(1); bNextTodo.onclick=nextTodo;
bClearBoxes.onclick=()=>{ vd(cur().id).boxes={}; save(); render(); };
document.querySelectorAll(".lay").forEach(b=>b.onclick=()=>{
  layers[b.dataset.r]=!layers[b.dataset.r]; b.classList.toggle("on"); drawCanvas(); });
drawRule.onchange=()=>{ cv.style.cursor = drawRule.value?"crosshair":"pointer"; };
[fQueue,fRule,fState,fSource].forEach(el=>el.onchange=()=>{pos=0;applyFilters();});
fText.oninput=()=>{pos=0;applyFilters();};
jump.onchange=()=>{ const t=jump.value.trim();
  const n=parseInt(t,10);
  if(!isNaN(n) && n>=1 && n<=view.length){ pos=n-1; }
  else { const i=view.findIndex(k=>DATA[k].id===t || DATA[k].id.endsWith(t)); if(i>=0) pos=i; }
  jump.value=""; render(); };
bHelp.onclick=()=>help.style.display="flex";
help.onclick=e=>{ if(e.target===help) help.style.display="none"; };

document.addEventListener("keydown",e=>{
  if(/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
  // Shortcuts used to keep working behind the gate, quietly changing state that
  // could not be written anywhere.
  if(gate.style.display==="flex") return;
  const k=e.key.toLowerCase(); const d=cur(); if(!d) return;
  if(["1","2","3","4"].includes(k)){ const r="rule_"+k; const v=vd(d.id);
    v.rules[r] = v.rules[r]==="y" ? "n" : v.rules[r]==="n" ? "" : "y"; save(); render(); e.preventDefault(); }
  else if(k==="a") setDec("accept");
  else if(k==="r") setDec("reject");
  else if(k==="u") setDec("unsure");
  else if(k==="c"){ const v=vd(d.id); v.caption_ok = v.caption_ok==="y"?"n":v.caption_ok==="n"?"":"y"; save(); render(); }
  else if(k==="h"){ const v=vd(d.id); v.hard=!v.hard; save(); render(); }
  else if(k==="n") nextTodo();
  else if(e.key==="ArrowRight") step(1);
  else if(e.key==="ArrowLeft") step(-1);
  else if(k==="?") help.style.display="flex";
});

// ---------------------------------------------------------------- the saved file
function b1000(b){ return "["+b.map(c=>Math.round(c*1000)).join(", ")+"]"; }
function payloadJSON(){
  const out={};
  for(const d of DATA){
    const v=state[d.id];
    if(!v || (!v.decision && !v.notes && !Object.keys(v.rules||{}).length
        && !Object.keys(v.boxes||{}).length && !Object.keys(v.reasons||{}).length)) continue;
    out[d.id]=v;
  }
  return JSON.stringify({reviewer:reviewer||"reviewer",
    saved_at:new Date().toISOString(), corpus_key:META.corpus_key,
    n_decided:Object.values(state).filter(v=>v.decision).length,
    n_touched:Object.keys(out).length, verdicts:out}, null, 1);
}
// Kept, unused by the app itself: it is the one place that documents, in code, the
// exact column set a verdict maps onto -- which is what build_review.py's CSVs use
// and what any later join has to produce.
function buildRows(){
  const rows=[];
  for(const d of DATA){
    const v=state[d.id]; if(!v || (!v.decision && !v.notes && !Object.keys(v.rules||{}).length
        && !Object.keys(v.boxes||{}).length && !Object.keys(v.reasons||{}).length)) continue;
    out[d.id]=v;
    const row={new_image_id:d.id, image_file:d.img.split("/").pop(), mocs_image_id:d.mid,
      file_name:d.fn, source:d.src, run:d.run,
      proposed_rules:RULES.filter(r=>d.r[r]&&d.r[r].p).join(" "),
      n_flagged:RULES.filter(r=>d.r[r]&&d.r[r].p).length, caption:d.cap};
    for(const r of RULES){
      row[r+"_proposed"]=d.r[r]&&d.r[r].p?1:0;
      row[r+"_reason"]=(d.r[r]&&d.r[r].reason)||"";
      row[r+"_boxes_1000"]=((d.r[r]&&d.r[r].boxes)||[]).map(b1000).join("; ");
    }
    row.mocs_categories=(d.cats||[]).join(" ");
    row.mocs_suggested_rule4_box_1000=(d.r4||[]).map(b1000).join("; ");
    row.image_path_original=d.orig||"";
    row.verify_decision=v.decision||"";
    for(const r of RULES) row["verify_"+r]=(v.rules||{})[r]||"";
    row.verify_caption_ok=v.caption_ok||"";
    row.verify_corrected_reason=Object.entries(v.reasons||{})
        .filter(([,t])=>t&&t.trim()).map(([r,t])=>r+": "+t.trim()).join("; ");
    row.verify_corrected_reason_json=JSON.stringify(v.reasons||{});
    row.verify_corrected_boxes_1000=Object.entries(v.boxes||{})
        .flatMap(([r,bs])=>bs.map(b=>r+":"+b1000(b))).join("; ");
    row.verify_corrected_boxes_json=JSON.stringify(v.boxes||{});
    row.verify_difficulty=v.hard?"hard":"";
    row.verify_notes=v.notes||"";
    rows.push(row);
  }
  return rows;
}
// ---------------------------------------------------------------- file pickers
const FS_OK = (typeof window.showSaveFilePicker === "function");
const PICK_TYPES = [{description:"Review results (JSON)",
                     accept:{"application/json":[".json"]}}];

function askName(){
  if(!reviewer) reviewer = (prompt("Your name (recorded in the file):","")||"reviewer").trim();
  return reviewer;
}
async function pickSaveFile(){
  try{
    askName();
    fileHandle = await window.showSaveFilePicker(
        {suggestedName:"review_results.json", types:PICK_TYPES});
    await writeFile();
    gate.style.display="none";
    render();
  }catch(e){ if(e && e.name!=="AbortError") showGate("Could not use that file: "+e.message); }
}
async function pickOpenFile(){
  try{
    const [h] = await window.showOpenFilePicker({types:PICK_TYPES, multiple:false});
    const j = JSON.parse(await (await h.getFile()).text());
    // A file from a DIFFERENT package would merge ids that are not in DATA, show
    // nothing on screen, and give no clue why. Check, and report what landed.
    if(j.corpus_key && j.corpus_key!==META.corpus_key &&
       !confirm("That file came from a DIFFERENT review package.\n\n  file: "+j.corpus_key+
                "\n  this: "+META.corpus_key+"\n\nOpen anyway?")) return;
    const ids=new Set(DATA.map(d=>d.id)); const v=j.verdicts||j;
    let n=0, skipped=0;
    for(const k in v){ if(ids.has(k)){ state[k]=v[k]; n++; } else skipped++; }
    fileHandle = h;                       // keep writing back to this SAME file
    reviewer = j.reviewer || reviewer;
    await writeFile();
    gate.style.display="none";
    applyFilters();
    alert("Loaded "+n+" record(s)."+(skipped?"\n"+skipped+" not in this package, ignored.":"")+
          "\n\nSaving back to this file from now on.");
  }catch(e){ if(e && e.name!=="AbortError") showGate("Could not open that file: "+e.message); }
}
bSaveFile.onclick=pickSaveFile; gSave.onclick=pickSaveFile;
bOpenFile.onclick=pickOpenFile; gOpen.onclick=pickOpenFile;

// Only fires if a write is still queued, or there is work in memory with nowhere to
// put it. Without the second condition it warned the instant the page opened, while
// the gate was still up and there was nothing to lose -- an alarm that cries wolf is
// worse than none, because it trains people to click through the real one.
window.addEventListener("beforeunload",e=>{
  if(pending || (!fileHandle && Object.keys(state).length)){
    e.preventDefault(); e.returnValue="";
  }
});

// ---------------------------------------------------------------- boot
(function(){
  const qs=new Set(); DATA.forEach(d=>(d.q||[]).forEach(q=>qs.add(q)));
  const order=["sample","tier1","tier2","tier3","negatives"];
  fQueue.innerHTML='<option value="">all queues</option>'+
    order.filter(q=>qs.has(q)).map(q=>`<option value="${q}">${q}</option>`).join("");
  fQueue.value = qs.has("sample") ? "sample" : "";

  setMode(false);
  applyFilters();

  // No file, no reviewing. The gate cannot be dismissed any other way, so there is
  // no path where someone works for an hour into nothing.
  if(!FS_OK){
    document.querySelector("#gate>div").innerHTML =
      "<h2 style='margin-top:0'>Please use Chrome or Edge</h2>"+
      "<p>This tool saves your work straight to a file on your computer, which this "+
      "browser does not support. Firefox and Safari cannot run it.</p>"+
      "<p class='muted'>Copy the folder's address into Chrome or Edge and open "+
      "<code>index.html</code> there.</p>";
    gate.style.display="flex";
    return;
  }
  showGate("");
})();
</script>
</body>
</html>
"""


def build_html(title: str, data_json: str, meta_json: str) -> str:
    """str.replace, never .format -- the CSS is full of braces."""
    return (HTML_TEMPLATE
            .replace("__TITLE__", title)
            .replace("__DATA__", data_json)
            .replace("__META__", meta_json))
