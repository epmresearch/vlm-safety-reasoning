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
  * AUTOSAVE TO localStorage ON EVERY KEYSTROKE, because losing six hours of review
    to a closed tab would be worse than any amount of slowness. An unexported-count
    badge nags past 50 decisions.
  * EXPORT IS LOSSLESS JSON + a readable CSV carrying the same columns the
    spreadsheet path produces, so whichever way the review happens, the file that
    comes back downstream is the same shape.
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
  body{margin:0;font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
       background:var(--bg);color:var(--fg);overflow:hidden}
  button{font:inherit;cursor:pointer;border:1px solid var(--line);background:var(--panel2);
         color:var(--fg);border-radius:6px;padding:5px 10px}
  button:hover{border-color:var(--accent)}
  button.on{background:var(--accent);border-color:var(--accent);color:#fff}
  select,input,textarea{font:inherit;background:var(--panel2);color:var(--fg);
         border:1px solid var(--line);border-radius:6px;padding:5px 8px}
  .bar{display:flex;gap:8px;align-items:center;padding:8px 12px;background:var(--panel);
       border-bottom:1px solid var(--line);flex-wrap:wrap}
  .grow{flex:1}
  .wrap{display:flex;height:calc(100vh - 53px)}
  .left{flex:1;min-width:0;display:flex;flex-direction:column;background:#0e1013}
  .canvasbox{flex:1;position:relative;display:flex;align-items:center;justify-content:center;
             overflow:hidden}
  canvas{max-width:100%;max-height:100%;cursor:crosshair}
  .right{width:430px;flex-shrink:0;overflow-y:auto;background:var(--panel);
         border-left:1px solid var(--line);padding:12px}
  .sec{margin-bottom:14px;padding-bottom:12px;border-bottom:1px solid var(--line)}
  .sec:last-child{border:0}
  h3{margin:0 0 7px;font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim)}
  .row{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
  .chip{display:inline-block;padding:1px 7px;border-radius:99px;font-size:11px;
        background:var(--panel2);border:1px solid var(--line);color:var(--dim)}
  .rule{border:1px solid var(--line);border-radius:8px;padding:8px;margin-bottom:7px;
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
  .big{font-size:15px;padding:9px 14px;font-weight:600}
  .caption{background:var(--panel2);border:1px solid var(--line);border-radius:8px;
           padding:8px;font-size:13px}
  .badge{background:var(--no);color:#fff;border-radius:99px;padding:1px 7px;font-size:11px}
  #help{position:fixed;inset:0;background:rgba(0,0,0,.82);display:none;z-index:9;
        align-items:center;justify-content:center}
  #help>div{background:var(--panel);border:1px solid var(--line);border-radius:12px;
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
  <strong>__TITLE__</strong>
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
  <input id="fText" placeholder="search caption / id" style="width:170px">
  <span class="grow"></span>
  <span id="prog" class="muted"></span>
  <span id="unsaved"></span>
  <button id="bExport" class="big">Export</button>
  <button id="bImport">Import</button>
  <button id="bHelp">?</button>
  <input type="file" id="fileIn" accept=".json" style="display:none">
</div>

<div class="wrap">
  <div class="left">
    <div class="canvasbox"><canvas id="cv"></canvas></div>
    <div class="bar" style="border-top:1px solid var(--line);border-bottom:0">
      <span class="muted">layers</span>
      <button class="lay on" data-r="rule_1"><span class="sw" style="background:var(--r1)"></span>rule_1</button>
      <button class="lay on" data-r="rule_2"><span class="sw" style="background:var(--r2)"></span>rule_2</button>
      <button class="lay on" data-r="rule_3"><span class="sw" style="background:var(--r3)"></span>rule_3</button>
      <button class="lay on" data-r="rule_4"><span class="sw" style="background:var(--r4)"></span>rule_4</button>
      <button class="lay on" data-r="mocs"><span class="sw" style="background:var(--mocs)"></span>MOCS r4</button>
      <span class="grow"></span>
      <span class="muted">draw box for</span>
      <select id="drawRule">
        <option value="">off</option>
        <option value="rule_1">rule_1</option><option value="rule_2">rule_2</option>
        <option value="rule_3">rule_3</option><option value="rule_4">rule_4</option>
      </select>
      <button id="bClearBoxes">clear my boxes</button>
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

    <div class="sec">
      <h3>Caption</h3>
      <div class="caption" id="cap"></div>
      <div class="row" style="margin-top:6px">
        <span class="muted">accurate?</span>
        <button class="y" data-cap="y">yes</button>
        <button class="n" data-cap="n">no</button>
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

<div id="help"><div>
  <h2 style="margin-top:0">How to review</h2>
  <p>Every row is a <strong>model proposal, not a label</strong>. Your decision is what makes it data.</p>
  <h3>Two things that are easy to get wrong</h3>
  <ol>
    <li><strong>Judge all four rules on every image you accept</strong>, not only the ones the
      model flagged. About 1 image in 10 violates rule&nbsp;1 (no hard hat / uncovered
      shoulders or legs). Accepting a row while leaving rule&nbsp;1 unset, when the image
      really does show it, puts a false negative into the strongest rule in the project.</li>
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
  <p>Your work saves automatically in this browser after every click. <strong>Press
     Export regularly anyway</strong> &mdash; that downloads the file you send back.
     Clearing browser data would otherwise lose everything. <em>Import</em> reloads an
     exported file so you can carry on where you left off, or on another machine.</p>
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
let exportedAt = 0;
const img = new Image();

// ---------------------------------------------------------------- persistence
function load(){ try{ state = JSON.parse(localStorage.getItem(LS_KEY)) || {}; }catch(e){ state={}; } }
function save(){ try{ localStorage.setItem(LS_KEY, JSON.stringify(state)); }
                 catch(e){ console.warn("localStorage full/blocked", e); } }
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
  const pending=Object.values(state).filter(v=>v.decision).length-exportedAt;
  unsaved.innerHTML = pending>50 ? `<span class="badge">${pending} unexported — press Export</span>` : "";
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
function setDec(x){ const d=cur(); if(!d) return; const v=vd(d.id);
  v.decision = v.decision===x ? "" : x; v.ts=new Date().toISOString(); save();
  if(v.decision) nextTodo(); else render(); }
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
  const k=e.key.toLowerCase(); const d=cur(); if(!d) return;
  if(["1","2","3","4"].includes(k)){ const r="rule_"+k; const v=vd(d.id);
    v.rules[r] = v.rules[r]==="y" ? "n" : v.rules[r]==="n" ? "" : "y"; save(); render(); e.preventDefault(); }
  else if(k==="a") setDec("accept");
  else if(k==="r") setDec("reject");
  else if(k==="u") setDec("unsure");
  else if(k==="c"){ const v=vd(d.id); v.caption_ok = v.caption_ok==="y"?"n":v.caption_ok==="n"?"":"y"; save(); render(); }
  else if(k==="h"){ const v=vd(d.id); v.hard=!v.hard; save(); render(); }
  else if(k==="n") nextTodo();
  else if(k==="e") doExport();
  else if(e.key==="ArrowRight") step(1);
  else if(e.key==="ArrowLeft") step(-1);
  else if(k==="?") help.style.display="flex";
});

// ---------------------------------------------------------------- export
function b1000(b){ return "["+b.map(c=>Math.round(c*1000)).join(", ")+"]"; }
function doExport(){
  const rows=[], out={};
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
  const rev=(localStorage.getItem("mocs_reviewer")||"").trim()
        || (prompt("Your name (recorded in the export):","")||"reviewer");
  localStorage.setItem("mocs_reviewer",rev);
  dl("review_results.json", JSON.stringify(
      {reviewer:rev, exported_at:new Date().toISOString(), corpus_key:META.corpus_key,
       n_decided:rows.filter(r=>r.verify_decision).length, verdicts:out}, null, 1));
  if(rows.length){
    const cols=Object.keys(rows[0]);
    const csv=[cols.join(",")].concat(rows.map(r=>cols.map(c=>{
      const s=String(r[c]??""); return /[",\n]/.test(s) ? '"'+s.replace(/"/g,'""')+'"' : s;
    }).join(","))).join("\n");
    dl("review_results.csv","﻿"+csv);
  }
  exportedAt=Object.values(state).filter(v=>v.decision).length; render();
}
function dl(name,text){
  const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([text],{type:"text/plain;charset=utf-8"}));
  a.download=name; a.click(); setTimeout(()=>URL.revokeObjectURL(a.href),1500);
}
bExport.onclick=doExport;
bImport.onclick=()=>fileIn.click();
fileIn.onchange=()=>{ const f=fileIn.files[0]; if(!f) return; const fr=new FileReader();
  fr.onload=()=>{ try{ const j=JSON.parse(fr.result);
      const v=j.verdicts||j;
      // Importing an export from a DIFFERENT package used to "succeed" silently:
      // every id merged into state, none of them in DATA, nothing appeared on screen,
      // and the reviewer had no way to tell. Check, and report what actually landed.
      if(j.corpus_key && j.corpus_key!==META.corpus_key &&
         !confirm("That file came from a DIFFERENT review package.\n\n  file: "+j.corpus_key+
                  "\n  this: "+META.corpus_key+"\n\nImport anyway?")) { fileIn.value=""; return; }
      const ids=new Set(DATA.map(d=>d.id));
      let n=0, skipped=0;
      for(const k in v){ if(ids.has(k)){ state[k]=v[k]; n++; } else skipped++; }
      save(); exportedAt=Object.values(state).filter(x=>x.decision).length; applyFilters();
      alert("Imported "+n+" record(s)."+
            (skipped? "\n"+skipped+" id(s) are not in this package and were ignored." : "")+
            (j.reviewer? "\nReviewer: "+j.reviewer : "")+
            (j.exported_at? "\nExported: "+j.exported_at : ""));
    }catch(e){ alert("Could not read that file: "+e); } };
  fr.readAsText(f); fileIn.value=""; };
window.addEventListener("beforeunload",e=>{
  const pending=Object.values(state).filter(v=>v.decision).length-exportedAt;
  if(pending>0){ e.preventDefault(); e.returnValue=""; }
});

// ---------------------------------------------------------------- boot
(function(){
  const qs=new Set(); DATA.forEach(d=>(d.q||[]).forEach(q=>qs.add(q)));
  const order=["sample","tier1","tier2","tier3","negatives"];
  fQueue.innerHTML='<option value="">all queues</option>'+
    order.filter(q=>qs.has(q)).map(q=>`<option value="${q}">${q}</option>`).join("");
  fQueue.value = qs.has("sample") ? "sample" : "";
  load(); exportedAt=Object.values(state).filter(v=>v.decision).length;
  applyFilters();
  if(!localStorage.getItem("mocs_seen_help")){ help.style.display="flex";
    localStorage.setItem("mocs_seen_help","1"); }
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
