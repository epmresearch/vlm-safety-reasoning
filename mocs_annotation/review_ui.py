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
  * THE ACTION BUTTONS ARE PINNED, NEVER SCROLLED. The right panel is a scrolling
    body plus a fixed footer. When every section lived in one scrolling column the
    Use it / Discard buttons fell below the fold on a 1366x768 laptop -- so the app
    read as though it had no decision buttons at all, which is exactly how it was
    reported.
  * THE UI STATES WHAT IT IS RECORDING, rather than explaining it in a help page.
    Every rule shows, in words, the fact your yes/no just asserted ("false alarm --
    model was wrong"), and the pinned bar names what is still unanswered. Prose in
    a help overlay is read once; a chip under your thumb is read every image.
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
  /* Scrolling body + PINNED footer -- see the module docstring. The decision buttons
     must be on screen at every window height; they are the point of the app. */
  .right{width:390px;flex:0 0 390px;min-height:0;background:var(--panel);
         border-left:1px solid var(--line);display:flex;flex-direction:column}
  .rscroll{flex:1 1 auto;min-height:0;overflow-y:auto;padding:10px}
  .rfoot{flex:0 0 auto;background:var(--panel2);border-top:2px solid var(--line);
         padding:8px 10px}
  @media (max-width:1250px){ .right{width:340px;flex:0 0 340px} }
  .sec{margin-bottom:10px;padding-bottom:9px;border-bottom:1px solid var(--line)}
  .sec:last-child{border:0;margin-bottom:0}
  h3{margin:0 0 5px;font-size:10px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim)}
  .step{background:var(--accent);color:#fff;border-radius:4px;padding:1px 5px;
        letter-spacing:0;margin-right:6px;font-size:10px}
  .row{display:flex;gap:5px;align-items:center;flex-wrap:wrap}
  .chip{display:inline-block;padding:1px 6px;border-radius:99px;font-size:11px;
        background:var(--panel2);border:1px solid var(--line);color:var(--dim)}
  /* The "what you just recorded" chips. Colour carries the meaning at a glance,
     the words carry it unambiguously. */
  .st{display:inline-block;padding:1px 7px;border-radius:99px;font-size:11px;
      border:1px solid var(--line);background:var(--panel2);color:var(--dim)}
  .st.todo{color:#ffd28a;border-color:var(--maybe);background:#3a2a00;font-weight:600}
  .st.good{color:#8ef0b0;border-color:var(--ok);background:#123d22}
  .st.bad {color:#ffb3b3;border-color:var(--no);background:#3d1212}
  .st.add {color:#cfe0ff;border-color:var(--accent);background:#132844}
  .rule{border:1px solid var(--line);border-radius:8px;padding:6px 7px;margin-bottom:5px;
        background:var(--panel2)}
  .rule.prop{border-left:4px solid var(--line)}
  .rule[data-r="rule_1"].prop{border-left-color:var(--r1)}
  .rule[data-r="rule_2"].prop{border-left-color:var(--r2)}
  .rule[data-r="rule_3"].prop{border-left-color:var(--r3)}
  .rule[data-r="rule_4"].prop{border-left-color:var(--r4)}
  .rule.unset{border-color:var(--maybe)}
  .rule .hd{display:flex;align-items:center;gap:6px;margin-bottom:5px}
  .rule .nm{font-weight:600}
  .reason{color:var(--fg);font-size:13px;margin:4px 0}
  .muted{color:var(--dim);font-size:12px}
  .y{border-color:var(--ok)} .y.on{background:var(--ok);border-color:var(--ok);color:#06210f}
  .n{border-color:var(--no)} .n.on{background:var(--no);border-color:var(--no);color:#2a0606}
  .m{border-color:var(--maybe)} .m.on{background:var(--maybe);border-color:var(--maybe);color:#2a1a00}
  textarea{width:100%;min-height:46px;resize:vertical}
  .big{font-size:14px;padding:6px 11px;font-weight:600}
  .caption{background:var(--panel2);border:1px solid var(--line);border-radius:8px;
           padding:7px;font-size:12.5px;max-height:120px;overflow-y:auto}
  #help,#gate{position:fixed;inset:0;background:rgba(0,0,0,.82);display:none;z-index:9;
        align-items:center;justify-content:center}
  #gate{background:rgba(10,11,14,.97);z-index:10}
  #help>div,#gate>div{background:var(--panel);border:1px solid var(--line);border-radius:12px;
            padding:22px;max-width:680px;max-height:86vh;overflow:auto}
  kbd{background:var(--panel2);border:1px solid var(--line);border-radius:4px;
      padding:1px 6px;font:12px ui-monospace,monospace}
  table{border-collapse:collapse;width:100%} td{padding:3px 6px;vertical-align:top}
  th{padding:3px 6px;text-align:left;font-size:11px;text-transform:uppercase;
     letter-spacing:.06em;color:var(--dim);border-bottom:1px solid var(--line)}
  .sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;
      vertical-align:-1px}
</style>
</head>
<body>

<div class="bar">
  <select id="fQueue" title="Which batch of photos to work through"></select>
  <select id="fRule" title="Show only photos where the model flagged this rule">
    <option value="">all rules</option>
    <option value="rule_1">model flagged rule_1</option>
    <option value="rule_2">model flagged rule_2</option>
    <option value="rule_3">model flagged rule_3</option>
    <option value="rule_4">model flagged rule_4</option>
  </select>
  <select id="fState">
    <option value="">all</option><option value="todo">not done yet</option>
    <option value="done">done</option>
  </select>
  <select id="fSource">
    <option value="">val+test</option><option value="val">val</option><option value="test">test</option>
  </select>
  <input id="fText" placeholder="search" style="width:100px">
  <span class="grow"></span>
  <span id="prog" class="muted"></span>
  <span id="smode" class="chip"></span>
  <button id="bSaveFile" class="big" title="Start a new results file">Save to file&hellip;</button>
  <button id="bOpenFile" class="big" title="Carry on from a file you saved earlier">Open&hellip;</button>
  <button id="bHelp" class="big" title="How to use this (or press ?)">? Help</button>
</div>

<div class="wrap">
  <div class="left">
    <div class="canvasbox"><canvas id="cv"></canvas></div>
    <div class="bar" style="border-top:1px solid var(--line);border-bottom:0">
      <span class="muted">show boxes</span>
      <button class="lay on" data-r="rule_1" title="show / hide rule_1 boxes"><span class="sw" style="background:var(--r1)"></span>1</button>
      <button class="lay on" data-r="rule_2" title="show / hide rule_2 boxes"><span class="sw" style="background:var(--r2)"></span>2</button>
      <button class="lay on" data-r="rule_3" title="show / hide rule_3 boxes"><span class="sw" style="background:var(--r3)"></span>3</button>
      <button class="lay on" data-r="rule_4" title="show / hide rule_4 boxes"><span class="sw" style="background:var(--r4)"></span>4</button>
      <button class="lay on" data-r="mocs" title="show / hide the human-drawn MOCS box"><span class="sw" style="background:var(--mocs)"></span>MOCS</button>
      <span class="grow"></span>
      <span class="muted">draw a better box for</span>
      <select id="drawRule" title="Pick a rule, then drag a box on the photo">
        <option value="">(off)</option>
        <option value="rule_1">rule_1</option><option value="rule_2">rule_2</option>
        <option value="rule_3">rule_3</option><option value="rule_4">rule_4</option>
      </select>
      <button id="bClearBoxes" title="Remove every box you drew on this photo">clear mine</button>
    </div>
  </div>

  <div class="right">
   <div class="rscroll">
    <div class="sec">
      <div class="row" style="justify-content:space-between">
        <strong id="rid">-</strong>
        <span><span class="chip" id="rsrc"></span> <span class="chip" id="rrun"></span></span>
      </div>
      <div class="muted" id="rq" style="margin-top:4px"></div>
    </div>

    <div class="sec" id="secCap">
      <h3><span class="step">Step 1</span>Does the caption match this photo? <kbd>C</kbd></h3>
      <div class="caption" id="cap"></div>
      <div class="row" style="margin-top:6px">
        <button class="y" data-cap="y" title="The caption describes this photo correctly">yes</button>
        <button class="n" data-cap="n" title="The caption is wrong, or describes a different scene">no</button>
        <span id="capSt" class="st"></span>
      </div>
    </div>

    <div class="sec">
      <h3><span class="step">Step 2</span>Is each rule broken in THIS photo? <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd><kbd>4</kbd></h3>
      <div class="muted" style="margin:-2px 0 7px">
        Answer for <strong>all four</strong>, on every photo, by looking at the photo.
        <strong>yes</strong> = it really is broken here; <strong>no</strong> = it is not.
        This is <em>not</em> &ldquo;do I agree with the model&rdquo; &mdash; the coloured line
        under each pair of buttons tells you exactly what you just recorded.</div>
      <div id="rules"></div>
    </div>

    <div class="sec">
      <h3>Extra information</h3>
      <div class="muted" id="cats"></div>
      <div class="muted" id="r4hint" style="margin-top:4px"></div>
      <textarea id="notes" placeholder="notes for me (optional)" style="margin-top:7px"></textarea>
    </div>
   </div>

   <div class="rfoot">
      <div class="row" style="margin-bottom:6px">
        <span class="step">Step 3</span><span id="todoChip" class="st"></span>
      </div>
      <div class="row">
        <button class="dec big y" data-dec="accept"
                title="My answers above are right - keep this photo and my answers">Use it <kbd>A</kbd></button>
        <button class="dec big n" data-dec="reject"
                title="Nobody could judge this photo - blurry, too dark, or not a construction scene">Discard <kbd>R</kbd></button>
        <button class="dec big m" data-dec="unsure"
                title="Leave it and come back later">Unsure <kbd>U</kbd></button>
        <span class="grow"></span>
        <button id="bHard" class="m" title="Mark as difficult, for a second opinion">hard <kbd>H</kbd></button>
      </div>
      <div class="muted" style="margin-top:5px">
        <strong>Use it</strong> = my answers are right &mdash; press it <em>even when every
        rule is &ldquo;no&rdquo;</em>. <strong>Discard</strong> = the photo itself is unusable
        (rare). Either one saves and jumps to the next photo.</div>
      <div class="row" style="margin-top:7px">
        <button id="bPrev" title="Previous photo">&larr; prev</button>
        <button id="bNext" title="Next photo">next &rarr;</button>
        <button id="bNextTodo" title="Skip to the next photo you have not done">next not-done <kbd>N</kbd></button>
        <span class="grow"></span>
        <input id="jump" style="width:80px" placeholder="go to #" title="Type a position number, or an image id">
      </div>
   </div>
  </div>
</div>

<div id="gate"><div>
  <h2 style="margin-top:0">First: choose where your work is saved</h2>
  <p>Everything you do is written straight to a file on your own computer, as you go.
     Nothing is kept in the browser, so nothing can be lost by closing a tab or clearing
     browsing data &mdash; but you have to pick that file before you start.</p>
  <div class="row" style="margin:18px 0">
    <button id="gSave" class="big">Save to new file&hellip;</button>
    <button id="gOpen" class="big">Open saved file&hellip;</button>
  </div>
  <p class="muted"><strong>First time?</strong> Press <em>Save to new file&hellip;</em> and
     save it as <code>review_results.json</code> somewhere you will remember, such as your
     Documents folder. The instructions open straight afterwards.<br>
     <strong>Coming back?</strong> Press <em>Open saved file&hellip;</em> and pick that same
     file &mdash; your work reappears and keeps saving to it.</p>
  <p id="gateWarn" style="color:#ff8a8a"></p>
</div></div>

<div id="help"><div>
  <h2 style="margin-top:0">How to review &mdash; everything, on one page</h2>
  <p>A computer model looked at each photo and <strong>guessed</strong> which safety rules it
     breaks. Those guesses are often wrong. You decide what is actually true, and
     <strong>your answers are what become the dataset</strong>.</p>
  <p class="muted">Three steps per photo, then it moves on by itself. About 5&ndash;8 seconds
     each once you get into a rhythm.</p>

  <h3>Step 1 &mdash; the caption <kbd>C</kbd></h3>
  <p>One sentence describing the photo. <strong>yes</strong> if it matches what you see,
     <strong>no</strong> if it is wrong or describes a different scene.</p>

  <h3>Step 2 &mdash; the four rules <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd><kbd>4</kbd></h3>
  <p>For each rule: <strong>is it broken in this photo, yes or no?</strong> Look at the photo
     and answer. It is <em>not</em> a vote on whether the model was right &mdash; though your
     answer does decide that, and the app says so back to you:</p>
  <table>
    <tr><th>the model</th><th>what you see</th><th>press</th><th>the app records</th></tr>
    <tr><td>flagged rule&nbsp;1</td><td>there really is a PPE breach</td><td><strong>yes</strong></td><td><span class="st good">confirmed &mdash; model was right</span></td></tr>
    <tr><td>flagged rule&nbsp;1</td><td>there is no breach</td><td><strong>no</strong></td><td><span class="st bad">false alarm &mdash; model was wrong</span></td></tr>
    <tr><td>said nothing about rule&nbsp;2</td><td>there <em>is</em> a harness breach</td><td><strong>yes</strong></td><td><span class="st add">you added it &mdash; model missed it</span></td></tr>
    <tr><td>said nothing about rule&nbsp;2</td><td>no harness breach</td><td><strong>no</strong></td><td><span class="st good">agreed &mdash; not broken</span></td></tr>
  </table>
  <p class="muted">A typical photo ends up rule_1 <em>yes</em>, rules 2&ndash;4 <em>no</em>.
     Four answers every time &mdash; a rule left blank is recorded as <em>not broken</em>,
     so skipping one on a photo that really does show it puts a mistake into the data.</p>

  <h3>The four rules, in full</h3>
  <table>
    <tr><td><span class="sw" style="background:var(--r1)"></span><strong>rule_1</strong></td>
        <td>a person <em>on foot</em> is missing basic PPE &mdash; no hard hat, or clothing
            that leaves the shoulders or legs uncovered</td></tr>
    <tr><td><span class="sw" style="background:var(--r2)"></span><strong>rule_2</strong></td>
        <td>a person <em>working at height</em> (scaffold, roof, beam, ladder) is not wearing
            a safety harness</td></tr>
    <tr><td><span class="sw" style="background:var(--r3)"></span><strong>rule_3</strong></td>
        <td>an open excavation, trench, pit or floor edge has no guard rail, barrier or
            warning marking</td></tr>
    <tr><td><span class="sw" style="background:var(--r4)"></span><strong>rule_4</strong></td>
        <td>a person is standing inside the operating radius / blind spot of an excavator or
            other heavy machine</td></tr>
  </table>
  <p><strong>The test for all four:</strong> answer <em>yes</em> only if you can point at the
     specific person, edge or machine at fault. If you cannot see who or what is to blame,
     the answer is <em>no</em>.</p>

  <h3>Step 3 &mdash; use it or discard it <kbd>A</kbd> <kbd>R</kbd> <kbd>U</kbd></h3>
  <table>
    <tr><td style="white-space:nowrap"><strong>Use it</strong> <kbd>A</kbd></td>
        <td>&ldquo;my answers above are correct&rdquo;. This is the normal ending for almost
            every photo &mdash; press it <strong>even when all four rules are
            &ldquo;no&rdquo;</strong>, because a confirmed-safe photo is as useful to us as a
            violation.</td></tr>
    <tr><td style="white-space:nowrap"><strong>Discard</strong> <kbd>R</kbd></td>
        <td>the <em>photograph</em> is unusable: too blurry, too dark, or not a construction
            scene at all. <strong>Not</strong> for &ldquo;the model was wrong&rdquo; &mdash;
            that is simply <em>no</em> on the rules, then <em>Use it</em>. You should rarely
            need this.</td></tr>
    <tr><td style="white-space:nowrap"><strong>Unsure</strong> <kbd>U</kbd></td>
        <td>leave it and come back later.</td></tr>
    <tr><td style="white-space:nowrap"><strong>hard</strong> <kbd>H</kbd></td>
        <td>use freely, on top of any of the above: &ldquo;this one was genuinely
            ambiguous&rdquo;. Knowing which were hard is useful &mdash; much better than
            agonising over it.</td></tr>
  </table>
  <p class="muted">The orange bar just above those buttons always names what is still
     unanswered, and the app stops you pressing <em>Use it</em> while anything is blank.</p>

  <h3>Boxes</h3>
  <table>
    <tr><td style="white-space:nowrap"><span class="sw" style="background:var(--mocs)"></span><strong>green &ldquo;MOCS&rdquo;</strong></td>
        <td><strong>drawn by a human</strong>, from the source dataset. Trust it over the
            model's.</td></tr>
    <tr><td>solid colour</td><td>the model's guess &mdash; often on the wrong thing</td></tr>
    <tr><td>dashed</td><td>a box <strong>you</strong> drew. Click it to delete it.</td></tr>
  </table>
  <p>If a rule really is broken but the model's box is on the wrong object, pick that rule in
     <em>&ldquo;draw a better box for&rdquo;</em> under the photo and drag a new box. Drawing a
     box also sets that rule to <em>yes</em>.</p>

  <h3>Reasons</h3>
  <p>If you turn <strong>on</strong> a rule the model did not propose, a reason box appears in
     that card. Please write one sentence naming <em>who or what</em> is at fault and
     <em>what</em> the breach is &mdash; e.g. &ldquo;The worker on the left is on foot without a
     hard hat.&rdquo; For a rule the model <em>did</em> propose, leave it blank unless its
     sentence is wrong.</p>

  <h3>Keyboard</h3>
  <table>
    <tr><td><kbd>1</kbd><kbd>2</kbd><kbd>3</kbd><kbd>4</kbd></td><td>that rule: yes &rarr; no &rarr; blank</td></tr>
    <tr><td><kbd>C</kbd></td><td>caption: yes &rarr; no &rarr; blank</td></tr>
    <tr><td><kbd>A</kbd></td><td>Use it</td></tr>
    <tr><td><kbd>R</kbd></td><td>Discard</td></tr>
    <tr><td><kbd>U</kbd></td><td>Unsure</td></tr>
    <tr><td><kbd>H</kbd></td><td>flag as hard</td></tr>
    <tr><td><kbd>&larr;</kbd> <kbd>&rarr;</kbd></td><td>previous / next photo</td></tr>
    <tr><td><kbd>N</kbd></td><td>next photo you have not done</td></tr>
    <tr><td><kbd>?</kbd></td><td>open this page &mdash; <kbd>Esc</kbd> closes it</td></tr>
  </table>

  <h3>Saving, and sending your work back</h3>
  <p>Your work goes <strong>straight into a file on your own computer</strong>, automatically,
     every time you change anything. Nothing is stored in the browser, so closing the tab or
     clearing browsing data cannot lose it.</p>
  <p><strong>Starting out:</strong> press <em>Save to file&hellip;</em> and save it as
     <code>review_results.json</code> somewhere you will remember.<br>
     <strong>Coming back:</strong> press <em>Open&hellip;</em> and pick that same file. Your
     work reappears and keeps saving to it.</p>
  <p>The chip in the toolbar shows the filename and the time of the last save. If it ever
     turns red and says
     <span class="chip" style="background:#4d1414;border-color:#ff5252;color:#ffb3b3">NOT SAVING</span>,
     stop and choose the file again.</p>
  <p class="muted">When you finish a session, email that <code>.json</code> file back. There is
     nothing to export &mdash; it is already up to date. Order of work: <strong>sample</strong>
     first (150 photos; it tells us how accurate the model is), then <strong>tier1</strong>.</p>
  <button class="big" onclick="document.getElementById('help').style.display='none'">Close &mdash; start reviewing</button>
</div></div>

<script>
const DATA = __DATA__;
const META = __META__;
const RULES = ["rule_1","rule_2","rule_3","rule_4"];
const COL = {rule_1:"#00E5FF",rule_2:"#FFEA00",rule_3:"#FF3D00",rule_4:"#D500F9",mocs:"#00E676"};

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

// What is still unanswered on this image. ONE function, used by the pinned bar and
// by the Use-it guard, so the warning can never disagree with what the bar showed.
function missingOn(v){
  const m = RULES.filter(r=>!v.rules[r]);
  if(!v.caption_ok) m.push("caption");
  return m;
}

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

// The four things a yes/no can MEAN, spelled out. This is the standing answer to
// "what does yes mean when the model did not propose the rule?" -- printed under the
// buttons, every image, instead of buried in a help page nobody re-reads.
function ruleStatus(proposed, val){
  if(!val)     return ["todo", "not judged yet"];
  if(proposed) return val==="y" ? ["good","confirmed — model was right"]
                                : ["bad", "false alarm — model was wrong"];
  return         val==="y" ? ["add", "you added it — model missed it"]
                           : ["good","agreed — not broken"];
}
function render(){
  const d=cur();
  prog.textContent = view.length
    ? `${pos+1} / ${view.length}   ·   ${Object.values(state).filter(v=>v.decision).length} done of ${DATA.length}`
    : "nothing matches these filters";
  if(!d){ ctx.clearRect(0,0,cv.width,cv.height); rid.textContent="-"; rules.innerHTML="";
          todoChip.className="st"; todoChip.textContent=""; return; }
  const v=vd(d.id);
  rid.textContent=d.id; rsrc.textContent=d.src; rrun.textContent=d.run;
  rq.textContent="queues: "+(d.q||[]).join(", ");
  cap.textContent=d.cap||"(no caption)";
  cats.textContent="MOCS categories: "+((d.cats||[]).join(", ")||"none (the test split carries no MOCS annotations)");
  r4hint.textContent=(d.r4&&d.r4.length)
      ? "green box = human-drawn worker+machine region — prefer it for rule_4"
      : "no human-drawn box for this photo";
  document.querySelectorAll("[data-cap]").forEach(b=>b.classList.toggle("on",v.caption_ok===b.dataset.cap));
  const cs = !v.caption_ok      ? ["todo","not judged yet"]
           : v.caption_ok==="y" ? ["good","caption is correct"]
                                : ["bad", "caption is wrong"];
  capSt.className="st "+cs[0]; capSt.textContent=cs[1];
  document.querySelectorAll(".dec").forEach(b=>b.classList.toggle("on",v.decision===b.dataset.dec));
  bHard.classList.toggle("on",!!v.hard);
  notes.value=v.notes||"";

  rules.innerHTML = RULES.map(r=>{
    const p=!!(d.r[r]&&d.r[r].p), val=v.rules[r]||"";
    const mine=(v.boxes[r]||[]).length;
    const st=ruleStatus(p,val);
    return `<div class="rule ${p?'prop':''} ${val?'':'unset'}" data-r="${r}">
      <div class="hd"><span class="sw" style="background:${COL[r]}"></span>
        <span class="nm">${r}</span>
        <span class="chip">${p?'model flagged it':'model said nothing'}</span>
        <span class="grow" style="flex:1"></span>
        <button class="y rv ${val==='y'?'on':''}" data-r="${r}" data-v="y"
                title="${r} IS broken in this photo">yes</button>
        <button class="n rv ${val==='n'?'on':''}" data-r="${r}" data-v="n"
                title="${r} is NOT broken in this photo">no</button>
      </div>
      <div><span class="st ${st[0]}">${st[1]}</span></div>
      ${p?`<div class="reason">${esc(d.r[r].reason||"")}</div>
           <div class="muted">${(d.r[r].boxes||[]).length} model box(es)${mine?` · ${mine} of yours`:""}</div>`
         :`<div class="muted" style="margin-top:4px">${mine?`${mine} box(es) you drew`:"the model did not mention this rule"}</div>`}
      ${val==="y"?`<input class="rsn" data-r="${r}" value="${esc(v.reasons[r]||"")}"
          style="width:100%;margin-top:6px" placeholder="${p
            ? "reason wrong? write a better one (optional)"
            : "REASON PLEASE — one sentence: who/what is at fault, and what the breach is"}">`:""}
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

  // The pinned bar always names what is left, in the same words the guard will use.
  const miss=missingOn(v);
  if(miss.length){
    todoChip.className="st todo";
    todoChip.textContent="still to judge: "+miss.join(", ");
  } else {
    const needR=RULES.filter(r=>v.rules[r]==="y" && !(d.r[r]&&d.r[r].p)
                                && !(v.reasons[r]||"").trim());
    if(needR.length){ todoChip.className="st add";
                      todoChip.textContent="all judged · please add a reason for "+needR.join(", "); }
    else { todoChip.className="st good";
           todoChip.textContent="all 4 rules + caption judged — ready"; }
  }

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
  // A "Use it" with anything left unset is the one silently damaging outcome: a rule
  // nobody looked at is recorded as "not broken" (a false negative in the data), and
  // an unjudged caption cannot be used as a training target at all. Discard needs
  // neither, and if the reviewer watches the orange bar this never fires.
  if(x==="accept" && v.decision!==x){
    const miss = missingOn(v);
    if(miss.length && !confirm(
        "Not judged yet: "+miss.join(", ")+".\n\n"+
        "An unjudged rule is recorded as NOT broken, and an unjudged caption cannot "+
        "be used at all. Press Cancel and answer them first — keys 1-4 for the rules, "+
        "C for the caption.\n\n"+
        "Use it anyway?")) return;
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
  if(e.key==="Escape"){ help.style.display="none"; return; }
  if(help.style.display==="flex") return;   // reading the instructions is not reviewing
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
// "Has this record been touched at all?" -- one predicate, so the saved file and the
// row builder can never disagree about what counts as work worth keeping. caption_ok
// and hard are included: judging only the caption, or only flagging a photo as hard,
// is still work, and the earlier version silently dropped both.
function touched(v){
  return !!(v && (v.decision || v.notes || v.hard || v.caption_ok
                  || Object.keys(v.rules||{}).length
                  || Object.keys(v.boxes||{}).length
                  || Object.keys(v.reasons||{}).length));
}
function payloadJSON(){
  const out={};
  for(const d of DATA){ const v=state[d.id]; if(touched(v)) out[d.id]=v; }
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
    const v=state[d.id]; if(!touched(v)) continue;
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
  const firstTime = !fileHandle;
  try{
    askName();
    fileHandle = await window.showSaveFilePicker(
        {suggestedName:"review_results.json", types:PICK_TYPES});
    await writeFile();
    gate.style.display="none";
    render();
    // Choosing a brand-new file means a brand-new reviewer, almost always. Show the
    // instructions once, unprompted, rather than hoping the "?" button gets pressed.
    if(firstTime) help.style.display="flex";
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
