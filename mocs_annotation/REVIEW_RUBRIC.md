# Review rubric — what counts as a violation, per rule

**The decision framework for reviewing MOCS auto-annotations.**

Three sources, in order of authority:

1. **The HuggingFace dataset card** (`LouisChen15/ConstructionSite`) — the only place the four
   rules are written out in full, with their qualifying conditions. Quoted verbatim below.
2. **The paper** (Chen & Zou, *Data-Centric Engineering* 2026, doi:10.1017/dce.2026.10044).
3. **The ground truth itself** — all 1,325 reasons across all 10,013 images
   (6,308 train + 701 val + 3,004 test). This is what the annotator *actually did*, which in
   two places is broader than the card's literal wording.

Your labels become training data for a model scored against this dataset's test split. A
judgement that is correct on a real site but absent from the ground truth teaches a class the
evaluation never rewards, and lands as a false positive. **Label what *this dataset* calls a
violation.**

> **Provenance worth knowing:** every annotation was made by **one** person — a MASc Civil
> Engineering student at UBC. That makes the labels internally consistent but idiosyncratic:
> where the card and the annotator's practice disagree, follow the practice, because that is
> what the test set contains. Captions were ~2,000 hand-written, the rest GPT-4 5-shot then
> human-corrected.

> **Verified against the paper, 2026-10-07.** The paper's Table 4 publishes per-rule counts
> for the test and training sets. All **eight** numbers reproduce exactly from our local copy
> — PPE 323 | 677, harness 25 | 59, edge protection 63 | 109, blind spot 24 | 46 — which also
> confirms that the paper's "training set" (7,009) is our `train` + `val` (6,308 + 701).
> Note Table 4's caption calls these "occurrences"; they are **image counts**. Box counts are
> higher (test rule_2 = 38 boxes across 25 images).

## Evidence base

```
                images with it      reasons          safe images
rule_1              1,000              999
rule_2                 84               84           8,736 of 10,013  (87.2%)
rule_3                172              172
rule_4                 70               70
```

## Calibration — read this before anything else

**The trigger being present does not mean the rule is broken.** Measured on the real ground
truth: given that the thing the rule is about is visibly in the image, the rule is violated
only about **one time in ten**.

| if the image shows… | rule is violated |
|---|---:|
| a person on foot | rule_1 — **13%** (854 of 6,785) |
| a worker **on** a scaffold/roof/formwork/frame/beam | rule_2 — **10%** (50 of 511) |
| an excavation, trench or pit | rule_3 — **6%** (96 of 1,744) |
| a person near an excavator | rule_4 — **7%** (26 of 369) |

So "there's a worker on a scaffold" or "there's an excavator and a person" is the **start** of
the judgement, not the end. Each rule below has the extra conditions that separate the 10%
from the 90% — and the teacher's prompt never mentioned most of them, so it over-flags in
exactly these ways.

**If you are confirming most of what the model proposed, you are running too hot.**

---

## RULE 1 — basic PPE

> **Card:** *"Use of basic PPE when on foot at construction sites (hard hats, properly worn
> clothes covering **shoulders and legs**, **shoes that can cover toes**, high-visibility
> retroreflective vests **at night**, face shield or safety glasses **when cutting, welding,
> grinding, or drilling**)."*

The card lists exactly five PPE items. The ground truth matches it almost perfectly:

| breach | card item | citations | share |
|---|---|---:|---:|
| No hard hat | hard hats | 838 | 83.9% |
| No hi-vis vest — **night only** | retroreflective vests at night | 113 | 11.3% |
| Legs uncovered — shorts, short pants, trousers above the ankle | clothes covering legs | 88 | 8.8% |
| Straw hat or cloth cap instead of a hard hat | hard hats | 53 | 5.3% |
| Shoulders/torso uncovered — sleeveless, bare chest, jacket worn open | clothes covering shoulders | 18 | 1.8% |
| Slippers, sandals, open shoes | **shoes that can cover toes** | 15 | 1.5% |
| No eye/face protection while grinding, cutting, welding | face shield when cutting/welding/grinding/drilling | 7 | 0.7% |

These **overlap and do not sum** — 123 reasons cite two breaches at once (*"wearing shorts
and slippers"*, *"a straw hat and shorts"*). One reason in 999 is generically worded
(*"Person on the left not using PPE."*); one rule_1 in the train split (`0013097`) has a box
and **no reason at all**.

### Does NOT count — zero citations in 999 reasons

- **Short sleeves / bare arms.** The card says *shoulders and legs*, not arms. T-shirts appear
  only as descriptors: *"The worker with a blue **t-shirt** … is wearing **shorts**."*
  "Shoulders uncovered" means sleeveless or bare-chested.
- **No gloves** — not a card item, 0 citations.
- **No mask or respirator** — not a card item, 0 citations.
- **No hi-vis in daylight.** The card says *at night*, and the data agrees: **108** of the 113
  hi-vis citations are on night images, **5** on day. There are only 202 night images in the
  whole corpus, so a night image without hi-vis is a rule_1 roughly **half** the time.
- **No eye protection when not cutting/welding/grinding/drilling.** The task gates it.

### Who counts as "on foot"

- **Counts:** standing, walking, squatting or sitting on the ground or on a structure. Also
  *beside* a vehicle — *"The worker standing next to the cabin of the red dump truck…"* — and
  even *"The worker **riding a bike** on the left…"*
- **Does not count:** a **seated operator or driver inside a cab**. Zero citations mention an
  operator, a driver, or anyone "in the cab" in 999 reasons. If the only bare head is behind
  glass, rule_1 is **no**.

### Reason wording

One sentence, person identified by position or appearance, naming the **specific garment**:

> *"The worker on the right is not wearing a hard hat."*
> *"The person on the left is wearing shorts."*
> *"None of the workers are wearing high-visibility vests when working at night."*

Never "protective clothing" or a bare "PPE" — the ground truth names the item, and
`reward_reasoning` scores similarity against that phrasing.

---

## RULE 2 — safety harness

> **Card:** *"Use of safety harness when working from a height of three meters **and the edges
> are without any edge protection**."*

**Two conditions, both required.** The second one is easy to miss: a worker at height on a
platform that *has* a guardrail is **not** a rule_2 violation, harness or no harness.

### The structure must be static

| structure named | citations |
|---|---:|
| scaffold / scaffolding | 43 |
| generic "structure" / "on top of" | 33 |
| roof / rooftop | 11 |
| metal or steel frame structure | 6 |
| formwork | 4 |
| steel beam | 3 |
| ladder | 1 |
| **machine, bucket, boom, excavator, truck bed** | **0** |

Ten reasons name no structure at all — cases where the caption already established the height.

### Does NOT count

- **A worker in or on machine plant** — excavator bucket, boom, loader arm, truck bed. **Zero
  of 84.** However unsafe it looks, this dataset does not call it rule_2. If the machine is an
  **excavator in operation**, it is **rule_4** instead.
- **A guarded edge.** Scaffold with a handrail, roof with a parapet, platform with a barrier →
  the card's second condition fails.
- **Low work.** The card says *three meters*. The paper lists that threshold as a known
  limitation for VLMs, so don't try to measure it — use the structure test. Scaffold, roof,
  formwork, frame, beam, ladder qualifies. A kerb, a pallet, a soil pile does not.

> **Note:** only **2 of 84** rule_2 images also carry a rule_3 violation, so the annotator did
> *not* require rule_3 to fire alongside. Read the clause as "the working position has no
> guardrail", judged within rule_2 itself.

### Decision test — three gates, all must fail

Being on a structure is **not** the test. 450 images show a worker on a scaffold, roof or
formwork and are **not** rule_2; only 50 are. Run all three gates every time:

> **1. Height.** Could a fall from there seriously injure them — roughly a storey or more?
>    First lift of a scaffold, a low formwork deck, a single-storey roof → **`no`**.
>    (The card says three metres. Don't measure; judge.)
>
> **2. Edge protection.** Any **guardrail, handrail, toe board, parapet, or safety netting at
>    the working edge**? → **`no`**. This is the card's second condition and the one the
>    teacher's prompt never mentions, so it is where the model over-flags hardest.
>
> **3. Harness.** A body harness, lanyard or lifeline visible on the worker? → **`no`**.
>
> **Only when all three fail → `yes`.**

Expect roughly **1 in 10** proposals to survive. Caption wording does not help here — "on top
of" and "upper level" appear just as often in the negatives (27%) as the positives (32%). You
have to look at the picture.

> *"The workers on top of the scaffolding are not wearing safety harnesses."*

---

## RULE 3 — edge protection

> **Card:** *"Adoption of edge protection or edge warning including guardrails, fences, for
> underground projects **three meters in depth** with **steep retaining wall** and for human
> to stand."*

**This rule is about the site condition, not a person.** In all 172 reasons, **zero** mention
a worker. Nobody needs to be near the edge — the unprotected edge alone is the violation.

### Where practice is broader than the card

The card says *underground projects*. The annotator also applied it to **above-ground concrete
platform edges** — 22 citations. Follow the practice: both count.

| feature | citations | share |
|---|---:|---:|
| **excavation / trench / pit / foundation** (underground) | 129 | 75.0% |
| **concrete platform / floor / slab edge** (above ground) | 22 | 12.8% |
| river bank / water / slope / embankment | 10 | 5.8% |
| stairwell / floor opening / shaft | 4 | 2.3% |

The word *edge* appears in **164 of 172** reasons (95%). Rule 3 is about an **edge**.

### What satisfies it (so you mark `no`)

The verb each reason uses, mutually exclusive, all 172:

```
"not guarded"                  86        "not fenced"                6
"not protected"                44        "not warned nor fenced"     4
"not protected nor warned"     31        "not guarded … warned"      1
```

So **any** of a guardrail, fence, barrier, or warning marking/tape along the edge → `no`.

**Shoring is not edge protection.** Vertical sheeting or soldier piles hold the *wall* up;
they are not a guard at the lip. An excavation with sheeted walls and no rail at the top is
still a rule_3.

### Decision test — three gates, all must fail

Only **96 of 1,473** images whose caption names an excavation, trench or pit are rule_3 — 7%.
Digging being present is nowhere near enough.

> **1. A defined, steep edge** — a cut face or lip, not a battered slope. → else **`no`**
> **2. Deep enough that falling in is a real fall** (the card says three metres). → else **`no`**
> **3. Nothing at the top** — no rail, fence, barricade, mesh, tape or cones. → else **`no`**
>
> **All three fail → `yes`**, whether or not anyone is near it.

**The one-line test:** *if I walked across this site looking at my phone, could I fall into a
hole — and is there anything there to stop or warn me?*

### The real `no` categories, from the ground truth

| what you'll see | why it's `no` |
|---|---|
| *"a **sloped** excavation site"* | battered sides — you could walk down them, there is no lip |
| *"a pile of **rocks**"*, *"**rubble**"*, *"a **quarry**"* | material being moved, not a pit with an edge |
| *"a sidewalk and a **barrier**"*, *"blue metal **fences**"* | guarded — the rule is satisfied |
| *"a close-up of an excavator bucket"* | no edge in frame, so nothing to judge |
| *"a large excavation with **wooden shoring** on the sides"* | shoring holds the soil; it is not a guard — but see the trap below |

**The trap: shoring is not guarding.** Sheet piling, soldier piles, timber shoring and the
rebar cage being built all retain the *soil*. None of them stops a person at ground level
walking over the lip. A pit with fully sheeted walls and a bare top edge is **still `yes`** —
what makes the shoring example above a `no` is that no unguarded lip is visible in it.

> *"The right edge of the excavation is not guarded."*
> *"The edge of the concrete platform in the middle is not guarded."*

---

## RULE 4 — blind spot

> **Card:** *"Appearance of worker in the blind spots of the operator **and within the
> operation radius** of **excavators in operation, or excavators with operators inside**."*

Three conditions, all required: **excavator**, **in operation or manned**, **worker in the
blind spot and inside the operation radius**.

### The machine must be an excavator

**70 of 70** reasons name an excavator. Truck, dump truck, loader, crane, roller, bulldozer,
generic "machine": **0 citations each**.

A worker at the rear of a dump truck, beside a loader, or under a crane is a real hazard and
is **not rule_4 in this dataset**. Mark `no`.

### The excavator must be working

Parked, empty, engine-off → **no**, even if someone is standing right beside it. The card
makes this explicit; the reasons assume it (*"the excavator in operation"*, *"the moving
excavator"*, 7 say so outright).

### How the ground truth phrases it

| phrase | citations |
|---|---:|
| "in the blind spot" | 56 |
| "within the operation radius" | 41 |
| "close to" / "too close to" | 22 |
| "at the back of" / "behind" | 6 |
| "in operation" / "moving" | 7 |

The dataset writes **"operation radius"**, not "operating radius".

### Judging distance from a photo

No metre threshold survives a 2D image. Use the machine's own body as the ruler:

> Measure from the excavator's **slew centre** (middle of the house, between the tracks) to
> the person, in **excavator body-lengths**.
>
> - **Under ~2 body-lengths → inside.** Mark yes.
> - **3+ body-lengths → outside.** Mark no.
> - **Check depth.** If the person looks much larger relative to the machine than their real
>   distance would allow, they are far closer to the camera and the true gap is bigger.

**Behind or beside** the machine, where the operator faces the bucket, is the strongest case —
that's the tail-swing zone, and it is what "blind spot" means.

> *"The worker on the left is in the blind spot and operation radius of the excavator."*

---

## Use it vs Discard — the costliest mistake in the whole review

**`Discard` means the PHOTOGRAPH is unusable.** Too blurry to judge, too dark, masked over the
thing that matters, or not a construction scene. It should be **rare — a few percent**.

**`Discard` does NOT mean "the model was wrong."** That is `no` on the rules, then **Use it**.

Two separate costs:

1. **A discarded image produces no training data at all.** It is dropped from `dataset_rows`.
   A photo where the model over-flagged and you corrected it to `no` is a **confirmed-safe
   row** — and with 87% of ConstructionSite being safe images, those are worth as much as a
   violation.
2. **Discarding the model's mistakes silently inflates its measured precision.** Keep the hits
   and discard the misses and the surviving rows say the teacher was right nearly every time.
   The `sample` queue exists to produce an honest precision number; discarding false alarms is
   the single action that destroys it.

> If you can see the scene well enough to answer the four rules, the answer is **Use it** —
> even when all four are `no`, and **especially** when the model was wrong.

## Cross-cutting rules

**"Can't tell" is `no`.** If you have to zoom and squint to decide whether there's a hard hat,
the answer is `no`. The prompt's own test: *answer yes only if you can point at the specific
person, edge or machine at fault.* Over-flagging is the dominant failure mode the dataset was
built to measure — don't add to it.

**One verdict per rule, however many people.** Three workers without hats = one rule_1 `yes`,
one sentence covering all three — and the box count follows the table below.

### How many boxes — the convention differs per rule

Measured on every ground-truth violation in the corpus:

| rule | boxes per violation | distribution | convention |
|---|---|---|---|
| rule_1 | mean **1.66**, up to 13 | 1→662, 2→164, 3→95, 4+→79 | **one box per person.** 312 of 410 multi-person reasons carry multiple boxes |
| rule_2 | mean **1.74**, up to 6 | 1→52, 2→15, 3→9, 4+→8 | **one box per person**, same as rule_1 |
| rule_3 | mean **1.16** | 1→**153**, 2→11, 3→8 | **one box** on the edge or excavation region |
| rule_4 | mean **1.01** | 1→**69**, 2→1 | **always one box** — see below |

**rule_4 takes exactly one box, even when several workers are at risk.** Of all 70 rule_4
violations in the dataset, **69 have a single box**, and that includes the ones whose reason
names two or three people (*"Two workers on the left are in the blind spot and operation
radius of the excavator"* → one box). The GT box covers the **group**, not each individual.
Earlier guidance in this project to add a second rule_4 box per worker was wrong — adding
boxes GT does not have costs you on `reward_violation_grounding`, which is weighted 0.317.

rule_3's box is the **edge or excavation region** and should be generous — GT examples run
40–60% of the frame, e.g. `[[130,20,730,970]]` in `[0,1000]` scale.

**Two rules on one person is normal.** A bare-headed worker beside a working excavator is
rule_1 *and* rule_4, with a separate box on each.

**The base rate is 87% safe.** Most photos break nothing. A run of `no` is expected.

---

## Quick card

Base rate given the trigger is visible: **~1 in 10 for every rule.** If you're confirming most
proposals, you're running too hot.

| | yes when | no when | hit rate |
|---|---|---|---:|
| **1** | **on foot**, and: no hard hat · shorts or trousers above the ankle · straw hat/cap instead of a hard hat · sleeveless or bare chest · slippers/sandals (toes exposed) · no hi-vis **at night** · no eye protection **while cutting/welding/grinding/drilling** | short sleeves · no gloves · no mask · no hi-vis in daylight · the only bare head is a **seated operator** | 13% |
| **2** | **all three** fail: ① a fall would really hurt (≈a storey+) ② **no rail, toe board, parapet or net** at the working edge ③ **no harness/lanyard visible** — on a scaffold, roof, formwork, frame, beam or ladder | any one gate passes · on a **bucket, boom, loader or truck bed** · low scaffold or single-storey roof · standing on soil, a kerb or a pallet | 10% |
| **3** | **all three** fail: ① a defined **steep** edge ② deep enough to be a real fall ③ **nothing** at the top — excavation/trench/pit or raised slab/platform edge. **No person required** | battered/sloped sides · rock pile, rubble or quarry · a rail, fence or tape is visible · no edge in frame · (shoring alone does **not** make it `yes` — look for the bare lip) | 7% |
| **4** | person within ~2 body-lengths of an **excavator's** slew centre, **excavator in operation or manned**, person behind/beside it | the machine is a **truck, loader, crane, roller or bulldozer** · the excavator is **parked and empty** · 3+ body-lengths away · person much closer to the camera than the machine | 7% |

**Boxes:** rule_1 and rule_2 → **one per person**. rule_3 → **one** generous box on the edge.
rule_4 → **exactly one**, covering the group even if several workers are at risk.
