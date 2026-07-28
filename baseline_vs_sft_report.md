# Baseline vs. SFT Evaluation Report
### Construction Site 10k - VLM Fine-Tuning Analysis

---

## 1. Scope and Methodology Note

This report compares the **untuned 2B baseline VLM (unsloth/Qwen3-VL-2B-Instruct)** against the **SFT (supervised fine-tuned) checkpoint of the same model**, both evaluated at **repetition penalty = 1.0**.

All quality metrics below are computed **after** a structural JSON repair pipeline is applied to raw baseline model outputs. Raw (pre-repair) validity is reported separately in Section 2, since it is the more honest measure of what the model natively produces.

Test set: 3,004 records.

---

## 2. Structural Output Validity

### 2.1 Raw output validity (before repair)

| | Baseline | SFT |
|---|---|---|
| Valid JSON+schema, no repair needed | 15.28% (459 / 3004) | 98.04% (2945 / 3004) |
| Required repair, successfully fixed | 84.05% (2525 / 3004) | 1.96% (59 / 3004) |
| Unrepairable (invalid schema) | 0.67% (20 / 3004) | 0.00% (0 / 3004) |

The baseline produces a well-formed, schema-compliant response **natively** only 15.3% of the time — the other ~84% require a repair script to become usable. SFT is natively valid 98.0% of the time, a **~6.4x improvement** in raw format compliance, and leaves zero unrepairable outputs (vs. 20 for baseline).

### 2.2 Post-repair validity (used for all downstream metrics)

| | Baseline | SFT |
|---|---|---|
| JSON validity rate | 99.33% (2984/3004) | 100.00% (3004/3004) |
| Schema adherence rate | 99.33% (2984/3004) | 100.00% (3004/3004) |

After repair, both models look nearly identical (99.3% vs. 100%). **This is the key reason raw validity (2.1) — not post-repair validity — is the metric that should be foregrounded in the paper**: post-repair numbers mask a real and large difference in native output reliability.

---

## 3. Captioning Quality (GT average = 48.5 words)

| Metric | Baseline | SFT | Δ (abs) | Δ (relative) |
|---|---|---|---|---|
| BERTScore Precision | 0.2778 | 0.3500 | +0.0722 | +25.9% |
| BERTScore Recall | 0.4010 | 0.4561 | +0.0551 | +13.7% |
| BERTScore F1 | 0.3382 | 0.4019 | +0.0637 | **+18.8%** |
| METEOR | 0.1886 | 0.2045 | +0.0159 | +8.5% |
| CIDEr-D | 0.0778 | 0.1478 | +0.0700 | **+90.0%** |
| CLIPScore | 0.7781 | 0.7732 | −0.0049 | −0.6% (flat) |
| Avg. words/caption | 73.33 | 59.58 | −13.75 | length reduced |
| Min / Max words | 1 / 216 | 26 / 115 | — | far tighter spread |

**Reading:** SFT improves every content-overlap metric (BERTScore F1, METEOR, CIDEr-D), most sharply CIDEr-D (~2x), which is the metric most sensitive to matching the exact phrasing/n-grams. CLIPScore (image-text alignment, not a reference-comparison metric) is essentially flat — both models describe the *correct scene content* similarly well; the gap is in how closely the *phrasing* matches the reference style.

Neither model matches the ground-truth caption length of 48.5 words: baseline over-generates by +51.2% (73.3 vs 48.5), SFT over-generates by +22.8% (59.6 vs 48.5). SFT is closer but still verbose relative to GT. The baseline's extreme max (216 words) and min (1 word) indicate output instability that SFT largely eliminates (max 115, min 26).

---

## 4. Grounding / Visual Localization

Grounding is evaluated per object class (Excavator, Rebar, White Hard Hat) on two axes: **presence detection** (did the model correctly say the object is/isn't there) and **mask IoU when the object exists** (how accurate is the localization given detection is attempted).

### 4.1 Presence detection (precision / recall / F1) — per class

| Class | Metric | Baseline | SFT | Δ (relative, F1) |
|---|---|---|---|---|
| **Excavator** (GT=1080) | Precision | 0.8057 | 0.9711 | |
| | Recall | 0.5528 | 0.9657 | |
| | **F1** | **0.6557** | **0.9684** | **+47.7%** |
| **Rebar** (GT=327) | Precision | 0.2650 | 0.7452 | |
| | Recall | 0.3242 | 0.7156 | |
| | **F1** | **0.2916** | **0.7301** | **+150.4%** |
| **White Hard Hat** (GT=314) | Precision | 0.1385 | 0.8667 | |
| | Recall | 0.3217 | 0.8280 | |
| | **F1** | **0.1937** | **0.8469** | **+337.2%** |

Confusion counts for context (TP / FP / FN out of existing GT):

| Class | Baseline TP/FP/FN | SFT TP/FP/FN |
|---|---|---|
| Excavator (GT 1080) | 597 / 144 / 483 | 1043 / 31 / 37 |
| Rebar (GT 327) | 106 / 294 / 221 | 234 / 80 / 93 |
| White Hard Hat (GT 314) | 101 / 628 / 213 | 260 / 40 / 54 |

**Reading:** The baseline's biggest failure mode is **false positives on rare classes** — it over-predicts white-hard-hat presence (628 FPs against only 314 true instances), driving precision down to 0.14. SFT nearly eliminates this (40 FPs), while also cutting false negatives roughly in half or better across all three classes. The improvement is **largest for the two minority/imbalanced classes** (rebar: 846 train examples; hard hat: 680 train examples) — exactly where a base model with no domain adaptation would be expected to struggle most.

### 4.2 Aggregate presence detection

| Metric | Baseline | SFT | Δ (relative) |
|---|---|---|---|
| Precision (micro) | 0.4299 | 0.9105 | +111.8% |
| Recall (micro) | 0.4672 | 0.8931 | +91.2% |
| F1 (micro) | 0.4478 | 0.9017 | **+101.4%** |
| F1 (macro) | 0.3803 | 0.8485 | **+123.1%** |

SFT roughly **doubles** presence-detection F1 in aggregate.

### 4.3 Mask IoU quality (when object exists) — per class, micro-averaged

| Class | Baseline | SFT | Δ (relative) |
|---|---|---|---|
| Excavator | 0.4278 | 0.8744 | +104.4% |
| Rebar | 0.1057 | 0.4464 | **+322.4%** |
| White Hard Hat | 0.1215 | 0.6379 | **+425.1%** |
| **Aggregate (micro mean)** | **0.3355** | **0.7732** | **+130.5%** |
| Aggregate (macro mean) | 0.1815 | 0.5759 | +217.3% |

**Reading:** This is the sharpest signal. Not only does SFT detect objects far more reliably (Section 4.1), when it does localize an object the mask quality is dramatically better — more than doubled for excavator, more than quadrupled for rebar and hard hat. This indicates fine-tuning taught the model genuine spatial grounding for this domain, not just a bias toward saying "yes."

**Caveat:** Rebar (mask IoU 0.446) and White Hard Hat (0.638) remain well below Excavator (0.874) even after fine-tuning. These are the two rarer classes in training (846 and 680 examples vs. 2,415 for excavator) — a residual capability gap likely tied to data volume, and a natural target for the RL phase.

---

## 5. Violation Identification

### 5.1 Aggregate identification (has a violation of this rule occurred — yes/no)

| Metric | Baseline | SFT | Δ (relative) |
|---|---|---|---|
| Precision (micro) | 0.1292 | 0.7790 | **+502.7%** |
| Recall (micro) | 0.7356 | 0.3241 | **−55.9%** |
| F1 (micro) | 0.2199 | 0.4578 | **+108.2%** |
| Precision (macro) | 0.0710 | 0.3067 | +332.0% |
| Recall (macro) | 0.2661 | 0.1922 | −27.8% |
| F1 (macro) | 0.0838 | 0.2299 | +174.4% |

**Reading — this is the important nuance.** SFT does **not** uniformly dominate here. The baseline achieves much higher *recall* (0.74 vs 0.32) but at the cost of extremely poor precision (0.13) — it is essentially over-flagging violations indiscriminately. SFT flips this: high precision (0.78), lower recall (0.32). F1 favors SFT because precision improved by a larger margin than recall was sacrificed, but **a system that misses ~68% of true violations is a real limitation**, not just a metric footnote — this is a problem, and is a natural target for reward shaping in the GRPO phase.

### 5.2 Per-rule identification F1

| Rule | Test N | Baseline P/R/F1 | SFT P/R/F1 | Δ F1 (relative) |
|---|---|---|---|---|
| Rule 1 (PPE) | 323 | 0.130 / 0.985 / 0.230 | 0.852 / 0.409 / 0.552 | +140.2% |
| Rule 2 (Harness) | 25 | 0.154 / 0.080 / 0.105 | 0.375 / 0.360 / 0.367 | +248.9% |
| Rule 3 (Edge Protection) | 63 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | both fail |
| Rule 4 (Blind Spot) | 24 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | both fail |
| Rule 0 (no violation) | — | 0.986 / 0.213 / 0.350 | 0.905 / 0.985 / 0.943 | +169.4% |

**Reading:** SFT improves substantially on the two rules with meaningful training support (Rule 1: 677 train examples; Rule 2: 59). **Both models score zero on Rule 3 and Rule 4** — with only 109 and 46 training examples respectively (and 63 / 24 in test), this is a data-scarcity failure common to both, not something fine-tuning fixed. This is worth flagging explicitly as an unresolved gap.

Rule 0 (correctly recognizing "no violation present") improves from F1 0.350 to 0.943 — this single number largely explains the aggregate precision gain in 5.1: the baseline was rarely willing/able to correctly say nothing was wrong (recall 0.213), while SFT does so reliably (recall 0.985).

### 5.3 IoU-conditioned identification (violation correctly identified AND correctly localized, IoU threshold 0.25)

| Metric | Baseline | SFT | Δ (relative) |
|---|---|---|---|
| Precision (micro) | 0.0432 | 0.5746 | +1230% |
| Recall (micro) | 0.2460 | 0.2391 | −2.8% (flat) |
| F1 (micro) | 0.0735 | 0.3377 | **+359.3% (~4.6x)** |
| F1 (macro) | 0.0323 | 0.1546 | +378.5% |

Under the stricter joint criterion (must both identify **and** correctly localize the violation), SFT's advantage is largest of any metric family (~4.6x on F1), driven almost entirely by the precision gain — recall is essentially unchanged between the two models under this stricter test. This reinforces the same story as 5.1: **SFT's gain is concentrated in precision/specificity, not in catching more true violations.**

---

## 6. Reasoning Text Quality (GT average = 13.3 words)

| Metric | Baseline (micro) | SFT (micro) | Δ (relative) |
|---|---|---|---|
| BERTScore Precision | 0.2123 | 0.7988 | +276.3% |
| BERTScore Recall | 0.2065 | 0.7090 | +243.2% |
| BERTScore F1 | 0.2082 | 0.7532 | **+261.8% (~3.6x)** |
| METEOR | 0.1292 | 0.3176 | +145.8% |
| CIDEr-D | 0.0043 | 2.2558 | baseline ≈ 0; SFT substantial |
| CLIPScore | 0.5521 | 0.6261 | +13.4% |
| Avg. words | 23.96 | 10.45 | vs GT 13.3 |

**Length calibration:** baseline overshoots GT by +80.2% (23.96 vs 13.3 words); SFT undershoots by −21.4% (10.45 vs 13.3). In absolute deviation from GT, SFT (2.85 words off) is far closer than baseline (10.66 words off).

**Macro-averaged** (equal weight per rule, rather than per-sample):

| Metric | Baseline | SFT | Δ (relative) |
|---|---|---|---|
| BERTScore F1 | 0.1827 | 0.3768 | +106.4% |
| METEOR | 0.1089 | 0.1574 | +44.6% |
| CIDEr-D | 0.0070 | 0.9945 | baseline ≈ 0; SFT substantial |
| CLIPScore | 0.3415 | 0.3285 | −3.8% (flat) |

**Per rule (only rules with valid identifications — Rule 3/4 are 0.0 for both, consistent with Section 5.2's finding that neither model correctly identifies any Rule 3/4 violation, so no reasoning text is scored against them):**

| Rule | Baseline BERTScore F1 | SFT BERTScore F1 | Δ (relative) |
|---|---|---|---|
| Rule 1 | 0.2062 | 0.7532 | +265.3% |
| Rule 2 | 0.5244 | 0.7542 | +43.8% |

**Reading:** SFT's reasoning-text quality gain is large on a per-sample (micro) basis — nearly 4x BERTScore F1, with CIDEr-D moving from essentially zero to a substantial positive value, indicating the baseline's free-text justifications share almost no n-gram overlap with the reference style, while SFT's do. Combined with the length calibration above, SFT has learned both the *content* and the *register/brevity* of the target reasoning field; the baseline has learned neither.

---

## 7. Summary Table — Headline Metrics

| Metric | Baseline | SFT | Relative Change |
|---|---|---|---|
| Raw JSON/schema validity | 15.3% | 98.0% | +542% |
| Captioning BERTScore F1 | 0.338 | 0.402 | +18.8% |
| Captioning CIDEr-D | 0.078 | 0.148 | +90.0% |
| Grounding presence F1 (micro) | 0.448 | 0.902 | +101.4% |
| Grounding mask IoU, exist (micro mean) | 0.336 | 0.773 | +130.5% |
| Violation ID F1 (micro) | 0.220 | 0.458 | +108.2% |
| Violation ID recall (micro) | 0.736 | 0.324 | **−55.9%** |
| Violation ID, IoU-conditioned F1 (micro) | 0.074 | 0.338 | +359.3% |
| Reasoning BERTScore F1 (micro) | 0.208 | 0.753 | +261.8% |

---

## 8. Conclusions and Next-Phase (GRPO/RL) Implications

**Overall conclusion:** Fine-tuning delivers large, consistent gains over the baseline across format compliance, captioning fidelity, spatial grounding, and reasoning-text quality — improvements ranging from roughly +19% to +4.6x depending on the metric family, with the largest gains concentrated in grounding IoU and reasoning-text similarity.

**The one genuine trade-off:** SFT trades recall for precision on violation identification (recall drops from 0.74 to 0.32 while precision rises from 0.13 to 0.78). The baseline's high recall is a symptom of over-triggering (it also has far worse precision and a near-collapsed ability to correctly say "no violation," Rule 0 recall 0.21), not a capability worth preserving — but it does mean **SFT alone leaves ~68% of true violations undetected**, which is a real limitation for a safety-inspection use case.
Violation recall, particularly for Rule 1 (0.409) and the entirely-failed Rules 3 and 4 (0.0 for both models, likely a data-scarcity issue given 109/46 training examples respectively).