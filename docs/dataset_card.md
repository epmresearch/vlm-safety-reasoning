# Dataset Card: ConstructionSite 10k

- **Source**: `LouisChen15/ConstructionSite`, derived from MOCS, annotated by a UBC MASc Civil Engineering student.
- **Size**: 10,013 images (7,009 train / 3,004 test) — used exactly as provided, no modification to structure or annotations.
- **License**: CC-BY-NC-4.0
- **Tasks used in Project 1**: rule violation identification, image captioning, visual grounding, attribute classification.



===


## Side-by-side, normalized to percentages

| Category | Attribute | Train (n=7009) | Test (n=3004) |
|---|---|---|---|
| Rule Violations | Rule 1 (PPE) | 9.66% | 10.79% |
| | Rule 2 (Harness) | 0.84% | 0.83% |
| | Rule 3 (Edge Protect) | 1.56% | 2.10% |
| | Rule 4 (Blind Spot) | 0.66% | 0.80% |
| | **Any violation** | **12.7%** | **14.5%** |
| | **No violation** | **87.3%** | **85.5%** |
| Camera Distance | short | 22.3% | 45.3% |
| | mid | 72.2% | 43.6% |
| | long | 5.5% | 11.1% |
| | *(unlabeled)* | 0% | 0.03% (1 image) |
| Illumination | normal | 84.0% | 80.8% |
| | underexposed | 9.9% | 12.7% |
| | overexposed | 3.9% | 5.1% |
| | night | 2.3% | 1.4% |
| Object Presence | excavator | 2415 | 1080 |
| | rebar | 846 | 327 |
| | white hard hat | 680 | 314 |
| Quality of Info | rich | 56.8% | 43.0% |
| | poor | 43.2% | 57.0% |
| View | elevation | 89.7% | 96.4% |
| | plan | 10.3% | 3.6% |

## The one finding that matters most: this is a deliberate distribution shift, not noise

Two categories flip almost entirely between splits:

- **Camera distance**: train is dominated by *mid* distance (72%), test is dominated by *short* distance (45%) — a near-complete reversal
- **Quality of info**: train is majority *rich* (57%), test is majority *poor* (57%) — also flipped

This actually matches something already in your project plan — recall the Gantt/pptx explicitly calls for an evaluation split of "seen vs unseen sites/hazard types." What you're looking at is very likely evidence that the dataset creator deliberately built test to be *harder and different* from train — testing generalization to novel camera distances and lower-information-quality images, not just held-out samples from the same distribution.

**This is good news for research validity, but it changes what your numbers mean.** A model that does well on this test set is demonstrating real generalization, not just memorization — which is a stronger result for your paper. But it also means: don't be surprised or alarmed if your SFT/GRPO accuracy on test looks meaningfully worse than whatever you see on a train-derived validation slice. That gap isn't necessarily a bug — it may be the dataset doing exactly what it was designed to do.

## Rule violation rates are consistent between splits — that's reassuring

Rule 1, Rule 2, and Rule 4 proportions are close between train/test (within ~1 percentage point). Rule 3 is a bit higher in test (2.10% vs 1.56%) but with only 63 test examples, that's a small enough sample that the difference could easily be natural variance rather than a real shift. This consistency is good — it means your earlier concern (severe class imbalance on rule violations) is a property of the *dataset itself*, not an artifact of a bad split, so the mitigations from before still stand:

- Report per-rule precision/recall, not aggregate accuracy
- Rule 2 (25 test examples) and Rule 4 (24 test examples) are genuinely small samples — expect wide error bars, and say so explicitly in your results rather than presenting them with false confidence

## The one data-quality flag: that single "None" camera_distance value in test

One test image has no camera_distance label at all. Worth a quick check:
```python
for r in dataset['test']:
    if r['camera_distance'] not in ['short distance', 'mid distance', 'long distance']:
        print(r['image_id'], repr(r['camera_distance']))
```
This tells you whether it's a genuine missing-label edge case (in which case your preprocessing needs to handle `None`/unexpected values gracefully rather than crashing) or just a labeling quirk like extra whitespace.

## What to actually do with this, concretely

1. **Update `docs/dataset_card.md`** with this full comparison table — this is exactly the kind of dataset documentation a reviewer or committee would want to see, and it directly supports a "seen vs unseen" framing in your eventual paper's methodology section.
2. **Reframe your evaluation narrative**: instead of just "Base vs SFT vs SFT+GRPO accuracy," consider explicitly slicing results by camera_distance and quality_of_info on the test set, since that's where the real generalization test is happening. Your `evaluation/error_analyzer.py` currently doesn't break down by attribute at all — add that.
3. **Don't panic if test scores come in lower than train-derived validation scores** — check first whether the gap concentrates on *short-distance* or *poor-quality* images specifically, which would confirm it's the expected generalization-difficulty effect rather than a bug in your pipeline.
4. **Keep the train/val split (from the earlier answer) drawn only from train**, so your iterative tuning never touches this specific hard-generalization test set until you're ready for a real, final number.





> notebooks/01_dataset_exploration.ipynb
The right move is to exclude them from reward computation, not silently include them (a zero-area box would make IoU always 0 or undefined/NaN and could poison your GSPO reward signal for those specific samples). Add a filter step to your data pipeline:
4 degenerate (zero-width or zero-height) bounding boxes identified out of [N] total — annotation artifacts where the annotator's drag ended at the start coordinate on one axis. Excluded from grounding reward computation; images retained for all other tasks.
These are degenerate/zero-area bounding boxes — annotation errors where one dimension collapsed to a single line instead of a box. Looking at the actual numbers:
RowBox [ymin, xmin, ymax, xmax]What happened5014, rule_1[0.68, 0.19, 0.68, 0.2...]ymin == ymax == 0.68 → zero height (width ≈ 0.01 is fine)6857, white_hard_hat[0.95, 0.45, 0.95, 0.4...]ymin == ymax == 0.95 → zero height11654, excavator[0.93, 0.95, 0.94, 0.9...]xmin ≈ xmax ≈ 0.95 → zero width (height ≈ 0.01 is fine)13188, rebar[0.9, 0.43, 0.91, 0.43]xmin == xmax == 0.43 → zero width
So in every case, one edge of the box (top/bottom or left/right) was placed at the exact same coordinate — the annotator likely dragged a box and released at (or very near) the starting point on one axis, producing a "line" instead of a rectangle. This is a genuine annotation artifact, not a bug in your validation code — your validate_box() function is working exactly as intended by catching these.


Found it — this exactly explains your 324 vs 323 mismatch, and it's more useful than just closing the discrepancy. Two separate things showed up here:
1. The mystery solved: image 0000167
index 17, image_id 0000167, bounding_box=[], reason="The two workers on the right are too close to...", num_boxes=0
This image has a Rule 1 violation with a written reason but no bounding box at all (bounding_box: []). Your code counts this as a valid rule_1 violation because the dict itself isn't None — but the paper's Table 4 count (323) most likely only counts violations that have a complete annotation (reason and grounding box). This one incomplete annotation is your entire +1 discrepancy. Mystery solved — not a bug, not random drift, a single incompletely-annotated sample.
Action: this image needs a decision for your reward design, not just a log entry — see below.
2. Bigger and more important finding: 67 images have multiple violators for the same rule
This is actually more significant for your GSPO reward design than the 324-vs-323 thing. It means your rule_1_violation.bounding_box field is a list of boxes, not a single box — a single image can have 2 or even 3 separate PPE violators, each needing to be grounded. That has a direct implication for how you compute grounding reward: an IoU reward function comparing "the model's box" to "the ground truth box" needs to handle multiple valid ground-truth boxes per image, typically via max-IoU-across-boxes or a matching/assignment scheme (like Hungarian matching if the model outputs multiple boxes too). If you'd designed the reward assuming one-box-per-violation, it would've silently under-rewarded correct answers on these 67 images.

Issue	----- Scope of fix	 -----  What to do
4 degenerate (zero-area) boxes	----- Box-level, not sample-level	----- Drop just that box from the list. Keep the image, keep other annotations. Log in manifest.
0000167-style: reason present, box list empty	----- Task-level, not sample-level	-----  Keep the sample for classification/captioning/reasoning reward. Exclude only from the grounding/IoU term of the reward for that one rule-instance. Don't drop the image.
324 vs 323 count vs paper	----- Already resolved	----- Explained by the one zero-box case above. No further action, just a one-line note in your log.


How to Guarantee 100% Valid JSON
If you want to absolutely bulletproof your pipeline so it never outputs a broken JSON (even if the model gets confused by a weird image), relying on indents won't save you.

Instead, you can enforce Constrained Decoding during your PyTorch inference loop. By passing a strict JSON schema mask to the generation config, you force the logits processor to only allow tokens that form mathematically valid JSON, completely blocking the model from generating markdown ticks or hallucinated keys.

https://www.google.com/search?q=for+VLM+fine+tuning+are+there+ore+better+VLM+then+these+supported+by+unsloth+that+are+really+better+then+these+or+these+are+ok%3F+of+same+sizes%3A+Qwen3-VL-2B-Instruct%2C+4B%2C+and+8B&sca_esv=1c22e5aede185c87&rlz=1C1GCEA_enPK1198PK1198&sxsrf=APpeQnsySspgh89kBozirvJn-vxycILOgA%3A1783777942158&source=chrome.ob&fbs=ABfTbFVyMZGZf1hfvX9uKjN_-G8cn05EoNqnRUpRtqDK_L3JteSLxt4cUK996luNBLqJEG7JzIq3Tint9vCIuZ7YwNtgqIF7l6VojXC1frYKGmk3gfODzSGQd2ICf44DSLNtvx-UyH9cPf1AnqnzfGUEJTUnpSzkckqgqBFVqWGLqBbzScnkh6qhsSxtObj0Dv33z0ZmXpUz&aep=1&ntc=1&sa=X&ved=2ahUKEwiDuebU4sqVAxXYP_sDHYr6NZMQ2J8OegQIERAD&biw=1366&bih=607&dpr=1&mstk=AUtExfBHKf6QXtW6B13aSlmxRVrTNwSH2XQblX9CYDEjnQanFm31azaIPbkpAi49WDQwm_ieiYF2Bcn4P2ATPYq_dSdAhsVDPXFYJEvS0clISKYDMIHTmgU-zeTqsWU1lU6gyX6ZuAuNZ24gkDbqUaM2PDeADIKR-qkF7PDnYVhvS8rn_Oa9_8GOUT676qxi4J7T0Jyil1S23pG9nEjk3mqO9a5Osx60_-Z-fpmTq2k4ydlbRddmtuOxZgPfb6rQCuOF0nMYjGF_6LhSnXFqEgdWCDUdfW6LWSZ82H7wp4Je6_2MEygk1W31hTpnBlI_AeC9L5r75ck7u4n_Ng&csuir=1&mtid=rUpSasOOCfyRkdUPvM6B2AY&udm=50