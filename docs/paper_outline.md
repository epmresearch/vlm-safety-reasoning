# Paper Outline

## 1. Abstract
Brief summary of the unified VLM approach for construction safety monitoring, highlighting the reduction in inference cost and alignment benefits.

## 2. Introduction
- The critical need for automated safety inspection in construction.
- Limitations of multi-stage pipelines (Object Detection -> Classification -> NLP).
- Our contribution: A unified prompt framework using Qwen3-VL and LoRA.

## 3. Related Work
- VLMs in industrial applications.
- Instruct tuning and RLHF/GRPO.
- Object detection and captioning metrics in VLM research.

## 4. Methodology
### 4.1 Unified Prompt Design
Explanation of the JSON output schema: captioning, grounding (bboxes), and violation reasoning.
### 4.2 Supervised Fine-Tuning
Unsloth optimizations, `UnslothVisionDataCollator`, and `train_on_responses_only` strategy.
### 4.3 GRPO for Structural Alignment
Future inclusion: how rewards based on JSON validity, IoU matching, and BERTScore are combined.

## 5. Experimental Setup
- Dataset: LouisChen15/ConstructionSite.
- Models: Qwen3-VL (2B, 4B, 8B).
- Evaluation Metrics: Structural validty, BERTScore, CLIPScore, IoU, Precision/Recall/F1.

## 6. Results
- Baseline vs. SFT comparison across tiers.
- Scaling laws (2B -> 4B -> 8B).
- Ablation on GRPO impact.

## 7. Error Analysis
- Edge cases: heavy occlusion, bad lighting.
- False positive/negative breakdown.

## 8. Conclusion
Summary of impact and future directions.
