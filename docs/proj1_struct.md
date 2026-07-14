Project 1 — Final Structure (Fine-Tuning Study Only)
Before the tree, here's the thinking that shaped it, because the "why" matters more than the folder names.
What Project 1 actually needs (and what it doesn't)
Project 1 = fine-tune VLMs on the existing ConstructionSite 10k schema (rule violations, captioning, grounding, attributes), add GSPO/GRPO on top, produce a comparison paper. That's it. No API, no dashboard, no agents, no smart-glasses, no knowledge base. Your old structure (and the one Khashayar's plan implies) was written for the combined 4-study project — so agents/, api/, dashboard/, knowledge_base/, react_app/ all get deleted. Keeping them "just in case" clutters the repo and signals scope creep to anyone reviewing your GitHub for a thesis application.
Second thing I changed: the old structure organizes around "Study 1/2/3/4." Project 1 doesn't have studies anymore — it has tasks (rule violation, captioning, grounding, attributes) and model strategy (multi-size vs task-specialized), which is still undecided. So the structure needs to be strategy-agnostic — it should work whether you end up fine-tuning 3 sizes of one model on all tasks, or one model per task. I did this by making registry.py the single place that maps model_id → (task, size, checkpoint), so the decision is a config change, not a restructuring.
Third: your dataset is used as-is, unmodified. That means no annotation guide/taxonomy building — that was Khash's job for the old custom dataset. Here your data/ layer is about conversion (HF dataset → per-task training format), not annotation creation.

A) GitHub Repository (source of truth — code only, nothing heavy)
vlm-safety-finetuning/                  # Project 1 repo
│
├── README.md                           # What this repo is, how to run it end-to-end
├── requirements.txt                    # Pinned versions (unsloth, trl, transformers, peft, etc.)
├── .env.example                        # HF_TOKEN=, WANDB_API_KEY=  (no real values)
├── .gitignore                          # .env, __pycache__, data/, checkpoints/, *.ipynb_checkpoints
│
├── configs/
│   ├── base.yaml                       # drive_root, hf_org, wandb_project — shared across everything
│   ├── model_strategy.yaml             # THE decision switch: "multi_size" | "task_specialized"
│   │                                   # This is the one file you flip once the strategy is finalized
│   ├── tasks/
│   │   ├── rule_violation.yaml         # task-specific: prompt name, reward weights, output schema
│   │   ├── captioning.yaml
│   │   ├── grounding.yaml
│   │   └── attributes.yaml
│   ├── sft.yaml                        # lr, epochs, batch size, LoRA r/alpha/targets
│   └── grpo.yaml                       # num_generations, reward weights, max_completion_length
│
├── core/
│   ├── config.py                       # load_config() — merges base.yaml + task yaml + strategy
│   ├── logging.py                      # get_logger() — one logger used everywhere
│   ├── io.py                           # get_drive_path(), safe_save_json(), append_to_csv()
│   └── wandb_utils.py                  # init_run(), log_results()
│
├── data/
│   ├── schemas.py                      # Pydantic contracts: RawSample, SFTSample, GRPOPrompt,
│   │                                   # per-task output schema (RuleViolationOutput, etc.)
│   ├── loader.py                       # load_construction_dataset(cfg) → uses HF's existing
│   │                                   # 7009/3004 train/test split AS-IS — no re-splitting
│   ├── preprocessor.py                 # to_sft_format(sample, task, cfg) — one function per task
│   │                                   # converts dataset fields (rule_X_violation, bbox, caption,
│   │                                   # attributes) into chat-format training examples
│   └── prompt_templates.py            # ALL prompt text as named constants, one set per task
│
├── models/
│   ├── registry.py                     # single dict: model_id → {hf_path, task, size, lora_path}
│   │                                   # reads model_strategy.yaml to decide which entries exist
│   ├── vlm_wrapper.py                  # UnifiedVLMWrapper.generate(image, prompt) → str
│   │                                   # handles Qwen2.5-VL / SmolVLM / MiniCPM differences
│   ├── sft_trainer.py                  # run_sft(cfg, task) → adapter saved + pushed to Hub
│   └── grpo_trainer.py                 # run_grpo(cfg, task, reward_fns) → adapter saved + pushed
│
├── rewards/                             # GSPO/GRPO rewards, derived straight from dataset schema
│   ├── json_validity.py
│   ├── rule_violation_accuracy.py      # matches rule_1..4 violation labels
│   ├── grounding_iou.py                # IoU vs ground-truth bbox
│   ├── caption_quality.py              # similarity/LLM-judge vs reference caption
│   └── attribute_accuracy.py           # illumination/camera_distance/view/quality match
│
├── evaluation/
│   ├── metrics.py                      # per-task metric functions
│   ├── evaluator.py                    # ModelEvaluator.run(model, task, dataset) → CSV + W&B
│   └── error_analyzer.py               # categorize_failures() per task
│
├── experiments/                         # entry points only — no logic, <80 lines each
│   ├── run_baseline.py                 # --task --model_id
│   ├── run_sft.py                      # --task --model_id
│   ├── run_grpo.py                     # --task --model_id
│   └── compare_results.py              # builds the base vs SFT vs SFT+GSPO comparison table
│
├── notebooks/                           # orchestration/exploration ONLY, outputs cleared pre-commit
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_baseline_inspection.ipynb
│   ├── 03_reward_function_dev.ipynb
│   ├── 04_sft_orchestration.ipynb       # !python experiments/run_sft.py --task ...
│   ├── 05_grpo_orchestration.ipynb
│   └── 06_comparison_and_figures.ipynb
│
├── scripts/
│   ├── setup_colab.sh                  # clone/pull repo, mount Drive, copy .env, pip install
│   ├── clear_notebook_outputs.sh       # run before every commit
│   ├── push_adapter_to_hub.py
│   └── generate_figures.py
│
├── tests/
│   ├── test_data/
│   │   └── test_preprocessor.py        # each task's to_sft_format() with known input/output
│   ├── test_rewards/                   # one test file per reward function
│   └── test_evaluation/
│       └── test_metrics.py
│
└── docs/
    ├── architecture.md                 # how data → model → reward → eval connects
    ├── dataset_card.md                 # ConstructionSite 10k, used as-is, license CC-BY-NC-4.0
    ├── experiment_log.md               # weekly log — becomes your Methods section
    ├── meeting_notes/
    │   └── 2026-06-29_kickoff.md
    └── model_cards/
        └── template.md
Why no configs/study1_*.yaml, configs/study2_*.yaml: those were per-study in the 4-study plan. Here the axis is per-task, so configs/tasks/*.yaml replaces it directly.
Why model_strategy.yaml exists as its own file: because you told me the multi-size vs task-specialized decision isn't final. Isolating it means when Pouya/Khash decide, you change one YAML value, not refactor code.

B) Google Drive (data + checkpoints + results — never in git, too large)
MyDrive/
└── vlm-finetuning-project1/
    │
    ├── secrets/
    │   └── .env                        # HF_TOKEN, WANDB_API_KEY — copied into Colab at session start
    │
    ├── datasets/
    │   ├── raw/                        # HF cache — LouisChen15/ConstructionSite, unmodified
    │   ├── processed/
    │   │   ├── rule_violation/
    │   │   │   ├── sft_train.parquet
    │   │   │   └── sft_test.parquet
    │   │   ├── captioning/
    │   │   ├── grounding/
    │   │   └── attributes/
    │   └── split_manifest.json         # records that we used HF's native 7009/3004 split, untouched
    │
    ├── checkpoints/
    │   ├── rule_violation/
    │   │   ├── sft_v1/  sft_v2/  grpo_v1/
    │   ├── captioning/
    │   ├── grounding/
    │   └── attributes/
    │
    ├── results/
    │   ├── rule_violation/  captioning/  grounding/  attributes/
    │   │   ├── baseline_eval.csv
    │   │   ├── sft_eval.csv
    │   │   ├── grpo_eval.csv
    │   │   └── comparison_table.csv
    │   └── error_analyses/
    │
    ├── figures/                        # paper-ready PNG + PDF
    │
    └── logs/
        ├── colab_session_log.txt       # append-only: session start/stop, what ran, where it died
        └── error_log.txt
One folder per task under datasets/processed, checkpoints, and results — this stays correct regardless of which model strategy you pick, since even task-specialized and multi-size runs both need per-task buckets.

C) How Local PC ↔ GitHub ↔ Colab ↔ Drive sync (the part that actually breaks most student projects)
The rule that prevents everything from diverging: GitHub is the only source of truth for code. Drive is the only source of truth for data/checkpoints/results. Nothing else. Colab never holds anything permanently — it's a disposable compute box that pulls both at the start of a session and pushes both at the end.

Local PC (your main dev environment)

Clone vlm-safety-finetuning once.
Write/edit all .py files, configs, prompts here in your IDE (not in Colab cells).
Commit + push to GitHub. This is where code review and version history live.


Colab (compute only)

Every session starts by running scripts/setup_colab.sh, which:

git clone (or git pull if the repo is already present in the Colab runtime/local disk)
mounts Drive
copies secrets/.env from Drive into the runtime
pip install -r requirements.txt
sets PYTHONPATH so data/, models/, etc. import cleanly


Notebooks in notebooks/ only call experiments/*.py — they never contain real logic. If you write a fix in a Colab cell, it must be moved into the repo and pushed before it counts as "done," or it disappears when the runtime resets.
Anything the run produces (checkpoints, CSVs, logs) writes straight to the Drive paths from core/io.py, so a Colab disconnect never loses results.


The one discipline that keeps this robust: never treat Colab as a place where code lives. It's stateless by design (session resets, disconnects, timeouts). Code lives in GitHub, artifacts live in Drive, Colab is just where the GPU is. Adapters also get pushed to the HF Hub from scripts/push_adapter_to_hub.py as a second backup beyond Drive.

A concrete session template for notebooks/04_sft_orchestration.ipynb:
!bash scripts/setup_colab.sh
!python experiments/run_sft.py --task rule_violation --model_id qwen25-7b
That's the entire notebook. Everything else is in the repo.