# Capability complexes and zigzag persistence along language-model training — reproducibility package

Repository: <https://github.com/Shengda21/zigzag>

This repository contains the code, task specifications, preregistration documents, per-item evaluation outputs, and
analysis scripts behind the manuscript

> *Zigzag Persistent Homology of Capability Complexes along Language-Model Training Trajectories*.

The paper itself is not part of this repository. Everything needed to recompute every number and figure of the paper from
the raw per-item model outputs is here; everything needed to regenerate those outputs (training and inference) is here as
well, except the public model weights, which are downloaded from the Hugging Face Hub.

## 1. Layout

The tree mirrors the working layout used during the project, so that all scripts run unchanged from the repository root.

| Path | Content |
|---|---|
| `crosslingual/code/` | **Topology and statistics core.** `cw_zigzag.py` (finite CW complexes over F₂, cellular homology, union/intersection zigzag barcodes by the rank formula, event classifiers), `zz_analyze.py` (evaluation-file loader, cell scores), `zz_robust.py` (item-cluster bootstrap S2, temporal permutation test S3, preregistered criteria of the forgetting experiment, Pythia and grokking lines), unit tests, the monotonisation counterexample, the cross-lingual training/evaluation code and its preregistration. |
| `stage_ab/code/` | `zz_train.py` (LoRA fine-tuning loop), `zz_score.py` (exact scoring of model outputs) and tests. |
| `complex_worlds/` | Forgetting-and-recovery experiment (internal name RCN12_FORGET3): task-world generator (`code/spec_world.py`), training/evaluation runner (`code/run_complex.py`), specifications (`specs/*.json`), generated worlds 635–637 and the X2 variant (`data/.../COMPLEX.json`, `GROUND_TRUTH.json`, `PROBES.jsonl`, `TRAIN_STREAM.jsonl.gz`), preregistration documents and the freeze record `FREEZE_CONFIRM2.json`. Worlds 631–634 (development and first confirmation) are included as specifications only. |
| `grokking/` | Modular-arithmetic experiment: one-layer Transformer (`code/tiny_transformer.py`), runner, specifications, generated worlds 705/706, preregistration draft. |
| `public_ckpt/` | Pythia-1.4B experiment: oracle, probe generator, inference runner (`code/run_public.py`), complex specification (`spec/COMPLEX.json`, `spec/ATOMIC_TASKS.md`), the 4000 probes (`data/PROBES.jsonl`), preregistration and freeze record. |
| `cloud/` | Orchestration used for the rented H800 and the LAN RTX 3090 (`lan3090_driver.sh`: download → size check → inference → delete, per checkpoint) and the session reports of the four GPU sessions (in Chinese). No credentials. |
| `cloud_return_round4/` | **Evidence.** Per-item evaluation outputs of the six confirmation arms (`complex_worlds/runs/RCN12_FORGET3/world63{5,6,7}_S{1,2}/eval_*.jsonl.gz`, 49 checkpoints each) with their final LoRA adapters (`final/`), the two continued-pretraining arms (`RCN12_FORGET3_X2`), the first 33 Pythia checkpoints, the archived analyses (`analysis_r4/`: robust.json files, support table, figures) and the Pythia download logs. |
| `cloud_return_round5/` | The 22 remaining Pythia checkpoints plus the re-evaluated checkpoint 76000 (RTX 3090 session), the merged 55-checkpoint run (`public_ckpt/runs_merged/`), the archived 55-checkpoint analysis (`analysis_r5/`), logs and software versions. |
| `cloud_return_round2/grokking/runs_confirm/` | The ten grokking confirmation runs: evaluation files on the 1000-step grid used in the paper plus the final checkpoint (step 99999), per-checkpoint summaries at the full 250-step resolution (used by the refinement check), run configurations, training logs. |
| `cloud_return_20260923/` | The eight cross-lingual runs (per-item evaluations and final LoRA adapters) and their descriptive analyses. |
| `paper_sec4/` | Scripts that produce the paper's numerical checks: `mv_counts.py` (Theorem 4.1 verified at every bridge of every arm and threshold; artifact dimension of the historical union), `theta_sweep.py` (threshold sweep, bootstrap perturbation scale, δ-robust counts), `pythia_h0_bars.py`, `check_grok_tier2.py`; and their archived outputs (`*.json`). `NUMBERS_SOURCES.md` maps every number of the examples section to a file. |
| `paper_scripts/` | `summarise_forget3.py` (cross-arm table behind Fig. 3), `fill_pythia_numbers.py` (the numbers of the Pythia section as LaTeX macros), `make_figures.py` (all figures), the archived figures. |
| `docs/` | The exploration log of the GPU sessions (development → confirmation, every change recorded) and the original experimental protocol. In Chinese. |
| `reproduce.py`, `verify_against_archived.py`, `compare_replication.py` | One-command reproduction of the analysis chain; exact comparison with the archived outputs; the cross-hardware replication check. |

Evaluation files are stored gzipped (`eval_<step>.jsonl.gz`); the loader reads them directly. Training streams of the task worlds are gzipped as well (`python tools/gunzip_data.py` restores them; only needed to re-run training); probe files (`PROBES.jsonl`) and specifications are stored uncompressed. `reproduction/analysis_dependencies/` holds the earlier, independent implementation of the homology computations that the unit tests cross-check against. Each line is one probe
with the prompt, the reference answer, the model's greedy completion and its exact-match score.

## 2. Reproducing the analysis (CPU only)

```bash
pip install -r requirements.txt          # numpy, matplotlib
python reproduce.py --quick              # ~10 min: unit tests, Pythia robust analyses, Theorem 4.1 checks, threshold sweep, figures
python reproduce.py                      # ~60–90 min: additionally the bootstrap analyses of the forgetting, X2 and grokking arms and the Pythia H0 bars
python verify_against_archived.py        # exact comparison with the shipped results (all seeds are fixed)
python compare_replication.py            # H800 vs RTX 3090 agreement on Pythia checkpoint 76000
```

All homology is computed exactly over F₂ (Gaussian elimination on bit vectors); zigzag barcodes use the rank formula of
Lemma 2.4 of the paper on the run-length-compressed node sequence. Bootstrap (B = 1000) and permutation (1000) seeds are
fixed, so `verify_against_archived.py` reports identity, not approximate agreement.

## 3. From the paper to the files

| Paper | Script / data |
|---|---|
| §7.1 preregistered criteria, supports, S3 | `zz_robust.py --line forget3` on `cloud_return_round4/complex_worlds/runs/RCN12_FORGET3` → `analysis_r4/forget3_confirm2_SFT/*.robust.json`, `support_table.md` |
| §7.1 union/intersection barcodes, transition class, Theorem 4.1 check at all bridges, Proposition 4.4 artifact dimension | `paper_sec4/mv_counts.py` → `mv_counts.json` |
| §7.1 continued-pretraining variant | `analysis_r4/forget3_confirm2_X2/` (criteria with `--time-scale 2`) |
| §7.2 threshold sweep, bootstrap ε, δ-robust counts; Fig. 3 | `paper_sec4/theta_sweep.py` → `theta_sweep.json`; `paper_scripts/summarise_forget3.py` → `forget3_summary.json` |
| §7.3 count vs. organisation; Fig. 4 | `cloud_return_20260923/analysis/crosslingual/world50*/analysis.json` (`zz_analyze.py --design crosslingual`), drawn by `make_figures.py` |
| §7.4 grokking; Fig. 5 | `zz_robust.py --line grok` → `analysis_r4/grokking_confirm_posthoc/`; level (ii) refinement: `paper_sec4/check_grok_tier2.py` |
| §7.5 Pythia-1.4B, 55 checkpoints; Fig. 6 | `zz_robust.py --line public` on `cloud_return_round5/public_ckpt/runs_merged` → `analysis_r5/pythia_runs/`; `paper_sec4/pythia_h0_bars.py` → `pythia_h0_bars_55.json`; degree-one transition classes at θ = 0.85/0.9: `analysis_r5/pythia_runs/h1_transition_support.json`; numbers: `paper_scripts/fill_pythia_numbers.py` |
| Appendix: cross-hardware replication | `compare_replication.py` |
| Appendix: amendments and deviations | `complex_worlds/PREREGISTRATION_FORGET3_CONFIRM2_20260924.md` (amendments 1–3 and their justification), `cloud/SESSION_REPORT_R4_20260925.md`, `docs/EXPLORATION_LOG_20260923.md`, `cloud_return_round5/public_ckpt/MERGE_NOTE.json` |

## 4. Provenance of the evaluation data

| Experiment | Model | Hardware, date | Runs | Directory |
|---|---|---|---|---|
| Forgetting and recovery (confirmation 2) | Qwen2.5-1.5B-Instruct + LoRA r16 | rented H800, 2026-09-24/25 | worlds 635/636/637 × seeds 1/2 | `cloud_return_round4/complex_worlds/runs/RCN12_FORGET3` |
| Continued-pretraining variant (X2) | Qwen2.5-1.5B (base) + LoRA, full-sequence loss | H800, 2026-09-24/25 | worlds 635/636, seed 1 | `cloud_return_round4/complex_worlds/runs/RCN12_FORGET3_X2` |
| Cross-lingual curriculum | Qwen2.5-1.5B-Instruct + LoRA | H800, 2026-09-22 | worlds 501/502 × {MONO, MULTI} × 2 seeds | `cloud_return_20260923/crosslingual/runs` |
| Grokking | 1-layer Transformer, 231,808 parameters, from scratch | H800, 2026-09-24 | p ∈ {101, 109} × 5 seeds, 10⁵ steps | `cloud_return_round2/grokking/runs_confirm` |
| Pythia-1.4B, checkpoints 1k–76k and 143k | EleutherAI/pythia-1.4b (inference only, bf16, greedy) | H800, 2026-09-25 | 33 checkpoints | `cloud_return_round4/public_ckpt/runs/pythia-1.4b` |
| Pythia-1.4B, checkpoints 79k–142k (+76k replication) | same | RTX 3090 workstation, 2026-10-21/22 (PyTorch 2.14.1+cu130, transformers 4.57.6) | 23 checkpoints | `cloud_return_round5/public_ckpt/runs/pythia-1.4b` |

The merged 55-checkpoint Pythia run uses the H800 evaluation of checkpoint 76000; the RTX 3090 evaluation of the same
checkpoint is kept for the replication check (3749 of 4000 greedy completions and 3966 of 4000 correctness values identical).

## 5. Re-running the GPU experiments

Model weights are not included. Qwen2.5-1.5B-Instruct, Qwen2.5-1.5B and the Pythia-1.4B checkpoints (`revision=step<N>`)
are public on the Hugging Face Hub; the general reading-comprehension data of the cross-lingual experiment (Belebele,
XQuAD; both CC BY-SA 4.0) are fetched by `crosslingual/download_general.py`. Run `python tools/gunzip_data.py` before training.

* Forgetting experiment: `python complex_worlds/code/run_complex.py --spec RCN12_FORGET3 --world 635 --seed 1 --micro 16 --save-every 256 --eval-batch 256` (the queue used on the H800 is `code/queue_complex.py`, see `cloud/deploy_round4.py`); the X2 variant adds `--format raw --fewshot 3 --model <path to Qwen2.5-1.5B>`.
* Grokking: see `grokking/README.md` (`code/run_grok.py --preset classic`, 10⁵ steps, evaluation every 250 steps).
* Cross-lingual: see `crosslingual/code/README_cw_zigzag.md` and `crosslingual/PREREGISTRATION_CROSSLINGUAL_20260922.md`.
* Pythia: `python public_ckpt/code/run_public.py --model EleutherAI/pythia-1.4b --revisions step1000,step2000,... --device cuda --batch_size 64 --out <dir>`; `cloud/lan3090_driver.sh` wraps this with a resumable download when huggingface.co is not reachable directly.

Each run writes `eval_<step>.jsonl` files in exactly the format shipped here, so the analysis chain of Section 2 applies to
new runs without modification.

## 6. What is not included, and sizes

* Trained weights of the grokking runs: the one-layer Transformer was not checkpointed (the runs retrain in minutes from the
  shipped data and `RUN_CONFIG.json`). The LoRA adapters of all fine-tuned arms are included (`final/adapter_model.safetensors`,
  about 70 MB each; 16 files).
* Grokking evaluation files at the 250-step resolution (the 1000-step grid used in the paper and the final checkpoint are shipped;
  the per-checkpoint summaries at 250-step resolution are included because the level-(ii) refinement check needs them).
* Development runs (worlds 601–634, 701–704, 801–808) other than their specifications; their analyses are summarised in
  `docs/EXPLORATION_LOG_20260923.md` and the session reports.

The package is about 1.7 GB, of which the sixteen adapter files account for 1.1 GB; no single file exceeds 100 MB. GitHub warns about files above 50 MB, so Git LFS is recommended for `*.safetensors` if the repository is hosted there. The complete analysis chain (`python reproduce.py`) was run inside this repository on 2026-10-22 and `verify_against_archived.py` reported every archived result as identical.

## 7. Licence and citation

Licence to be chosen by the authors before publication (suggested: MIT for code, CC BY 4.0 for data and documents).
Third-party data: Belebele and XQuAD are distributed under CC BY-SA 4.0 by their authors; Pythia checkpoints under Apache 2.0;
Qwen2.5 under the Qwen licence.

---

### 中文说明

本仓库是论文《Zigzag Persistent Homology of Capability Complexes along Language-Model Training Trajectories》的可复现材料：
代码（拓扑与统计核心、任务世界生成、训练与评测脚本）、任务规格、预注册文档与冻结记录、论文用到的全部检查点的逐题评测输出、
存档的分析结果，以及生成论文每个数字和每张图的脚本。不含论文正文和模型权重。

* `python reproduce.py --quick` 约 10 分钟完成不含长时间 bootstrap 的分析链；`python reproduce.py` 完整复算约 1–1.5 小时；
  `python verify_against_archived.py` 与存档结果逐项比对（随机种子固定，结果应完全一致）。
* 目录结构与项目工作目录一致，各脚本无需改动即可运行；评测文件以 gzip 存放，分析代码直接读取。
* 第 3 节的表给出论文各节、各图与脚本和数据文件的对应关系；第 4 节给出每批数据的模型、硬件与日期。
