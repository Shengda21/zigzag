# Capability complexes and zigzag persistence along language-model training — reproducibility package

Repository: <https://github.com/Shengda21/zigzag>

This repository contains the experiment code, task specifications, and
evaluation results for *Zigzag Persistent Homology of Capability Complexes
along Language-Model Training Trajectories*. The original analyses run from
the per-item outputs in this repository. [Follow-up experiments](followup/README.md)
cover explicit-operand and visible-state composition, exact chain checks,
and three seven-operator assessments. The paper and the large follow-up
archives are stored separately.

## 1. Layout

The tree mirrors the working layout used during the project, so that all scripts run unchanged from the repository root.

| Path | Content |
|---|---|
| `crosslingual/code/` | **Topology and statistics core.** `cw_zigzag.py` (finite CW complexes over F₂, cellular homology, union/intersection zigzag barcodes by the rank formula, event classifiers), `zz_analyze.py` (evaluation-file loader, cell scores), `zz_robust.py` (item-cluster bootstrap, temporal permutation test, criteria of the forgetting experiment, Pythia and grokking lines), unit tests, the monotonisation counterexample, the cross-lingual training/evaluation code and its preregistration. |
| `stage_ab/code/` | `zz_train.py` (LoRA fine-tuning loop), `zz_score.py` (exact scoring of model outputs) and tests. |
| `complex_worlds/` | Forgetting-and-recovery experiment (internal name RCN12_FORGET3): task-world generator (`code/spec_world.py`), training/evaluation runner (`code/run_complex.py`), specifications (`specs/*.json`), generated worlds 635–637 and the X2 variant (`data/.../COMPLEX.json`, `GROUND_TRUTH.json`, `PROBES.jsonl`, `TRAIN_STREAM.jsonl.gz`), preregistration documents and the freeze record `FREEZE_CONFIRM2.json`. Worlds 631–634 (development and first confirmation) are included as specifications only. |
| `grokking/` | Modular-arithmetic experiment: one-layer Transformer (`code/tiny_transformer.py`), runner, specifications, generated worlds 705/706, preregistration draft. |
| `public_ckpt/` | Pythia-1.4B experiment: oracle, probe generator, inference runner (`code/run_public.py`), complex specification (`spec/COMPLEX.json`, `spec/ATOMIC_TASKS.md`), the 4000 probes (`data/PROBES.jsonl`), preregistration and freeze record. |
| `cloud/` | Orchestration used for the rented H800 and the LAN RTX 3090 (`lan3090_driver.sh`: download → size check → inference → delete, per checkpoint) and the session reports of the four GPU sessions (in Chinese). No credentials. |
| `cloud_return_round4/` | **Evidence.** Per-item evaluation outputs of the six confirmation arms (`complex_worlds/runs/RCN12_FORGET3/world63{5,6,7}_S{1,2}/eval_*.jsonl.gz`, 49 checkpoints each) with their final LoRA adapters (`final/`), the two continued-pretraining arms (`RCN12_FORGET3_X2`), the first 33 Pythia checkpoints, the archived analyses (`analysis_r4/`: robust.json files, support table, figures) and the Pythia download logs. |
| `cloud_return_round5/` | The 22 remaining Pythia checkpoints plus the re-evaluated checkpoint 76000 (RTX 3090 session), the merged 55-checkpoint run (`public_ckpt/runs_merged/`), the archived 55-checkpoint analysis (`analysis_r5/`), logs and software versions. |
| `cloud_return_round2/grokking/runs_confirm/` | The ten grokking confirmation runs: evaluation files on the 1000-step grid used in the paper plus the final checkpoint (step 99999), per-checkpoint summaries at the full 250-step resolution (used by the refinement check), run configurations, training logs. |
| `cloud_return_20260923/` | The eight cross-lingual runs (per-item evaluations and final LoRA adapters) and their descriptive analyses. |
| `paper_sec4/` | Scripts for the original studies' numerical checks: `mv_counts.py` (bridge relation and historical-union artifact dimension), `theta_sweep.py` (threshold sweep, bootstrap perturbation scale, δ-robust counts), `pythia_h0_bars.py`, `check_grok_tier2.py`; and their archived outputs (`*.json`). `NUMBERS_SOURCES.md` describes the older draft's numbers and numbering; use the current mapping in Section 3 below. |
| `paper_scripts/` | `summarise_forget3.py` (cross-arm forgetting summary), `fill_pythia_numbers.py` (Pythia LaTeX macros), `make_figures.py` (original-study figures), and the archived figures. |
| `followup/` | Follow-up analyses, experiment protocols, training and scoring code, result snapshots, and archive indexes. See the [follow-up guide](followup/README.md). |
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

All homology is computed exactly over F₂ (Gaussian elimination on bit vectors); zigzag barcodes use the rank formula on
the run-length-compressed node sequence. Bootstrap (B = 1000) and permutation (1000) seeds are fixed, so
`verify_against_archived.py` reports identity for its archived original-study outputs. The later analyses in `followup/`
have separate commands and dependencies in `followup/README.md`; they are not run by `reproduce.py`.

## 3. Experiment and analysis guide

| Manuscript anchor | Script / data |
|---|---|
| `sec:ex-forget`; Supplement S1 | `zz_robust.py --line forget3` on `cloud_return_round4/complex_worlds/runs/RCN12_FORGET3`; `paper_scripts/summarise_forget3.py`; `paper_sec4/mv_counts.py` |
| `sec:ex-sweep`; Supplements S2--S3 | `paper_sec4/theta_sweep.py`; `followup/e0_e3/`, `followup/confidence/`, `followup/behavior/`, `followup/e5_cpu_audit/` contain the later bootstrap, finite-population, matched-behavior and exact-chain checks |
| `sec:ex-count`; Supplement S3 | `cloud_return_20260923/analysis/crosslingual/world50*/analysis.json`, `zz_analyze.py --design crosslingual`, original figure scripts |
| `sec:ex-grok`; Supplement S3 | `zz_robust.py --line grok`, `analysis_r4/grokking_confirm_posthoc/`, `paper_sec4/check_grok_tier2.py` |
| `sec:ex-pythia`; Supplement S3 | merged 55-checkpoint outputs in `cloud_return_round5/public_ckpt/runs_merged`; `analysis_r5/pythia_runs/`; `paper_sec4/pythia_h0_bars.py`; `paper_scripts/fill_pythia_numbers.py` |
| Supplement S4 (`supp:details`) | Original protocol amendments, hardware comparison and session records; `compare_replication.py` |
| `sec:ex-e4`; Supplement S5 | Frozen `followup/e4_v4/` source and result addendum, plus `followup/e4_full_invariant_pilot/`; large raw archives listed separately |
| `sec:ex-composition`; Supplement S6 | `followup/composition/` source bundle and verified archive/adapter index; visible-state controls, fixed failures and all six training sources |
| `sec:ex-labelled`, `rem:rank-design`; Supplements S7--S8 | `followup/seven_operator/`: exact rank-design checks, frozen six-source primary/fixed-dose/new-pattern protocols, execution and audit code, archive indexes, and results for all 18 cases. |

The E4-v4 confirmation met the primitive-edge equivalence criterion. Its
external-effect and predictive-utility criteria were unmet. The [seven-operator
study](followup/seven_operator/README.md) passed all three assessments in each
of the six task sources. The fixed-dose control used the same number of
updates in both exchange phases. The disjoint-pattern assessment reused its
fitted weights.

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

## 6. Data availability

* Trained weights of the grokking runs: the one-layer Transformer was not checkpointed (the runs retrain in minutes from the
  shipped data and `RUN_CONFIG.json`). The LoRA adapters of all fine-tuned arms are included (`final/adapter_model.safetensors`,
  about 70 MB each; 16 files).
* Grokking evaluation files at the 250-step resolution (the 1000-step grid used in the paper and the final checkpoint are shipped;
  the per-checkpoint summaries at 250-step resolution are included because the level-(ii) refinement check needs them).
* Development runs (worlds 601–634, 701–704, 801–808) other than their specifications; their analyses are summarised in
  `docs/EXPLORATION_LOG_20260923.md` and the session reports.
* The follow-up archive indexes identify the input banks, checkpoints, fitted
  weights and raw predictions needed for full replay. The indexes contain
  filenames and digests. A public download location is currently unavailable.

## 7. Contributors and third-party assets

- Shengda21 (`160199828+Shengda21@users.noreply.github.com`)

Third-party data: Belebele and XQuAD are distributed under CC BY-SA 4.0 by their authors; Pythia checkpoints under Apache 2.0;
Qwen2.5 under the Qwen licence.

---

### 中文说明

仓库提供原始实验的逐题评测与复算脚本，以及后续实验的协议、代码和结果。
七算子实验包含主确认、等步数对照和留出程序模式评测，共 18 组，全部通过预设判据。
[后续实验指南](followup/README.md) 列出了各项复算命令及所需数据。大型原始归档另存，当前暂无公开下载地址。
