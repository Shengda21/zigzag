# 算例一节中数字的来源

2026-09-26 起以《persistent_homo_and_llm.pdf》（2026-09-24 版，英文）为准；编号引用（定理 3.7、命题 3.5/3.15/3.18、注 3.16/3.19、例 3.17、定义 3.10/3.11、式 (3.3)）均指该版。

所有路径相对 `zigzag_v5/`。复算脚本均为只读分析，不修改数据。

## 文件

| 文件 | 内容 |
|---|---|
| `paper_sec4/sec4_examples.tex` | 算例一节中文初稿（日常语言，按新版论文编号锚定），建议作为第 4 节 |
| `paper_sec4/supplement.tex` | 一页补充材料（按论文 §3.3 末尾的四项报告要求组织） |
| `paper_sec4/figure_notes.tex` / `图意说明.pdf` | 四张图的图意说明，每图一页，含术语对照表 |
| `paper_sec4/sec4_suanli*.tex`、`sec3_stats_revision.tex` | 对旧版（9 月 17 日中文稿）写的早期版本，仅存档 |
| `paper_sec4/sec3_stats_revision.tex` | 统计程序修订文本：(A) §3.1 统计程序段，(B) 识别假设 (d)，(C) 定义 3.3，(D) §3.3 末句，(E) 结论句 + 建议补充段 |
| `paper_sec4/new_refs.tex` | 新增参考文献 [23]–[28] |
| `paper_sec4/main_preview.tex` / `.pdf` | 预览（`xelatex` 两遍）；正式并入时只需 `\input{sec4_suanli}` 并按标注替换段落 |
| `paper_sec4/make_paper_figures.py` | 生成 `figures/fig_*.pdf`（中文标注） |
| `paper_sec4/check_grok_tier2.py` | grokking 第 (ii) 级事后核对，输出 `check_grok_tier2.json` |

## §4.1 设置

| 数字 | 来源 |
|---|---|
| LoRA r16、α32、dropout 0.05、全部投影；AdamW 2e-4、20 步预热、余弦到 0.1、裁剪 1.0 | `stage_ab/code/zz_train.py`（LORA、优化器、lr_at） |
| 每步 64 条及其构成；3712 条探针；39 个胞腔 | `complex_worlds/specs/RCN12_FORGET3.json`；`complex_worlds/data/RCN12_FORGET3/world635/MANIFEST.json`、`COMPLEX.json` |
| 联合任务的干预变体 | `complex_worlds/code/spec_world.py::joint_task` |
| 修订 1–3 | `complex_worlds/PREREGISTRATION_FORGET3_CONFIRM2_20260924.md` 末尾；`FREEZE_CONFIRM2.json` |

## §4.2 算例一（内部代号 FORGET3）

| 数字 | 来源 |
|---|---|
| 表 4 支持率、时序结构 p | `cloud_return_round4/analysis_r4/forget3_confirm2_SFT/*.robust.json`、`forget3_confirm2_X2/*.robust.json`；汇总在 `support_table.md` |
| 事件时刻：160–240 / 304 / 336–384 / 496 / 496–544；面 208–240 | 同上 `scores_point`（阈值 0.8 进入、0.6 退出） |
| 条形码 [216,296]、[344,488]、[504,768]；单调化五条；交集桥接三段 | 同上 `detail.zigzag['0.8']`；交集桥接用 `zz_robust.fast_barcode(..., 'intersection')` 在 K_B 上复算 |
| 原边与替换边从不同时在场（0 次） | 在六臂 × 七阈值 × 全部检查点上逐一检查 |
| 历史并集五个环：真实支撑维数 2（原三角形、替换三角形）、伪类维数 3（2026-10-22 更正，原稿误作“另外四条都不存在”） | `paper_sec4/mv_counts.json` 中 `hist_union_final`（`mv_counts.py`，六臂 × 七阈值一致） |
| R5 4/6、0.72–0.96；637 种子 1 的 208/240/304 | `robust.json` 中的 `support.c6_mechanism`；K_C 事件在 θ=0.8 复算 |
| R4 重新张开、R3 迁移窗口 | `robust.json` 中的 `detail.R4_*`、`transfer_R3` |
| X2 各数值（0.73–0.97、≤0.09、0.56–0.81、0.73–0.98、≤0.25、解析率 ≥0.96） | `forget3_confirm2_X2/*.robust.json` 中的 `scores_point`；`eval_1536.summary.json` |
| 旧零模型最小 p 0.35–0.41 | `cloud_return_round2/analysis/RCN12_FORGET3_confirm/*/analysis.json` 中的 `null_model` |
| 端点总数 5–7 倍 | `cloud_return_round2/analysis/null_total.py` 的输出（`EXPLORATION_LOG` 中有记录） |
| 第一次确认事后 3/4、0.78 | `cloud_return_round4/analysis_r4/forget3_first_confirm_posthoc/` |

## §4.3 跨语言

| 数字 | 来源 |
|---|---|
| 97 / 48 / 30 / 18；终点 (1,0)；W 17–33；18 次减少都只移除 zh/@I 胞腔 | `cloud_return_20260923/analysis/crosslingual/world50*/analysis.json` 的 `trajectory`，用 `cw_zigzag.crosslingual_complex()` 在 θ=0.8 复算（Betti 补齐到两项） |
| 复形与桥接分数定义 | `crosslingual/PREREGISTRATION_CROSSLINGUAL_20260922.md` §4 |

## §4.4 grokking

| 数字 | 来源 |
|---|---|
| 231,808 参数、lr 1e-3、wd 1.0、batch 2048、1e5 步、每 250 步评测、p=101/109、训练比例 0.4 | `cloud_return_round2/grokking/runs_confirm/*/RUN_CONFIG.json` |
| 第 (i) 级：10/10、支持度 ≥ 0.999、安慰剂、S3 | `cloud_return_round4/analysis_r4/grokking_confirm_posthoc/*.robust.json` |
| 出生 3000–33000、加法族 3000–13000、4/10、1/10、最大差 9000 | 同上 `detail.birth_step_08` 与 `support.g2_*` |
| 第 (ii) 级 66/70；覆盖 66 个全部 7 阈值 | `paper_sec4/check_grok_tier2.json` |
| 迁移任务最高 0.027 | `runs_confirm/*/eval_*.summary.json` 中的 `modp:transfer:task*` |

## §4.5 Pythia

| 数字 | 来源 |
|---|---|
| 33 个检查点；22 次点估计退出；稳健出生；H_P1 不成立；S3 | `cloud_return_round4/analysis_r4/pythia_runs/pythia-1.4b.robust.json`（连续网格版本见 `pythia_runs_sub/`） |
| 复形 17/23/9、4000 条探针 | `public_ckpt/spec/COMPLEX.json`、`public_ckpt/data/PROBES_MANIFEST.json` |
| 预注册与偏离 | `public_ckpt/PREREGISTRATION_PYTHIA_20260924.md`；`EXPLORATION_LOG_20260923.md`（第四轮） |

## 图 4 下半部分（Pythia，θ=0.6 的 H₀ 条形码）

| 数字 | 来源 |
|---|---|
| zigzag 有限区间 8 个、单调化 5 个；共有的 5 个是 6k–19k 步的合并 | `paper_sec4/pythia_h0_bars.json`（由 `pythia_h0_bars.py` 生成，B=1000） |
| zigzag 独有的 3 个区间（13k、13k–16k、49k），支持率 0.67–0.71 | 同上 `zigzag_only_finite` |
| 98% 的重复中 zigzag 有限区间多于单调化 | 同上 `bootstrap.share_zigzag_more_finite` |
| 造成分裂的两条边及分数（0.54–0.69） | `analysis_r4/pythia_runs/pythia-1.4b.robust.json` 中的 `scores_point`（e_caesar3_caesar3upper、e_reverse_upperreverse） |
