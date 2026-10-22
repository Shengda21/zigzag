# 运行记录 — 2026-09-22 云端实验值守

值守对象：seetacloud RTX PRO 6000 96GB 云 GPU，五条并发 lane（ab401、ab402、rcn、xl、pub）。
本记录仅报告数字与预注册判据的判定结果，不下科学结论。

## 0. 时间线概要

- 第一段（部署后 ~15:40 UTC 之前）：正常运行约 **4.8 小时**，累计约 **33 元人民币**。
- 期间云服务器被**人为强制重启**（非 OOM、非磁盘耗尽崩溃）。重启后工作区被迁移到扩容后的数据盘（`/root/autodl-tmp/zz`，`/root/zz` 改为指向它的符号链接），guard 以新 BUDGET（6 小时上限）重新计时。
- 第二段（重启后 15:40:23 UTC 起）：运行到 `collect_all.py --finish` 完成时（约 17:17:28 UTC）为止，elapsed ≈ **1.62 小时**，session_estimated_cny ≈ **11.3 元**（按 6.98 元/小时估算）。`status_all.py` 随即报 `ConnectionRefusedError`，确认机器已关机（guard 在 `ALL_WORK_COMPLETE.json` + `LOCAL_RETURN_VERIFIED.json` 均存在后立即关机，未到 5 分钟宽限期）。
- **两段合计：约 6.4 小时，约 44–45 元人民币**（第二段数字为 collect_all --finish 触发时刻的估算，机器随后几秒内关机，未能再拿到一次刷新的 LIFECYCLE_STATUS.json）。

## 1. 各 lane 完成情况与门槛（GATE）结果

### 1.1 stage_ab（ab401 / ab402）— 仅 Stage A（COMMON → FORMAT），Stage B 未进入

两个世界的 **FORMAT 门槛均未通过**，按预注册规则（COMMON/FORMAT 门槛失败跳过该世界的 B 阶段）直接完成 lane。`QUEUE_COMPLETE_401.json` / `QUEUE_COMPLETE_402.json` 均记录 `skipped: "Stage A format gate failed"`。

| run | gate.passed | 备注 |
|---|---|---|
| world401_COMMON | **True**（atomic edges 全 1.0，threshold 0.9） | 目标原子边全部满分 |
| world401_FORMAT | **False**（min=0.8125, threshold=0.9） | 见下方修复记录 |
| world402_COMMON | **True**（atomic edges 全 1.0） | |
| world402_FORMAT | **False**（min=0.8125, threshold=0.9） | 与 401 几乎同分 |

修复记录：重启前的首次 FORMAT 跑（`world40{1,2}_FORMAT_v1_noreplay_FAILED`，gate.passed=False）atomic edge scores **全部为 0.0**——经诊断为执行层 bug（缺少逐步重放），以 **EXECUTION_REPAIR_02（16/步重放）** 修复后重跑，atomic edge scores 变为 **0.8125–1.0**（min=0.8125），仍低于 0.9 门槛，但已是真实的"僵局未过"而非此前的系统性零分。两世界的 Stage B（DEP1/IND1/DEP2/IND2 等）**未运行**（按规则跳过），因此本记录没有"联合、修复"分数可报——Stage A 门槛即已判定为不通过。

### 1.2 complex_worlds（rcn，RCN12）— 8/8 arm 全部完成，GATE 全部通过

`complex_worlds/QUEUE_COMPLETE.json` 已生成。

| run | gate.passed | taught_edges_template3 (min~max) | control_edges (min~max) | latent_cells_final (min~max) |
|---|---|---|---|---|
| world601_S1 | True | 1.000~1.000 (n=13) | 0.000~0.000 (n=2) | 0.000~0.000 (n=7) |
| world601_S2 | True | 1.000~1.000 | 0.000~0.000 | 0.000~0.031 |
| world602_S1 | True | 0.969~1.000 | 0.000~0.000 | 0.000~0.031 |
| world602_S2 | True | 1.000~1.000 | 0.000~0.031 | 0.000~0.000 |
| world603_S1 | True | 0.969~1.000 | 0.000~0.000 | 0.000~0.000 |
| world603_S2 | True | 0.969~1.000 | 0.000~0.000 | 0.000~0.000 |
| world604_S1 | True | 1.000~1.000 | 0.000~0.000 | 0.000~0.000 |
| world604_S2 | True | 1.000~1.000 | 0.000~0.000 | 0.000~0.062 |

8 个 run 全部通过：taught edges（template3）持续接近满分，control edges 与 latent cells 的残留泄漏都接近 0。

### 1.3 crosslingual（xl）— 8/8 arm 全部完成，GATE 仅 1/8 通过

`crosslingual/QUEUE_COMPLETE.json` 已生成（501: MULTI1→MONO1→MULTI2→MONO2；502: MONO1→MULTI1→MONO2→MULTI2）。

| run | gate.passed | en_edges min | en_edges_t3 min | zh_edges min | ru_edges (典型值) | ja_edges min | general(en/zh/ru) |
|---|---|---|---|---|---|---|---|
| world501_MONO1 | **True** | 0.979 | 1.000 | 1.000 | 0~0.0625 | 0.500 | 0.875/0.75/0.656 |
| world501_MONO2 | False | 0.958 | **0.875** | 1.000 | 0~0.031 | 0.8125 | — |
| world501_MULTI1 | False | 0.958 | **0.875** | 1.000 | 0~0.625 | 0.500 | 0.875/0.781/0.781 |
| world501_MULTI2 | False | **0.792** | 0.4375 | 1.000 | 0 | 0.531 | 0.875/0.781/— |
| world502_MONO1 | False | **0.688** | 0.0625 | 0.969 | 0~0.344 | 0.0 | 0.75/0.75/0.656 |
| world502_MONO2 | False | 0.875 | 0.625 | 1.000 | 0~0.125 | 0.4375 | 0.75/—/— |
| world502_MULTI1 | False | 0.875 | 0.625 | 1.000 | 0~0.781 | 0.906 | — |
| world502_MULTI2 | False | 0.792 | 0.375 | **0.875** | 0~0.688 | 0.656 | — |

判据（推断自代码行为）：gate 要求 en_edges 与 en_edges_template3 的 5 个边分数都 ≥0.9（zh_edges 基本恒为满分，未构成瓶颈）；ru_edges / ja_edges / general 为诊断性迁移分数，**不参与 pass/fail 判定**，仅记录。

**值得注意的一致模式**：8 个 run 中 **ru_edges 几乎全部接近 0**（多数边分数为 0.0，仅少数达 0.03–0.78），而 zh_edges 几乎全部接近满分；ja_edges 表现介于两者之间且波动大。这一模式在全部 8 个 run 中高度一致，值得关注（可能是迁移设计使然，也可能是探针/数据侧的系统性因素，本记录不做判断）。

### 1.4 public_ckpt（pub）— 8/24 revision 完成（时间到点终止，非错误）

按收尾流程给予 20 分钟宽限期后 `pkill -f 'run_[p]ublic.py'` 停止。已完成 8 个 revision（其余 16 个未跑，保留原样未强行推进）：

| step | overall accuracy (采样) |
|---|---|
| 1 | 0.000 |
| 1000 | 0.020 |
| 4000 | — |
| 16000 | 0.217 |
| 32000 | 0.236 |
| 64000 | — |
| 96000 | 0.303 |
| 143000 | 0.304 |

`cell_scores.py` 对已完成的 8 个 checkpoint 生成 49-cell 分数表（`cloud_return_20260923/analysis/public_ckpt/cell_scores.json`），49-cell 均值随预训练步数上升：step1 mean=0.000 → step143000 mean=0.471（区间内单调不减）。`transfer_scores.json` 记录了 3 个迁移任务的分数（`decimal_to_binary_to_hex_to_english`、`iso_to_us_to_text_to_weekday`、`reverse_then_upper_then_caesar3`）。

已知瓶颈：hf-mirror 镜像下载慢（单个 revision 的模型/分词器下载观测到 10–50 分钟不等，其中 `step1`/`step64000` 首次尝试还遭遇过 `ChunkedEncodingError` 连接中断，重试后自行恢复），因此 24 个 revision 在给定时间内只完成 1/3。

## 2. 本地独立核验（collect_all.py）

两次运行均 `verified: true`：

- 第一次（pub 仍在下载 step64000 时执行，增量式）：1348 files / 2660.1 MB，eval_files=580（0 mismatch），adapters=22（0 mismatch）。
- 第二次（pub 停止后补收新增 revision）：1353 files / 2664.8 MB，eval_files=582（0 mismatch），adapters=22（0 mismatch）。

未触发 `BASE` 路径 ENOENT 问题（符号链接对 SFTP 遍历工作正常），未需要修改 `collect_all.py` 的 `BASE` 行。

## 3. 本地分析摘要（`zz_analyze.py --perm 200`）

三个分析均正常运行完毕（stage_ab 的 COMMON/FORMAT 目录也被分析，属预期行为），无报错。

- **RCN12**：8 个 run 的 verdict 全部为 `undetermined`，`t_star=[0,16]`，`thresholds_supporting=0`，`min_p=0`，ledger 中所有 cell（A1A3/B2B4/A3B1/B4C1/fA123/fA143/fB234/A2B2/C2B4）在全部阈值（0.6~0.9）上 `first_step=null, stable=false`，无 `control_violations`。**注意**：这与 §1.2 中 RCN12 自身 GATE（taught_edges_template3/control_edges/latent_cells_final）**全部 8/8 通过**形成对照——GATE 使用的是任务自带的边/胞腔分数判据，而 `zz_analyze.py` 使用的是基于置换检验的时间窗口判据，二者判据不同、样本量也有限（--perm 200），这一差异建议核实是否符合预期，本记录不做归因判断。
- **crosslingual**：8 个 run 中 `world501_MULTI1` 与 `world502_MONO1` 的 verdict 为 `synchronous`（`thresholds_supporting=5` 和 `7`），其余 6 个为 `undetermined`；`min_p=0`（全部 run）。注意 verdict="synchronous" 与 §1.3 中该 run 的 GATE 结果（两者均为 `passed=False`）是两套独立判据，不应混同。
- **stage_ab**：6 个 run（含旧的 `_v1_noreplay_FAILED` 结果）verdict 全部 `undetermined`，`thresholds_supporting=0`，`min_p=0`。

完整逐 run 明细见 `cloud_return_20260923/analysis/{RCN12,crosslingual,stage_ab}/SUMMARY.json`。

## 4. 执行层事件与修复记录（完整时间线见 `cloud/MONITOR_LOG.md`）

1. **`status_all.py` 格式串 bug**：首次调用报 `TypeError: not enough arguments for format string`，已修复（脚本本身问题，非科学参数）。
2. **`status_all.py` pub revision 文件名解析 bug**：pub 产出首个 `eval_143000.summary.json` 后，`int(p.stem.split('_')[1])` 因文件名含两个 `.` 后缀解析出 `'143000.summary'` 导致 `ValueError`。已修正为 `int(p.name.split('_')[1].split('.')[0])`（`cloud/status_all.py` 第57行），仅为监控脚本的展示逻辑修复，不涉及实验代码或科学参数。
3. **云服务器被人为强制重启**（~15:32 UTC 左右）：所有训练进程与 guard 一并消失，磁盘曾观察到一度低于 2G（此前正在下降但尚未触发自动清理规则）。经确认为人为操作，非 OOM/磁盘崩溃。亲自完成重连、数据盘扩容迁移、guard 与四条训练 lane 重新拉起（含 EXECUTION_REPAIR_02）。此事件期间 rcn（RCN12）与 ab401/ab402 的 COMMON 阶段结果已在重启前完成并保留在磁盘上，未受影响；ab401/ab402 的 FORMAT 阶段与 xl 的 502_MONO2/502_MULTI2 需要重跑/续跑。
4. **`pkill -f run_[p]ublic.py` 自匹配问题**：首次以 `pkill -f 'run_public.py'` 停止 pub 进程时，因该 pkill 命令自身的 SSH 调用命令行文本中也包含 `run_public.py` 子串，导致其误杀自身所在的 shell/连接（表现为 `cloud_remote2.py` 返回 exit 127、无输出）。改用括号转义写法 `pkill -f 'run_[p]ublic.py'` 后正常执行，未造成数据损失（train 进程本身此时已经在预定的 20 分钟宽限期后被正常清理）。
5. 磁盘空间：重启前一度从 20G 降到最低约 1.9G（锯齿式波动，run 完成时会回收 step_* 中间 checkpoint），**未达到需要人工清理的触发条件（一直高于或临界 2G 阈值时机器很快被人为强制重启，未执行任何删除操作）**；重启后迁移到扩容数据盘，全程 82–85G 富余，未再出现磁盘压力。
6. 阶段 A 门槛在重跑后仍然失败——按预注册规则，这是停止信号而非错误，未做任何进一步处理。

未做任何科学参数（步数、门槛、种子、数据）修改；未触碰 `/root/autodl-tmp` 下的旧结果。

## 5. 最终关机核验

`collect_all.py --finish` 执行成功，远端确认 `ALL_WORK_COMPLETE.json` 与 `LOCAL_RETURN_VERIFIED.json` 均存在。随后立即 `status_all.py` 返回 `ConnectionRefusedError [WinError 10061]`，确认 guard 已按预期在两个完成标志文件齐备后立即关机（未使用 5 分钟宽限或手动 `shutdown -h now`）。

## 6. 需关注的异常点（仅报告现象，不作解释）

- stage_ab 的 FORMAT 阶段在 EXECUTION_REPAIR_02 之前 atomic edge scores 全部为精确的 0.0（而非接近 0 的低分），修复后变为 0.81–1.0 区间——说明修复前的结果可能不是真实的"模型学不会"，而是执行层缺陷导致的伪阴性；修复后的结果（min 0.8125 vs 门槛 0.9）是否仍需进一步复核，建议判断。
- crosslingual 全部 8 个 run 中 `ru_edges` 一致地接近 0，而 `zh_edges` 一致地接近满分，模式高度一致，值得核实是否符合预期设计。
- RCN12 的任务自带 GATE（8/8 通过）与 `zz_analyze.py` 的置换检验 verdict（8/8 `undetermined`，`min_p=0`）不一致，建议核实两套判据的适用范围是否匹配 RCN12 的设计（`--perm 200` 的统计功效是否足够也可一并核实）。
- pub 仅完成 8/24 revision（受 hf-mirror 带宽限制），剩余 16 个 revision 未评估。
- 强制重启导致 ab401/ab402 的 FORMAT 阶段与 xl 的最后两个 arm（502_MONO2、502_MULTI2）需要重新计算；这部分结果对应的是重启后（应用 EXECUTION_REPAIR_02 后）的新跑数据，而非重启前的旧跑数据。
