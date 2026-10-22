# 第二轮实验运行记录（2026-09-23）

值守范围：租用 H800（seetacloud，9.98 元/小时）机器上并发的四条初始 lane，以及期间陆续追加的探究迭代 lane。全程只做执行层修复，未修改任何科学参数（步数、课程表、门槛、阈值、种子、数据），未触碰 `/root/autodl-tmp` 与 `/root/zz`（上一轮数据）。按中途的指令更新，本次**不执行 `collect_all.py --finish`，不关机**，机器在报告完成时仍在线。

## 0. 总耗时与费用

- 截至最后一次核实（`status_all.py`，UTC 16:51）：`elapsed=17.26h`，`cost≈172.2 CNY`（9.98 元/小时 × 17.26h）。
- 原始预算上限（8 小时）于 01:23Z 撤销，`BUDGET.json` 改为 `session_hours_limit=72`，探究模式。
- 期间机器持续在线，未发生因预算触发的 STOP_COMPUTE / 硬关机。

## 1. 探究迭代的追加时间线

| 时间(UTC) | 内容 | 核实方式 |
| --- | --- | --- |
| 01:23 | 撤销 8h 预算上限，`BUDGET.json.session_hours_limit=72`，收尾改为不 `--finish`/不关机 | 远端直接读取 BUDGET.json 核实，note 字段确认 |
| 01:25 | 新增 `forget23_621`/`forget23_622`：RCN12_FORGET2(621/622)→自动转 RCN12_FORGET3(631/632)，`--micro 8` | ps 进程 + 日志核实 |
| 01:49 | `bio`→`bio2`：原 BIO 无问答训练探针恒为 0，终止后启动 BIO2(803/804) | 首次核实与消息声称有约 8 分钟延迟（进程未及时消失），复查后确认为真实过渡，非注入 |
| 07:57 | 探究迭代 3：六条 FORGET 系列队列已全 QUEUE_COMPLETE，要求先跑增量 `collect_all.py` + 三条 complex_worlds 分析；同时终止 grok(MODP_LATENTSUB) 改跑 `grok3_701`/`grok3_702`（classic 预设 10 万步）与 `addonly`（MODP_ADDONLY） | ps + 日志核实；**发现 grok3_701/702 因与旧 MODP 运行目录同名+resume 误加载导致 1 层/2 层权重不匹配崩溃**，已记录并报告，随后确认同一根因并改用 `--runs-dir runs_classic` 重启 |
| 08:09 | bio2 完成，启动 `bio3_805`/`bio3_806`（BIO3，1536 步） | ps + RUNNING_BIO3_*.json 核实 |

## 2. 四条 lane 完成情况

| lane/规格 | 完成标志 | 结果 |
| --- | --- | --- |
| complex_worlds RCN12_FORGET (611/612) | `QUEUE_COMPLETE_RCN12_FORGET_611/612.json` | S1+S2 全部 gate **passed=true** |
| complex_worlds RCN12_FORGET2 (621/622) | `QUEUE_COMPLETE_RCN12_FORGET2_621/622.json` | S1 gate FAIL（阴性）→ S2 gate passed=true |
| complex_worlds RCN12_FORGET3 (631/632) | `QUEUE_COMPLETE_RCN12_FORGET3_631/632.json` | S1+S2 全部 gate **FAIL**（完整阴性结果） |
| biography BIO2 (803/804) | `biography/QUEUE_COMPLETE.json` (spec=BIO2) | 8 臂全部完成（801 两臂作 BIO 记录保留，802 因 bio→bio2 切换被中止未计入） |
| biography BIO3 (805/806) | `QUEUE_COMPLETE_BIO3_805/806.json` | 8 臂全部完成，1536 步 |
| grokking scan_grok | `SCAN_SUMMARY.json` | 3×3 网格（wd∈{0.1,0.5,1.0}×frac∈{0.3,0.4,0.5}），8/9 grok |
| grokking MODP (默认预设,30000步) | 9/10 完成，1 个无法分析 | world701 全 5 种子 + world702 种子1-4 完成；**world702_seed5 分析流程两次独立复现卡死（远端+本地），训练/评测数据完整但未产出 COMPLETE.json/verdict** |
| grokking MODP_ADDONLY | 10/10 完成 | world703/704 × 种子1-5 |
| grokking MODP(classic,100000步,runs_classic) | 6/6 完成 | world701/702 × 种子1-3 |

`collect_all.py` 最终核验（增量两次）：`{"files": 20249, "verified": true, "eval_files": 8780, "eval_mismatch": [], "adapters": 30, "adapter_mismatch": []}`，全部通过独立重评分与 adapter 哈希核验。

## 3. 执行层修复记录（按发生顺序，全部数字，未改科学参数）

1. **磁盘清理**（05:12 UTC，4.3G→13G free）：对已 `QUEUE_COMPLETE` 的 8 个 complex_worlds run（RCN12_FORGET 4 个 + RCN12_FORGET2 4 个）删除 `step_*` 中间 adapter（每个 run 13 个快照，共约 7.7GB），保留 `final/`、`eval_*`、`*.json`；对 7 个已完成的 grokking MODP run 只保留最后一个 `ckpt_*.pt`（回收约 1GB）。
2. **grok MODP_world702_seed5 卡死重启**（07:24 UTC）：与另外 4 个同类运行对比，正常运行从训练结束到写出 `COMPLETE.json` 耗时 81-87 秒，该运行同一阶段耗时 76 分钟仍未完成（CPU 时间持续增长、非 D 状态、GPU 占用极低，排除资源竞争，`nproc=176`）。判定为分析阶段卡死，kill 后重启剩余队列（跳过已完成的 scan_grok）。
3. **grok MODP_world702_seed5 二次卡死，判定跳过**（07:49 UTC）：重启后队列正常快速通过其余 9 个运行，但再次在同一运行卡死（6.5 分钟无输出，远超基线），确认为该运行数据的确定性问题而非瞬时故障。跳过该运行，直接启动 MODP_LATENTSUB 队列。
4. **发现并报告 grok3_701/702 目录冲突崩溃**（07:57 UTC）：新起的 `--preset classic` 命令与已完成的默认预设运行共用同一输出目录（`MODP_world701_seed1` 等），`resume=True` 误加载不兼容权重（1 层 vs 2 层）崩溃。判定为确定性目录冲突而非可通过“同命令重启”解决的瞬时错误，记录并报告；随后确认并改用 `--runs-dir runs_classic` 解决。
5. **`collect_all.py` 后台任务超时后续传**（09:24、13:29 UTC 两次）：本地 `collect_all.py` 下载在约 40-45 分钟处被后台任务机制终止（非脚本报错）；因其下载逻辑按文件大小跳过已下载文件（幂等），直接重跑续传至完成，`verified: true`。
6. **`collect_all.py` 路径匹配边界情况（`runs_classic` 未被下载 `.jsonl.gz`）**：`wanted()` 按路径子串 `/runs/` 判断是否下载 eval 原始数据，但 `runs_classic` 目录名不含该子串，导致只下载了 `.json` 摘要、没下载 `.jsonl.gz` 原始探针数据（首次分析报 `ValueError: negative dimensions`）。判定为新建目录名与既有脚本路径匹配逻辑的边界情况，**未修改 `collect_all.py` 代码**，改用独立 SFTP 脚本单独补拉 6 个 run 共 2406 个 `eval_*.jsonl.gz` 文件后分析成功。
7. **本地 zz_analyze.py 对 BIO3（8 臂，96 checkpoint/run）单次运行超过后台任务时限**：首次调用在约 60 分钟处被终止，5/8 臂已完成；未改任何脚本或参数，改为对剩余 3 臂单独分析并手动合并 `SUMMARY.json`（从各 run 的 `analysis.json` 重建汇总字段）完成全部 8 臂。
8. **本地对 `MODP_world702_seed5` 直接调用 `analyze_run()` 复现同一卡死**：绕过 `main()` 的跳过逻辑直接分析该 run，本地运行 43 分钟无进展（内存稳定在 13.4GB，非 OOM），与两次远端卡死构成第三次独立复现。判定为该 run 数据在拓扑分析代码路径中的确定性问题（具体代码位置未诊断，未修改代码），终止该次分析，记录为未解决项。

## 4. RCN12_FORGET / FORGET2 / FORGET3（θ=0.8，共 12 个 run，`--perm 300`）

`zz_analyze.py` 判定：**全部 12 个 run，`synchrony.verdict = undetermined`**（`t_star` 恒为观测窗口第一个 transition，因为预注册的“≥6 个潜在胞腔在≥5 个阈值上同步出现”判据在全部 run 中都不成立——潜在胞腔 `A1A3,B2B4,A2B2,C2B4,fA123,fA143,fB234` 在全部 12 个 run、全部 θ∈[0.6,0.9]、全部 checkpoint 上恒为 0.0；`A3B1,B4C1` 起始小幅非零后迅速跌至 0-0.14 区间、从未达到 0.6 阈值）。ledger 中除阴性对照外全部 `first_step: null`。

GATE（taught_edges_template3 / control_edges / latent_cells_final，终值）：

- **RCN12_FORGET**（611_S1/S2, 612_S1/S2）：4/4 gate **passed=true**；`taught_edges_template3` 全部 =1.0；`control_edges`(A2B2,C2B4) 全部 =0.0；`latent_cells_final` 除个别 run 的 `A3B1`(0.062)/`B4C1`(0.094) 微小非零外全为 0。
- **RCN12_FORGET2**（621,622）：S1 均 gate FAIL（`B1E1`/`B4E4`/`C4A1` 等未达 0.9 门槛，如 621_S1 的 B1E1=0.219、C4A1=0.625），S2 均 passed=true。`latent_cells_final` 除 622_S1 的 `fA123`=0.016/`fA143`=0.008 外均为 0。
- **RCN12_FORGET3**（631,632）：S1+S2 全部 FAIL（`B1B2r/B2B3r/B1B3r/B3B4r` 替换边终值恒为 0.0，其余 taught edges 接近满分但因替换边未达标而不通过）。`latent_cells_final` 全为 0。

θ=0.8 下 B 域四边（B1B2, B2B3, B1B3, B3B4）进入(in)/掉出(out)/恢复(back)步（数字为训练步，`None` 表示未发生）：

**RCN12_FORGET（无遗忘窗口，边进入后不再掉出）**
| run | B1B2 | B2B3 | B1B3 | B3B4 |
| --- | --- | --- | --- | --- |
| 611_S1 | in=176 | in=176 | in=176 | in=192 |
| 611_S2 | in=176 | in=160 | in=176 | in=160,out=176,back=192 |
| 612_S1 | in=160 | in=160 | in=160 | in=144 |
| 612_S2 | in=192 | in=144 | in=176 | in=208 |

**RCN12_FORGET2**
| run | B1B2 | B2B3 | B1B3 | B3B4 |
| --- | --- | --- | --- | --- |
| 621_S1 | in=192,out=304,back=496 | in=192,out=320,back=496 | in=176,out=208,back=224 | in=208,out=304,back=496 |
| 621_S2 | in=176,out=304,back=496 | in=176,out=320,back=496 | in=176,out=304,back=496 | in=192,out=304,back=496 |
| 622_S1 | in=176,out=192,back=208 | in=192,out=304,back=496 | in=208,out=304,back=496 | in=208,out=304,back=496 |
| 622_S2 | in=192,out=304,back=496 | in=176,out=208,back=224 | in=192,out=304,back=496 | in=208,out=304,back=496 |

**RCN12_FORGET3**（原边 B1B2 等 + 替换边 B1B2r 等）
| run | B1B2 | B2B3 | B1B3 | B3B4 | B1B2r | B2B3r | B1B3r | B3B4r |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 631_S1 | in=160,out=304,back=512 | in=176,out=304,back=512 | in=160,out=304,back=512 | in=176,out=304,back=528 | in=352,out=496,back=None | in=368,out=496,back=None | in=368,out=496,back=None | in=352,out=496,back=None |
| 631_S2 | in=192,out=304,back=512 | in=176,out=304,back=512 | in=176,out=304,back=496 | in=192,out=304,back=512 | in=368,out=496,back=None | in=352,out=496,back=None | in=352,out=496,back=None | in=352,out=496,back=None |
| 632_S1 | in=192,out=304,back=512 | in=192,out=304,back=496 | in=192,out=304,back=496 | in=208,out=304,back=512 | in=368,out=496,back=None | in=368,out=496,back=None | in=384,out=496,back=None | in=368,out=496,back=None |
| 632_S2 | in=192,out=304,back=512 | in=208,out=304,back=496 | in=176,out=304,back=512 | in=208,out=304,back=512 | in=352,out=496,back=None | in=352,out=496,back=None | in=352,out=496,back=None | in=352,out=496,back=None |

即：原 B 域边在 FORGET2/FORGET3 中均呈现“教→约 step 208-320 掉出→约 step 496-528 恢复”的模式；FORGET3 的替换边呈现“约 step 352-384 教入→约 step 496 掉出→768 步内未再恢复（终值 0.0）”。

fC123（联合面）在 128/288/352/480/544/768 步的取值：

| run | 128 | 288 | 352 | 480 | 544 | 768 |
| --- | --- | --- | --- | --- | --- | --- |
| FORGET 611_S1 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| FORGET 611_S2 | 0.000 | 0.969 | 1.000 | 1.000 | 1.000 | 1.000 |
| FORGET 612_S1 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| FORGET 612_S2 | 0.000 | 0.969 | 1.000 | 1.000 | 1.000 | 1.000 |
| FORGET2 621_S1 | 0.000 | 0.938 | 0.969 | 1.000 | 1.000 | 1.000 |
| FORGET2 621_S2 | 0.000 | 0.969 | 1.000 | 1.000 | 0.969 | 1.000 |
| FORGET2 622_S1 | 0.000 | 0.969 | 1.000 | 1.000 | 0.969 | 1.000 |
| FORGET2 622_S2 | 0.000 | 0.969 | 1.000 | 0.812 | 0.000 | 1.000 |
| FORGET3 631_S1 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| FORGET3 631_S2 | 0.000 | 0.969 | 1.000 | 1.000 | 1.000 | 1.000 |
| FORGET3 632_S1 | 0.000 | 0.969 | 1.000 | 1.000 | 0.656 | 1.000 |
| FORGET3 632_S2 | 0.000 | 0.844 | 1.000 | 1.000 | 0.500 | 0.969 |

θ=0.8 下 β₀/β₁（primary 复形）在 288/304/320/352/480/496/512/528/544 步的轨迹要点（RCN12_FORGET2/3 显示明显的“连通分量数先降后升再降、H1 先升后降”模式，对应教→忘→复的拓扑变化；RCN12_FORGET 因无遗忘窗口，288 步后 β₀=2/β₁=2 保持不变）：

- FORGET（4 run）：288 步起 (β0,β1) 稳定为 (2,2) 直至 768。
- FORGET2 621_S1：288(7,2)→304(10,0)→320(9,1)→352(6,1)→480(5,1)→496(2,2)→512-544 保持(2,2)。
- FORGET2 622_S2：288(6,2)→304(13,0)→320(10,0)→352(7,1)→480(5,1)→496(2,3)→512-544(2,3)→768(2,2)。
- FORGET3 631_S1：288(2,2)→304(6,1)→320(5,1)→352(3,1)→480(2,2)→496(6,2)→512(3,2)→528-544(2,2)。
- FORGET3 632_S1：288(2,2)→304(8,0)→320(5,1)→352(5,1)→480(2,2)→496(3,2)→512-544(2,3)→768(2,2)。
（完整 12 个 run 的逐 checkpoint β0/β1/β2 见各 `cloud_return_round2/analysis/RCN12_FORGET*/world*/states.csv`。）

## 5. biography BIO2（803/804）/ BIO3（805/806）（`--perm 300`）

BIO 与 BIO2 差异：原 BIO 规格未包含问答训练，问答探针恒为 0（判定继续跑是浪费后终止，改跑 BIO2）；BIO2 起补上问答训练。BIO3（805/806，1536 步，是 BIO2 的 512 步的 3 倍）在此基础上进一步加长训练。

`synchrony.verdict`：

- BIO2（8 臂）：**全部 undetermined**。`t_star`：803_AUG1_S1[416,432]，803_AUG1_S2[0,16]，803_AUG5_S1[416,432]，803_AUG5_S2[0,16]，804_AUG1_S1[496,512]，804_AUG1_S2[320,336]，804_AUG5_S1[416,432]，804_AUG5_S2[480,496]。
- BIO3（8 臂）：**7 个 undetermined + 1 个 synchronous**（本轮唯一 synchronous 判定：`world805_AUG5_S1`，`t_star=[416,432]`）。`t_star`：805_AUG1_S1[384,400]，805_AUG1_S2[688,704]，805_AUG5_S1 **synchronous**[416,432]，805_AUG5_S2[0,16]，806_AUG1_S1[480,496]，806_AUG1_S2[576,592]，806_AUG5_S1[352,368]，806_AUG5_S2[928,944]。

4 条提取边（q_birth_year, q_birth_city, q_employer, q_major）与直达两跳边（q_employer_city_direct）终值 / 首次≥0.5 的步；安慰剂（q_favorite_number）终值；留出迁移（transfer_q_*）终值：

**BIO2**
| run | q_birth_year | q_birth_city | q_employer | q_major | q_employer_city_direct | 安慰剂终值 |
| --- | --- | --- | --- | --- | --- | --- |
| 803_AUG1_S1 | 0.064/None | 0.646/336 | 0.562/464 | 0.549/496 | 0.027/None | 0.000 |
| 803_AUG1_S2 | 0.285/None | 0.652/336 | 0.590/464 | 0.604/400 | 0.012/None | 0.000 |
| 803_AUG5_S1 | 0.023/None | 0.896/336 | 0.930/384 | 0.902/384 | 0.016/None | 0.000 |
| 803_AUG5_S2 | 0.059/None | 0.865/352 | 0.896/368 | 0.910/384 | 0.027/None | 0.000 |
| 804_AUG1_S1 | 0.182/None | 0.727/240 | 0.715/288 | 0.562/400 | 0.027/None | 0.004 |
| 804_AUG1_S2 | 0.180/None | 0.768/256 | 0.797/272 | 0.539/400 | 0.008/None | 0.012 |
| 804_AUG5_S1 | 0.039/None | 0.861/336 | 0.912/368 | 0.691/464 | 0.004/None | 0.000 |
| 804_AUG5_S2 | 0.021/None | 0.887/336 | 0.939/368 | 0.857/400 | 0.016/None | 0.012 |

**BIO3**
| run | q_birth_year | q_birth_city | q_employer | q_major | q_employer_city_direct | 安慰剂终值 |
| --- | --- | --- | --- | --- | --- | --- |
| 805_AUG1_S1 | 0.883/480 | 0.969/256 | 0.916/336 | 0.801/320 | 0.016/None | 0.000 |
| 805_AUG1_S2 | 0.879/464 | 0.979/256 | 0.939/352 | 0.678/400 | 0.008/None | 0.000 |
| 805_AUG5_S1 | 1.000/592 | 1.000/336 | 1.000/352 | 1.000/384 | 0.008/None | 0.000 |
| 805_AUG5_S2 | 1.000/656 | 1.000/336 | 1.000/368 | 1.000/384 | 0.004/None | 0.000 |
| 806_AUG1_S1 | 0.705/624 | 0.975/256 | 0.787/384 | 0.598/400 | 0.016/None | 0.000 |
| 806_AUG1_S2 | 0.633/672 | 0.957/256 | 0.844/368 | 0.627/496 | 0.039/None | 0.000 |
| 806_AUG5_S1 | 0.996/544 | 1.000/304 | 1.000/320 | 0.992/352 | 0.020/None | 0.000 |
| 806_AUG5_S2 | 1.000/576 | 1.000/304 | 1.000/320 | 1.000/368 | 0.016/None | 0.000 |

AUG1 vs AUG5 对比（数字）：AUG5 各 run 的 4 项提取边终值系统性高于同 world/seed 的 AUG1（如 805：AUG5_S1/S2 四项终值均为 1.000，AUG1_S1/S2 为 0.68-0.98）；`q_birth_year` 在 BIO2 全部 8 臂中终值均 <0.29 且首次≥0.5 全为 None（即从未学会），在 BIO3 全部 8 臂中终值 0.633-1.000、首次≥0.5 步 464-672（AUG1）/ 544-656（AUG5），即该字段仅在 BIO3（1536 步，更长训练）才被学会。

直达两跳边 `q_employer_city_direct`：**全部 16 臂（BIO2+BIO3）终值 0.004-0.039，首次≥0.5 全部为 None**——两跳直达捷径在任何 run 中都未涌现。安慰剂（`q_favorite_number`）与留出人物迁移（`transfer_q_*`）在全部 16 臂中终值均 ≤0.062，无假阳性。

## 6. grokking

### 6.1 scan_grok 执行参数扫描（MODP world701，p=97，仅确定执行区域，非确证管线）

3×3 网格（wd × frac），20000 步/格，`grok_threshold=0.9`：

| wd\frac | 0.3 | 0.4 | 0.5 |
| --- | --- | --- | --- |
| 0.1 | 未grok (0.000) | grok@19000 (0.984) | grok@9400 (1.000) |
| 0.5 | grok@8400 (0.998) | grok@3400 (1.000) | grok@2200 (1.000) |
| 1.0 | grok@4200 (1.000) | grok@2400 (0.977) | grok@1400 (1.000) |

### 6.2 各正式运行的 `COMPLETE.json.grok_step_estimate`（首个检查点评测值≥0.9 的步；`None`=该任务在观测步数内未达标）

**MODP（默认预设，30000 步，9 个成功 run；`train_acc` 为末批次训练 token 准确率，`placebo`=P_X 终值）**
| run | train_acc | placebo | 已grok的任务(步) | 未grok的任务 |
| --- | --- | --- | --- | --- |
| 701_seed1 | 0.942 | 0.013 | P_N=300,P_Pn=300,P_Ps=200,P_I=300,P_Pi=300,f_sub=800,f_comm=1000,f_div=800 | P_S,P_D,P_M,Pn_D,Ps_S,Pi_Q,P_Q |
| 701_seed2 | 0.930 | 0.010 | P_N=300,P_Pn=300,P_Ps=200,P_I=300,P_Pi=300,f_sub=1900,f_comm=3300,f_div=2500 | 同上7项 |
| 701_seed3 | 0.929 | 0.010 | P_N=300,P_Pn=300,P_Ps=300,P_I=300,P_Pi=400,f_sub=1900,f_comm=2800,f_div=2200 | 同上7项 |
| 701_seed4 | 0.935 | 0.012 | P_N=300,P_Pn=300,P_Ps=300,P_I=300,P_Pi=300,f_sub=3000,f_comm=3000,f_div=1100 | 同上7项 |
| 701_seed5 | 0.937 | 0.015 | P_N=400,P_Pn=400,P_Ps=300,P_I=400,P_Pi=400,f_sub=2400,f_comm=2400,f_div=1600 | 同上7项 |
| 702_seed1 | 0.891 | 0.011 | P_N=300,P_Pn=400,P_Ps=300,P_I=300,P_Pi=300,f_sub=900,f_comm=4100,f_div=1000 | 同上7项 |
| 702_seed2 | 0.916 | 0.012 | P_N=400,P_Pn=400,P_Ps=300,P_I=400,P_Pi=400,f_sub=2100,f_comm=2100,f_div=1800 | 同上7项 |
| 702_seed3 | 0.953 | 0.010 | **全部15项均grok**：P_S=23600,P_D=23800,P_M=21800,P_N=400,P_Pn=400,Pn_D=23800,P_Ps=300,Ps_S=22800,P_I=400,P_Pi=400,Pi_Q=21800,P_Q=21400,f_sub=1000,f_comm=1200,f_div=1000 | 无 |
| 702_seed4 | 0.927 | 0.008 | P_N=400,P_Pn=400,P_Ps=300,P_I=400,P_Pi=400,f_sub=1800,f_comm=1900,f_div=1800 | 同上7项 |
| 702_seed5 | **无法产出**（训练正常完成，分析流程两次独立卡死，详见第3节修复记录第2/3/8条） | | | |

即：30000 步预算内，`P_S,P_D,P_M,Pn_D,Ps_S,Pi_Q,P_Q` 这 7 个较难任务在 8/9 个可分析 run 中都未 grok，仅 702_seed3 全部 grok。

**MODP classic 预设（100000 步，6 个 run，`--runs-dir runs_classic`）**
| run | train_acc | placebo | 未grok任务 |
| --- | --- | --- | --- |
| 701_seed1 | 0.949 | 0.008 | P_S, P_D |
| 701_seed2 | 0.936 | 0.009 | 无（全部grok，P_S@3000,P_D@7750） |
| 701_seed3 | 0.951 | 0.009 | 无（全部grok，P_S@4500,P_D@11250） |
| 702_seed1 | 0.947 | 0.014 | P_D |
| 702_seed2 | 0.941 | 0.010 | 无（全部grok，P_S@20000,P_D@43000） |
| 702_seed3 | 0.943 | 0.015 | 无（全部grok，P_S@2500,P_D@2750） |

即：延长到 100000 步后，`P_S`/`P_D` 这两个在默认预设下从未 grok 的任务，在 4/6（P_S）与 4/6（P_D）个 classic run 中转为 grok（步数范围 2500-43000）。

**MODP_ADDONLY（10 run，world703/704 × 种子1-5）**：train_acc 0.941-0.992，placebo 0.002-0.015；`P_S` 任务在**全部 10 个 run 中均未 grok**，其余任务无 grok_step_estimate 字段返回（该 spec cell 集合较小）。

### 6.3 zz_analyze 同步判据（`--perm 100`，共 25 个可分析 run：MODP 9 + MODP_classic 6 + MODP_ADDONLY 10）

**全部 25 个 run 的 `synchrony.verdict = undetermined`**（`t_star` 恒为观测窗口第一个 transition：MODP/ADDONLY 为 [0,100]，MODP_classic 为 [0,250]，即预注册的多阈值同步判据在任何 run 中都未被触发；`grok_step_estimate` 反映的是单阈值单 cell 的"曾经跨过 0.9"，两者是不同指标，不应混同解读）。

## 7. 未解决问题

1. **`MODP_world702_seed5` 无法产出 zigzag 分析结果**：训练与逐 checkpoint 评测数据完整（`TRAIN_LOG.jsonl` + 300 个 `eval_*.jsonl.gz`/`.summary.json`），但 `zz_analyze.analyze_run()` 对该 run 数据的计算在远端（两次，累计约 83 分钟）与本地（一次，43 分钟）共三次独立尝试中全部卡死/未完成，具体代码路径未诊断（按规则未修改代码）。可通过其 `eval_*.summary.json` 手动查阅逐 checkpoint 的准确率汇总，但完整的 β₀/β₁/ledger/verdict 缺失。
2. **`grok3_701`/`grok3_702` 首次启动崩溃**报告后已另行修复（改用 `--runs-dir runs_classic`），不构成遗留问题，但记录在案供参考。
3. 报告中的 β₀/β₁ 摘录仅覆盖 FORGET 系列指定的关键步（128/288/352/480/544/768 及相邻教/忘/复步），完整逐 checkpoint 数据见各 `states.csv`；未逐一核对 grokking 25 个 run 的完整 `states.csv`（`--perm 100` 已按第一轮指令的预期使用，避免过长本地计算时间）。
4. 机器仍在线（未执行 `--finish`，未关机），等待后续安排。

## 8. 关键产物路径（本机 `zigzag_v5/` 下）

- 监控日志：`cloud/MONITOR_LOG_R2.md`
- 回收数据：`cloud_return_round2/`（`complex_worlds/runs/`、`biography/runs/`、`grokking/runs/` 与 `runs_classic/`）
- 核验结果：`cloud_return_round2/LOCAL_RETURN_VERIFIED.json`
- 本地分析：
  - `cloud_return_round2/analysis/RCN12_FORGET/`、`RCN12_FORGET2/`、`RCN12_FORGET3/`（`--perm 300`）
  - `cloud_return_round2/analysis/BIO2/`、`BIO3/`（`--perm 300`）
  - `cloud_return_round2/analysis/grokking_MODP/`、`grokking_MODP_classic/`、`grokking_MODP_ADDONLY/`（`--perm 100`）
