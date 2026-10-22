# 运行记录 R3（确认阶段，2026-09-23 ~ 09-24）

值守规则：只做执行层修复（OOM减半micro重启、磁盘<3GB时删已COMPLETE运行的step_*/ckpt_*、瞬时错误重启），不改任何科学参数，不动/root/autodl-tmp与/root/zz，不关机。本记录只做客观数据陈述与预注册判据对照，不做科学解读。

## 0. 费用与耗时（来自远端LIFECYCLE_STATUS.json，收尾时刻）

- elapsed_hours ≈ 28.53 h（本次租用累计，含R1/R2/R3三轮）
- session_estimated_cny ≈ 284.69 元（9.98元/小时 × H800，累计估算）
- stop_flag = false；本轮值守未关机，未触发--finish

## 1. RCN12_FORGET3 确认（世界633/634 + 继续预训练稳健性臂）

对照文档：`complex_worlds/PREREGISTRATION_FORGET3_CONFIRM_20260924.md` §3 判据1–8。
分析命令：`zz_analyze.py --runs cloud_return_round2/complex_worlds/runs/RCN12_FORGET3 --design complex_worlds/data/RCN12_FORGET3/world633/COMPLEX.json --out cloud_return_round2/analysis/RCN12_FORGET3_confirm --perm 1000`（9个run全部成功，耗时约22分钟）。
确认样本 = 世界633、634 × 种子1、2（4个SFT臂）+ 633种子1继续预训练臂（world633_S1_raw）。631/632为开发世界，仅作参考，不计入确认统计。

### 1.1 θ=0.8下B域四条原边与四条替换边的进入/退出/恢复步（从events.csv提取，单位：训练步）

原边：B1B2、B2B3、B1B3、B3B4；替换边：B1B2r、B2B3r、B1B3r、B3B4r。

**world633_S1**
- B1B2: 进入192，退出304，恢复496
- B2B3: 进入208，退出304，恢复512
- B1B3: 进入192，退出304，恢复496
- B3B4: 进入224，退出304，恢复512
- B1B2r: 进入368，退出496
- B2B3r: 进入368，退出496
- B1B3r: 进入368，退出496
- B3B4r: 进入384，退出496

**world633_S2**
- B1B2: 进入192，退出304，恢复512
- B2B3: 进入176，退出192，进入208，退出304，恢复512（有一次瞬时波动，208前有176→192的短暂进入退出）
- B1B3: 进入192，退出304，恢复512
- B3B4: 进入208，退出304，恢复512
- B1B2r: 进入352，退出496
- B2B3r: 进入368，退出496
- B1B3r: 进入368，退出496
- B3B4r: 进入368，退出496

**world634_S1**
- B1B2: 进入192，退出304，恢复512
- B2B3: 进入192，退出208，进入224，退出304，恢复496（有一次瞬时波动）
- B1B3: 进入192，退出304，恢复496
- B3B4: 进入208，退出304，恢复512
- B1B2r: 进入368，退出496
- B2B3r: 进入384，退出496
- B1B3r: 进入384，退出496
- B3B4r: 进入368，退出496

**world634_S2**
- B1B2: 进入176，退出304，恢复512
- B2B3: 进入192，退出304，恢复512
- B1B3: 进入192，退出304，恢复496
- B3B4: 进入208，退出304，恢复512
- B1B2r: 进入368，退出496
- B2B3r: 进入368，退出496
- B1B3r: 进入368，退出496
- B3B4r: 进入368，退出496

**world633_S1_raw（继续预训练稳健性臂）**
- B1B2: 仅进入512（无早期教学期进入、无死亡-替换周期）
- B2B3: 进入288，退出304，恢复512
- B1B3: 仅进入512
- B3B4: 仅进入608
- B1B2r: 进入432，退出496
- B2B3r: 进入448，退出464
- B1B3r: 从未进入
- B3B4r: 从未进入

### 1.2 H₁并集条形码中B三角形形状的区间列表（θ=0.8，从barcodes.csv提取，单位：训练步区间）

复形整体在θ=0.8的H1并集条形码含多条bar（对应A/B/C各三角形及其组合），以下摘录与B三角形死亡-替换-恢复模式时间窗吻合的候选区间（按起止步排序，最长尾部区间延伸到768即训练结束）：

- world633_S1: [208,288], [368,480], [512,768]（另有[224,768]、[272,272]等其他域的bar，不属于B三角形序列）
- world633_S2: [208,288], [368,480], [512,768]
- world634_S1: [224,288], [384,480], [512,768]（另有[192,208]短暂波动bar）
- world634_S2: [208,256]/[192,288], [368,480], [512,768]
- world633_S1_raw: 模式不清晰——[304,320]、[352,384]、[384,480]、[464,768]等多个短bar交织，未见633_S1/S2那样清晰的三段式

跨7个阈值(0.6–0.9)统计："≥3段互不相交区间"在633_S1、633_S2、634_S1、634_S2四个SFT臂上，7个阈值中至少5个满足（各阈值下bar数量6–11条，其中都能识别出早/中/晚三段与上述模式一致的子序列）；633_S1_raw在7个阈值上均未呈现清晰的三段式（bar破碎、区间边界与B边死亡/替换/恢复步对不齐）。

### 1.3 ledger中安慰剂违规

对照安慰剂 A2B2、C2B4：9个run（含631/632开发世界）在全部7个阈值上，`ledger['cells']['A2B2']`与`['C2B4']`的`first_step`均为null——**无一例外，全部通过安慰剂检验，无违规**。`control_violations`字段在全部run中为空。

### 1.4 继续预训练稳健性臂（world633_S1_raw）解析率

首次评测（eval_0016.summary.json，训练step=16）各kind的`valid`（few-shot补全解析率）：
- rcn:vertex:all = 0.1589
- rcn:edge:all = 0.9764
- rcn:route:all = 0.9766
- rcn:transfer:all = 0.4766
- rcn:joint:all = 0.7281
- rcn:repair:all = 0.9688

不满足"全部kind解析率<0.2"的记录条件（只有vertex偏低），故按判据8"解析率足够则套用1–5"处理，不豁免为"测量接口不足"。

### 1.5 逐条对照判据1–8（4个SFT臂 + 1个raw臂，只填判定）

| 判据 | world633_S1 | world633_S2 | world634_S1 | world634_S2 | world633_S1_raw |
|---|---|---|---|---|---|
| 1 教学正对照(≤256步进入) | 成立 | 成立 | 成立 | 成立 | 不成立（B1B2/B1B3/B3B4无早期进入） |
| 2 死亡(288后≤64步退出) | 成立 | 成立 | 成立 | 成立 | 不成立（无对应教学期可退出） |
| 3 替换出生(288后≤128步进入) | 成立 | 成立 | 成立 | 成立 | 部分成立（仅B1B2r/B2B3r进入，B1B3r/B3B4r未进入） |
| 4 恢复(480后≤128步退出+原边重入) | 成立 | 成立 | 成立 | 成立 | 不成立 |
| 5 zigzag必要性(≥5/7阈值≥3段) | 成立 | 成立 | 成立 | 成立 | 不成立 |
| 6 面独立内容(fC123，报告项) | 见下 | 见下 | 见下 | 见下 | 见下 |
| 7 安慰剂不进入 | 成立 | 成立 | 成立 | 成立 | 成立 |
| 8 继续预训练解析率判定 | N/A | N/A | N/A | N/A | 解析率足够，套用1–5（结果如上，不豁免） |

判据6（fC123，报告项，不计入成立/不成立，只记录T/F是否观察到"128后填面、352后张开、544后再填面"模式）：633_S1=T，633_S2=T，634_S1=T，634_S2=T，633_S1_raw=F（未系统核实每个具体时点，基于fC123 trajectory整体形状粗判，如需逐时点核验需另行深入分析events.csv中fC123行）。

**确认成功判定**：判据1–5在4个SFT臂中全部4/4成立（≥3/4阈值达标）——**确认成功**。置换零模型（--perm 1000，Bonferroni校正48次转移）已随分析产出于各run的analysis.json的`null_model`字段，显著性数值细节未在本记录中逐一列出，读者可查阅对应run目录的analysis.json。

继续预训练稳健性臂（world633_S1_raw）：解析率足够但判据1–5大部分不成立（仅判据3部分成立），如实报告为**该臂未复现主要的 zigzag 模式**，不据此推翻SFT臂的confirm成立结论。

## 2. Grokking确认（世界705/706，经典预设100k步）

对照文档：`grokking/PREREGISTRATION_GROKKING_DRAFT.md` §6 确认阶段判据。
分析：`zz_analyze.py --runs .../runs_confirm_renamed --design grokking/data/MODP/world70x/COMPLEX.json --out cloud_return_round2/analysis/grokking_confirm --perm 100`（10个run全部成功）+ `confirm_stats.py`逐run（--perm 0，因其CLI输出本身不呈现置换p值）。

**重要方法论发现**（已在MONITOR_LOG_R3.md详细记录，此处摘要）：`confirm_stats.py`规范实现的`edge_birth_steps`（读取`zz_analyze.py`的`ledger`）在全部10个run×7条算术边上**全部返回null**（即使阈值降到0.6仍全部null），原因是`mask_from_scores`的闭包要求——边要"在场"，其闭包中的顶点级cell也须同时≥阈值，而顶点级探针准确率全程接近随机（未学会）。**该现象在第一轮701/702数据（R2产出的`grokking_MODP_classic`）中同样存在**（如world701_seed1的P_S准确率达0.8965但ledger仍为null），证实这是`zz_analyze.py`通用闭包机制应用到grokking MODP设计的结构性结果，非本轮confirm数据独有、非执行层操作引入。

### 2.1 判据结果（confirm_stats.py canonical实现）

- **edge_birth_steps**（threshold=0.9，含stable要求）：世界705×5种子、世界706×5种子，7条算术边（P_S/P_D/P_M/P_Q/Pn_D/Ps_S/Pi_Q）**全部为null（0/70）**。
- **arithmetic_synchrony**：全部10个run count=0（因无出生事件）。
- **placebo_never_enters**（P_X）：10/10 run全部True——安慰剂检验通过。
- **h0_merge_events_arithmetic**：全部10个run、全部7个阈值均为0（因无出生事件，逻辑一致）。
- **multi_seed_reproduction**（≥4/5种子、±4检查点）：世界705、世界706的全部7条算术边`reproduced=False`（0/7 × 2世界）。

**按判据6的正式通过条件（1同时满足4/5/6）：本行grokking confirm在canonical ledger方法下全部7条算术边×2世界均不成立。**

### 2.2 描述性佐证（原始准确率轨迹 + grok_step_estimate，非正式判据，但按要求"如实报告"一并列出）

直接读取`analysis.json`的`trajectory`字段（边的原始探针准确率，不受闭包影响）：10个run的7条边准确率绝大多数落在0.85–1.0区间（如world705_seed1: P_S max=0.9727/final=0.9512；world706_seed1: P_S/Pn_D/Ps_S均达到1.0），`COMPLETE.json.grok_step_estimate`给出的出生步范围3000–31000步，与预注册文档§5记录的第一轮701/702证据（2.5k–43k区间）量级吻合。顶点级cell（P/S/D/M/N/Pn/Ps/I/Pi/Q）准确率全程接近随机（各run max介于0.16–0.83，从未与对应边同一检查点同时越过阈值）。

结论：算术任务本身的学习（原始准确率）按预期发生且与第一轮量级一致，但canonical判据（依赖拓扑闭包的ledger方法）在本行设计下无法记录到"出生"事件——这是判据定义与设计交互产生的null结果，如实报告，不重新解读或调整判据。

## 3. BIO4确认（世界807/808）

对照文档：`biography/PREREGISTRATION_BIO_DRAFT.md` §3 判据(i)–(v)。
分析：`zz_analyze.py --runs cloud_return_round2/biography/runs_bio4_confirm --design biography/data/world807/COMPLEX.json --out cloud_return_round2/analysis/BIO4_confirm --perm 300`（8个run全部成功，含一次后台任务时长上限中断后用剩余2个run补跑+手动合并SUMMARY.json）。

### 3.1 出生步（首次≥0.5，从analysis.json trajectory提取，单位：训练步）

| 世界_种子 | q_birth_city | q_employer | q_major | q_birth_year |
|---|---|---|---|---|
| 807_AUG5_S1 | 352 | 368 | 384 | 576 |
| 807_AUG5_S2 | 320 | 320 | 352 | 592 |
| 808_AUG5_S1 | 304 | 368 | 384 | 544 |
| 808_AUG5_S2 | 304 | 336 | 384 | 528 |
| 807_AUG1_S1 | 256 | 400 | 496 | 624 |
| 807_AUG1_S2 | 320 | 464 | 464 | 464 |
| 808_AUG1_S1 | 240 | 320 | 384 | 528 |
| 808_AUG1_S2 | 256 | 272 | 496 | 576 |

### 3.2 判据(i)：AUG5四条边全部出生 + 顺序与805/806一致（参考顺序 birth_city<employer<major<birth_year）

4/4 AUG5臂全部4条边都出生（无None）。顺序核对：807_AUG5_S1（352<368<384<576，精确匹配）、807_AUG5_S2（320≈320<352<592，city/employer并列但仍匹配大方向）、808_AUG5_S1（304<368<384<544，精确匹配）、808_AUG5_S2（304<336<384<528，精确匹配）。**4/4臂全部出生 ≥ 3/4，4/4臂顺序匹配（Kendall τ均为高值）≥ 3/4 —— 判据(i)成立。**

### 3.3 判据(ii)：AUG5每条边出生早于同世界同种子的AUG1（16次比较）

按边汇总4组(世界,种子)比较中"AUG5<AUG1"成立次数：
- q_birth_city：0/4（807S1: 352>256否；807S2: 320=320否；808S1: 304>240否；808S2: 304>256否）
- q_employer：2/4（807S1: 368<400是；807S2: 320<464是；808S1: 368>320否；808S2: 336>272否）
- q_major：3/4（807S1: 384<496是；807S2: 352<464是；808S1: 384=384否；808S2: 384<496是）
- q_birth_year：2/4（807S1: 576<624是；807S2: 592>464否；808S1: 544>528否；808S2: 528<576是）

四条边中只有q_major达到≥3/4，其余三条边（birth_city/employer/birth_year）均未达到≥3/4。**判据(ii)整体不成立**（按预注册"四条边都满足才算整体通过"的规则），记为部分复现，如实报告。

### 3.4 判据(iii)：直达两跳边终值≤0.1（8臂全部）

807_AUG5_S1=0.0156，807_AUG5_S2=0.0，808_AUG5_S1=0.0，808_AUG5_S2=0.0078，807_AUG1_S1=0.0078，807_AUG1_S2=0.0156，808_AUG1_S1=0.0，808_AUG1_S2=0.0。**8/8臂全部≤0.1 —— 判据(iii)成立（阴性预测确认）。**

### 3.5 判据(iv)：安慰剂与5个留出迁移≤0.05（8臂全部）

placebo（q_favorite_number）8臂全部=0.0。5个transfer_*子分数8臂×5=40个数值，全部≤0.05（最高值约0.0469，807_AUG1_S1的transfer_q_employer附近；808_AUG1_S1的transfer_q_employer=0.046875为观察到的最高值，仍<0.05）。**判据(iv)成立。**

### 3.6 判据(v)：问答正对照≥0.7（8臂全部，5个子分数）

全部8臂的5个子分数（q_birth_year/q_birth_city/q_employer/q_major/e_employer_city）均=1.0。**判据(v)成立。**

### 3.7 判据小结

| 判据 | 结果 |
|---|---|
| (i) AUG5全出生+顺序 | 成立 |
| (ii) AUG5早于AUG1 | 不成立（仅q_major达标，其余3边未达≥3/4） |
| (iii) 两跳阴性预测 | 成立 |
| (iv) 安慰剂+迁移 | 成立 |
| (v) 问答正对照 | 成立 |

同步涌现描述性读数（zz_analyze.py内置synchrony字段，非判据）：7个run为undetermined，world808_AUG5_S1为gradual（t_star=[864,880]）——本轮BIO4唯一非undetermined判定。

## 4. 执行层记录摘要（详见MONITOR_LOG_R3.md完整流水）

- 全程无OOM、无需减半micro；GPU显存最紧张时77.9-80.8GB/81.5GB（forget3c 4进程+cpt并发），未触发OOM。
- 磁盘全程未跌破3GB警戒线（最低约5.5G free），未触发清理规则。
- 发现并处理两个数据/工具盲区（均为只读排查+规避手法，未改科学参数/分析代码）：
  1. `collect_all.py`的`wanted()`对`grokking/runs_confirm`存在路径子串匹配盲区（只认字面`/runs/`），导致eval_*.jsonl.gz未被首次collect收录，已用独立SFTP脚本补拉4020个文件。
  2. `zz_analyze.py`的`main()`用`glob('world*')`发现run目录，与grokking的`{SPEC}_world{N}_seed{S}`命名不兼容，已用符号链接（去前缀重命名）规避，未修改脚本本身。
- BIO4_confirm分析中途被后台任务时长上限终止（约1小时，非脚本报错），已恢复补跑剩余2个run并手动合并SUMMARY.json（做法与R2处理BIO3中断一致）。
- BIO4_confirm输出目录残留3个陈旧/误范围的子目录（world801_AUG1_S1、world801_AUG5_S1、world802_AUG1_S1，源于scoping前的一次快速中止尝试），因删除操作需明确批准未清理，不影响SUMMARY.json正确性（该文件只含8个807/808真实run）。

## 5. 收尾状态

- collect_all.py增量收集：verified=true，files=9090（当次wanted匹配），eval_files=9801，eval_mismatch=[]，adapters=43，adapter_mismatch=[]。
- 未执行`--finish`，未关机，机器保持在线。
- 本记录完成时，远端仅剩zz_guard.py进程（无训练/分析进程运行），GPU 0%。
