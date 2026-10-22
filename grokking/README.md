# 模运算群复形 + grokking（方案 A）

目录：`zigzag_v5/grokking/`。对应 `GAP_ANALYSIS_AND_PLANS_20260923.md` 方案 A：用从零训练的小
Transformer 在模 p 运算上制造真实的 grokking（记忆先行、留出泛化滞后跃迁），把"留出准确率跨阈值"
当作候选拓扑涌现（定义 3.3 ii）的正例来源，而不是像 RCN12 那样只测试"结构上从未训练的边"。

- Python 3.12（torch 2.5 CPU、numpy）。
- 不修改 `zigzag_v5/` 下其他目录的任何文件；`crosslingual/code/zz_analyze.py`、`cw_zigzag.py`、
  `stage_ab/code/zz_score.py` 均按现状直接 `import` 使用。

## 目录结构

```
grokking/
  specs/MODP.json              # 主规格：p=97(世界701)/113(世界702)，所有真实运算 taught
  specs/MODP_LATENTSUB.json    # 变体：P_D 与 f_sub 结构性 latent（从不训练）
  specs/MODP_ADDONLY.json      # 最小规格：只有 P_S(加法)+P_X(安慰剂)，世界703/704，配 --preset classic
  code/
    modp_common.py             # 复形定义、算术、token 化、行构造、划分算法（唯一权威实现）
    make_world.py               # spec -> TRAIN_*.jsonl / PROBES.jsonl / COMPLEX.json / GROUND_TRUTH.json / MANIFEST.json
    tiny_transformer.py         # 1~2 层 causal Transformer（从零训练，无预训练权重）
    run_grok.py                  # 训练 + 周期性评测，写 eval_{step}.jsonl(.summary.json)
    face_scores.py               # 面的路线一致性分数（现为主读出，见"等式捷径"一节）
    queue_grok.py                 # 世界701/702/705/706 × 种子1-5 队列，写 COMPLETE.json
    scan_grok.py                   # GPU 日超参扫描（仅加法，wd×frac 网格），只定参数不判定
    confirm_stats.py                # 确认阶段判据统计量（边出生/同步/安慰剂/H0合并/跨种子复现）
    test_grokking.py                 # 全部断言测试，见下
  data/<SPEC>/world<W>/          # 生成的数据（见"数据"一节）
  runs/                            # 训练输出（含 CPU dryrun 证据 runs/dryrun_test、runs/classic_dryrun_test）
```

## 复形

11 个顶点：10 个是有序对 `(a,b) mod p` 的"表示"，第 11 个 `X` 是安慰剂边 `P_X` 唯一的落点
（纯粹的控制顶点，不代表任何真实运算）：

| 顶点 | 含义 | 是否二元组 |
| --- | --- | --- |
| P | (a,b) | 是 |
| S | a+b | 否 |
| D | a−b | 否 |
| M | a·b | 否 |
| N | −b | 否 |
| Pn | (a,−b) | 是 |
| Ps | (b,a) | 是 |
| I | b⁻¹（b≠0） | 否 |
| Pi | (a,b⁻¹) | 是 |
| Q | a·b⁻¹ | 否 |
| X | 固定查表值（与 (a,b) 独立均匀随机） | 否 |

13 条边（`task` 名即边名）：12 条真实运算 + 1 条安慰剂 `P_X`。

| 边 | src→dst | 运算符 token | 备注 |
| --- | --- | --- | --- |
| P_S | P→S | `+` | |
| P_D | P→D | `-` | |
| P_M | P→M | `*` | |
| P_N | P→N | `NEG` | 只用第二个分量 |
| P_Pn | P→Pn | `MKNEG` | 构造 (a,−b) |
| Pn_D | Pn→D | `+` | 对 (a,−b) 做加法 = a−b |
| P_Ps | P→Ps | `MKSWAP` | 构造 (b,a) |
| Ps_S | Ps→S | `+` | 对 (b,a) 做加法 = a+b |
| P_I | P→I | `INV` | b≠0 |
| P_Pi | P→Pi | `MKINV` | 构造 (a,b⁻¹)，b≠0 |
| Pi_Q | Pi→Q | `*` | 对 (a,b⁻¹) 做乘法 = a/b |
| P_Q | P→Q | `/` | 直接除法 |
| P_X | P→X | `X` | **安慰剂**：固定查表，与 (a,b) 独立均匀随机，见下 |

`+`、`-`、`*` 等运算符 token 在不同边之间**复用**（P_S 与 Pn_D 都用 `+`，P_Ps/Ps_S 的加法同理，
P_M 与 Pi_Q 都用 `*`）：这是有意为之——Pn_D/Ps_S/Pi_Q 本质上是把同一个"加法/乘法"原语作用在一个
经过变换的输入对上，模型如果真的学到了 mod-p 加法/乘法这个函数，就该在这些边上立即泛化；如果只是
背对子，则不会。这正是"留出对泛化 = grokking"要检验的东西。

3 个面（等式恒等式），`face_readout="joint"`：

| 面 | 恒等式 | 三角形（cycle，次级路线读数用） |
| --- | --- | --- |
| f_sub | a−b = a+(−b) | P_D, P_Pn, Pn_D |
| f_comm | a+b = b+a | P_S, P_Ps, Ps_S |
| f_div | a/b = a·b⁻¹ | P_Q, P_Pi, Pi_Q |

### 安慰剂 P_X：为什么是 P→X 的普通边、为什么不能是自环

P_X 的语义是"对 (a,b) 的一个与真实运算无关的固定查表"（`modp_common.placebo_view`：对每个 (a,b)
独立抽取一个 Uniform(0,p) 的数字，表存在 `MANIFEST.json.placebo_table_len`，由
`placebo_seed_offset+world` 确定性生成）。这样对任何固定的模型函数 f，留出准确率的期望严格等于
1/p（标签与输入独立），是最干净、最贴近题意（"留出正确率必须 ≈ 1/p"）的负对照实现。

**P_X 的陪域是新增的第 11 个顶点 `X`，边是普通的 `P→X`，不是自环。** 最初的实现图省事，把 P_X 声明
成 `P→P` 的自环以凑"顶点恰好 10 个"，但这违反了正则 CW 复形的约束（不允许单顶点自环），而且更严重
的是：一条自环边一旦（哪怕虚假地）进入复形，在 `cw_zigzag` 的 F2 无符号边界算法下会凭空产生一个新
的 H₁ 生成元——这恰好摧毁了安慰剂"永远不该改变拓扑"这个角色本身。改成真正的 `X` 顶点后：

- `X` 只被 `P_X` 这一条边触碰到，在图论意义上是一枚"悬挂叶子"（pendant leaf）；
- `GROUND_TRUTH.json` 的 `present` 集合是**闭包驱动**的（见下"手算 Betti"）：一个顶点只有在某个
  present 的边/面真正连到它时才会被计入，X 因此在除 `vertices_only` 外的每一个训练相关阶段里都
  天然缺席（`P_X` 恒为 `control`，从不进入 `present`）；
- 新增了一条显式断言（`make_world.ground_truth` 里的 `placebo_entered_no_betti_change`）：**强行**把
  `P_X`/`X` 塞进一个已经闭合的 present 集合，Betti 数必须不变——因为悬挂叶子既不会合并/拆分连通
  分量（β₀ 不变），也不会打开/封住任何环（β₁ 不变）。这不是训练结果，而是复形本身的结构保证，正是
  `P_X` 能充当"永远不该移动拓扑"的负对照的前提。`test_grokking.py` 对 MODP、MODP_LATENTSUB 两个规格
  都独立核对了这条断言。
- `X` 同时也有自己的读出探针 `vX`（128 项 × 2 模板，与其余顶点一样，用于健全性检查）。

### 手算 Betti（README 要求的"手算"部分）

`GROUND_TRUTH.json` 的 `present` 集合是**闭包驱动**的：一个顶点只有在某个 present 的边/面确实连到
它时才被计入（见 `make_world.ground_truth` 里的 `present()` 辅助函数），唯一的例外是专门的
`vertices_only` 基线（"只有全部 11 个声明的顶点，没有任何边"）。这避免了"复形里声明了 11 个顶点，但
只有 10 个会被任何一条 taught/latent 边碰到"这一事实，把 `X` 错误地当成一个永久孤立分量算进每个阶段
的 β₀ 里。

**MODP（全部 12 条真实边 taught，P_X 恒为 control，不进入 present 集合）：**

- `vertices_only`：11 个声明的顶点（10 个真实表示 + 控制顶点 `X`）互不相连 ⇒
  `[β0,β1] = [11, 0]`。
- 12 条真实边中，9 条从 P 直接发出（P_S,P_D,P_M,P_N,P_Pn,P_Ps,P_I,P_Pi,P_Q），另 3 条
  (Pn_D, Ps_S, Pi_Q) 把 Pn/Ps/Pi 各自连回 S/S/Q。P 与其余全部 9 个**真实**顶点直接相邻 ⇒ 这 10 个
  顶点构成的子图连通，`β0=1`；`X` 没有任何 taught 边碰到它，不在 present 集合里，不影响这个 β0。
  独立环数 = E−V+1 = 12−10+1 = **3**。这三个环恰好就是三个三角形 f_sub(P,D,Pn)、f_comm(P,S,Ps)、
  f_div(P,Q,Pi)；M、N 是只有一条边的叶子，不参与任何环。
  ⇒ `taught_no_faces = [1, 3]`（**"教过的边全部存在但面未填时 β1 应等于独立环数"，即 3，核对通过**）。
- 三个面各自把所在三角形的一个独立环"封住"（每个 2-cell 恰好杀掉一个 β1 的生成元，互不干扰）：
  `taught_with_faces = [1, 0]`（**"面填后减 3"，3→0，核对通过**）。
- **安慰剂不改变拓扑**：把 `P_X`（和它唯一连到的 `X`）强行加进 `taught_with_faces` 的 present 集合
  里，`X` 相对这个已经连通的子图只是一枚悬挂叶子（degree 1，只连一条边）——加一枚叶子既不会合并/
  拆分连通分量（β0 不变），也不会打开/封住任何环（β1 不变）。断言 `placebo_entered_no_betti_change`
  核实 `before == after == [1, 0]`。这是本次修正新增的检查，专门验证"安慰剂进入不改变 Betti 数"。

以上由 `make_world.ground_truth()` 用 `cw_zigzag.CWComplex` 实际计算并用 `assert` 与 `specs/MODP.json`
的 `expected_topology` 核对（`test_grokking.py` 独立复算一遍，避免"生成器与断言用同一份错误公式"）。

**MODP_LATENTSUB（P_D、f_sub 结构性 latent，从不进入训练流；`vertices_only` 同样是 [11, 0]）：**

- 只有 11 条真实边被训练（少了 P_D），D 只能经 Pn_D 到达，图仍连通 `β0=1`；独立环数
  = 11−10+1 = **2**（只剩 f_comm、f_div 两个三角形；f_sub 的三角形退化成一条开路径 P–Pn–D）。
  ⇒ `taught_no_faces = [1, 2]`。
- 训练 f_comm、f_div 两个面后：`taught_with_faces = [1, 0]`。
- 现在把结构性 latent 边 P_D **加回来**（模型如果能组合 P_Pn+Pn_D 隐式推出 a−b，这条边就该"出生"），
  但先不加 f_sub 面：V=10（真实顶点，`X` 仍不在 present 里）,E=12，2-cell 仍只有 2 个。独立环数变回
  12−10+1=3，其中 2 个已被封住，**新开一个环**：`plus_latent_edge_only = [1, 1]`。这就是一次真实的
  "边先出生、环先张开"事件——哪怕它是"结构性从未训练"而不是"留出泛化"的 latent，其拓扑效果与
  zigzag 想捕捉的"β1 先升后降"完全一致。
- 再把 f_sub 面加上：`plus_latent_edge_and_face = [1, 0]`，环闭合。同样验证了
  `placebo_entered_no_betti_change`（基线取 `plus_latent_edge_and_face`）：把 P_X/X 加进去，
  `[1,0] -> [1,0]`，不变。

以上六个数字（含两条 `placebo_entered_no_betti_change`）都由
`test_grokking.py::test_ground_truth_betti` 独立于生成器再断言一遍。

## 数据划分规则

对每个世界，先把 `Zp × Zp` 的全部 p² 个有序对按下列三个对合（involution）计算**轨道**（orbit）：
`neg(a,b)=(a,−b)`、`swap(a,b)=(b,a)`、`inv(a,b)=(a,b⁻¹)`（b=0 时视为不动点）。用种子
`split_seed_offset(900000)+world` 打乱轨道顺序，累积到 ≈`split_frac`(0.4) 的比例为止归入训练集，
其余归入留出集——**整条轨道一起分配**，因此天然保证：若 (a,b) 落在留出集，则 (a,−b)、(b,a)、
(a,b⁻¹)（b≠0 时）也都在留出集（`modp_common.build_split`，测试见
`test_grokking.py::test_split_closure`）。所有 13 条边、3 个面共用同一个划分。

这个闭包性质正是 Pn_D / Ps_S / Pi_Q / f_sub / f_comm / f_div 不能"抄近道"的原因：把它们的输入
（变换后的数值对）映射回原始索引后，训练集和留出集在**变换前后都是同一个集合**，模型必须真正学会
底层的 mod-p 运算，而不是通过"换一个门路撞见训练过的数值对"来作弊。

需要 b≠0（I、Pi、Q 相关的边与 f_div）额外过滤 b=0 的对；f_div 的"错误对照"分支还需要 a≠0（否则
`a·m` 对任意 m 都等于 0，等式判断会失真），已在 `make_world.filtered_pool` 处理。

## 训练器与超参

- 词表（`modp_common.build_vocab`）：数字 token `0..p-1`、12 个符号 token
  （`+ - * / NEG INV MKNEG MKSWAP MKINV X ?= =`）、11 个顶点标签 token（`[vP]...[vQ][vX]`）、1 个 pad。
  P=97 时词表大小 121，P=113 时 137。
- 模型（`tiny_transformer.py`）：1~2 层 causal self-attention（`torch.nn.functional.
  scaled_dot_product_attention(is_causal=True)`），d_model=128，4 头，MLP 512，均可通过 CLI 参数化。
  序列右侧补 pad；由于是因果注意力，pad 永远不会影响任何真实位置的表示，训练/推理都不需要额外的
  padding mask。
- 每一行的 prompt/answer 都来自同一套 `modp_common.encode_row`：prompt 取自 `row['prompt']`
  （空格分隔），answer 取自 `row['gold']`（逗号分隔，1 个 token 给单数字/面判定答案，2 个 token 给
  二元组答案如 P_Pn 的结果）。训练时只在 answer 对应的位置算交叉熵；评测时按 (prompt_len,
  answer_len) 分组做无 padding 的批量贪心解码（`run_grok.evaluate`）。
- 优化：AdamW，lr=1e-3，betas=(0.9,0.98)，weight_decay 可调（`--wd`，默认 1.0）。每步从**所有教过的
  边/面任务中均匀选任务、再从该任务训练池里均匀采样**组成 batch（默认 512，`run_grok.build_batches`）。
- 默认 30000 步，每 100 步评测全部探针并写 `eval_{step:05d}.jsonl` + `.summary.json`
  （`zz_score.summarize` 结果，另外并入 `face_scores.merge_face_cross_check` 的次级面路线分数）；每
  1000 步存一次 checkpoint（`ckpt_{step:05d}.pt`）与 `resume.pt`；`--resume` 从 `resume.pt` 续跑。
- `--dryrun`（等价于 `--dryrun-p 13 --steps 300 --eval-every 50 --ckpt-every 150`，除非显式覆盖）在
  CPU 上跑通，见"测试"一节的实测数字。
- 支持的 CLI：`--seed --steps --wd --frac --layers --d-model --n-heads --d-ff --batch-size
  --preset --only-tasks --resume --cpu`。`--frac` 会用 `make_world.build_data` 在内存里按新的
  `split_frac` **重新计算**划分/训练池/探针（不依赖磁盘上按 0.4 生成的规范数据），因此 `scan_grok.py`
  可以直接扫 frac∈{0.3,0.4,0.5} 而不需要为每个网格点单独跑 `make_world.py`。`--preset classic` 见下
  "最小规格与经典设置"一节；除 `--dryrun` 外，任何显式传入的超参数 flag 都会覆盖预设值。

## 最小规格与经典设置：MODP_ADDONLY + --preset classic

`specs/MODP_ADDONLY.json` 是从完整 MODP 复形里"抽出一根边"得到的最小规格：只保留 3 个顶点
`P、S、X`，2 条边 `P_S`（加法，taught）与 `P_X`（安慰剂，control），**没有面**。这正是 grokking
文献（Power et al. 2022 的模加法表、Nanda et al. 2023 的机制可解释性分析）实际使用的设置——单一
二元运算的完整查找表，全批梯度下降，长训练。把它和完整的 13 边/3 面 MODP 复形分开跑，既能用极低的
GPU 成本快速确认"这套 tokenizer/模型/训练循环在文献已知会 grok 的最简单场景下确实会 grok"，作为
排查用的最小反例/正例，又不用一开始就在整个复杂复形上烧显卡。

- 数据生成规则与 MODP 完全一致（40% 训练对，留出探针 512 对 × 2 模板，顶点读出），只是复形范围通过
  spec 里的 `"cell_subset": {"vertices": [...], "edges": [...], "faces": []}` 限定
  （`make_world.py` 新增的 `spec_vertex_names/spec_edges/spec_faces/spec_transfer_tasks` 辅助函数，
  MODP/MODP_LATENTSUB 不声明 `cell_subset` 时行为完全不变，仍使用 `modp_common` 里全部 11/13/3 个
  cell）。`transfer_tasks` 显式置空——迁移任务需要 `+ - *` 都学过，这里只训练了加法，没有意义。
- **顶点在场约定不同于 MODP**：`"vertex_present_mode": "blanket"`——3 个顶点太少，控制顶点 `X`
  是否被安慰剂边连通，本身就是要观察的量，而不是像 11 顶点的 MODP 里那样应该被隐藏掉的噪声
  （对比 README"复形"一节里 MODP 的闭包驱动 present 约定）。手算：
  - `vertices_only`：3 个顶点互不相连 ⇒ `[3, 0]`。
  - `taught_no_faces`（只有 `P_S` 在场，`P_X` 恒为 control）：`{P,S}` 连通成 1 个分量，`X`
    因为 blanket 模式仍然"在场"但没有任何边碰到它，独自成 1 个分量 ⇒ `β0=2, β1=0` ⇒ `[2, 0]`
    （没有面，`taught_with_faces` 数值相同）。
  - `all_present`（`P_S` 与 `P_X` 都强行加入）：P 同时是两条边的枢纽，`P-S-X` 三点连通成 1 个分量；
    E=2,V=3 ⇒ 独立环数 = 2−3+1=0 ⇒ `[1, 0]`。**这里"安慰剂进入"确实会改变 Betti 数**（`X` 从孤立
    点变成被并入主分量），是本规格特有的、有信息量的现象，因此 `make_world.ground_truth()` 对
    `vertex_present_mode=blanket` 的规格**不**做"安慰剂不改变拓扑"的不变量断言，而是把 `all_present`
    作为一个普通的、需要与 `expected_topology` 核对的阶段（`test_grokking.py::test_ground_truth_betti`
    有对应检查）。
- 已生成世界 703（p=97）、704（p=113）到 `data/MODP_ADDONLY/`。

`run_grok.py` / `queue_grok.py` 新增 `--preset classic`：1 层、d_model=128、4 头、d_ff=512、
AdamW（lr=1e-3, betas=(0.9,0.98)）、weight_decay=1.0、steps=50000、eval_every=200、ckpt_every=5000，
以及**全批训练**——每一步都用当前 spec/world 下*全部*教过任务的训练对（不是每步随机采样 512 条）。
全批张量在训练开始前只编码一次（`run_grok.make_full_batch`），之后每步原样复用，不产生额外开销。
任何显式传入的 CLI flag（`--layers`、`--wd`、`--batch-size`……）仍然覆盖预设值；`--dryrun` 的步数/
评测频率（300/50/150）优先级高于预设的 50000/200/5000，方便在 CPU 上快速验证预设本身接好了没有。

真正的"全批"只在训练池足够小时才现实：`run_grok.FULL_BATCH_CAP = 8192`——MODP_ADDONLY 单条边
的训练池只有 ~3800（p=97）/~5100（p=113）条，明显在这个上限之内，`--preset classic` 会把
`batch_size` 设成训练池的**确切大小**（`RUN_CONFIG.json` 里 `full_batch: true`）。完整 13 边/3 面
的 MODP 复形若也用 `--preset classic`，全部教过任务的训练池总和约 **71,320** 条（p=97 世界实测），
远超 8192，**按指示自动退化为 `batch_size=2048`** 的普通随机小批采样（`full_batch: false`，
`RUN_CONFIG.json.full_batch_note` 会记录这次回退和触发它的池大小）。这个判断是按"当次 spec/world/
`--only-tasks` 实际的训练池大小"动态做的，不是按 spec 名字硬编码，所以同一套代码对任何未来的规格都
适用。

```bash
python code/make_world.py --spec MODP_ADDONLY --worlds 703,704
python code/run_grok.py --spec MODP_ADDONLY --world 703 --seed 1 --preset classic
python code/queue_grok.py --spec MODP_ADDONLY --worlds 703,704 --seeds 1,2,3,4,5 --preset classic
```

CPU dryrun 实测（p=13，300 步，`--preset classic`，`test_grokking.py::test_classic_preset_dryrun`）：
训练池只有 168 条（P_S 84 + P_X 84），`full_batch=true`，300 步内训练 token 准确率就到 100%（单一
运算表 + 全批 + 1 层模型，收敛非常快，同一份对照也印证了"记忆先行"这一半 grokking 现象在此设置下
确实容易触发；留出泛化——grokking 的另一半——要看真正的 GPU 长训练）。

## 评测文件格式

`eval_{step:05d}.jsonl.gz`（gzip 压缩，`prompt` 字段被去掉以省空间——评分所需的字段已经在写盘前
算完，`prompt` 只是原始题面文本，重跑评测不需要它）：每行 = 一条 PROBES.jsonl 行 + `prediction`
（渲染后的答案字符串）+ `zz_score.score()` 的 4 个字段（`correct/valid_format/field_correct/
missing_fields/extra_fields`）。`crosslingual/code/zz_analyze.py` 的 `rows()`/`load_run()` 已原生
支持 `.gz`（与 `.jsonl` 混合都可以），不需要额外处理；这是磁盘空间相关的独立改动，与下面"面读出"
一节讨论的分析器兼容性无关。`eval_{step:05d}.summary.json`：`zz_score.summarize()` 的分组统计
（`modp:edge:all`、`modp:edge:taskP_S` 等），额外并入 `modp:face:taskf_sub` 等**次级**路线交叉校验
分数（见下）。

`COMPLEX.json` 的字段完全遵循 `complex_worlds/code/spec_world.py` 的范式（`cells / cell_kind /
latent_cells / control_cells / taught_edges / faces / transfer_tasks / main_templates`），额外带
`vocabulary: "modp"`、`face_readout: "routes"`（见下"等式捷径"一节——**已从 `"joint"` 改回
`"routes"`**）、`face_routes`（现在是主读出用的路线定义）。可以直接：

```bash
python crosslingual/code/zz_analyze.py --runs grokking/runs --design grokking/data/MODP/world701/COMPLEX.json --out <out>
```

或在代码里 `zz_analyze.analyze_run(run_dir, 'grokking/data/MODP/world701/COMPLEX.json', n_perm=1000)`。

### 等式捷径：为什么面读出从"等式比较任务"改回"路线一致性"

**最初的设计**（见 GAP_ANALYSIS 方案 A）：面分数 = 同一 (a,b) 上，两条路径各自的单字段边探针是否都
答对（如 f_sub 需要 `P_D(a,b)` 与 `Pn_D(P_Pn(a,b))` 都对）。因为对**确定性**的模运算，这个定义看起来
"闭包平凡"——一旦 P_D、P_Pn、Pn_D 三条边各自 grok，面分数似乎会自动跟着变成 1，不会有独立于边的 H1
事件——所以中途一度改成了**等式比较任务**：面本身是一个独立训练、独立留出评测的判断题，例如 f_sub 的
prompt 是 `a - b ?= a + n =`，`n=−b` 时 gold=1、否则 gold=0，`kind='joint'`、`override` 标出正确/
错误分支，`zz_analyze.design_from_spec` 的 `cells_fn` 对 `dim==2` 的 cell 本来就是
`min(acc(override=False), acc(override=True))`，当时认为这样完全不用改分析器。

**GPU 证据推翻了这个假设**（701/702，`--preset classic`，10 万步，6 次运行）：7 条算术边多数会
grok，但跨越步数很晚也很分散——P_M 4k–11k，P_Q 3.5k–16.75k，P_S 2.5k–20k，P_D 2.75k–43k（少数种子
未 grok）；world702 种子3 上 7 条算术边罕见地几乎同步，集中在 2.25k–4.25k 步。**但三个等式比较面
f_sub/f_comm/f_div 在全部 6 次运行里都在 500–2750 步就越过 0.9**——**比任何一条算术边都早**。原因是
这个判断题本身有捷径：`a − b ?= a + n` 为真当且仅当 `n = −b mod p`，模型只需要学会"取负"（`P_N` 这类
单元素运算，容易且快，实测约 200–400 步就会），**完全不需要学会减法**就能把比较题答对——等式比较
任务从来没有真正测到"这条边/这个恒等式本身"，只是测到了"取负"这个更简单的子技能，恰好足以让判断题
蒙对。这正是最初"闭包平凡"担忧的镜像版本：不是"面因为依赖边所以太晚"，而是"面绕开了边所以太早"——
两种失败模式都说明"面的分数应该独立反映它声称在测的那件事"这条设计原则很容易被违反，需要用真实训练
结果来检验，不能只靠推理。

**因此改回路线一致性**（`face_readout: "routes"`，`code/face_scores.py:route_consistency` 是权威实现）：
面分数 = 同一原始 (a,b) 上，`face_routes[面名]` 列出的**全部**边都答对（如 f_sub 要求 `P_D`、`P_Pn`、
`Pn_D` 三条边，在同一个 `item` 上全部正确）。每条探针行本来就带 `item` 字段（=原始 `[a,b]`，哪怕该行
实际输入的是变换后的数值，如 `Pn_D` 探测的是 `(a,−b)` 但 `item` 仍写 `[a,b]`），所以这个分数不需要
新的探针格式，`face_scores.route_consistency`/`merge_face_cross_check` 从最初就实现好了（当时是作为
"次级交叉校验"）。等式比较任务**保留不变**——继续训练、继续留出探针——但降级为次级、描述性的读数：
它现在能明确地展示"捷径何时被学会"（200–400 步），与"面真正对应的恒等式路线何时被学会"（理论上应
接近其所依赖的算术边）分开报告，本身就是一个有信息量的对照。

**分析器需要的扩展说明（这次是真的需要）**：`crosslingual/code/zz_analyze.py` 的 `design_from_spec`
从未读取过 `face_readout` 字段——它对 `dim==2` cell 硬编码走 `min(acc(override=False),
acc(override=True))`，也就是**仍然**在用等式比较任务的分数去驱动 `analyze_run()` 里面所有
基于面的 Betti/barcode/ledger 计算，哪怕 `COMPLEX.json` 已经声明 `face_readout: "routes"`。目前的
折中方案：
- **边/顶点**的 Betti、条形码、`event_ledger`：完全不受影响，仍然正确（`design_from_spec` 对
  `dim<=1` 的 cell 从来就是按 `task` 名直接算 accuracy，这条路径本来就没问题）。
- **面**的分数：`run_grok.py` 每个 checkpoint 已经把 `face_scores.merge_face_cross_check` 的结果并入
  `eval_*.summary.json`（键名 `modp:face:task<name>`）——这是**正确的**路线一致性分数，但它只进了
  summary.json，没有进 `analyze_run()` 自己算的 `trajectory`/`per_theta`/`ledger`（那些仍然用的是
  design_from_spec 内置的等式任务分数）。
- 本线的**确认阶段判据**（见 `PREREGISTRATION_GROKKING_DRAFT.md`）已经把"主事件"限定在**边**上
  （含 7 条算术边）+ H₀ 合并事件，不再依赖面的 Betti 数字做主判定，所以这个缺口不阻塞确认阶段的核心
  结论；但任何引用"面几时出生"的描述性结论，必须读 `modp:face:task<name>`（或直接调用
  `face_scores.route_consistency`），不能读 `analyze_run()` 的面轨迹/ledger。
- 真正修好这个缺口需要修改 `design_from_spec` 的 `cells_fn`（`dim==2` 分支改成对
  `spec.get('face_readout')=='routes'` 时改用路线一致性，否则保留原逻辑），这仍然是
  `crosslingual/code/zz_analyze.py` 里的改动，按本仓库的约束不由这条线的代码来改；已在
  `code/face_scores.py`、本节把需要改的确切位置和逻辑写清楚，供后续改分析器时直接对照实现。

## GPU 结果（第一轮）与确认阶段准备

第一轮 GPU 结果（701/702，`--preset classic`，10 万步，种子 1–3，共 6 次运行）：7 条算术边
（`P_S P_D P_M P_Q Pn_D Ps_S Pi_Q`，见下"算术边"定义）在多数运行里都 grok 了，但步数很晚、跨度很大
——P_M 4k–11k，P_Q 3.5k–16.75k，P_S 2.5k–20k，P_D 2.75k–43k 步（个别种子未 grok，记为 None）；
world702 种子3 是个例外，7 条算术边罕见地几乎同步集中在 2.25k–4.25k 步。等式比较面的捷径问题见上
"等式捷径"一节。另外，`MODP_ADDONLY`（**默认预设**，即没有加 `--preset classic`，30000 步，含安慰剂
对照）跑了 10 次，**全部未 grok**——这与该规格本来就是为搭配 `--preset classic`（1 层、全批、
wd=1.0、50000 步，文献设置）设计的这一点一致：默认预设是 2 层、`batch_size=512` 随机小批、且步数
更少，本就不是这个最小规格的目标配置；如果要用 `MODP_ADDONLY` 复核"这套代码在文献已知会 grok 的
设置下确实会 grok"，应该带 `--preset classic` 跑，而不是默认预设。

据此新增两个**确认世界** `MODP` 705（p=101）、706（p=109）——生成规则与 701/702 完全一致（同一个
`specs/MODP.json`，只是 `primes` 字典里多了这两个 world→p 映射；`split_seed_offset`/
`placebo_seed_offset` 仍然是 `offset+world`，world 号本身已经保证 705/706 用的是与 701/702 不同、
但同样确定性可复现的划分/安慰剂表种子），**不改动、不重新生成 701–704 的任何数据**（`COMPLEX.json`
的 `face_readout` 字段改动例外，见上，只改了 701/702 两个文件本身，其余 701/702/703/704 的
`TRAIN_*`/`PROBES`/`GROUND_TRUTH`/`MANIFEST` 一概未动，用 `MANIFEST.json.p` 核对过）。

新增 `code/confirm_stats.py`，把确认阶段判据（见 `PREREGISTRATION_GROKKING_DRAFT.md` "确认阶段判据
(GPU 证据后修订)"）实现成可直接跑的统计量，全部构建在**未修改**的 `zz_analyze.analyze_run()` 输出
之上（用它已经算好的 `ledger`/`per_theta[...]['transitions']`，不重新发明阈值/稳定性逻辑，也不用
碰 `zz_analyze.py`）：

- `ARITHMETIC_EDGES = ('P_S','P_D','P_M','P_Q','Pn_D','Ps_S','Pi_Q')`——7 条"算术边"，即真正需要
  算出正确数值才能答对的边（4 条直接运算 + 3 条"在变换后的数对上复用同一运算"的组合边）；其余真实边
  （`P_N P_Pn P_Ps P_I P_Pi`）是取负/交换/求逆/构造这类更简单的单元素或构造操作，不计入主事件集合，
  仍会在 `edge_births`/描述性统计里报告。
- `edge_birth_steps(res)`：每条边的"出生步"= `ledger` 里首次 ≥ 阈值且在下一检查点仍然稳定的那一步
  （不稳定的越阈直接算未出生）。
- `arithmetic_synchrony(res, tol_checkpoints=4)`：同步统计量——对每个观测到的检查点，数有多少条算术边
  的出生步落在 `±4 个检查点`（=该次跑的 `eval_every` 决定的步数，不是硬编码的 1000；`eval_every=250`
  时刚好是 ±1000 步）以内，取计数最大的检查点。
- `placebo_never_enters(res)`：安慰剂 `P_X` 在给定阈值下必须从未"出生"，否则判定失败。
- `h0_merge_events(res, cells=ARITHMETIC_EDGES)`：从 `per_theta[...]['transitions']` 里筛出
  `cw_zigzag.classify_h0_change` 标为 `'merge'` 且新增边属于算术边集合的那些转移（H₀ 合并事件）。
- `multi_seed_reproduction(births_by_seed, min_seed_frac=0.8, tol_checkpoints=4, eval_every=...)`：
  跨种子复现——一条边至少在 4/5 的种子里"出生"，且这些出生步彼此都落在 ±4 个检查点以内，才算复现。

```bash
python code/confirm_stats.py grokking/runs/MODP_world705_seed1 grokking/data/MODP/world705/COMPLEX.json --threshold 0.9 --tol-checkpoints 4 --perm 1000
```

`code/test_grokking.py::test_confirm_stats` 用手造的（非真实训练产出的）`ledger`/`per_theta` 结构
对上述 5 个函数做了确定性单元测试（出生/未出生/不稳定、同步窗口计数、安慰剂通过/失败、H0 合并事件
过滤、跨种子复现判定），不依赖任何一次真实 GPU 训练的结果。

## GPU 日命令

```bash
# 1) 数据已在本仓库生成好（下方"数据"一节的计数），无需在 GPU 机上重跑；如需重跑：
python code/make_world.py --spec MODP --worlds 701,702,705,706
python code/make_world.py --spec MODP_LATENTSUB --worlds 701,702
python code/make_world.py --spec MODP_ADDONLY --worlds 703,704

# 2)（可选，建议先做）超参扫描，找到会 grok 的 (wd, frac) 区域，只用于选执行参数
python code/scan_grok.py --spec MODP --world 701 --steps 20000

# 3) 正式队列（第一轮，已跑）：世界701/702 × 种子1-3，classic 预设，10 万步
python code/queue_grok.py --spec MODP --worlds 701,702 --seeds 1,2,3 --preset classic --steps 100000

# 3b) 确认阶段：补full 5 个种子、加两个新世界 705/706
python code/queue_grok.py --spec MODP --worlds 701,702,705,706 --seeds 1,2,3,4,5 --preset classic --steps 100000

# 4) MODP_ADDONLY 复核：必须带 --preset classic（默认预设 10 次全部未 grok，见上）
python code/queue_grok.py --spec MODP_ADDONLY --worlds 703,704 --seeds 1,2,3,4,5 --preset classic

# 5) 用现有分析器出条形码/事件表（面的 Betti 目前仍用等式任务分数，见"等式捷径"一节的说明）
python crosslingual/code/zz_analyze.py --runs grokking/runs --design grokking/data/MODP/world701/COMPLEX.json --out grokking/analysis/MODP

# 6) 确认阶段判据统计量（每个 world×seed 跑一次，再手动/脚本汇总跨种子复现）
python code/confirm_stats.py grokking/runs/MODP_world705_seed1 grokking/data/MODP/world705/COMPLEX.json --perm 1000
```

## 时间估算

CPU 实测（本机 `dryrun`：p=13，300 步，每 50 步评测全部 1408 条探针，2 层/d_model=128 模型）：
300 步 + 7 次评测总共约 **17~22 秒**，即每步（含摊销的评测开销）约 60ms。真实规模下单次评测的探针量
更大（p=97 世界一次评测共 19,712 条：边 13×1024=13312 + 顶点 11×256=2816 + 面 3×1024=3072 + 迁移
2×256=512），CPU 上单次
评测预计数秒到十几秒；30000 步 + 300 次评测的完整单个 (世界,种子) 在 CPU 上大约数十分钟量级。GPU 上
矩阵乘法和批量解码的常数因子远小于 CPU，`GAP_ANALYSIS` 估计"2 世界 × 5 种子 × 全部检查点探针 ≈ 1–2
小时 GPU"，与这里的 CPU 数字做粗略外推是一致的（batch_size=512、d_model=128 的模型在任何现代 GPU 上
都远没有跑满，单步应为个位数毫秒）。扫描网格（9 格 × 20000 步、只训 P_S 一条边、探针量小得多）预计
比正式队列快一个数量级。

## 已知限制

- 这是**development** 管线：没有世界×种子的置信推断（那部分留给 `zz_analyze.py` 的 `--perm` 与
  `PREREGISTRATION_GROKKING_DRAFT.md` 里的判据，本身运行在 GPU 产出的真实 eval 文件上）。
- `scan_grok.py` 的网格只在 world701(p=97)、只训 P_S 一条边上跑，用来"选参数"；它的 grok 判定用单一
  阈值 0.9，不是预注册判据，不能用来做任何confirmatory 结论。
- `X` 只有一条边（`P_X`）碰到它，且 `P_X` 恒为 control，永不进入任何真实的 present 集合，所以 `X`
  在 `taught_no_faces`/`taught_with_faces` 等阶段里始终缺席（不是"永远孤立地算进 β0"，而是根本不在
  present 集合里）——如果未来往这条线上加别的、会把 `X` 变成 taught/latent 的机制，需要重新检查
  `placebo_entered_no_betti_change` 这条断言是否还成立。
- `face_scores.py` 的路线交叉校验是"该 (task,item) 下**全部**探针行都对"这种较严格的定义（跨模板都要
  对），比逐行的 accuracy 更保守，仅供参考，不参与任何判定。
- MODP_ADDONLY 用 `vertex_present_mode: "blanket"`，语义与 MODP/MODP_LATENTSUB 的闭包驱动 present
  刻意不同（见"最小规格与经典设置"一节）；`make_world.ground_truth()` 据此只对 `blanket=false` 的
  规格做"安慰剂不改变拓扑"的不变量断言，`blanket=true` 时改为把强行并入所有 control cell 的阶段
  记成 `all_present` 并与 `expected_topology` 核对——两条路径共享同一份代码，但语义分叉，新增规格
  时需要想清楚该用哪一种。
- `--preset classic` 的"全批"只按训练池总行数与 `FULL_BATCH_CAP=8192` 比较来判断是否回退到
  `batch_size=2048`；这个上限是本次交付里选的一个经验值（GPU 显存/时间都还没有实测过——本仓库只在
  CPU 上跑过 dryrun），如果 GPU 日发现即使 8192 条全批也偏慢/偏大，或者更大的池子其实也扛得住，
  应该按实测调整这个常量，而不是想当然接受它。
- 本仓库随交付一起生成的真实规模数据（`data/MODP*/world70{1,2}`）已经过 `make_world.py` 内部的
  `assert`（划分闭包、探针自评分、GROUND_TRUTH Betti）和独立的 `test_grokking.py` 复核，但训练本身
  （grokking 是否真的出现、在哪一步）**尚未在 GPU 上跑过**，属于本次交付之外、下次租机执行的部分。
