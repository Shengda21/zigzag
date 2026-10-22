# cw_zigzag：通用有限 CW 复形 F2 同调与 zigzag 条形码

`cw_zigzag.py` 把 `reproduction/analysis_dependencies/full_topology.py`
中已验证、但硬编码为 11 个胞腔（原始"理论关系图" `vF vD vP vR / eFD eDP eFP
eFR eRP / fA fB`）的算法，逐一般化到任意有限 CW 复形：F2 精确同调、诱导
映射、augmented union/intersection zigzag 图、广义秩区间分解（barcode）、
端点计数闭式公式。**算法本身未改动**，只是把原来写死的 11 这个常量、以
及 `DIMS`/`BOUND`/`CLOSURE` 三个硬编码数组，替换成由调用方传入的胞腔列表
在 `CWComplex.__init__` 里现算出来的等价结构。`reproduction/` 目录下的任
何文件都未被修改；`test_cw_zigzag.py` 通过 `sys.path.insert` 引用它，逐
项与原始实现做交叉验证。

## 1. 安装与运行

只依赖 Python 标准库 + numpy（不需要 scipy/networkx 等）。

```bash
cd crosslingual/code
python test_cw_zigzag.py   # 应打印 5 组检查全部通过，最后一行 ALL TESTS PASSED
```

## 2. 基本概念

- **胞腔（cell）**：由字符串名字标识，例如 `'vF'`、`'eFD'`、`'fA@I'`。
- **维度（dim）**：非负整数；一个胞腔的边界胞腔必须全部是 `dim-1` 维。
- **边界（boundary）**：F2（无符号）边界，即胞腔的"直接下维面"集合，用其
  名字列表给出，不需要定向或系数。`__init__` 会检查每个边界胞腔维度严格
  为 `dim-1`，从而保证不可能出现环。
- **胞腔索引 / mask**：`CWComplex` 把输入列表里第 `i` 个胞腔的索引固定为
  `i`，此后所有"子复形"都表示成一个 Python 整数（bitmask）：第 `i` 位为
  1 表示该胞腔在场。与 `full_topology.py` 里 11 位 bitmask 的做法完全一
  致，只是位数从 11 变成 `len(cells)`。
- **闭子复形（closed subcomplex）**：一个 mask，若它包含某胞腔，则必须
  同时包含该胞腔的**整个拓扑闭包**（自身 + 递归的所有面）。本模块的几乎
  所有函数（`homology`、`betti`、`barcode`……）都要求输入 mask 是闭的，否
  则抛 `ValueError`。

## 3. `CWComplex` API

```python
cx = CWComplex(cells)   # cells: [(name:str, dim:int, boundary:list[str]), ...]
```

构造时的检查：胞腔名字不重复；每个边界名字须已在列表中出现且维度恰为
`dim-1`（否则报错并指出是哪个胞腔的哪条边界不对）。

属性：

| 属性 | 含义 |
|---|---|
| `names` | 胞腔名字元组，按输入顺序，下标即为该胞腔的整数索引 |
| `dims` | 每个胞腔的维度 |
| `bound` | 每个胞腔的"直接面" bitmask |
| `closure` | 每个胞腔的"完整拓扑闭包" bitmask（自身+递归所有面） |
| `n` | 胞腔总数 |

方法（均不修改传入的 mask，纯函数式）：

- `index(name) -> int`：名字转索引。
- `cells_in(mask) -> list[str]`：mask 里有哪些胞腔（按索引顺序）。
- `diff(a, b) -> {'added': [...], 'removed': [...]}`：从 `a` 到 `b` 增删
  了哪些胞腔名字。
- `is_closed(mask) -> bool`：是否为闭子复形。
- `close(mask) -> int`：mask 的拓扑闭包（**最小**包含它的闭子复形，即把
  每个已置位胞腔的 `closure[i]` 取并；与"mask 内最大的闭子复形"是另一种
  运算，本模块不需要后者，`mask_from_scores` 已能直接产出闭子复形）。
- `mask_from_scores(scores, threshold) -> int`：与
  `local_topology.mask_from_scores` 语义一致——胞腔进入当且仅当**它整个
  闭包内所有胞腔**分数都 `>= threshold`（对闭包取 min 再比较阈值）。返
  回值总是闭子复形。`scores` 是 `{name: float}` 字典，值须在 `[0,1]`。
- `homology(mask, k) -> (reps, coord)`：mask 上 k 维 F2 同调。`reps` 是
  一组代表元链（bitmask 元组），`len(reps)` 即 Betti 数；`coord` 把每个
  k 维闭链映到它的同调类编号，供 `induced` 使用。带 `lru_cache`。
- `betti(mask) -> list[int]`：`[β0, β1, ..., β_D]`，`D` 是 mask 中**实际
  出现**的最高胞腔维度（不是复形全局的最高维度）——例如 33 胞腔的跨语
  言复形里，只取 en 侧 11 个胞腔时没有 3 维胞腔在场，`betti` 返回长度 3
  的 `[1,0,0]`，而非长度 4。
- `induced(source, target, k) -> tuple[int,...]`：`source ⊆ target` 时
  `H_k(source) -> H_k(target)` 的矩阵（target 基下的列向量元组）。带
  `lru_cache`。
- `augmented(masks, bridge='union') -> list[int]`：把检查点 mask 序列插
  入并集/交集桥，得到长度 `2T-1` 的节点序列。
- `diagram(masks, k=1, bridge='union') -> (dims, arrows)`：augmented 序
  列上每个节点的 `H_k` 维度及相邻节点间诱导映射（type-A quiver）。
- `barcode(masks, k=1, bridge='union') -> tuple[(a,b), ...]`：对
  `diagram` 做广义秩区间分解得到条形码（区间端点见 §4 节点约定）。带
  `lru_cache`。
- `endpoints(masks, k=1, bridge='union', exclude_boundary=False) ->
  np.ndarray`：长度 `len(masks)-1` 的数组，第 `i` 项是"落在节点
  `{2i, 2i+1, 2i+2}` 上的区间端点总数"的闭式公式（不经过 `decompose`，
  与 `barcode` 得到的结果被证明恒等，测试里逐一核对）。`exclude_boundary`
  会把第一个/最后一个检查点的贡献清零（那两个端点只是"记录窗口边界"，
  不代表真实出生/死亡，参见 §5）。

## 4. 节点索引约定（偶数=检查点，奇数=桥）

给定 `T` 个检查点 mask，`augmented`/`diagram`/`barcode` 内部构造的序列
共有 `2T-1` 个节点：

```
节点:   0        1        2        3        4    ...  2T-2
含义:  ckpt0   bridge0  ckpt1   bridge1  ckpt2  ...   ckpt(T-1)
```

- **偶数节点 `2i`** 是第 `i` 个检查点本身的 mask；**奇数节点 `2i+1`**
  是检查点 `i`、`i+1` 之间的"桥"：`bridge='union'` 时为
  `mask_i | mask_{i+1}`，`'intersection'` 时为 `mask_i & mask_{i+1}`。
- 箭头方向取决于 `bridge`：union 桥总是"检查点 → 桥 → 检查点"（子复形到
  超复形的包含诱导映射）；intersection 桥反过来是"桥 → 检查点"（交集是
  子复形，包含方向相反）。

`barcode` 返回的每个区间 `(a, b)` 里，`a`、`b` 就是上述节点索引：`a` 为
偶数表示区间"出生"于某个真实检查点，为奇数则出生于桥；`b` 同理对应
"消亡"位置。`endpoints(..., exclude_boundary=True)` 会把 `a=0` 或
`b=2T-2` 的端点清零——它们只是观测窗口被截断的截尾，不是真正的出生/死
亡事件。

## 5. 事件分类：`classify_h1_change` / `classify_h0_change`

这两个函数比较两个闭子复形 `prev_mask -> next_mask`（只看 cell 层面的
增删，不要求二者互相包含），结合 Betti 数变化和胞腔增删的维度，给出一
个描述性标签。**这只是对"洞消失/合并"这类事件做粗分类，不构成拓扑证据
的稳健性检验**（见 §7）。

### `classify_h1_change(cx, prev_mask, next_mask)`

只在 β1 **下降**（`next` 的独立 1 维闭链数少于 `prev`）时才细分类别：

1. `edge_loss`：有任何 0 维或 1 维胞腔被**删除**——最常见也最"廉价"的
   洞消失原因，不是"填面"，只是把构成这个洞的边或点直接拿走了。
2. `clean_fill`（无删边删点的前提下）：新增了 2 维胞腔——"保留旧边界、
   干净地填上一个面"，通常是更有信息量的候选拓扑事件。
3. `bridge_only`（既没删边点，也没加面）：β1 下降纯粹是并集/交集桥造成
   的（两个检查点各自独立都有一个洞，取交集后洞不再同时闭合），属于
   "桥专有"效应，不对应任何单一检查点上的真实胞腔增删。

β1 不下降时：`h1_increase`（新洞出现）或 `no_change`。

返回 `{'h1_delta', 'category', 'added', 'removed'}`。

### `classify_h0_change(cx, prev_mask, next_mask)`

β0 **下降**（连通分量变少，即"合并"）：`merge`——新增了 1 维胞腔且没有
删除任何 0 维胞腔，典型的"加一条边把两块连起来"；否则 `vertex_loss`。

β0 **上升**（分裂/新分量出现）：

- `vertex_birth`：新增了孤立的 0 维胞腔（新分量由一个全新顶点开始）。
- 否则 `split`：删除了 1 维胞腔（把原本连通的一块断成两块）。
- 否则 `other_h0_increase`：兜底类别。

不变：`no_change`。

返回 `{'h0_delta', 'category', 'added', 'removed'}`。

## 6. 原始 11 胞腔复形与跨语言乘积复形

- `theory_graph_cells()` / `eleven_cell_complex()`：逐字复现
  `full_topology.py` 的 `DIMS`/`BOUND`/`CLOSURE`（测试中有逐元素相等断
  言）。四个顶点 `vF vD vP vR`，五条边 `eFD eDP eFP eFR eRP`，两个三角形
  `fA`（=F-D-P）、`fB`（=F-P-R，与 `fA` 共享边 `eFP`）。

- `product_with_interval(cells, ends=('en','zh'), bridge_tag='I')`：把一
  个 CW 复形 `G` 与区间 `I` 做"柱面/棱柱积" `K* = G × I`（F2 无符号版
  本）。对 `G` 里每个 `d` 维胞腔 σ（边界 `{τ_1,...,τ_m}`）：
  - 两个"端面"拷贝 `σ@en`、`σ@zh`（维度仍是 `d`，边界分别是
    `{τ_j@en}`、`{τ_j@zh}`）；
  - 一个"侧壁"胞腔 `σ@I`（维度 `d+1`），边界是
    `{τ_j@I} ∪ {σ@en, σ@zh}`——即 σ 边界扫出的侧面加上两个端面。

  `ends` 须恰好两个标签。`crosslingual_complex()` 即
  `product_with_interval(theory_graph_cells())`，共 `11×3=33` 个胞腔。
  因为 `I` 可缩，`G×I` 与 `G` 同伦等价（形变收缩到 `G×{en}`），故全体
  33 个胞腔的 Betti 数应与 `G` 一致（`[1,0,0]`），只是多了一层恒为 0 的
  `β3`（测试验证为 `[1,0,0,0]`）。

  `test_cw_zigzag.py` 第 3 组测试逐步构造从"只有 en 侧"到"en+zh 都全、
  且用桥胞腔连起来"的中间状态，核对每一步 Betti 数，并用
  `classify_h1_change` 验证"加回缺失的桥边 `eFD@I`"分类为 `clean_fill`。
  "两个不相交拷贝 + 4 个顶点桥 `vX@I`" 这一步的 Betti 数用 Euler 示性数
  独立核对为 `[1,3,0]`（8 个 0 维、14 个 1 维、4 个 2 维胞腔给出示性数
  `-2`，配合 `β0=1` 反推 `β1=3`）——与任务描述文字里的"β1=5"不符，测试
  以经过示性数验证过的数值为准；再加 5 个方块 `eXY@I` 之后的
  `β1=0, β2=2` 则与任务描述一致，同样通过示性数核对。

## 7. `permutation_null`：置换零模型

```python
permutation_null(cx, score_matrix, thresholds, n_perm, seed,
                  k=1, bridge='union', block=None, alpha=0.05) -> dict
```

> 注：交付说明的伪代码签名省略了目标 `CWComplex`；没有它无法从分数构造
> mask/计算 `endpoints`，因此本实现把 `cx` 作为**必填的第一个位置参数**。

### 统计量

`score_matrix` 是 `{胞腔名: [每个检查点的分数, ...]}`，所有胞腔序列长度
须相同（记为 `T`）。给定阈值 `theta_1..theta_M`，每个阈值下用
`cx.mask_from_scores` 把每个检查点的分数二值化成一个闭子复形 mask，再用
`cx.endpoints` 得到长度 `T-1` 的端点计数数组。观测统计量是阈值平均：

```
T_t = (1/M) * Σ_θ E_{k,t,θ}      t = 0..T-2（第 t 个"检查点过渡"）
```

`t` 与 §4 的节点索引约定一致：`T_t` 描述节点 `{2t, 2t+1, 2t+2}`（检查点
t、桥、检查点 t+1）附近的端点爆发强度。

### 零模型（置换过程）

每次置换重复：按 `t=0..T-2` 顺序依次处理每个检查点边界；每个边界上，对
每一组同步置换的胞腔（见 `block`）独立掷一次公平硬币，正面则把组内每个
胞腔在 `t`、`t+1` 两个检查点上的分数原地交换。因为每次操作都只是一次对
换，任何单个胞腔最终的时间序列仍是它自身原始 `T` 个取值的某个置换——边
际分布（该胞腔在这 `T` 个检查点上出现过的取值集合）被严格保留，只是重
新分配到哪个检查点。`n_perm` 次独立重复给出每个 `t` 上 `T_t` 的零分布。

- `block=None`（默认）：每个胞腔独立掷硬币。
- `block` 为字典 `{胞腔名: 组号}`：同组号的胞腔每个边界上共用一次硬币结
  果（全交换或都不换）。
- `block` 为组的可迭代对象（如 `[['a','b'], ['c']]`）：等价写法，未出现
  的胞腔各自单独成组。

### 返回值

```python
{'observed': {t: float},         # T_t（观测值）
 'p_values': {t: float},         # (1 + #{T_perm[t] >= T_obs[t]}) / (n_perm + 1)
 'null_quantiles': {t: {'q05', 'q50', 'q95': float}},
 'n_perm', 'thresholds', 'k', 'bridge', 'n_transitions', 'alpha',
 'bonferroni_alpha': alpha / n_transitions,
 'significant_bonferroni': {t: bool}}  # p_values[t] <= bonferroni_alpha
```

p 值用"大于等于"单边检验（`T_t` 越大代表越多端点/事件挤在这个过渡上，
越不像随机噪声），并用 `+1` 平滑避免零 p 值；Bonferroni 阈值按测试的过
渡数 `T-1`（而不是阈值数 `M`，阈值已经被平均掉了）做族系误差校正。

`test_cw_zigzag.py` 第 5 组测试构造了 11 个孤立点胞腔、12 个检查点的分
数矩阵：一个让全部 11 个胞腔在同一过渡（`t=5`）同步从低分跳到高分，另
一个让 11 个胞腔各自在 11 个不同过渡上单独跳变。`B=200`、固定
`seed=12345` 下，前者在 `t=5` 处的 p 值（约 0.030）明显小于后者所有 `t`
上的最小 p 值（约 0.682），说明零模型能区分"同步跃迁"与"错峰跃迁"。

### 与论文定义 3.3 的对应关系和局限

原论文定义 3.3（见仓库内 `assessment_20260920/拓扑证据复核.md` 引用的
`tmp/pdfs/original_paper.txt` 第 177–183 行）把证据分三层：**真实出
生/死亡**（原始条形码里的一条区间）→ **候选拓扑涌现**（需跨提示改写、
跨训练种子、预先固定且覆盖足够密的阈值网格、检查点增密、以及**预先注
册并校准的零模型**这几项稳健性检验）→ **候选能力涌现**（还需与独立迁
移任务表现在预定窗口内同步）。

`permutation_null` 只实现了"零模型"这一项稳健性检验里数学上最核心的部
分——给定固定的分数时间序列和固定的阈值网格，检验某个过渡上的端点爆发
是否显著超出随机重排水平。它**不能单独**把一个原始拓扑事件提升为"候选
拓扑涌现"，原因包括：

1. **预注册**：零模型须预先注册、跨提示/种子校准，而不是事后针对某一
   次观测到的检查点序列现算——本函数是通用工具，是否"预注册"取决于调
   用方的实验流程，模块不代为保证。
2. **阈值覆盖**：本函数对一组阈值取平均（对应"预定阈值覆盖"要求的一部
   分），但阈值网格本身、以及"6/7 阈值同时出现"这类覆盖率判据，仍需调
   用方按论文口径单独设计和报告。
3. **检查点增密**：零模型只在调用方给定的检查点密度上工作；粗网格下显
   著与否都不能替代细网格复现。
4. **独立训练重复 vs. 网格计算对象数**：`n_perm` 次置换、`M` 个阈值、
   `T-1` 个过渡，都是同一批已训练权重上的重计算对象，不是独立训练重
   复——"计算对象数≠独立训练重复数"。
5. **候选能力涌现**（第三层）完全在本模块范围之外：不涉及独立迁移任务
   表现，也不做"事件窗口 vs 独立指标同步"的检验。

因此，`permutation_null` 的 p 值应理解为"给定观测协议后，某过渡上拓扑
事件强度是否超出置换零分布"这一条狭义、描述性统计证据，而非论文定义
3.3 意义上"候选拓扑涌现"或"候选能力涌现"的充分判据。

## 8. 测试覆盖一览（`test_cw_zigzag.py`）

1. **交叉验证**：`eleven_cell_complex()` 的 `dims/bound/closure` 与
   `full_topology.py` 对应数组逐元素相等；全部 2048 个 mask 上
   `is_closed` 一致；全部 83 个闭子复形上 `betti` 与
   `full_topology.homology` 三维度数一致且满足欧拉示性数恒等式；随机
   300 组长度 2–5 的闭子复形序列，在 `k∈{0,1}`、
   `bridge∈{union,intersection}`（含两种 `exclude_boundary`）下
   `barcode`（排序后）与 `endpoints` 与 `full_topology` 完全相等。
2. **玩具断言复现**：`full_topology.tests()` 里手写的区间断言、外环
   H1、全复形 Betti、100 个随机已知区间分解 quiver，在通用类上复现。
3. **乘积复形**：33 胞腔；见 §6 的逐步 Betti 数核对与 `clean_fill` 事件
   分类核对。
4. **`mask_from_scores` 闭包语义**：`fA` 自身分数高、但边界边 `eDP` 分
   数低时不进入 mask，且与 `local_topology.mask_from_scores` 逐位一致。
5. **`permutation_null`**：§7 的同步 vs 错峰场景，`B=200`、固定
   `seed`，打印并断言同步场景在其跳变过渡上的 p 值显著小于错峰场景所
   有过渡上的最小 p 值。

运行 `python test_cw_zigzag.py` 依次打印每组检查摘要，最后打印完整 JSON
汇总和 `ALL TESTS PASSED`；任何断言失败都会抛出异常并以非零退出码终止。
