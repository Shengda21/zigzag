# public_ckpt：公开预训练检查点序列的纯推理 zigzag 探针

本目录为"在公开预训练检查点序列上做纯推理的 zigzag 持续同调分析"构建了
一套自然语言原子能力探针集与推理 runner。全程**只做推理，不做任何梯度
更新/微调**：给定一串同一模型家族在训练过程中留档的 HuggingFace revision
（如 Pythia 的 `step1000`、`step2000`、...），逐个 revision 加载、跑探针、
打分、卸载，最终得到"每个检查点 × 每个原子能力胞腔"的分数矩阵，供后续
用 `crosslingual/code/cw_zigzag.py` 做 zigzag 条形码分析。

不修改 `zigzag_v5/` 下除 `public_ckpt/` 以外的任何文件；只是把
`stage_ab/code/zz_score.py`（评分器）和 `crosslingual/code/cw_zigzag.py`
（CW 复形/同调代数）作为库导入使用。

## 1. 目录结构

```
public_ckpt/
  spec/
    ATOMIC_TASKS.md      # 复形定义的详细说明（3 个域、17 顶点、23 边、9 面）
    COMPLEX.json          # 复形本身：{vertices, edges:[{name,src,dst}], faces:[{name,cycle,composition_tasks}]}
  code/
    oracle.py              # 纯 Python oracle：所有 gold 的唯一来源；也能重新生成 COMPLEX.json
    make_probes.py         # oracle.py -> data/PROBES.jsonl（few-shot 探针）
    run_public.py          # 加载某个 revision -> 贪心解码 -> 打分 -> eval_<step>.jsonl / eval_<step>.summary.json
    cell_scores.py         # 一个 runs 目录的全部 eval_*.jsonl -> cell_scores.json / transfer_scores.json
  data/
    PROBES.jsonl            # 4000 行探针（构造后已用 oracle 自检：全部 correct==1）
    PROBES_MANIFEST.json    # 行数 / sha256 / 各维度计数
  runs/<model>/              # run_public.py 的输出（每个 revision 一对 eval_*.jsonl + summary）
  SMOKE_TEST.md               # CPU 冒烟测试记录（pythia-70m，3 个 revision，--limit 12）
```

## 2. 任务集与复形

3 个域、17 个顶点（表示）、23 条边（单步变换）、9 个面（两条组合路径的一
致性）；另有 3 个不属于复形的"迁移"任务（3 步组合，独立题项，检验泛化）。
完整定义、每条边/面用的具体函数、以及"为什么这些交换律精确成立"的自检
结果见 [`spec/ATOMIC_TASKS.md`](spec/ATOMIC_TASKS.md)。复形本身是
[`spec/COMPLEX.json`](spec/COMPLEX.json)，可以直接喂给
`cw_zigzag.CWComplex`：

```bash
cd public_ckpt/code
python oracle.py   # 重新生成 COMPLEX.json 并打印: cells=49 (17+23+9), betti=[3,0,0]
```

`betti=[3,0,0]`：3 个域互不相连、且每个域内部的所有"来回两条路径"都被对
应的面填平——这是拓扑层面对复形自身合法性的验证，与任何模型无关。

三个域概览：

| 域 | 基础表示 | 顶点数 | 边数 | 面数（一致性关系） |
|---|---|---|---|---|
| `string` | 5–8 字母小写英文单词（固定 320 词表） | 8 | 10 | 3：`upper∘reverse=reverse∘upper`、`caesar3∘upper=upper∘caesar3`、`reverse∘caesar3=caesar3∘reverse` |
| `number` | 十进制整数 10–999 | 5 | 7 | 3：`decimal→{binary,hex,roman,english}` 之间的路径一致性（如 `decimal→binary→hex` 与 `decimal→hex`） |
| `date` | 1901–2099 年间的 ISO 日期 | 4 | 6 | 3：`iso→{us,text,weekday}` 之间的路径一致性 |

## 3. 探针格式（`data/PROBES.jsonl`）

每行是一个能被 `stage_ab/code/zz_score.score(row, prediction)` 直接打分
的 JSON 对象，字段：

- `id`：全局唯一
- `task`：边名 / 顶点名 / 组合任务名 / 迁移任务名
- `kind`：`'edge' | 'vertex' | 'composition' | 'transfer'`
- `vocabulary`：固定为 `'public'`
- `template`：模板编号（int）
- `prompt`：完整的 few-shot 补全式提示词
- `gold`：oracle 算出的标准答案（纯字符串，未做大小写/空白归一化）
- `schema`：按域区分（`oracle.py::schema_for_domain`）——字符串域用
  `['exact']`（`zz_score` 对 `'exact'` 做大小写敏感、只去首尾空白的精确
  匹配），数字域/日期域用 `['text']`（NFKC + 小写 + 去标点 + 去空白后的
  规范化精确匹配）。见 §6 "已修正" 说明为什么字符串域需要区别对待。
- `field_length`：固定为 `1`
- 额外字段（供 `cell_scores.py` 分析用，`zz_score` 不使用）：`domain`、
  `item`（题项本身，便于复现）、`face`（仅 `composition` 行非空）、
  `path_index`（仅 `composition` 行非空，0/1 标记两条路径中的哪一条）

行数构成（共 4000 行，见 `data/PROBES_MANIFEST.json`）：

| kind | 数量 | 构成 |
|---|---|---|
| `edge` | 2208 | 23 边 × 32 题项 × 3 模板 |
| `vertex` | 544 | 17 顶点 × 16 题项 × 2 模板 |
| `composition` | 1152 | 9 面 × 2 条路径 × 32 题项（两条路径共享同一批题项）× 2 模板 |
| `transfer` | 96 | 3 个迁移任务 × 32 题项 × 1 模板 |

`make_probes.py` 在写盘前会对**全部**行执行
`zz_score.score(row, row['gold'])['correct'] == 1` 自检，任何一行没通过
都会直接抛异常——保证探针集本身没有 oracle/模板层面的错误。

## 4. few-shot 模板设计

Pythia / OLMo-2 这类是纯基座模型，不能用 chat 模板，所以全部用**补全式**
3-shot 提示（`make_probes.py::TEMPLATES`），演示例子固定取自"构造集"（见
`spec/ATOMIC_TASKS.md` §4），与被打分的题项（取自"独立集"）互不相交：

```
# 模板 0：Q/A 风格
Task: convert the word to uppercase.

Q: mountain
A: MOUNTAIN

Q: picture
A: PICTURE

Q: orchard
A: ORCHARD

Q: raspberry
A:

# 模板 1：箭头风格（无空行）
Task: ...
mountain -> MOUNTAIN
picture -> PICTURE
orchard -> ORCHARD
raspberry ->

# 模板 2：input/output 标签风格
Task: ...
input: mountain | output: MOUNTAIN
...
input: raspberry | output:
```

`run_public.py` 贪心解码 `max_new_tokens=24`，取生成内容的第一行（遇到
`\n` 截断）作为预测。数字域/日期域用 `schema=['text']`，比较前会去掉全
部空白，前导空格完全不影响打分；字符串域用 `schema=['exact']`，只去首
尾空白（`pred.strip() == gold.strip()`），所以生成内容中间不能有多余空
格或标点，但由于我们的字符串域 gold 都是单个不含空格的 token，生成后第
一行内部本就不会有空格，前导/尾随空白一样不影响打分。

## 5. 如何在单卡 H800 上跑一整轮

```bash
python code/run_public.py \
  --model EleutherAI/pythia-410m \
  --revisions step0,step1,step2,step4,step8,step16,step32,step64,step128,step256,step512,step1000,step13000,step25000,step36000,step48000,step60000,step72000,step84000,step96000,step108000,step119000,step131000,step143000 \
  --device cuda --batch_size 64 --out runs/pythia-410m
python code/cell_scores.py runs/pythia-410m
```

对 `pythia-1b` / `allenai/OLMo-2-0425-1B` 同理，把 `--model` 换掉即可；
**OLMo-2 的 revision 命名与 Pythia 不同**（不是简单的 `stepN`，常见形式
是 `stepNNNN-tokensNNNB` 或按 `main`/`revisions` 里列出的具体名字），跑
之前务必先用

```python
from huggingface_hub import HfApi
print(sorted(b.name for b in HfApi().list_repo_refs("allenai/OLMo-2-0425-1B").branches))
```

确认真实可用的 revision 名，不要假设和 Pythia 一样。`run_public.py`
对不存在的 revision 会捕获异常、写一个 `eval_<label>.error.json` 并继续
跑下一个 revision，不会让整轮任务中断。

### 检查点列表建议

Pythia 官方一共留了 154 个检查点：`step0`，然后 `step1..step512` 中 2 的
幂（`1,2,4,...,512`，共 10 个），之后每 1000 步一个直到 `step143000`（共
143 个）。全跑 154 个太贵；建议**早期对数密集 + 后期线性等距**，共 24
个，训练动态早期（loss 下降最快、能力可能"涌现"）密一点，后期稀一点：

```
早期（11 个，2 的幂）: step0, step1, step2, step4, step8, step16, step32, step64, step128, step256, step512
后期（13 个，等距约 12000 步一个）: step1000, step13000, step25000, step36000, step48000,
    step60000, step72000, step84000, step96000, step108000, step119000, step131000, step143000
```

OLMo-2 系列没有"2 的幂"式的早期检查点，需按其自身发布的 revision 列表
自行挑选早/中/晚各若干个，思路一致即可。

### 时间估算

单个检查点的推理时间 ≈ `探针条数 / batch_size` 个 batch × 每 batch 的
`prefill(few-shot 提示，约 100–250 token) + max_new_tokens(24) 生成` 时
间。本仓库跑过的唯一真实数据点是 CPU 上 pythia-70m 的 12 行子集（见
`SMOKE_TEST.md`），单卡 H800 上没有实测——下面是按数量级给出的**粗估**，
正式跑一整轮 24 个检查点前，建议先用 `--limit 200` 或 `--limit 400` 在
目标模型 + 目标 GPU 上跑一个检查点计时，再线性外推到全部 4000 行 ×
24 个检查点：

- `pythia-410m` / `pythia-1b`：H800 上大概率是"模型加载（几秒到几十
  秒）远小于生成时间"；4000 行、`batch_size=64`、`max_new_tokens=24` 的
  生成量级是 `4000/64 ≈ 63` 个 batch，每 batch 生成 24 步、每步一次前
  向——1B 参数模型在 H800 上单步前向是毫秒级，粗估每个检查点 1–5 分钟；
  24 个检查点合计 **半小时到 2 小时**左右。
- `OLMo-2-0425-1B` 结构和量级与 `pythia-1b`接近，估算类似；实际吞吐还
  受 `transformers` 版本、是否装了 flash-attention、tokenizer 是否变长
  padding 效率等影响，务必以实测的 pilot 为准。

## 6. 已知限制

1. **已修正：字符串域现在用 `schema=['exact']` 评分**。`zz_score.py` 新
   增了大小写敏感的 `'exact'` schema（`field_equal` 只做
   `pred.strip() == gold.strip()`，不做小写化/去标点）；
   `oracle.py::schema_for_domain` 让字符串域的全部探针行（顶点/边/组合/
   迁移）都用它，数字域/日期域保留 `'text'`（理由见
   `spec/ATOMIC_TASKS.md` §5：那两个域的标点/空白是任务无关的格式噪声，
   而大小写从不承载语义）。`make_probes.py::self_check_case_sensitivity`
   对 4 条纯大小写边（`oracle.PURE_CASE_EDGES`：`e_word_upper` /
   `e_word_alternate` / `e_reverse_upperreverse` /
   `e_caesar3_caesar3upper`）共 384 行做回归断言——"原样照抄输入"作为预
   测必须得 0 分——现在 100% 通过（见 `data/PROBES_MANIFEST.json::
   case_sensitivity_check_rows`）。涉及 `reverse`/`caesar3` 的边/面/迁移
   任务从一开始就不受这个问题影响。
2. **基座模型对 few-shot 格式很敏感**：同一个任务换一种模板措辞（比如
   `Q:/A:` 换成 `input:/output:`），尤其是早期/很小的检查点，正确率可能
   有明显差异——这正是我们保留 3 个模板、并在 `cell_scores.py` 里按
   `task/template` 两个维度都能拆开统计的原因，不建议只看单模板的分数。
3. **题项长度/复杂度要克制**：数字域限定在 10–999（3 位以内），字符串
   域限定在 5–8 字母；更长的输入会显著增加基座模型在补全式提示下的失败
   率，且和"检查点训练进度"这个我们真正关心的自变量混杂在一起，弱化了
   信号。
4. **已修正：`--limit` 现在按 kind 分层抽样**：`run_public.py::
   load_probes` 把 `limit` 平均分给 `edge`/`vertex`/`composition`/
   `transfer` 四种 kind（各kind 内部仍是等距抽样，保证覆盖多种
   task/face），不会再出现"全部命中 edge、transfer 一条都没采到"的情况
   （`SMOKE_TEST.md` 记录了修正前后的对比）。`composition` kind 额外做了
   配对感知抽样（`_stratified_composition_sample`）：优先抽出至少一组
   `(face, item, template)` 的两条路径都在场的"完整配对"，再用等距抽样
   补齐剩余名额——这样即使 `--limit` 很小，`cell_scores.py::score_faces`
   也大概率能算出至少一个非空的 face 分数，而不是把 face 分数全部跳
   过。极小的 `--limit`（远小于 kind 数或 face 数）仍然只能保证一到两组
   完整配对，不代表该 face 的真实一致性率——这一点和之前一样，仍建议只
   把 `--limit` 用于管线冒烟测试，正式评测不设 `--limit`。
5. **本仓库未做过真实 GPU 实测**：`run_public.py` 和 `cell_scores.py` 只
   在 CPU + pythia-70m + `--limit 12` 上跑通过；H800 上的吞吐/显存表现
   未实测，§5 的时间估算是数量级粗估，不是基准测试结果。
