# CPU 冒烟测试记录

真实跑通记录，非估算。目的仅为验证 `run_public.py` -> `cell_scores.py`
这条管线能跑通、能打分、能幂等跳过、能在小样本下正确覆盖 `composition`/
`transfer`，**不代表模型能力**（70m 模型 + 三个检查点，正确率理应很低，
也确实很低）。

> 本记录是**重跑**版本：字符串域探针 schema 从 `['text']` 改成
> `['exact']`（`zz_score.py` 新增的大小写敏感评分类型）之后，
> `data/PROBES.jsonl` 换了内容（sha256 变了），旧的
> `runs/pythia-70m/eval_*.jsonl` 是用旧 schema 打分的、已不能代表当前代
> 码，因此先删除旧输出目录再重新跑。`run_public.py::load_probes` 同时把
> `--limit` 的抽样方式从"全量等距抽样"改成了"按 kind 分层抽样"（细节
> 见 README.md §6 item 4），本记录也顺带验证了这个改动。

## 运行环境

- Python 3.12
- torch 2.5.0+cpu，transformers 5.17.0，huggingface_hub 1.7.2
- 设备：`--device cpu`
- 模型：`EleutherAI/pythia-70m`，revision：`step1000`、`step32000`、
  `step143000`（HF 缓存里已有上一轮下载的权重，本轮**未重新下载**）
- 探针：`data/PROBES.jsonl` 全量 4000 行（新 sha256，见
  `data/PROBES_MANIFEST.json`），`--limit 12` 做**按 kind 分层**抽样

## 命令

```bash
cd public_ckpt/code
# 1) 用新 schema 重新生成探针 + 自检（含大小写回归断言）
python make_probes.py
# 2) 删除旧的（用旧 schema 打分的）冒烟输出
python -c "import os,glob; [os.remove(f) for f in glob.glob('../runs/pythia-70m/*')]"
# 3) 重跑冒烟（模型已缓存，无需重新下载）
python run_public.py --model EleutherAI/pythia-70m \
  --revisions step1000,step32000,step143000 \
  --device cpu --limit 12 --out ../runs/pythia-70m
python cell_scores.py ../runs/pythia-70m
```

## `make_probes.py` 自检结果

```
wrote data/PROBES.jsonl (4000 rows, sha256=5b2e5365f362...)
wrote data/PROBES_MANIFEST.json
by_kind: {'composition': 1152, 'edge': 2208, 'transfer': 96, 'vertex': 544}
by_domain: {'date': 1120, 'number': 1248, 'string': 1632}
case-sensitivity self-check: 384 pure-case-transform rows all correctly score 0
for a verbatim-copy prediction (edges: ['e_caesar3_caesar3upper',
'e_reverse_upperreverse', 'e_word_alternate', 'e_word_upper'])
```

- 全部 4000 行仍然对自身 gold 自检 `correct==1`（`self_check`，和之前一
  样）。
- 新增的 `self_check_case_sensitivity`：对 4 条纯大小写边（`upper` /
  `alternate` 相关）共 384 行，把"原样照抄输入"当作预测喂给
  `zz_score.score`，全部 384 行都正确地得到 `correct==0`——如果这里改
  回 `schema=['text']` 重跑，这条断言会失败（旧 schema 下照抄会被误判为
  `correct==1`），这就是对"修复确实生效"的直接证明。
- schema 按域拆分核对：`string` 域 1632 行全部是 `schema=['exact']`，
  `number`（1248 行）/ `date`（1120 行）全部是 `schema=['text']`（逐行
  统计验证过，无一例外）。

## 每个 revision 的耗时

| revision | 模型加载耗时 | 12 行推理耗时 | 单行推理耗时 | 采样行整体正确率 |
|---|---|---|---|---|
| `step1000`   | 11.7 s | 25.6 s | 2.131 s/row | 0/12 = 0.000 |
| `step32000`  | 10.2 s | 25.7 s | 2.142 s/row | 1/12 = 0.083 |
| `step143000` | 9.8 s  | 25.7 s | 2.141 s/row | 0/12 = 0.000 |

三个 revision 总计约 **83 秒**——模型权重已在 HF 缓存里（上一轮冒烟测试
下载过），本轮加载耗时从之前的 180–310 秒降到 10–12 秒，纯粹是"读本地
缓存 vs 首次联网下载"的差异，推理本身的耗时（约 2.13–2.14 秒/行）和上
一轮几乎完全一致，符合预期（模型结构/大小没变）。三次都没有触发
`run_public.py` 里的 `except` 分支，没有生成任何 `*.error.json`。

幂等性：`step1000` 的 `eval_1000.summary.json` 已存在时重复执行同一条命
令，`run_public.py` 打印 `summary already exists ... skipping`，不重新
加载模型——和上一轮验证结果一致。

## `--limit 12` 分层抽样覆盖情况

`load_probes` 把 12 平均分给 4 种 kind，实际采样结果：

```
by kind: {'composition': 3, 'edge': 3, 'transfer': 3, 'vertex': 3}
```

composition 的 3 行里**包含一组完整配对**（`face_caesar3_upper` 的两条
路径 `caesar3_then_upper` / `upper_then_caesar3`，同一题项 `onion`、同一
模板 0）加 1 行额外样本（`caesar3_then_upper`，题项 `dynamics`）；
transfer 的 3 行**每个迁移任务各 1 行**。相比修正前那次冒烟测试（全量等
距抽样，12 个索引 0 个落进 transfer、3 个落进 composition 但没有任何一
对配对完整），这次 `face_caesar3_upper` 和全部 3 个迁移任务都有非空分
数——分层抽样的修正生效。

## `cell_scores.json`（三个 revision）

```json
{
  "1000":   {"e_binary_hex": 0.0, "e_decimal_roman": 0.0, "e_upper_caesar3upper": 0.0,
             "date_iso": 0.0, "num_decimal": 0.0, "str_caesar3_upper": 0.0,
             "face_caesar3_upper": 0.0},
  "32000":  {"e_binary_hex": 0.0, "e_decimal_roman": 0.0, "e_upper_caesar3upper": 0.0,
             "date_iso": 1.0, "num_decimal": 0.0, "str_caesar3_upper": 0.0,
             "face_caesar3_upper": 0.0},
  "143000": {"e_binary_hex": 0.0, "e_decimal_roman": 0.0, "e_upper_caesar3upper": 0.0,
             "date_iso": 0.0, "num_decimal": 0.0, "str_caesar3_upper": 0.0,
             "face_caesar3_upper": 0.0}
}
```

7 个胞腔（3 边 + 3 顶点 + **1 面，非空**）在三个 revision 都有分数。绝大
多数是 0.0（70m 模型在这几个检查点上确实答不对这些题——预期之内）；唯
一的例外是 `step32000` 的顶点 `date_iso` 得了 1.0：查看对应的
`eval_32000.jsonl` 一行，模型正确地把 few-shot 演示里"原样复述日期"的
模式迁移到了第 4 个（测试）日期上（`1917-04-02` -> `1917-04-02`，
`schema=['text']`，纯字符串照抄，不需要理解日期语义），是一次真实的正
确预测，不是打分逻辑的问题。`face_caesar3_upper` 三个 revision 都是
0.0——70m 模型完全答不对"凯撒+3 再转大写"这类两步组合，符合直觉。

`transfer_scores.json`：三个 revision 的三个迁移任务全部 0.0（同样符合
预期，70m 模型答不对三步组合）：

```json
{
  "1000":   {"decimal_to_binary_to_hex_to_english": 0.0, "iso_to_us_to_text_to_weekday": 0.0, "reverse_then_upper_then_caesar3": 0.0},
  "32000":  {"decimal_to_binary_to_hex_to_english": 0.0, "iso_to_us_to_text_to_weekday": 0.0, "reverse_then_upper_then_caesar3": 0.0},
  "143000": {"decimal_to_binary_to_hex_to_english": 0.0, "iso_to_us_to_text_to_weekday": 0.0, "reverse_then_upper_then_caesar3": 0.0}
}
```

## 结论

- `run_public.py` / `cell_scores.py`：加载、批量贪心解码、按行截断首个换
  行、`zz_score.score` 打分（`'exact'` 与 `'text'` 两种 schema 都验证过
  实际生效）、写 `eval_<step>.jsonl` / `.summary.json`、幂等跳过、顶点/
  边准确率聚合、面配对逻辑、迁移任务聚合——全部验证通过，三个 revision
  均无异常。
- 分层抽样修正验证通过：`--limit 12` 现在稳定覆盖全部 4 种 kind，并且
  至少包含一组完整的 face 路径配对，`cell_scores.json` 里 `face_*` 和
  `transfer_scores.json` 不再是空的。
- 大小写敏感评分修正验证通过：`make_probes.py` 的
  `self_check_case_sensitivity` 断言（384 行，"照抄输入必须得 0 分"）
  100% 通过；`string` 域 1632 行全部确认是 `schema=['exact']`。
- 遇到的问题：无下载失败、无缺失 revision、无打分异常。
