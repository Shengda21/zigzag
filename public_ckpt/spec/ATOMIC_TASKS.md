# 原子能力探针：复形定义

本文档描述 `public_ckpt` 用于对公开预训练检查点序列做 zigzag 持续同调分析
的"原子能力复形"（CW complex）。所有 gold 均由 `public_ckpt/code/oracle.py`
中的纯 Python 函数算出（不依赖任何模型），复形本身由
`public_ckpt/spec/COMPLEX.json` 描述，可被 `oracle.load_complex()` 直接读入
`crosslingual/code/cw_zigzag.py` 的 `CWComplex` 并验证闭合性与 Betti 数。

运行 `python oracle.py` 会重新生成 `COMPLEX.json` 并打印自检结果，当前输出：

```
cells: 49 (vertices=17, edges=23, faces=9)
betti of full complex: [3, 0, 0]
oracle self-checks passed: {'edges': 575, 'faces': 225, 'transfer': 75}
```

`betti = [3, 0, 0]`：3 个域互不相连（β0=3），且每个域内部所有 1-圈都被 2-胞
腔"填平"（β1=0）——这正是下面 9 个"一致性"面存在的直接后果。

## 1. 约定

- **顶点（0-胞腔）**：一种"表示"，即某个基础项（word / 十进制数 / ISO 日
  期）经过若干次单步变换后得到的字符串。每个顶点都有一个纯函数
  `vertex_value(vertex, item) -> str`。
- **边（1-胞腔）**：一次单步变换，`src -> dst`，由 `EDGE_FN[edge_name]`
  实现（输入是 `src` 顶点的字符串值，输出应等于 `dst` 顶点的字符串值——
  这一致性由 `validate_oracle()` 对所有边、所有采样项断言）。
- **面（2-胞腔）**：从该域的基础顶点出发的两条不同路径（1~2 条边），二者
  必须到达同一个目标顶点且字符串完全相等——这就是论文里"两种组合顺序结
  果相同"的可验证形式。面的边界（`cycle`）恰好是两条路径用到的边的并集，
  在 F2 意义下构成一个闭合的 3-圈（三角形）或 4-圈（正方形）。

## 2. 三个域

### 2.1 字符串域（`string`）

- 基础项：`public_ckpt/code/oracle.py::WORDS`，固定的 320 个 5–8 字母小
  写英文单词表（构造集 30 词 / 独立测试集 290 词，见 §4）。
- 顶点（8 个）：`str_word`（原词）、`str_upper`（大写）、`str_reverse`
  （反转）、`str_caesar3`（凯撒 +3 移位，保留大小写）、`str_alternate`
  （隔位大写）、`str_upper_reverse`、`str_caesar3_upper`、
  `str_reverse_caesar3`（三个"两步组合"的落点顶点）。
- 边（10 条）：`e_word_upper`、`e_word_reverse`、`e_word_caesar3`、
  `e_word_alternate` 四条"从原词出发"的单步边，另加 6 条"第二跳"边（例
  如 `e_upper_upperreverse`：输入已经是大写词，做反转）。
- 面（3 个，均为 4 边正方形）：
  - `face_upper_reverse`：`upper∘reverse = reverse∘upper`
  - `face_caesar3_upper`：`caesar3∘upper = upper∘caesar3`
  - `face_reverse_caesar3`：`reverse∘caesar3 = caesar3∘reverse`

  三个交换律在纯 Python 层面是精确成立的（大小写、反转、凯撒移位互不
  干扰），`validate_oracle()` 对采样项逐一断言。

### 2.2 数字域（`number`）

- 基础项：十进制整数 `10..999`（`NUMBER_POOL`，990 个，构造集 30 / 独立
  测试集 960）。
- 顶点（5 个）：`num_decimal`、`num_binary`、`num_hex`（小写 `a-f`）、
  `num_roman`（罗马数字）、`num_english`（英文数词，美式，无 "and"）。
- 边（7 条）：4 条"从十进制出发"的边，加 `e_binary_hex`、
  `e_binary_roman`、`e_hex_english` 三条"第二跳"边。
- 面（3 个，均为 3 边三角形，路径一致性）：
  - `face_decimal_binary_hex`：`decimal→binary→hex` 与 `decimal→hex`
  - `face_decimal_binary_roman`：`decimal→binary→roman` 与
    `decimal→roman`
  - `face_decimal_hex_english`：`decimal→hex→english` 与
    `decimal→english`

  这些一致性在数学上是平凡但精确的（进制/罗马数字/英文数词都是十进制值
  的双射编码，二次编码不会引入偏差），仍然用 `validate_oracle()` 逐项断
  言，作为规约自检而非假设。

### 2.3 日期域（`date`）

- 基础项：`10..999`? 不——是 1901–2099 年间的合法 ISO 日期字符串
  （`DATE_POOL`，220 个，构造集 30 / 独立测试集 190）；月份名/星期名全部
  硬编码为英文，不依赖系统 locale。
- 顶点（4 个）：`date_iso`（`YYYY-MM-DD`）、`date_us`（`MM/DD/YYYY`）、
  `date_text`（`March 5, 2021` 风格）、`date_weekday`（`Friday` 风格）。
- 边（6 条）：`e_iso_us`、`e_iso_text`、`e_iso_weekday`、`e_us_text`、
  `e_us_weekday`、`e_text_weekday`（4 个顶点两两之间除 `us↔text` 外的完
  全图，足以支撑 3 个三角形）。
- 面（3 个，均为 3 边三角形）：
  - `face_iso_us_text`：`iso→us→text` 与 `iso→text`
  - `face_iso_us_weekday`：`iso→us→weekday` 与 `iso→weekday`
  - `face_iso_text_weekday`：`iso→text→weekday` 与 `iso→weekday`

  注意后两个面共享同一条"直达边"任务 `iso_to_weekday`（1 跳路径），这是
  有意为之：同一个基线任务被两个不同的三角形分别拿来跟两条不同的两跳路
  径比较，`face` 字段用于区分探针行属于哪个面。

## 3. 迁移任务（不属于复形，独立评估）

三个三步组合任务，`kind='transfer'`，检验模型能否把已验证的两步一致性能
力泛化到训练/评测中从未出现过的第三跳：

- `reverse_then_upper_then_caesar3`（字符串域）：`caesar3(upper(reverse(w)))`
- `decimal_to_binary_to_hex_to_english`（数字域）：串起
  `e_decimal_binary -> e_binary_hex -> e_hex_english` 三条已有边
- `iso_to_us_to_text_to_weekday`（日期域）：串起
  `e_iso_us -> e_us_text -> e_text_weekday` 三条已有边

## 4. 构造集 / 独立集划分

每个域的基础项池被固定种子（`_split_pool`，见 `oracle.py`）划分为两个不
相交子集：

- **构造集**（`DEMO_ITEMS[domain]`，30 项/域）：只用于 few-shot 提示模板
  里的演示例子，从不出现在任何被打分的探针行里。
- **独立集**（`TEST_ITEMS[domain]`，字符串 290 / 数字 960 / 日期 190）：
  所有边/顶点/面/迁移任务的题项都从这里采样（`make_probes.py` 用
  `sha256(key)` 派生的确定性种子采样，同一 key 每次运行结果相同）。

这保证任何被打分的题项都不曾作为演示例子出现过。不同任务之间题项允许重
叠（例如同一个词既是 `e_word_upper` 的题项、也是某个面的题项）——这不构
成泄漏，因为整个流程是纯推理（无梯度更新），检查点在评估前从未见过这份
数据。

## 5. 已修正：字符串域用 `exact` 评分

`stage_ab/code/zz_score.py` 早先只有 `schema=['text']`：NFKC 归一化、
**小写化**、去标点、去空白后再比较。这导致 `upper`（转大写）和
`alternate`（隔位大写）这两个**纯大小写变换**——不改变字符恒等或顺
序——在打分时"什么都不做（原样照抄）"和"正确转换大小写"完全不可区
分，二者都会被记为 `correct=1`。

`zz_score.py` 现已增加 `schema` 的第三种取值 `'exact'`：`field_equal` 对
`'exact'` 做 `pred.strip() == gold.strip()`——大小写敏感、只去首尾空
白，不做 NFKC/小写化/去标点/去内部空白（`field_valid` 对 `'exact'` 仍只
要求非空）。`public_ckpt/code/oracle.py::schema_for_domain(domain)` 据此
按域分配 schema，`make_probes.py::make_row` 直接调用它：

- **字符串域**（`domain == 'string'`）→ `schema=['exact']`：该域全部
  gold 值都是不含空格/标点的单个 token（单词经过大小写/反转/凯撒变换的
  结果），切到 `'exact'` 在标点/空白维度上没有任何代价，换来的是
  `upper`/`alternate` 终于能与"原样照抄"区分开。
- **数字域 / 日期域**（`domain in ('number', 'date')`）→ 保持
  `schema=['text']`：这两个域的 gold 值经常带有对任务无关紧要的标点/空
  白（英文数词里的连字符 `forty-two`、长文本日期里的逗号
  `March 5, 2021`），而 `'exact'` 只去首尾空白、不做内部归一化，切过去
  会让这两个域对这些无关的格式差异变得脆弱；而且这两个域里大小写从不
  承载语义（数字、罗马数字、十六进制、星期名的对错都与大小写无关），
  `'text'` 的小写化不损失任何信号。

`public_ckpt/code/oracle.py::PURE_CASE_EDGES` 精确列出了所有"纯大小写"
边（`EDGE_FN` 是 `str_upper` 或 `str_alternate` 且属于字符串域）：
`e_word_upper`、`e_word_alternate`、`e_reverse_upperreverse`（对已反转
的词做大写）、`e_caesar3_caesar3upper`（对已凯撒变换的词做大写）—— 共 4
条。`make_probes.py::self_check_case_sensitivity()` 对这 4 条边的全部
384 行（4 边 × 96 行/边）做回归断言：把"原样照抄输入"当作预测，
`zz_score.score(row, test_input)['correct']` 必须等于 `0`。这条断言现在
100% 通过（`data/PROBES_MANIFEST.json::case_sensitivity_check_rows ==
384`），证明修复确实生效——在旧的 `schema=['text']` 下，这条断言会失败
（照抄会被误判为 `correct=1`）。

涉及 `reverse`/`caesar3` 的边/面/迁移任务从一开始就不受这个问题影响
（它们的组合路径最终字符串会因大小写以外的原因——字母恒等或顺序——而
在错误时不匹配），oracle 内部对 `upper_then_reverse = reverse_then_upper`
等式的断言也一直是大小写敏感、精确成立的（见 `validate_oracle()`）；这
次修正解决的是"评分器无法体现 oracle 已有的大小写敏感性"这一层，不涉
及 oracle 本身。
