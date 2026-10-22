"""Instruction templates for the cross-lingual worlds.

Languages: en (specialised layer is TRAINED only in English; templates 0/1/2
train, 3 held-out paraphrase), zh (probe language that builds the complex),
ru and ja (held-out transfer languages, never trained, never used to build the
complex). The general layer (Belebele / XQuAD) is trained in en/zh/ru under
MULTI and in en only under MONO; format-copy is trained in en/zh/ru always.

Answer formats are language-independent: code sequences, 0/1 flags, A-D
letters or exact text spans, so a probe in any language is scored by the same
zz_score rules.
"""
from __future__ import annotations

LANGS = ('en', 'zh', 'ru', 'ja')
TRAIN_LANGS = ('en', 'zh', 'ru')
PROBE_TEMPLATES = {'en': (0, 1, 3), 'zh': (0, 1), 'ru': (0, 1), 'ja': (0, 1)}

SUFFIX = {
    'en': {0: 'Give only the answer in the required format, no explanation.',
           1: 'Answer only; separate fields with vertical bars and items with commas.',
           2: 'Complete the task without any additional remarks; fill in only the requested fields.',
           3: 'Reply with the fields in order, using bars and commas as separators, and add nothing else.'},
    'zh': {0: '只给规定格式的答案，不要解释。',
           1: '请仅填写答案；字段用竖线分隔，序列内部用逗号分隔。'},
    'ru': {0: 'Дайте только ответ в требуемом формате, без объяснений.',
           1: 'Только ответ; поля разделяйте вертикальной чертой, элементы — запятыми.'},
    'ja': {0: '指定形式の答えのみを出力し、説明はしないでください。',
           1: '答えのみを書いてください。フィールドは縦棒、要素はカンマで区切ります。'},
}

VERTEX = {
    'en': 'Read the code sequence {seq} and output the three-digit number of each code in the original order, without the letters.',
    'zh': '读取代码序列 {seq}，原顺序输出每个代码的三位数字编号，去掉字母。',
    'ru': 'Прочитайте последовательность кодов {seq} и выведите трёхзначный номер каждого кода в исходном порядке, без букв.',
    'ja': 'コード系列 {seq} を読み、各コードの3桁の番号を元の順序で、文字を除いて出力してください。',
}

EDGE = {
    'en': {0: 'Using the fixed codebook you have learned, convert the {s} sequence {seq} into a {d} sequence.',
           1: 'Apply the learned fixed codebook: map the {s} sequence {seq} to its {d} sequence.',
           2: 'Translate the {s} sequence {seq} into the corresponding {d} sequence under the fixed codebook.',
           3: 'According to the memorised fixed codebook, what is the {d} sequence for the {s} sequence {seq}?'},
    'zh': {0: '按已学习且固定的代码词典，把 {s} 序列 {seq} 转换成 {d} 序列。',
           1: '请依据固定词典，将 {s} 序列 {seq} 映射为对应的 {d} 序列。'},
    'ru': {0: 'Используя выученный фиксированный кодовый словарь, преобразуйте последовательность {s} {seq} в последовательность {d}.',
           1: 'По фиксированному словарю сопоставьте последовательности {s} {seq} соответствующую последовательность {d}.'},
    'ja': {0: '学習済みの固定コードブックに従って、{s} 系列 {seq} を {d} 系列に変換してください。',
           1: '固定辞書に基づき、{s} 系列 {seq} に対応する {d} 系列を答えてください。'},
}

JOINT = {
    'en': dict(body='Starting from the F sequence {F}, compare F->P with F->{m}->P. {ins} Output: direct P sequence|actual intermediate {m} sequence|P sequence obtained through that intermediate state|whether the two endpoints are identical (1 or 0).',
               override='Explicitly replace the intermediate state with {mid}.', keep='Keep the intermediate state obtained from the codebook.'),
    'zh': dict(body='从 F 序列 {F} 出发，比较 F->P 与 F->{m}->P。{ins}输出：直接P序列|实际中间{m}序列|沿该中间状态得到的P序列|两个终点是否相同(1或0)。',
               override='将中间状态明确替换为 {mid}。', keep='保持按词典得到的中间状态。'),
    'ru': dict(body='Начиная с последовательности F {F}, сравните F->P и F->{m}->P. {ins} Выведите: прямую последовательность P|фактическую промежуточную последовательность {m}|последовательность P, полученную через это промежуточное состояние|совпадают ли две конечные точки (1 или 0).',
               override='Явно замените промежуточное состояние на {mid}.', keep='Сохраните промежуточное состояние, полученное по словарю.'),
    'ja': dict(body='F 系列 {F} から出発し、F->P と F->{m}->P を比較してください。{ins}出力：直接のP系列|実際の中間{m}系列|その中間状態を経て得られるP系列|二つの終点が同一か(1または0)。',
               override='中間状態を {mid} に明示的に置き換えてください。', keep='コードブックから得られる中間状態を保持してください。'),
}

REPAIR = {
    'en': dict(body='Starting from the F sequence {F}, execute F->{m}->P. {ins} Observed intermediate state {om}, endpoint {op}. Output: expected intermediate {m} sequence|expected P endpoint|earliest error (none 0 / intermediate 1 / endpoint 2).'),
    'zh': dict(body='从 F 序列 {F} 出发执行 F->{m}->P。{ins}观察中间状态 {om}，终点 {op}。输出：应有中间{m}序列|应有P终点|最早错误(无错0/中间1/终点2)。'),
    'ru': dict(body='Начиная с последовательности F {F}, выполните F->{m}->P. {ins} Наблюдаемое промежуточное состояние {om}, конечная точка {op}. Выведите: ожидаемую промежуточную последовательность {m}|ожидаемую конечную точку P|самую раннюю ошибку (нет 0 / промежуточная 1 / конечная 2).'),
    'ja': dict(body='F 系列 {F} から F->{m}->P を実行してください。{ins}観測された中間状態 {om}、終点 {op}。出力：あるべき中間{m}系列|あるべきP終点|最初の誤り(なし0/中間1/終点2)。'),
}

FORMAT = {
    'en': 'Copy fields only, no codebook reasoning. Output the fixed slots: slot A = intermediate D, slot B = successor P, slot C = direct P, slot E = equality flag. The input fields are, in order: {fields}. Output exactly four fields separated by vertical bars, no explanation.',
    'zh': '只做字段复制，不做词典推理。请按固定槽位输出：槽A=中间D，槽B=后继P，槽C=直接P，槽E=相等标志。输入字段依次为：{fields}。只能输出四个字段，用竖线分隔，不能解释。',
    'ru': 'Только скопируйте поля, без рассуждений по словарю. Выведите фиксированные слоты: слот A = промежуточное D, слот B = последующее P, слот C = прямое P, слот E = флаг равенства. Входные поля по порядку: {fields}. Выведите ровно четыре поля через вертикальную черту, без объяснений.',
}

BELEBELE = {
    'en': '{passage}\n\nQuestion: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer with the letter only.',
    'zh': '{passage}\n\n问题：{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n只回答选项字母。',
    'ru': '{passage}\n\nВопрос: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nОтветьте только буквой.',
}

XQUAD = {
    'en': '{context}\n\nQuestion: {q}\nAnswer with the exact span from the passage only.',
    'zh': '{context}\n\n问题：{q}\n只用文中原文片段回答。',
    'ru': '{context}\n\nВопрос: {q}\nОтветьте только точным фрагментом из текста.',
}

PAD_PHRASE = {'en': ' Ignore padding:', 'zh': ' 忽略填充：', 'ru': ' Игнорируйте заполнение:', 'ja': ' パディングは無視してください：'}


def suffix(lang, t):
    return SUFFIX[lang][t]


def vertex_prompt(lang, t, seq):
    return VERTEX[lang].format(seq=seq) + ' ' + suffix(lang, t)


def edge_prompt(lang, t, s, d, seq):
    return EDGE[lang][t].format(s=s, d=d, seq=seq) + ' ' + suffix(lang, t)


def joint_prompt(lang, t, F, m, mid, override):
    j = JOINT[lang]
    ins = j['override'].format(mid=mid) if override else j['keep']
    return j['body'].format(F=F, m=m, ins=ins) + ' ' + suffix(lang, t)


def repair_prompt(lang, t, F, m, mid, override, om, op):
    j = JOINT[lang]
    ins = j['override'].format(mid=mid) if override else j['keep']
    return REPAIR[lang]['body'].format(F=F, m=m, ins=ins, om=om, op=op) + ' ' + suffix(lang, t)


def format_prompt(lang, t, fields):
    return FORMAT[lang].format(fields=fields) + ' ' + suffix(lang, t)


def belebele_prompt(lang, row):
    return BELEBELE[lang].format(passage=row['flores_passage'].strip(), q=row['question'].strip(), a=row['mc_answer1'].strip(),
                                 b=row['mc_answer2'].strip(), c=row['mc_answer3'].strip(), d=row['mc_answer4'].strip())


def xquad_prompt(lang, context, q):
    return XQUAD[lang].format(context=context.strip(), q=q.strip())
