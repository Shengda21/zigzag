"""Pure-Python oracles for the public-checkpoint zigzag probe set.

This module is the single source of truth for the "atomic capability"
complex used by ``make_probes.py`` (deliverable 2) and ``cell_scores.py``
(deliverable 3). It defines, for three domains (string / number / date):

  * a set of **vertices** -- named representations, each computable from a
    base item (a word / an integer / an ISO date string) via a chain of
    **edges** -- named single-step transformations (``EDGE_FN``);
  * a set of **faces** -- pairs of composition paths from a domain's base
    vertex to a shared target vertex, whose outputs must agree exactly
    (this is the "consistency" claim a 2-cell encodes);
  * three **transfer** tasks -- 3-step compositions used only as an
    independent generalisation probe, not part of the CW complex.

No prediction, model or template logic lives here: this module only
computes *gold values* from *oracle functions*, and exposes a uniform
``TASKS`` registry so that ``make_probes.py`` can generate probe rows
without knowing which domain a task belongs to.

Everything here is deterministic pure Python (no numpy/torch), so it can
be imported both by the probe generator and by any downstream re-scoring
script without side effects.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Make the sibling ``cw_zigzag`` module importable without copying/modifying
# anything under zigzag_v5/crosslingual/.
# ---------------------------------------------------------------------------
_ZZ_ROOT = Path(__file__).resolve().parents[2]
_CW_CODE_DIR = _ZZ_ROOT / 'crosslingual' / 'code'
if str(_CW_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CW_CODE_DIR))

SPEC_DIR = Path(__file__).resolve().parents[1] / 'spec'
COMPLEX_JSON_PATH = SPEC_DIR / 'COMPLEX.json'


def _seed_for(key: str) -> int:
    """Deterministic integer seed derived from a string key (stable across
    processes/Python versions, unlike the builtin ``hash()``)."""
    return int(hashlib.sha256(key.encode('utf8')).hexdigest()[:16], 16)


# ===========================================================================
# String domain
# ===========================================================================

WORDS: List[str] = [
    'academy', 'account', 'address', 'adequate', 'airline', 'arrival', 'bacteria', 'banana',
    'calcium', 'chair', 'chamber', 'champion', 'channel', 'chess', 'classify', 'cloak',
    'collapse', 'colonial', 'comment', 'commerce', 'commonly', 'conduct', 'congress', 'country',
    'cream', 'creation', 'creative', 'cultural', 'currency', 'dessert', 'diagnose', 'dialogue',
    'diameter', 'diamond', 'diary', 'discuss', 'disease', 'display', 'dividend', 'dominion',
    'drama', 'dreamers', 'dress', 'dropdown', 'dynamics', 'economy', 'educated', 'eloquent',
    'endeavor', 'episode', 'escape', 'escaped', 'essence', 'eternal', 'evening', 'exercise',
    'explicit', 'extended', 'familiar', 'family', 'fancy', 'fashion', 'favorite', 'feeling',
    'fence', 'fiction', 'flexible', 'florist', 'flower', 'flute', 'focus', 'football',
    'forecast', 'fragment', 'freedom', 'friend', 'garbage', 'garden', 'generate', 'glacier',
    'globe', 'glory', 'gorilla', 'grace', 'graphite', 'grasp', 'group', 'guest',
    'harvest', 'honey', 'humor', 'hundred', 'husband', 'imagine', 'imperial', 'impulse',
    'incense', 'index', 'insight', 'inspired', 'instance', 'interact', 'isolated', 'jacket',
    'jolly', 'judge', 'knock', 'label', 'library', 'lifetime', 'lockdown', 'longhorn',
    'maritime', 'melodic', 'message', 'meter', 'migraine', 'minimize', 'mission', 'mistake',
    'modern', 'moist', 'monkey', 'mouth', 'nature', 'neighbor', 'nerve', 'netscape',
    'nobility', 'noble', 'number', 'nutshell', 'obvious', 'olive', 'ongoing', 'onion',
    'operate', 'orchard', 'order', 'otter', 'outbreak', 'outdated', 'overlook', 'overpass',
    'overseas', 'oxygen', 'paradox', 'particle', 'partner', 'paste', 'patrol', 'pattern',
    'pearl', 'peculiar', 'penalty', 'pivot', 'platform', 'playful', 'point', 'position',
    'positive', 'possible', 'powerful', 'presence', 'pressure', 'pride', 'program', 'progress',
    'protocol', 'province', 'prune', 'punch', 'quality', 'question', 'radiator', 'rally',
    'range', 'reading', 'rebel', 'receipt', 'receiver', 'redesign', 'region', 'relative',
    'relaxing', 'relieve', 'religion', 'renovate', 'replace', 'republic', 'require', 'rescue',
    'resemble', 'resident', 'respond', 'restore', 'reunion', 'rewrite', 'right', 'ripple',
    'risky', 'river', 'rocky', 'romance', 'royal', 'salsa', 'salty', 'sample',
    'science', 'screw', 'seaside', 'season', 'seating', 'secret', 'section', 'seminar',
    'sequence', 'service', 'session', 'shape', 'shell', 'shock', 'sight', 'silence',
    'simple', 'skate', 'sleep', 'slice', 'slope', 'smart', 'smile', 'smoke',
    'soldier', 'speak', 'spectra', 'spectrum', 'spell', 'spending', 'spike', 'stadium',
    'stain', 'stake', 'stamina', 'stamp', 'stand', 'standing', 'standout', 'stardom',
    'starfish', 'starting', 'station', 'stimulus', 'stock', 'stone', 'storage', 'store',
    'straight', 'stray', 'strip', 'stubborn', 'study', 'stunning', 'suite', 'superior',
    'supply', 'surface', 'surgery', 'suspense', 'sustain', 'swamp', 'sweep', 'sweeping',
    'syllable', 'symptom', 'table', 'taste', 'taxpayer', 'teach', 'teacher', 'teamwork',
    'template', 'temple', 'tennis', 'thing', 'ticket', 'today', 'tower', 'track',
    'trail', 'tranquil', 'transfer', 'trouble', 'truth', 'tunnel', 'unity', 'unripe',
    'unstable', 'update', 'upright', 'valiant', 'various', 'vicinity', 'video', 'vintage',
    'visual', 'voice', 'volatile', 'voucher', 'weather', 'welfare', 'wellness', 'westward',
    'whatever', 'wheel', 'while', 'whole', 'widow', 'wireless', 'worry', 'yield',
]
assert len(WORDS) == len(set(WORDS)) == 320
assert all(len(w) in range(5, 9) and w.isalpha() and w.islower() for w in WORDS)


def str_upper(s: str) -> str:
    """Uppercase every letter (case transform; order/identity unchanged)."""
    return s.upper()


def str_reverse(s: str) -> str:
    """Reverse the character order of the string."""
    return s[::-1]


def str_caesar3(s: str) -> str:
    """Caesar-cipher shift every letter by +3, wrapping within its own case
    alphabet (``a``..``z`` and ``A``..``Z`` separately); non-letters pass
    through unchanged."""
    out = []
    for c in s:
        if 'a' <= c <= 'z':
            out.append(chr((ord(c) - 97 + 3) % 26 + 97))
        elif 'A' <= c <= 'Z':
            out.append(chr((ord(c) - 65 + 3) % 26 + 65))
        else:
            out.append(c)
    return ''.join(out)


def str_alternate(s: str) -> str:
    """Alternate-case: uppercase characters at even positions (0, 2, 4, ...),
    lowercase at odd positions."""
    return ''.join(c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(s))


STRING_OPS: Dict[str, Callable[[str], str]] = {
    'upper': str_upper, 'reverse': str_reverse, 'caesar3': str_caesar3, 'alternate': str_alternate,
}

# ===========================================================================
# Number domain
# ===========================================================================

_ONES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
          'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen',
          'eighteen', 'nineteen']
_TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
_ROMAN_VALS: Tuple[Tuple[int, str], ...] = (
    (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'), (50, 'L'),
    (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I'),
)


def to_roman(n: int) -> str:
    """Classic greedy Roman-numeral encoding, valid for 1 <= n < 4000."""
    if not (0 < n < 4000):
        raise ValueError(f'to_roman only supports 1..3999, got {n}')
    out = []
    for val, sym in _ROMAN_VALS:
        while n >= val:
            out.append(sym)
            n -= val
    return ''.join(out)


def to_english(n: int) -> str:
    """Number-to-English-words, American style, no 'and' (0 <= n <= 999)."""
    if not (0 <= n <= 999):
        raise ValueError(f'to_english only supports 0..999, got {n}')
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + ('' if o == 0 else '-' + _ONES[o])
    h, rem = divmod(n, 100)
    s = _ONES[h] + ' hundred'
    return s if rem == 0 else s + ' ' + to_english(rem)


def to_binary(n: int) -> str:
    """Plain binary string, no ``0b`` prefix."""
    return format(n, 'b')


def to_hex(n: int) -> str:
    """Plain lowercase hexadecimal string, no ``0x`` prefix."""
    return format(n, 'x')


NUMBER_POOL: List[int] = list(range(10, 1000))  # 990 integers

# ===========================================================================
# Date domain
# ===========================================================================

MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
               'September', 'October', 'November', 'December']
WEEKDAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def iso_value(d: date) -> str:
    """ISO-8601 date string, e.g. ``2021-03-05`` (locale-independent)."""
    return d.isoformat()


def us_value(d: date) -> str:
    """US date format, e.g. ``03/05/2021`` (locale-independent, hardcoded)."""
    return f'{d.month:02d}/{d.day:02d}/{d.year:04d}'


def text_value(d: date) -> str:
    """Long text date, e.g. ``March 5, 2021`` (locale-independent, no leading
    zero on the day)."""
    return f'{MONTH_NAMES[d.month - 1]} {d.day}, {d.year}'


def weekday_value(d: date) -> str:
    """Full English weekday name, e.g. ``Friday`` (locale-independent)."""
    return WEEKDAY_NAMES[d.weekday()]


def parse_iso(s: str) -> date:
    """Inverse of :func:`iso_value`."""
    return date.fromisoformat(s.strip())


def parse_us(s: str) -> date:
    """Inverse of :func:`us_value`."""
    m, d, y = s.strip().split('/')
    return date(int(y), int(m), int(d))


def parse_text(s: str) -> date:
    """Inverse of :func:`text_value` (parses our own canonical format only)."""
    month_part, rest = s.strip().split(' ', 1)
    day_part, year_part = rest.split(',')
    month = MONTH_NAMES.index(month_part) + 1
    return date(int(year_part.strip()), month, int(day_part.strip()))


def _make_date_pool(n: int, seed: int) -> List[str]:
    """``n`` distinct pseudo-random valid ISO dates in 1901-2099."""
    import calendar
    rng = random.Random(seed)
    out = set()
    while len(out) < n:
        year = rng.randint(1901, 2099)
        month = rng.randint(1, 12)
        day = rng.randint(1, calendar.monthrange(year, month)[1])
        out.add(date(year, month, day).isoformat())
    return sorted(out)


DATE_POOL: List[str] = _make_date_pool(220, seed=20260904)

# ===========================================================================
# Edge functions: name -> (current representation string -> next string)
# ===========================================================================

EDGE_FN: Dict[str, Callable[[str], str]] = {
    # string domain
    'e_word_upper': str_upper,
    'e_word_reverse': str_reverse,
    'e_word_caesar3': str_caesar3,
    'e_word_alternate': str_alternate,
    'e_upper_upperreverse': str_reverse,       # apply reverse to an already-uppercased word
    'e_reverse_upperreverse': str_upper,       # apply upper to an already-reversed word
    'e_caesar3_caesar3upper': str_upper,       # apply upper to an already-caesar3'd word
    'e_upper_caesar3upper': str_caesar3,       # apply caesar3 to an already-uppercased word
    'e_reverse_reversecaesar3': str_caesar3,   # apply caesar3 to an already-reversed word
    'e_caesar3_reversecaesar3': str_reverse,   # apply reverse to an already-caesar3'd word
    # number domain
    'e_decimal_binary': lambda s: to_binary(int(s)),
    'e_decimal_hex': lambda s: to_hex(int(s)),
    'e_decimal_roman': lambda s: to_roman(int(s)),
    'e_decimal_english': lambda s: to_english(int(s)),
    'e_binary_hex': lambda s: to_hex(int(s, 2)),
    'e_binary_roman': lambda s: to_roman(int(s, 2)),
    'e_hex_english': lambda s: to_english(int(s, 16)),
    # date domain
    'e_iso_us': lambda s: us_value(parse_iso(s)),
    'e_iso_text': lambda s: text_value(parse_iso(s)),
    'e_iso_weekday': lambda s: weekday_value(parse_iso(s)),
    'e_us_text': lambda s: text_value(parse_us(s)),
    'e_us_weekday': lambda s: weekday_value(parse_us(s)),
    'e_text_weekday': lambda s: weekday_value(parse_text(s)),
}

EDGES: Dict[str, Dict[str, str]] = {
    'e_word_upper': dict(src='str_word', dst='str_upper', domain='string'),
    'e_word_reverse': dict(src='str_word', dst='str_reverse', domain='string'),
    'e_word_caesar3': dict(src='str_word', dst='str_caesar3', domain='string'),
    'e_word_alternate': dict(src='str_word', dst='str_alternate', domain='string'),
    'e_upper_upperreverse': dict(src='str_upper', dst='str_upper_reverse', domain='string'),
    'e_reverse_upperreverse': dict(src='str_reverse', dst='str_upper_reverse', domain='string'),
    'e_caesar3_caesar3upper': dict(src='str_caesar3', dst='str_caesar3_upper', domain='string'),
    'e_upper_caesar3upper': dict(src='str_upper', dst='str_caesar3_upper', domain='string'),
    'e_reverse_reversecaesar3': dict(src='str_reverse', dst='str_reverse_caesar3', domain='string'),
    'e_caesar3_reversecaesar3': dict(src='str_caesar3', dst='str_reverse_caesar3', domain='string'),
    'e_decimal_binary': dict(src='num_decimal', dst='num_binary', domain='number'),
    'e_decimal_hex': dict(src='num_decimal', dst='num_hex', domain='number'),
    'e_decimal_roman': dict(src='num_decimal', dst='num_roman', domain='number'),
    'e_decimal_english': dict(src='num_decimal', dst='num_english', domain='number'),
    'e_binary_hex': dict(src='num_binary', dst='num_hex', domain='number'),
    'e_binary_roman': dict(src='num_binary', dst='num_roman', domain='number'),
    'e_hex_english': dict(src='num_hex', dst='num_english', domain='number'),
    'e_iso_us': dict(src='date_iso', dst='date_us', domain='date'),
    'e_iso_text': dict(src='date_iso', dst='date_text', domain='date'),
    'e_iso_weekday': dict(src='date_iso', dst='date_weekday', domain='date'),
    'e_us_text': dict(src='date_us', dst='date_text', domain='date'),
    'e_us_weekday': dict(src='date_us', dst='date_weekday', domain='date'),
    'e_text_weekday': dict(src='date_text', dst='date_weekday', domain='date'),
}

EDGE_HEADERS: Dict[str, str] = {
    'e_word_upper': 'Task: convert the word to uppercase.',
    'e_word_reverse': 'Task: reverse the letters of the word.',
    'e_word_caesar3': 'Task: apply a Caesar cipher shift of 3 (a->d, ..., z->c), keeping the original letter case.',
    'e_word_alternate': 'Task: capitalize every other letter, starting from the first letter (positions 0,2,4,... uppercase; the rest lowercase).',
    'e_upper_upperreverse': 'Task: reverse the letters. The input is already uppercase.',
    'e_reverse_upperreverse': 'Task: convert to uppercase. The input is already reversed.',
    'e_caesar3_caesar3upper': 'Task: convert to uppercase. The input already has a Caesar+3 shift applied.',
    'e_upper_caesar3upper': 'Task: apply a Caesar cipher shift of 3. The input is already uppercase.',
    'e_reverse_reversecaesar3': 'Task: apply a Caesar cipher shift of 3. The input is already reversed.',
    'e_caesar3_reversecaesar3': 'Task: reverse the letters. The input already has a Caesar+3 shift applied.',
    'e_decimal_binary': 'Task: convert the decimal number to binary (base 2), no prefix.',
    'e_decimal_hex': 'Task: convert the decimal number to hexadecimal (base 16, lowercase a-f), no prefix.',
    'e_decimal_roman': 'Task: convert the decimal number to a Roman numeral.',
    'e_decimal_english': 'Task: spell the decimal number out in English words.',
    'e_binary_hex': 'Task: the input is a binary number; convert it to hexadecimal (base 16, lowercase), no prefix.',
    'e_binary_roman': 'Task: the input is a binary number; convert it to a Roman numeral.',
    'e_hex_english': 'Task: the input is a hexadecimal number; spell its decimal value out in English words.',
    'e_iso_us': 'Task: convert the ISO date (YYYY-MM-DD) to US format (MM/DD/YYYY).',
    'e_iso_text': 'Task: convert the ISO date (YYYY-MM-DD) to long text format (Month D, YYYY).',
    'e_iso_weekday': 'Task: give the day of the week for the ISO date (YYYY-MM-DD).',
    'e_us_text': 'Task: the input is a US-format date (MM/DD/YYYY); convert it to long text format (Month D, YYYY).',
    'e_us_weekday': 'Task: the input is a US-format date (MM/DD/YYYY); give the day of the week.',
    'e_text_weekday': 'Task: the input is a long text date (Month D, YYYY); give the day of the week.',
}

# ===========================================================================
# Vertices
# ===========================================================================

VERTEX_DOMAIN: Dict[str, str] = {
    'str_word': 'string', 'str_upper': 'string', 'str_reverse': 'string', 'str_caesar3': 'string',
    'str_alternate': 'string', 'str_upper_reverse': 'string', 'str_caesar3_upper': 'string',
    'str_reverse_caesar3': 'string',
    'num_decimal': 'number', 'num_binary': 'number', 'num_hex': 'number', 'num_roman': 'number',
    'num_english': 'number',
    'date_iso': 'date', 'date_us': 'date', 'date_text': 'date', 'date_weekday': 'date',
}

BASE_VERTEX: Dict[str, str] = {'string': 'str_word', 'number': 'num_decimal', 'date': 'date_iso'}

# canonical edge-path from the domain's base vertex to each non-base vertex
VERTEX_PATH: Dict[str, List[str]] = {
    'str_word': [],
    'str_upper': ['e_word_upper'],
    'str_reverse': ['e_word_reverse'],
    'str_caesar3': ['e_word_caesar3'],
    'str_alternate': ['e_word_alternate'],
    'str_upper_reverse': ['e_word_upper', 'e_upper_upperreverse'],
    'str_caesar3_upper': ['e_word_caesar3', 'e_caesar3_caesar3upper'],
    'str_reverse_caesar3': ['e_word_reverse', 'e_reverse_reversecaesar3'],
    'num_decimal': [],
    'num_binary': ['e_decimal_binary'],
    'num_hex': ['e_decimal_hex'],
    'num_roman': ['e_decimal_roman'],
    'num_english': ['e_decimal_english'],
    'date_iso': [],
    'date_us': ['e_iso_us'],
    'date_text': ['e_iso_text'],
    'date_weekday': ['e_iso_weekday'],
}

BASE_RENDER: Dict[str, Callable] = {
    'string': lambda item: item,          # item is already the raw lowercase word
    'number': lambda item: str(item),     # item is a python int
    'date': lambda item: item,            # item is already the ISO string
}


def run_path(domain: str, path: Sequence[str], item) -> str:
    """Apply a chain of edge names (see :data:`EDGE_FN`) starting from the
    domain's base rendering of `item`."""
    v = BASE_RENDER[domain](item)
    for edge in path:
        v = EDGE_FN[edge](v)
    return v


def vertex_value(vertex: str, item) -> str:
    """Canonical string value of `vertex` for base item `item`."""
    domain = VERTEX_DOMAIN[vertex]
    return run_path(domain, VERTEX_PATH[vertex], item)


VERTEX_HEADER = 'Task: repeat the input exactly, unchanged.'

# ===========================================================================
# Faces: two composition paths (from the domain base vertex) that must agree
# ===========================================================================

# task_name -> dict(domain, path=[edge names from base vertex], header)
COMPOSITION_TASKS: Dict[str, Dict] = {
    'upper_then_reverse': dict(domain='string', path=['e_word_upper', 'e_upper_upperreverse'],
        header='Task: first convert the word to uppercase, then reverse the letters.'),
    'reverse_then_upper': dict(domain='string', path=['e_word_reverse', 'e_reverse_upperreverse'],
        header='Task: first reverse the letters of the word, then convert it to uppercase.'),
    'caesar3_then_upper': dict(domain='string', path=['e_word_caesar3', 'e_caesar3_caesar3upper'],
        header='Task: first apply a Caesar cipher shift of 3, then convert the result to uppercase.'),
    'upper_then_caesar3': dict(domain='string', path=['e_word_upper', 'e_upper_caesar3upper'],
        header='Task: first convert the word to uppercase, then apply a Caesar cipher shift of 3.'),
    'reverse_then_caesar3': dict(domain='string', path=['e_word_reverse', 'e_reverse_reversecaesar3'],
        header='Task: first reverse the letters of the word, then apply a Caesar cipher shift of 3.'),
    'caesar3_then_reverse': dict(domain='string', path=['e_word_caesar3', 'e_caesar3_reversecaesar3'],
        header='Task: first apply a Caesar cipher shift of 3, then reverse the letters.'),
    'decimal_via_binary_to_hex': dict(domain='number', path=['e_decimal_binary', 'e_binary_hex'],
        header='Task: convert the decimal number to binary, then convert that binary value to hexadecimal; give the final hexadecimal value.'),
    'decimal_to_hex': dict(domain='number', path=['e_decimal_hex'], header=EDGE_HEADERS['e_decimal_hex']),
    'decimal_via_binary_to_roman': dict(domain='number', path=['e_decimal_binary', 'e_binary_roman'],
        header='Task: convert the decimal number to binary, then convert that binary value to a Roman numeral; give the final Roman numeral.'),
    'decimal_to_roman': dict(domain='number', path=['e_decimal_roman'], header=EDGE_HEADERS['e_decimal_roman']),
    'decimal_via_hex_to_english': dict(domain='number', path=['e_decimal_hex', 'e_hex_english'],
        header='Task: convert the decimal number to hexadecimal, then spell that value out in English words; give the final English words.'),
    'decimal_to_english': dict(domain='number', path=['e_decimal_english'], header=EDGE_HEADERS['e_decimal_english']),
    'iso_via_us_to_text': dict(domain='date', path=['e_iso_us', 'e_us_text'],
        header='Task: convert the ISO date to US format, then convert that US-format date to long text format; give the final long text date.'),
    'iso_to_text': dict(domain='date', path=['e_iso_text'], header=EDGE_HEADERS['e_iso_text']),
    'iso_via_us_to_weekday': dict(domain='date', path=['e_iso_us', 'e_us_weekday'],
        header='Task: convert the ISO date to US format, then give the day of the week for that date; give the final weekday.'),
    'iso_to_weekday': dict(domain='date', path=['e_iso_weekday'], header=EDGE_HEADERS['e_iso_weekday']),
    'iso_via_text_to_weekday': dict(domain='date', path=['e_iso_text', 'e_text_weekday'],
        header='Task: convert the ISO date to long text format, then give the day of the week for that date; give the final weekday.'),
}

# face name -> dict(cycle=[edge names forming the closed boundary],
#                    composition_tasks=[pathA task, pathB task])
FACES: Dict[str, Dict] = {
    'face_upper_reverse': dict(
        cycle=['e_word_upper', 'e_upper_upperreverse', 'e_word_reverse', 'e_reverse_upperreverse'],
        composition_tasks=['upper_then_reverse', 'reverse_then_upper']),
    'face_caesar3_upper': dict(
        cycle=['e_word_caesar3', 'e_caesar3_caesar3upper', 'e_word_upper', 'e_upper_caesar3upper'],
        composition_tasks=['caesar3_then_upper', 'upper_then_caesar3']),
    'face_reverse_caesar3': dict(
        cycle=['e_word_reverse', 'e_reverse_reversecaesar3', 'e_word_caesar3', 'e_caesar3_reversecaesar3'],
        composition_tasks=['reverse_then_caesar3', 'caesar3_then_reverse']),
    'face_decimal_binary_hex': dict(
        cycle=['e_decimal_binary', 'e_binary_hex', 'e_decimal_hex'],
        composition_tasks=['decimal_via_binary_to_hex', 'decimal_to_hex']),
    'face_decimal_binary_roman': dict(
        cycle=['e_decimal_binary', 'e_binary_roman', 'e_decimal_roman'],
        composition_tasks=['decimal_via_binary_to_roman', 'decimal_to_roman']),
    'face_decimal_hex_english': dict(
        cycle=['e_decimal_hex', 'e_hex_english', 'e_decimal_english'],
        composition_tasks=['decimal_via_hex_to_english', 'decimal_to_english']),
    'face_iso_us_text': dict(
        cycle=['e_iso_us', 'e_us_text', 'e_iso_text'],
        composition_tasks=['iso_via_us_to_text', 'iso_to_text']),
    'face_iso_us_weekday': dict(
        cycle=['e_iso_us', 'e_us_weekday', 'e_iso_weekday'],
        composition_tasks=['iso_via_us_to_weekday', 'iso_to_weekday']),
    'face_iso_text_weekday': dict(
        cycle=['e_iso_text', 'e_text_weekday', 'e_iso_weekday'],
        composition_tasks=['iso_via_text_to_weekday', 'iso_to_weekday']),
}

# ===========================================================================
# Transfer tasks: independent 3-step compositions, not part of the complex
# ===========================================================================


def _transfer_string(word: str) -> str:
    return str_caesar3(str_upper(str_reverse(word)))


def _transfer_number(n: int) -> str:
    return to_english(int(to_hex(int(to_binary(n), 2)), 16))


def _transfer_date(iso: str) -> str:
    return weekday_value(parse_text(text_value(parse_us(us_value(parse_iso(iso))))))


TRANSFER_TASKS: Dict[str, Dict] = {
    'reverse_then_upper_then_caesar3': dict(
        domain='string', fn=_transfer_string,
        header='Task: reverse the letters of the word, then convert to uppercase, then apply a Caesar cipher shift of 3; give the final result.'),
    'decimal_to_binary_to_hex_to_english': dict(
        domain='number', fn=lambda n: _transfer_number(int(n)),
        header='Task: convert the decimal number to binary, then to hexadecimal, then spell the value out in English words; give the final English words.'),
    'iso_to_us_to_text_to_weekday': dict(
        domain='date', fn=_transfer_date,
        header='Task: convert the ISO date to US format, then to long text format, then give the day of the week; give the final weekday.'),
}

# ===========================================================================
# Demo / test item pool splits (construction set vs held-out set)
# ===========================================================================

N_DEMO = dict(string=30, number=30, date=30)


def _split_pool(pool: Sequence, seed: int, n_demo: int) -> Tuple[List, List]:
    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    demo, test = shuffled[:n_demo], shuffled[n_demo:]
    return demo, sorted(test, key=lambda x: str(x))


DEMO_ITEMS: Dict[str, List] = {}
TEST_ITEMS: Dict[str, List] = {}
DEMO_ITEMS['string'], TEST_ITEMS['string'] = _split_pool(WORDS, 20260911, N_DEMO['string'])
DEMO_ITEMS['number'], TEST_ITEMS['number'] = _split_pool(NUMBER_POOL, 20260912, N_DEMO['number'])
DEMO_ITEMS['date'], TEST_ITEMS['date'] = _split_pool(DATE_POOL, 20260913, N_DEMO['date'])
assert not (set(DEMO_ITEMS['string']) & set(TEST_ITEMS['string']))
assert not (set(DEMO_ITEMS['number']) & set(TEST_ITEMS['number']))
assert not (set(DEMO_ITEMS['date']) & set(TEST_ITEMS['date']))

# ===========================================================================
# Unified task registry used by make_probes.py
# ===========================================================================


def compute(kind: str, name: str, item) -> Tuple[str, str]:
    """Return ``(input_text, gold_text)`` for a task of the given `kind`
    (``'edge' | 'vertex' | 'composition' | 'transfer'``) and `name`, applied
    to base item `item`.
    """
    if kind == 'edge':
        e = EDGES[name]
        return vertex_value(e['src'], item), vertex_value(e['dst'], item)
    if kind == 'vertex':
        v = vertex_value(name, item)
        return v, v
    if kind == 'composition':
        t = COMPOSITION_TASKS[name]
        inp = BASE_RENDER[t['domain']](item)
        return inp, run_path(t['domain'], t['path'], item)
    if kind == 'transfer':
        t = TRANSFER_TASKS[name]
        inp = BASE_RENDER[t['domain']](item)
        return inp, t['fn'](item)
    raise ValueError(f'unknown kind {kind!r}')


def domain_of(kind: str, name: str) -> str:
    """Domain ('string'/'number'/'date') of a task of the given kind/name."""
    if kind == 'edge':
        return EDGES[name]['domain']
    if kind == 'vertex':
        return VERTEX_DOMAIN[name]
    if kind == 'composition':
        return COMPOSITION_TASKS[name]['domain']
    if kind == 'transfer':
        return TRANSFER_TASKS[name]['domain']
    raise ValueError(f'unknown kind {kind!r}')


def header_of(kind: str, name: str) -> str:
    """Few-shot template header (task instruction) for a task."""
    if kind == 'edge':
        return EDGE_HEADERS[name]
    if kind == 'vertex':
        return VERTEX_HEADER
    if kind == 'composition':
        return COMPOSITION_TASKS[name]['header']
    if kind == 'transfer':
        return TRANSFER_TASKS[name]['header']
    raise ValueError(f'unknown kind {kind!r}')


def schema_for_domain(domain: str) -> str:
    """``zz_score`` schema kind to use for a domain's probe rows.

    ``string`` uses ``'exact'`` (case-sensitive, whitespace-stripped only):
    the string domain's gold values are single space-free tokens, so
    switching from ``'text'`` to ``'exact'`` costs nothing on the
    punctuation/whitespace front while finally making the pure case-only
    edges (``upper``, ``alternate``) distinguishable from a verbatim copy
    (``'text'`` lowercases before comparing, which made those edges
    unscoreable -- see spec/ATOMIC_TASKS.md Sec.5 and README.md Sec.6).

    ``number`` and ``date`` stay on ``'text'``: their gold values routinely
    contain punctuation/spacing that is incidental to the tested capability
    (a comma in ``'March 5, 2021'``, a hyphen in ``'forty-two'``), and
    ``'exact'`` only strips *outer* whitespace -- it would make those two
    domains fragile to harmless formatting variance (comma spacing, hyphen
    vs. space) that has nothing to do with whether the checkpoint computed
    the right value. Case is never the distinguishing feature in those two
    domains (digits, Roman numerals, weekday names, hex digits are all
    case-insensitive in meaning), so ``'text'``'s lowercasing loses no
    signal there.
    """
    return 'exact' if domain == 'string' else 'text'


# Edges whose EDGE_FN is a *pure* case transform (str_upper / str_alternate):
# exactly the edges that were unscoreable under the old 'text' schema
# (copying the input verbatim used to score correct=1 there). Used by
# make_probes.py's self-check to assert the 'exact' schema now catches this.
PURE_CASE_EDGES: List[str] = sorted(
    name for name, fn in EDGE_FN.items()
    if EDGES[name]['domain'] == 'string' and fn in (str_upper, str_alternate)
)


# ===========================================================================
# COMPLEX.json export / load
# ===========================================================================


def complex_spec() -> Dict:
    """Build the ``{vertices, edges, faces}`` dict matching COMPLEX.json's
    schema, from this module's registries (single source of truth)."""
    return dict(
        vertices=sorted(VERTEX_DOMAIN.keys()),
        edges=[dict(name=n, src=e['src'], dst=e['dst']) for n, e in sorted(EDGES.items())],
        faces=[dict(name=n, cycle=f['cycle'], composition_tasks=f['composition_tasks'])
               for n, f in sorted(FACES.items())],
    )


def export_complex_json(path: Path = COMPLEX_JSON_PATH) -> None:
    """Write :func:`complex_spec` to `path` as pretty JSON."""
    path.write_text(json.dumps(complex_spec(), ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def load_complex():
    """Read spec/COMPLEX.json and build a :class:`cw_zigzag.CWComplex`,
    verifying that the full complex is closed and that ``betti`` can be
    computed. Returns ``(cx, spec_dict)``.
    """
    import cw_zigzag
    spec = json.loads(COMPLEX_JSON_PATH.read_text(encoding='utf8'))
    cells: List[Tuple[str, int, List[str]]] = []
    for v in spec['vertices']:
        cells.append((v, 0, []))
    for e in spec['edges']:
        cells.append((e['name'], 1, [e['src'], e['dst']]))
    for f in spec['faces']:
        cells.append((f['name'], 2, f['cycle']))
    cx = cw_zigzag.CWComplex(cells)
    full_mask = (1 << cx.n) - 1
    if not cx.is_closed(full_mask):
        raise AssertionError('COMPLEX.json does not form a closed CW complex')
    betti = cx.betti(full_mask)
    return cx, spec, betti


# ===========================================================================
# Oracle self-consistency checks
# ===========================================================================


def validate_oracle(n_samples: int = 25) -> Dict[str, int]:
    """Assert, on `n_samples` items per domain, that:

      1. every edge's ``EDGE_FN`` applied to its source vertex value equals
         its destination vertex value (edges are internally consistent with
         :func:`vertex_value`);
      2. for every face, its two composition paths produce *identical*
         output strings for every sampled item (the commuting-square /
         path-consistency claim the face encodes);
      3. every transfer task runs without error and returns a non-empty
         string.

    Raises ``AssertionError`` on any mismatch. Returns a dict of counts
    checked, for logging.
    """
    counts = dict(edges=0, faces=0, transfer=0)
    samples = {d: (TEST_ITEMS[d][:n_samples] if len(TEST_ITEMS[d]) >= n_samples else TEST_ITEMS[d])
               for d in ('string', 'number', 'date')}

    for name, e in EDGES.items():
        for item in samples[e['domain']]:
            got = EDGE_FN[name](vertex_value(e['src'], item))
            want = vertex_value(e['dst'], item)
            if got != want:
                raise AssertionError(f'edge {name!r} inconsistent with vertex_value on item {item!r}: {got!r} != {want!r}')
            counts['edges'] += 1

    for face, f in FACES.items():
        task_a, task_b = f['composition_tasks']
        domain = COMPOSITION_TASKS[task_a]['domain']
        for item in samples[domain]:
            _, out_a = compute('composition', task_a, item)
            _, out_b = compute('composition', task_b, item)
            if out_a != out_b:
                raise AssertionError(f'face {face!r} paths disagree on item {item!r}: {task_a}={out_a!r} vs {task_b}={out_b!r}')
            counts['faces'] += 1

    for name, t in TRANSFER_TASKS.items():
        for item in samples[t['domain']]:
            _, out = compute('transfer', name, item)
            if not out:
                raise AssertionError(f'transfer task {name!r} produced empty output on item {item!r}')
            counts['transfer'] += 1

    return counts


if __name__ == '__main__':
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    export_complex_json()
    cx, spec, betti = load_complex()
    checks = validate_oracle()
    print(f'wrote {COMPLEX_JSON_PATH}')
    print(f'cells: {cx.n} (vertices={len(spec["vertices"])}, edges={len(spec["edges"])}, faces={len(spec["faces"])})')
    print(f'betti of full complex: {betti}')
    print(f'oracle self-checks passed: {checks}')
