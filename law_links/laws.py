"""Поиск упоминаний законов в тексте по базе law_aliases.json.

Упоминание закона собирается из компонентов в любом порядке:
вид акта (кодекс / ФЗ / закон / указ / распоряжение / основы / ФСБУ, ПБУ),
номер, дата и название. Склонения снимаются стеммингом, опечатки -
нечётким сравнением слов.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from rapidfuzz.distance import OSA

from .normalize import NUM, P, W, Tok, normalize, stem, tokenize

# --------------------------------------------------------------------------
# Типы актов
CODE, FZ, ZAKON, UKAZ, RASP, OSNOVY, STD = "code", "fz", "zakon", "ukaz", "rasp", "osnovy", "std"
LAWLIKE = {FZ, ZAKON, OSNOVY}

MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "мая": 5, "май": 5, "июн": 6, "июл": 7,
    "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}

COUNTRY_STEMS = {"рф", "росс", "российск", "федерац"}
STOP_STEMS = {"о", "об", "обо", "в", "во", "и", "на", "по", "с", "со", "к", "ко", "для", "из", "от", "до",
              "при", "за", "не", "или", "а", "же", "российск", "федерац", "рф", "закон", "федеральн",
              "указ", "президент", "отдельн", "некотор", "вопрос", "изменен", "внесен", "законодательн",
              "акт", "утвержден", "мер", "порядк", "положен", "особен"}

ANAPH_WORDS = {
    "настоящего", "настоящему", "настоящим", "настоящем", "настоящий",
    "указанного", "указанному", "указанным", "указанном", "указанный",
    "данного", "данному", "данным", "данном", "данный",
    "этого", "этому", "этим", "этом", "этот",
    "того", "тому", "тем", "том", "тот",
    "названного", "названному", "названным", "названном", "названный",
    "вышеуказанного", "вышеуказанному", "вышеуказанным", "вышеуказанном", "вышеуказанный",
    "вышеназванного", "вышеназванному", "вышеназванным", "вышеназванном", "вышеназванный",
    "упомянутого", "упомянутому", "упомянутым", "упомянутом", "упомянутый",
    "вышеупомянутого", "вышеупомянутому", "вышеупомянутым", "вышеупомянутом", "вышеупомянутый",
    "приведенного", "приведенному", "приведенным", "приведенном",
    "последнего", "последнему", "последним", "последнем",
    "соответствующего", "соответствующему", "соответствующим", "соответствующем",
    "этого же", "того же", "же", "самого", "ранее",
}

# Сокращения кодексов, которых нет в базе (или есть с ошибками)
EXTRA_ABBR = {
    "кас": 18, "грк": 4, "вк": 2, "взк": 3, "вздк": 3, "ктм": 20, "кввт": 19, "крфоап": 17,
    "коап": 17, "ск": 8, "лк": 14, "тк": 10, "крфобап": 17, "коапрф": 17, "тмк": 9,
}
# Устойчивые короткие названия законов ("Закон о банкротстве", "Закон об ОСАГО")
EXTRA_TITLES = {
    "о банкротстве": 386, "об осаго": 397, "об ао": 506, "об ооо": 470, "о контрактной системе": 210,
    "о закупках": 237, "об информации": 331,
}
# После этих слов сокращение относится к другому акту
ABBR_BLOCK_NEXT = {"еаэс", "тс", "ес"}

BARRIER_STEMS = {
    "конституц", "постановлен", "приказ", "договор", "соглашен", "конвенц", "регламент", "устав",
    "инструкц", "правил", "решен", "определен", "письм", "кодекс", "закон", "указ", "распоряжен",
    "положен", "протокол", "приговор", "контракт", "регламента", "директив", "стандарт", "методическ",
    "разъяснен", "обзор", "пленум", "декларац", "пакт", "хартия", "соглашение",
}

_TITLE_SPLIT_RE = re.compile(r"«(.*)»", re.S)
_REDACTION_RE = re.compile(r"\(в ред\..*?\)", re.S)
_NUM_RE = re.compile(r"№\s*(\d+)(?:-([A-Za-zА-Яа-яІ]+))?")
_DATE_RE = re.compile(r"от (\d{1,2})\.(\d{1,2})\.(\d{4})")
_STD_NUM_RE = re.compile(r"(ФСБУ|ПБУ)\s*(\d+)/(\d+)")


def canon_suffix(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.lower()
    if s in ("фз", "ф3", "фз.", "fz"):
        return "фз"
    if s in ("i", "1", "l", "і", "|"):
        return "i"
    if s in ("рп", "pп", "rp"):
        return "рп"
    return s


def canon_year(y: str) -> int:
    v = int(y)
    if len(y) == 2:
        v += 2000 if v < 50 else 1900
    return v


def text_keys(text: str) -> list[str]:
    """Последовательность ключей (стемов/чисел) для сравнения названий."""
    toks = tokenize(normalize(text))
    keys: list[str] = []
    for t in toks:
        if t.kind == W:
            s = stem(t.text)
            if s in ("рф", "росс"):
                keys.extend(("российск", "федерац"))
            else:
                keys.append(s)
        elif t.kind == NUM:
            keys.append(t.text)
    return keys


@dataclass
class LawInfo:
    id: int
    kind: str
    numbers: set = field(default_factory=set)       # (digits, suffix)
    dates: set = field(default_factory=set)         # (y, m, d)
    std_numbers: set = field(default_factory=set)   # (kind, a, b)
    titles: list = field(default_factory=list)      # list[tuple[str]]
    title_stems: set = field(default_factory=set)
    red_dates: set = field(default_factory=set)     # даты редакций "(в ред. ... от ...)"
    head_stems: set = field(default_factory=set)    # слова вне кавычек (для ФСБУ)


@dataclass
class Mention:
    start: int            # индекс первого токена
    end: int              # индекс токена после последнего
    kind: str             # тип акта, "anaphora" или "unknown"
    cands: frozenset = frozenset()
    law_id: Optional[int] = None
    atype: Optional[str] = None   # для анафоры: какой вид акта ищем
    strong: bool = True           # найден по однозначным признакам


class Trie:
    __slots__ = ("root",)

    def __init__(self):
        self.root: dict = {}

    def add(self, keys, value):
        node = self.root
        for k in keys:
            node = node.setdefault(k, {})
        node.setdefault(None, set()).add(value)

    def longest(self, keys_at, start: int, limit: int = 80):
        """keys_at(i) -> ключ или None (пропустить) или False (стоп)."""
        node = self.root
        best = None
        i = start
        n_used = 0
        while n_used < limit:
            k = keys_at(i)
            if k is False:
                break
            if k is None:
                i += 1
                continue
            nxt = node.get(k)
            if nxt is None:
                break
            node = nxt
            i += 1
            n_used += 1
            if None in node:
                best = (i, node[None])
        return best


class LawIndex:
    def __init__(self, aliases: dict):
        self.laws: dict[int, LawInfo] = {}
        self.code_trie = Trie()
        self.title_trie = Trie()
        self.abbr: dict[str, int] = {}
        self.by_number: dict = defaultdict(set)
        self.by_digits: dict = defaultdict(set)
        self.by_date: dict = defaultdict(set)
        self.by_std: dict = defaultdict(set)
        self.by_kind: dict = defaultdict(set)
        self.title_list: list[tuple[tuple, int]] = []
        self.stem_df: dict = defaultdict(int)
        self.code_adj: list[tuple[tuple, int]] = []   # (stems before "кодекс", id)
        self._fuzzy_cache: dict = {}
        self._build(aliases)

    # ------------------------------------------------------------------ build
    def _build(self, aliases: dict):
        for sid, names in aliases.items():
            lid = int(sid)
            first = names[0]
            if lid <= 20:
                kind = CODE
            elif first.startswith("Федеральный закон"):
                kind = FZ
            elif first.startswith("Указ"):
                kind = UKAZ
            elif first.startswith("Распоряжение"):
                kind = RASP
            elif first.startswith("Закон"):
                kind = ZAKON
            elif first.startswith("Основы"):
                kind = OSNOVY
            else:
                kind = STD
            info = LawInfo(lid, kind)
            self.laws[lid] = info
            self.by_kind[kind].add(lid)
            for a in names:
                for red in _REDACTION_RE.findall(a):
                    for m in _DATE_RE.finditer(red):
                        info.red_dates.add((int(m.group(3)), int(m.group(2)), int(m.group(1))))
                core = _REDACTION_RE.sub(" ", a)
                outside = re.sub(r'"[^"]*"|«.*»', " ", core)
                info.head_stems.update(text_keys(outside))
                if kind == CODE:
                    keys = text_keys(core)
                    if len(keys) == 1:
                        self.abbr[keys[0]] = lid
                    elif len(keys) == 3 and keys[1:] == ["российск", "федерац"] and len(keys[0]) <= 6 and keys[0].isalpha() and a.split()[0].isupper():
                        self.abbr[keys[0]] = lid
                    else:
                        self.code_trie.add(keys, lid)
                        if "кодекс" in keys:
                            k = keys.index("кодекс")
                            if k > 0:
                                self.code_adj.append((tuple(keys[:k]), lid))
                    continue
                # реквизиты только из "шапки": в названии бывают номера и даты других актов
                head = re.sub(r"«.*»", " ", core)
                for m in _NUM_RE.finditer(head):
                    info.numbers.add((m.group(1), canon_suffix(m.group(2))))
                for m in _DATE_RE.finditer(head):
                    info.dates.add((int(m.group(3)), int(m.group(2)), int(m.group(1))))
                for m in _STD_NUM_RE.finditer(core):
                    info.std_numbers.add((m.group(1).lower(), m.group(2), m.group(3)))
                titles = []
                m = _TITLE_SPLIT_RE.search(core)
                if m:
                    t = m.group(1)
                    titles.append(t)
                    inner = re.match(r'^(?:Федеральный закон|Закон Российской Федерации|Закон РФ|Закон)\s*"(.*)"\s*$', t)
                    if inner:
                        titles.append(inner.group(1))
                elif kind == STD:
                    for q in re.findall(r'"([^"]+)"', core):
                        titles.append(q)
                    if not titles and core.isupper():
                        titles.append(core)
                for t in titles:
                    keys = tuple(text_keys(t))
                    if keys and keys not in info.titles:
                        info.titles.append(keys)
            # абзац для кодексов не нужен
            for keys in info.titles:
                self.title_trie.add(keys, lid)
                self.title_list.append((keys, lid))
                info.title_stems.update(k for k in keys if k not in STOP_STEMS and not k.isdigit())
            for n in info.numbers:
                self.by_number[n].add(lid)
                self.by_digits[n[0]].add(lid)
            for d in info.dates:
                self.by_date[d].add(lid)
            for s in info.std_numbers:
                self.by_std[(s[1], s[2])].add(lid)
        for code_abbr, lid in EXTRA_ABBR.items():
            self.abbr.setdefault(code_abbr, lid)
        for title, lid in EXTRA_TITLES.items():
            if lid in self.laws:
                keys = tuple(text_keys(title))
                self.title_trie.add(keys, lid)
        # Названия: документная частота стемов для контекстного выбора
        for info in self.laws.values():
            for s in info.title_stems:
                self.stem_df[s] += 1
        self.n_laws = len(self.laws)
        self.code_adj_stems = {s for adj, _ in self.code_adj for s in adj}
        # индекс названий по стему для нечёткого поиска
        self.title_by_stem: dict = defaultdict(set)
        for i, (keys, lid) in enumerate(self.title_list):
            for k in set(keys):
                if k not in STOP_STEMS:
                    self.title_by_stem[k].add(i)
        # словарь стемов названий по первым/последним буквам - для поиска слов с опечатками
        self._stem_buckets: dict = defaultdict(set)
        for k in self.title_by_stem:
            if len(k) >= 4 and not k.isdigit():
                self._stem_buckets["^" + k[:2]].add(k)
                self._stem_buckets["$" + k[-2:]].add(k)
        self._near_cache: dict = {}

    def near_stems(self, key: str) -> tuple:
        """Стемы названий, отличающиеся от key опечаткой."""
        res = self._near_cache.get(key)
        if res is not None:
            return res
        out = []
        if len(key) >= 4 and not key.isdigit():
            lim = 1 if len(key) < 7 else 2
            pool = self._stem_buckets.get("^" + key[:2], set()) | self._stem_buckets.get("$" + key[-2:], set())
            for c in pool:
                if abs(len(c) - len(key)) <= 3 and fuzzy_eq(key, c):
                    out.append(c)
        res = tuple(out)
        if len(self._near_cache) > 100_000:
            self._near_cache.clear()
        self._near_cache[key] = res
        return res

    def fuzzy_title_ids(self, keys: tuple):
        """Кэшированное нечёткое сопоставление названия."""
        cache = self._fuzzy_cache
        if keys in cache:
            return cache[keys]
        res = _fuzzy_title_ids(self, keys)
        if len(cache) > 50_000:
            cache.clear()
        cache[keys] = res
        return res

    def idf(self, s: str) -> float:
        return math.log((self.n_laws + 1) / (self.stem_df.get(s, 0) + 1))

    @classmethod
    def from_file(cls, path: str) -> "LawIndex":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))


# --------------------------------------------------------------------------
# Формы слов-маркеров видов актов

def _forms(stem_: str, ends: str) -> set[str]:
    return {stem_ + e for e in ends.split("|")}


ZAKON_FORMS = _forms("закон", "|а|у|ом|е|ы|ов|ам|ами|ах")
UKAZ_FORMS = _forms("указ", "|а|у|ом|е|ы|ов|ам|ами|ах")
RASP_FORMS = _forms("распоряжени", "е|я|ю|ем|и|й|ям|ями|ях")
KODEKS_FORMS = _forms("кодекс", "|а|у|ом|е|ы|ов|ам|ами|ах")
OSNOVY_FORMS = _forms("основ", "ы||ам|ами|ах")
FED_FORMS = _forms("федеральн", "ый|ого|ому|ым|ом|ые|ых|ыми|ым|ая|ой|ую|ое")
POL_FORMS = _forms("положени", "е|я|ю|ем|и|й|ям|ями|ях")
STANDART_FORMS = _forms("стандарт", "|а|у|ом|е|ы|ов|ам|ами|ах")
DATE_SKIP = {"г", "года", "год", "году", "годом", "гг"}
NUMBER_SIGNS = {"№", "n", "no", "номер", "номером", "#"}
PRES_FORMS = _forms("президент", "|а|у|ом|е")
COUNTRY_FORMS = {"российской", "российская", "федерации", "федерация", "россии", "россия"}
ANAPH_FUZZY = {"настоящего", "указанного", "вышеуказанного", "вышеназванного", "упомянутого", "вышеупомянутого",
               "названного", "приведенного", "соответствующего", "настоящему", "указанному", "указанным",
               "настоящим", "данного"}
HEAD_VOCAB_SET = (ZAKON_FORMS | UKAZ_FORMS | RASP_FORMS | FED_FORMS | KODEKS_FORMS | PRES_FORMS | STANDART_FORMS
                  | POL_FORMS | COUNTRY_FORMS | ANAPH_FUZZY)


HEAD_VOCAB = None


@lru_cache(maxsize=100_000)
def canon_head(word: str) -> str:
    """Слово с опечаткой -> ближайшая форма слова-вида акта ("Укааз" -> "указ")."""
    global HEAD_VOCAB
    if HEAD_VOCAB is None:
        HEAD_VOCAB = sorted(HEAD_VOCAB_SET)
    if word in HEAD_VOCAB_SET or len(word) < 5:
        return word
    lim = 1 if len(word) < 8 else 2
    best = None
    for f in HEAD_VOCAB:
        if abs(len(f) - len(word)) > lim or f[0] != word[0] and f[1] != word[1]:
            continue
        d = OSA.distance(word, f, score_cutoff=lim)
        if d <= lim and (best is None or d < best[0]):
            best = (d, f)
    return best[1] if best else word


def fuzzy_eq(a: str, b: str) -> bool:
    if a == b:
        return True
    la = max(len(a), len(b))
    if la < 5:
        return False
    if min(len(a), len(b)) >= 5 and abs(len(a) - len(b)) <= 3 and (a.startswith(b) or b.startswith(a)):
        return True  # опечатка в окончании ломает стемминг: "окружаюещ" ~ "окружа"
    lim = 1 if la < 8 else 2
    return OSA.distance(a, b, score_cutoff=lim) <= lim


class MentionFinder:
    """Находит упоминания законов в токенах одного текста."""

    def __init__(self, index: LawIndex, toks: list[Tok], norm: str):
        self.ix = index
        self.toks = toks
        self.norm = norm
        self.n = len(toks)
        self._stems = [stem(t.text) if t.kind == W else t.text for t in toks]
        self.last_walk_fuzzy = 0
        self._canon = [canon_head(t.text) if t.kind == W else t.text for t in toks]

    # ------------------------------------------------------------- helpers
    def w(self, i: int) -> str:
        return self.toks[i].text if 0 <= i < self.n else ""

    def cw(self, i: int) -> str:
        """Слово с исправленной опечаткой (только для слов-видов актов)."""
        return self._canon[i] if 0 <= i < self.n else ""

    def st(self, i: int) -> str:
        return self._stems[i] if 0 <= i < self.n else ""

    def key_at(self, i: int):
        """Ключ для обхода префиксных деревьев: стем слова или число;
        пунктуация пропускается, перевод строки и конец текста - стоп."""
        if i >= self.n:
            return False
        t = self.toks[i]
        if t.kind == W:
            s = self._stems[i]
            if s in ("рф", "росс"):
                return "российск"  # дальше ожидается "федерац" - упрощение ниже
            return s
        if t.kind == NUM:
            return t.text
        if t.text in ('"', "'", "(", ")", ",", "-", "«", "»", ":", "."):
            return None
        return False

    def _skip_country(self, j: int) -> int:
        """Пропустить "РФ", "России", "Российской Федерации"."""
        w0 = self.cw(j)
        if w0 in ("рф", "россии", "россия", "российской", "российская", "росии"):
            if w0 in ("российской", "российская") and self.cw(j + 1) in ("федерации", "федерация"):
                return j + 2
            if w0 in ("российской", "российская"):
                return j
            return j + 1
        if self.w(j) == "р" and self.w(j + 1) == "." and self.w(j + 2) == "ф":
            return j + 3 + (self.w(j + 3) == ".")
        return j

    # ------------------------------------------------------------- codes
    def _title_keys_walk(self, trie: Trie, start: int, limit: int = 80, fuzzy: int = 2):
        """Самое длинное название из дерева, начинающееся в start: (j_after, ids) или None.

        fuzzy - сколько слов с опечаткой можно сопоставить нечётко; число использованных
        правок сохраняется в self.last_walk_fuzzy."""
        # дерево хранит "российск федерац" двумя ключами; в тексте "РФ" одно слово
        toks = self.toks
        stems = self._stems
        node = trie.root
        best = None
        best_fz = 0
        fz_used = 0
        i = start
        used = 0
        pending: list[str] = []
        while used < limit:
            if pending:
                k = pending.pop(0)
                adv = 0
            else:
                if i >= self.n:
                    break
                t = toks[i]
                if t.kind == W:
                    s = stems[i]
                    if t.text in ("n", "no", "номер") and i + 1 < self.n and toks[i + 1].kind == NUM:
                        i += 1  # "N 378" внутри названия = "№ 378"
                        continue
                    if s in ("рф", "росс"):
                        k = "российск"
                        pending = ["федерац"]
                    else:
                        k = s
                        if k not in node and self._canon[i] != t.text:
                            k = stem(self._canon[i])
                elif t.kind == NUM:
                    k = t.text
                elif t.text in ('"', "'", "(", ")", ",", "-", ":", ".", ";", "№", "/"):
                    i += 1
                    continue
                else:
                    break
                adv = 1
            nxt = node.get(k)
            if nxt is None and fz_used < fuzzy and adv and len(k) >= 4 and not k[0].isdigit():
                for ck, child in node.items():
                    if ck is not None and len(ck) >= 4 and fuzzy_eq(k, ck):
                        nxt = child
                        fz_used += 1
                        break
            if nxt is None:
                break
            node = nxt
            i += adv
            used += 1
            if None in node and not pending:
                best = (i, node[None])
                best_fz = fz_used
        self.last_walk_fuzzy = best_fz
        return best

    def try_code(self, i: int) -> Optional[Mention]:
        t = self.toks[i]
        if t.kind != W:
            return None
        # сокращения
        lid = self.ix.abbr.get(t.text)
        if lid is not None and (t.upper or t.text in ("коап", "крфоап") or self._country_follows(i + 1)
                                or self._ref_before(i)):
            if self.w(i + 1) in ABBR_BLOCK_NEXT:
                return Mention(i, i + 2, "unknown")
            if self.w(i + 1) == "-" and self.toks[i + 1].start == t.end:
                return None
            j = self._skip_country(i + 1)
            return Mention(i, j, CODE, frozenset([lid]), lid)
        # полные названия
        if self.cw(i) in KODEKS_FORMS or self.st(i) in self.ix.code_adj_stems or t.text.startswith("уголовно") or t.text.startswith("гражданско"):
            best = self._title_keys_walk(self.ix.code_trie, i, 12)
            if best:
                j, ids = best
                j = self._skip_country(j)
                lid = min(ids)
                return Mention(i, j, CODE, frozenset([lid]), lid)
        return None

    def _country_follows(self, j: int) -> bool:
        return self._skip_country(j) != j

    def _ref_before(self, i: int) -> bool:
        """Перед словом стоит номер ссылки (ст. 5 ук) - тогда сокращение в нижнем регистре допустимо."""
        k = i - 1
        while k >= 0 and self.toks[k].kind == P and self.toks[k].text in (".", ","):
            k -= 1
        return k >= 0 and (self.toks[k].kind == NUM or (self.toks[k].kind == W and len(self.toks[k].text) == 1
                                                       and k > 0 and self.toks[k - 1].text in (".", '"', " ")))

    def try_code_fuzzy(self, i: int) -> Optional[Mention]:
        """Опечатки в названии кодекса: "Налогвого коедкса"."""
        t = self.toks[i]
        if t.kind != W or len(t.text) < 5:
            return None
        if self.cw(i) not in KODEKS_FORMS:
            return None
        # слева прилагательные
        best = None
        for adj, lid in self.ix.code_adj:
            k = len(adj)
            if i - k < 0:
                continue
            ok = True
            pos = i - 1
            for a in reversed(adj):
                while pos >= 0 and self.toks[pos].text == "-":
                    pos -= 1
                if pos < 0 or self.toks[pos].kind != W or not fuzzy_eq(self.st(pos), a):
                    ok = False
                    break
                pos -= 1
            if ok and (best is None or k > best[0]):
                best = (k, lid, pos + 1)
        if best:
            _, lid, s = best
            j = self._skip_country(i + 1)
            return Mention(s, j, CODE, frozenset([lid]), lid, strong=False)
        # справа: "Кодекс об административных правонарушениях" с опечатками
        j0 = self._skip_country(i + 1)
        for lid, pat in RIGHT_CODE_PATTERNS:
            k = len(pat)
            if j0 + k > self.n or any(self.toks[j0 + q].kind != W for q in range(k)):
                continue
            words = [self.toks[j0 + q].text for q in range(k)]
            if all(_word_close(a, b) for a, b in zip(words, pat)):
                return Mention(i, self._skip_country(j0 + k), CODE, frozenset([lid]), lid, strong=False)
        return None

    # ------------------------------------------------------------- acts
    def _head(self, i: int):
        """Вид акта, начинающийся в i: (kind, j_after_head) или None."""
        w0 = self.cw(i)
        if w0 in FED_FORMS:
            j = i + 1
            if self.st(j) == "конституцион":
                j += 1
            if self.cw(j) in ZAKON_FORMS:
                return FZ, j + 1
            if self.cw(j) in STANDART_FORMS:
                return STD, j + 1
            return None
        if self.w(i) == "фз" or (w0 == "ф" and self.w(i + 1) == "3" and self.toks[i + 1].start == self.toks[i].end):
            # "127-ФЗ" - это суффикс номера, а не вид акта
            if i > 0 and self.w(i - 1) == "-":
                return None
            if i > 0 and self.toks[i - 1].kind == NUM and self.toks[i - 1].end == self.toks[i].start:
                return None
            return FZ, i + (1 if w0 == "фз" else 2)
        if w0 in ZAKON_FORMS:
            return "law", i + 1
        if w0 in UKAZ_FORMS:
            j = i + 1
            if self.cw(j) in PRES_FORMS:
                j += 1
            return UKAZ, j
        if w0 in RASP_FORMS:
            j = i + 1
            if self.cw(j) in PRES_FORMS:
                j += 1
            return RASP, j
        if self.w(i) == "рп" and (self.toks[i].upper or self.w(i + 1) in NUMBER_SIGNS
                                 or self.w(i + 1) in ('"', "'") or (self.w(i + 1) == "от" and self._parse_date(i + 2))):
            return RASP, i + 1
        if w0 in OSNOVY_FORMS and fuzzy_eq(self.st(i + 1), "законодательств"):
            return OSNOVY, i + 2
        if w0 in OSNOVY_FORMS and (t_cap(self.toks[i]) or self.w(i + 1) in ("№", "n")) \
                and self.w(i + 1) in ("№", "n", "от", '"'):
            return OSNOVY, i + 1
        if self.w(i) in ("фсбу", "пбу"):
            return STD, i + 1
        if w0 in POL_FORMS and self.w(i + 1) == "по" and (fuzzy_eq(self.st(i + 2), "бухгалтерск")
                                                          or self.st(i + 2) == "веден"):
            return STD, i + 1
        return None

    def _parse_date(self, j: int):
        """Дата в позиции j: (date, j_after) или None."""
        t = self.toks[j] if j < self.n else None
        if t is None or t.kind != NUM:
            return None
        parts = t.text.split(".")
        if len(parts) == 3 and all(parts) and len(parts[2]) in (2, 4):
            try:
                d, m, y = int(parts[0]), int(parts[1]), canon_year(parts[2])
            except ValueError:
                return None
            k = j + 1
            if self.w(k) in DATE_SKIP:
                k += 1
                if self.w(k) == ".":
                    k += 1
            return (y, m, d), k
        if len(parts) == 1 and len(t.text) <= 2 and j + 2 < self.n:
            mw = self.toks[j + 1]
            if mw.kind == W:
                mon = None
                for ms, mv in MONTHS.items():
                    if mw.text.startswith(ms) and (len(ms) > 2 or mw.text in ("мая", "май")):
                        mon = mv
                        break
                if mon and self.toks[j + 2].kind == NUM and len(self.toks[j + 2].text) == 4:
                    k = j + 3
                    if self.w(k) in DATE_SKIP:
                        k += 1
                        if self.w(k) == ".":
                            k += 1
                    return (int(self.toks[j + 2].text), mon, int(t.text)), k
        return None

    def _parse_number(self, j: int, signed: bool = True):
        """Номер акта в позиции j: ((digits, suffix), j_after).

        signed - перед номером был знак "№"; без него суффикс через пробел ("73 ФЗ") не считается.
        """
        if j >= self.n or self.toks[j].kind != NUM or "." in self.toks[j].text:
            return None
        digits = self.toks[j].text
        k = j + 1
        suffix = None
        if self.w(k) == "-" and k + 1 < self.n:
            nt = self.toks[k + 1]
            if nt.kind == W:
                if nt.text == "ф" and self.w(k + 2) == "3":
                    suffix, k = "фз", k + 3
                else:
                    suffix, k = canon_suffix(nt.text), k + 2
            elif nt.kind == NUM and nt.text in ("1",):
                suffix, k = "i", k + 2
            elif nt.kind == NUM:
                suffix, k = "#" + nt.text, k + 2   # чужой номер вида 20-456
        elif k < self.n and self.toks[k].kind == W:
            glued = self.toks[k].start == self.toks[j].end
            if (self.toks[k].text in ("фз", "рп") and (signed or glued)) or (self.toks[k].text == "i" and glued):
                suffix, k = canon_suffix(self.toks[k].text), k + 1
        return (digits, suffix), k

    def _parse_std_number(self, j: int):
        if j + 2 < self.n and self.toks[j].kind == NUM and self.w(j + 1) == "/" and self.toks[j + 2].kind == NUM:
            return (self.toks[j].text, self.toks[j + 2].text), j + 3
        return None

    def _parse_title(self, j: int):
        """Название в позиции j (после кавычки или "о/об"): (ids, j_after, exact)."""
        best = self._title_keys_walk(self.ix.title_trie, j, 120)
        if best:
            k, ids = best
            exact = self.last_walk_fuzzy == 0
            # закрывающая кавычка
            while k < self.n and self.w(k) in ('"', "'", "»"):
                k += 1
            return set(ids), k, exact
        return None

    def _quoted_title(self, j: int, min_keys: int = 1):
        """Название после открывающей кавычки: точное, а если точное совпадение - лишь начало
        текста в кавычках ("О безопасности" при опечатке дальше), то нечёткое по всему тексту."""
        ex = self._parse_title(j)
        if ex:
            k = ex[1]
            if k - 1 >= j and self.toks[k - 1].text in ('"', "'"):
                return ex
            fz = self._parse_title_fuzzy(j, min_keys)
            if fz and not (fz[0] <= ex[0]):
                return fz
            return ex
        return self._parse_title_fuzzy(j, min_keys)

    def _parse_title_fuzzy(self, j: int, min_keys: int = 1):
        """Нечёткое сравнение текста в кавычках с названиями."""
        # содержимое до закрывающей кавычки (кавычка, после которой не слово)
        k = j
        keys = []
        depth = 0
        while k < self.n and k - j < 200:
            t = self.toks[k]
            if t.text in ('"', "'"):
                nxt = self.toks[k + 1] if k + 1 < self.n else None
                glued_next = nxt is not None and nxt.kind in (W, NUM) and t.end == nxt.start
                if glued_next and t.space_before:
                    depth += 1          # внутренняя открывающая: сети "Интернет"
                elif not glued_next:
                    if depth == 0:
                        break
                    depth -= 1
            elif t.text == "\n":
                break
            elif t.kind == W:
                s = self._stems[k]
                if s in ("рф", "росс"):
                    keys.extend(("российск", "федерац"))
                else:
                    keys.append(s)
            elif t.kind == NUM:
                keys.append(t.text)
            k += 1
        if len(keys) < min_keys or len(keys) > 150:
            return None
        found = self.ix.fuzzy_title_ids(tuple(keys))
        if found is None:
            return None
        found, score = found
        if len(keys) < 4 and score < 0.999 and min_keys > 1:
            return None
        end = k + 1 if k < self.n and self.toks[k].text in ('"', "'") else k
        return found, end, False

    def try_act(self, i: int) -> Optional[Mention]:
        h = self._head(i)
        if not h:
            return None
        kind, j = h
        start = i
        if kind == STD and self.cw(i) in POL_FORMS:
            # "ПОЛОЖЕНИЕ ПО ВЕДЕНИЮ БУХГАЛТЕРСКОГО УЧЕТА ..." - название без кавычек целиком
            tt = self._parse_title(i)
            if tt:
                ids, k, _ = tt
                if len(ids) > 1:
                    k, ids = self._redaction(k, ids)
                return Mention(start, k, STD, frozenset(ids))
        number = date = std_num = None
        title_ids = None
        title_exact = True
        failed = False
        steps = 0
        std_words: set = set()
        if kind == STD:
            # слова вида стандарта ("ФЕДЕРАЛЬНЫЙ СТАНДАРТ", "ФСБУ", "ПБУ", "ПОЛОЖЕНИЕ") сужают выбор
            std_words.update(self.st(q) for q in range(i, j) if self.toks[q].kind == W)
        std_skipped = 0
        while j < self.n and steps < 14:
            steps += 1
            w_ = self.w(j)
            jj = self._skip_country(j)
            if jj != j:
                j = jj
                continue
            if self.cw(j) in PRES_FORMS:
                j += 1
                continue
            if kind == STD and self.toks[j].kind == W and w_ not in ("от",) and std_skipped < 8 \
                    and title_ids is None:
                std_words.add(self.st(j))
                std_skipped += 1
                j += 1
                continue
            if kind == STD and w_ in ("(", ")") and title_ids is None and std_num is None:
                j += 1
                continue
            if w_ == "от" and date is None:
                d = self._parse_date(j + 1)
                if d:
                    date, j = d
                    continue
                # замаскированная или битая дата: "от ДД.ММ.ГГГГ", "от <ДАТА>"
                k = j + 1
                while k < self.n and k - j <= 7 and self.toks[k].kind != W or (k < self.n and k - j <= 7 and self.w(k) in ("дд", "мм", "гггг", "гг", "дата", "дата1", "дата2", "дата3")):
                    if self.w(k) in ("№", "n", '"', "'"):
                        break
                    k += 1
                if k > j + 1:
                    j = k
                    continue
                break
            if w_ in NUMBER_SIGNS and number is None and std_num is None:
                k = j + 1
                if self.w(k) in NUMBER_SIGNS:
                    k += 1
                if kind == STD:
                    sn = self._parse_std_number(k)
                    if sn:
                        std_num, j = sn
                        continue
                nm = self._parse_number(k)
                if nm:
                    number, j = nm
                    continue
                failed = True
                break
            if kind == STD and std_num is None:
                sn = self._parse_std_number(j)
                if sn:
                    std_num, j = sn
                    continue
            if number is None and self.toks[j].kind == NUM and self.w(j + 1) == "-" and kind in (FZ, "law", RASP, ZAKON):
                nm = self._parse_number(j, signed=False)
                if nm and nm[0][1] in ("фз", "рп", "i"):
                    number, j = nm
                    continue
            if w_ in ('"', "'") and title_ids is None:
                tt = self._quoted_title(j + 1)
                if tt:
                    title_ids, j, title_exact = tt
                    continue
                failed = True
                break
            if w_ in ("о", "об", "обо") and title_ids is None:
                tt = self._parse_title(j)
                if tt:
                    title_ids, j, _ = tt
                    continue
                break
            if w_ in (",",) and j + 1 < self.n and self.w(j + 1) in ("от", "№", "n") and (number or date or title_ids):
                j += 1
                continue
            break
        if kind == OSNOVY:
            # в базе одни "Основы": о нотариате
            if self.w(j) == "о" and self.st(j + 1) == "нотариат":
                j += 2
            return Mention(start, j, OSNOVY, frozenset(self.ix.by_kind[OSNOVY]))
        cands = self._resolve(kind, number, date, std_num, title_ids, title_exact)
        if cands and kind == STD and std_words and len(cands) > 1:
            def _std_score(c):
                head = self.ix.laws[c].head_stems
                return sum(1 for w_ in std_words if w_ in head or any(fuzzy_eq(w_, h) for h in head))
            best = max(_std_score(c) for c in cands)
            cands = {c for c in cands if _std_score(c) == best}
        if cands and len(cands) > 1:
            j, cands = self._redaction(j, cands)
        if cands is None and failed:
            return Mention(start, j, "unknown")
        if cands is None and j < self.n and self.toks[j].kind == W and self.toks[j].cap \
                and _adj_like(self.toks[j].text) and self.cw(j) not in ANAPH_WORDS:
            # "Закона Краснодарского края", "Закона Московской области" - акт не из базы
            return Mention(start, j + 1, "unknown")
        if cands is None:
            # только вид акта, без реквизитов: анафора ("Закона", "Указа")
            atype = CODE if kind == CODE else kind
            return Mention(start, j, "anaphora", atype=atype)
        if not cands:
            return Mention(start, j, "unknown")
        return Mention(start, j, kind, frozenset(cands))

    def _redaction(self, j: int, cands):
        """"(в ред. Приказа Минфина РФ от 25.12.2019 N 253н)" - уточнение по датам редакций."""
        if not (self.w(j) == "(" and self.w(j + 1) == "в" and self.w(j + 2) in ("ред", "редакции")):
            return j, cands
        k = j + 3
        dates = set()
        while k < self.n and k - j < 80 and self.w(k) != ")":
            if self.w(k) == "от":
                d = self._parse_date(k + 1)
                if d:
                    dates.add(d[0])
            k += 1
        if k < self.n and self.w(k) == ")":
            k += 1
        if dates:
            narrowed = {c for c in cands if dates <= self.ix.laws[c].red_dates}
            if narrowed:
                cands = narrowed
        else:
            return j, cands
        return k, cands

    def _resolve(self, kind, number, date, std_num, title_ids, title_exact=True):
        ix = self.ix
        if number is None and date is None and std_num is None and title_ids is None:
            return None
        if kind == "law":
            type_ids = ix.by_kind[FZ] | ix.by_kind[ZAKON] | ix.by_kind[OSNOVY]
        else:
            type_ids = ix.by_kind[kind]
        if title_ids and not title_exact and (number is not None or date is not None):
            # название с опечатками: реквизиты надёжнее
            req = set(type_ids)
            if number is not None:
                digits, suf = number
                if suf and suf.startswith("#"):
                    req = set()
                else:
                    req &= ix.by_number.get((digits, suf), set()) if suf else ix.by_digits.get(digits, set())
            if date is not None:
                req &= ix.by_date.get(date, set())
            if req:
                both = req & title_ids
                return both or req
        cands = set(type_ids)
        if title_ids:
            narrowed = cands & title_ids
            cands = narrowed if narrowed else set(title_ids)
        if number is not None:
            digits, suf = number
            if suf and suf.startswith("#"):
                by_num = set()
            elif suf:
                by_num = ix.by_number.get((digits, suf), set())
            else:
                by_num = ix.by_digits.get(digits, set())
            narrowed = cands & by_num
            if narrowed:
                cands = narrowed
            elif title_ids and title_exact:
                pass  # номер с ошибкой, верим точному названию
            else:
                return set()
        if std_num is not None:
            narrowed = cands & ix.by_std.get(std_num, set())
            if narrowed:
                cands = narrowed
            elif not title_ids:
                return set()
        if date is not None:
            narrowed = cands & ix.by_date.get(date, set())
            if narrowed:
                cands = narrowed
            elif not title_ids and number is None:
                return set()
            elif title_ids and not title_exact:
                return set()
        return cands

    def try_bare(self, i: int) -> Optional[Mention]:
        """Номер "№ 127-ФЗ" или название в кавычках без вида акта."""
        t = self.toks[i]
        if t.text in NUMBER_SIGNS or t.kind == NUM:
            k = i + 1 if t.text in NUMBER_SIGNS else i
            if k < self.n and self.toks[k].kind == NUM:
                nm = self._parse_number(k, signed=t.text in NUMBER_SIGNS)
                if nm and nm[0][1] in ("фз", "рп"):
                    (digits, suf), j = nm
                    ids = self.ix.by_number.get((digits, suf), set())
                    # дата после номера
                    if self.w(j) == "от":
                        d = self._parse_date(j + 1)
                        if d:
                            date, j2 = d
                            narrowed = ids & self.ix.by_date.get(date, set())
                            if narrowed:
                                ids = narrowed
                            j = j2
                    if self.w(j) in ('"', "'"):
                        tt = self._quoted_title(j + 1)
                        if tt:
                            narrowed = ids & tt[0]
                            ids = narrowed or tt[0]
                            j = tt[1]
                    if ids:
                        return Mention(i, j, FZ if suf == "фз" else RASP, frozenset(ids))
                    return Mention(i, j, "unknown")
        if t.text in ('"', "'") and self._opening_quote(i):
            tt = self._quoted_title(i + 1, min_keys=3)
            if tt:
                ids, j, exact = tt
                if not exact or j - i >= 3 or self.toks[j - 1].text in ('"', "'"):
                    if len(ids) > 1:
                        j, ids = self._redaction(j, ids)
                    return Mention(i, j, "title", frozenset(ids))
        return None

    def _opening_quote(self, i: int) -> bool:
        """Кавычка открывающая: перед ней пробел/начало/скобка, после - слово без пробела."""
        t = self.toks[i]
        if i > 0 and not t.space_before and self.toks[i - 1].text not in ("(", "[", ":"):
            return False
        nxt = self.toks[i + 1] if i + 1 < self.n else None
        return nxt is not None and not nxt.space_before

    def try_anaphora(self, i: int) -> Optional[Mention]:
        w0 = self.cw(i)
        if w0 not in ANAPH_WORDS:
            return None
        j = i + 1
        if self.w(j) == "же":
            j += 1
        if self.w(j) in ("самого", "самому"):
            j += 1
        w1 = self.cw(j)
        atype = None
        if w1 in KODEKS_FORMS:
            atype = CODE
        elif w1 in ZAKON_FORMS:
            atype = "law"
        elif w1 in FED_FORMS and self.cw(j + 1) in ZAKON_FORMS:
            atype, j = "law", j + 1
        elif w1 == "фз":
            atype = "law"
        elif w1 in UKAZ_FORMS:
            atype = UKAZ
        elif w1 in RASP_FORMS:
            atype = RASP
        if atype is None:
            return None
        return Mention(i, j + 1, "anaphora", atype=atype)

    def try_barrier(self, i: int) -> Optional[Mention]:
        t = self.toks[i]
        if t.kind != W:
            return None
        s = self._stems[i]
        if s in BARRIER_STEMS or self.cw(i) in KODEKS_FORMS:
            # "Кодекса" без прилагательного - анафора к последнему кодексу
            if self.cw(i) in KODEKS_FORMS:
                prev = self.toks[i - 1] if i > 0 else None
                if prev is None or prev.kind != W or not _adj_like(prev.text):
                    return Mention(i, i + 1, "anaphora", atype=CODE)
            return Mention(i, i + 1, "unknown")
        if s.startswith("конституц"):
            return Mention(i, i + 1, "unknown")
        return None

    # ------------------------------------------------------------- main
    def find(self) -> list[Mention]:
        out: list[Mention] = []
        i = 0
        n = self.n
        while i < n:
            t = self.toks[i]
            if t.kind == P and t.text not in ('"', "'", "№", "#"):
                i += 1
                continue
            m = (self.try_code(i) or self.try_act(i) or self.try_bare(i) or self.try_anaphora(i)
                 or self.try_code_fuzzy(i))
            if m is None:
                m = self.try_barrier(i)
            if m:
                if out and m.start < out[-1].end:
                    # нечёткий кодекс мог захватить слова левее
                    m.start = out[-1].end
                out.append(m)
                i = max(m.end, i + 1)
            else:
                i += 1
        self._disambiguate(out)
        return out

    def _disambiguate(self, ms: list[Mention]):
        text_stems = None
        resolved: list[Mention] = []
        for m in ms:
            if m.kind == "anaphora":
                for prev in reversed(resolved):
                    if _compatible(m.atype, prev.kind):
                        m.law_id = prev.law_id
                        m.cands = frozenset([prev.law_id])
                        break
                if m.law_id is None:
                    m.kind = "unknown"
                continue
            if m.kind == "unknown":
                continue
            if m.law_id is None:
                if len(m.cands) == 1:
                    m.law_id = next(iter(m.cands))
                else:
                    seen = [p.law_id for p in resolved if p.law_id in m.cands]
                    if seen:
                        m.law_id = seen[-1]
                    else:
                        if text_stems is None:
                            text_stems = {s for s in self._stems if s and s[0].isalpha() and s not in STOP_STEMS}
                        m.law_id = self._by_context(m.cands, text_stems)
            m.kind = self.ix.laws[m.law_id].kind
            resolved.append(m)

    def _by_context(self, cands, text_stems) -> int:
        best, best_score = None, -1.0
        for lid in sorted(cands):
            info = self.ix.laws[lid]
            score = sum(self.ix.idf(s) for s in info.title_stems if s in text_stems)
            if score > best_score + 1e-9:
                best, best_score = lid, score
        return best


RIGHT_CODE_PATTERNS = [
    (17, ("об", "административных", "правонарушениях")),
    (18, ("административного", "судопроизводства")),
    (19, ("внутреннего", "водного", "транспорта")),
    (20, ("торгового", "мореплавания")),
]


def _word_close(a: str, b: str) -> bool:
    """Слово из текста a похоже на слово шаблона b (с учётом опечаток и падежа окончания)."""
    if a == b:
        return True
    if len(b) <= 3:
        return a == b
    sa, sb = stem(a), stem(b)
    if sa == sb:
        return True
    lim = 1 if len(b) < 7 else 2
    return OSA.distance(a, b, score_cutoff=lim) <= lim or OSA.distance(sa, sb, score_cutoff=lim) <= lim


def _compatible(atype: Optional[str], kind: str) -> bool:
    if atype == CODE:
        return kind == CODE
    if atype in ("law", FZ, ZAKON):
        return kind in (FZ, ZAKON, OSNOVY, "title")
    if atype == UKAZ:
        return kind in (UKAZ, "title")
    if atype == RASP:
        return kind in (RASP, "title")
    if atype == STD:
        return kind in (STD, "title")
    return False


def t_cap(t: Tok) -> bool:
    return t.cap


def _adj_like(w: str) -> bool:
    return w.endswith(("ого", "его", "ому", "ему", "ым", "им", "ом", "ем", "ый", "ий", "ой", "ая", "ую", "ые",
                       "ых", "ими", "ыми"))


def _seq_sim(a: list[str], b: list[str]) -> float:
    """Сходство последовательностей ключей с нечётким равенством слов (Dice)."""
    used = [False] * len(b)
    hit = 0
    for x in a:
        for k, y in enumerate(b):
            if not used[k] and (x == y or (len(x) >= 4 and len(y) >= 4 and fuzzy_eq(x, y))):
                used[k] = True
                hit += 1
                break
    return 2 * hit / (len(a) + len(b))


def _fuzzy_title_ids(ix: "LawIndex", keys: tuple):
    """Законы, название которых нечётко совпадает с ключами keys (или None)."""
    cand_idx = set()
    for key in set(keys):
        if key in STOP_STEMS:
            continue
        hit = ix.title_by_stem.get(key)
        if hit:
            cand_idx |= hit
        else:
            for near in ix.near_stems(key):
                cand_idx |= ix.title_by_stem[near]
    if not cand_idx:
        return None
    best_score, best_ids = 0.0, set()
    for ci in cand_idx:
        tk, lid = ix.title_list[ci]
        if abs(len(tk) - len(keys)) > max(2, len(tk) // 4):
            continue
        score = _seq_sim(keys, list(tk))
        hit = score * (len(keys) + len(tk)) / 2
        if hit / len(keys) < 0.75 or hit / len(tk) < 0.75:
            continue
        if score > best_score + 1e-9:
            best_score, best_ids = score, {lid}
        elif abs(score - best_score) < 1e-9:
            best_ids.add(lid)
    if best_score >= 0.8:
        return frozenset(best_ids), best_score
    return None
