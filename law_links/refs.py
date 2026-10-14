"""Разбор цепочек "пп. 1 п. 2 ч. 3 ст. 4" и их группировка."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz.distance import OSA

from .normalize import NUM, P, W, Tok

# уровни
ABZ, SUB, POINT, PART, ART = 0, 1, 2, 3, 4

_ART_FULL = {"статья", "статьи", "статье", "статью", "статьей", "статьею", "статей", "статьям", "статьями",
             "статьях", "статьи"}
_PART_FULL = {"часть", "части", "частью", "частей", "частям", "частями", "частях"}
_POINT_FULL = {"пункт", "пункта", "пункту", "пунктом", "пункте", "пункты", "пунктов", "пунктам", "пунктами",
               "пунктах"}
_SUB_FULL = {"подпункт", "подпункта", "подпункту", "подпунктом", "подпункте", "подпункты", "подпунктов",
             "подпунктам", "подпунктами", "подпунктах"}
_ABZ_FULL = {"абзац", "абзаца", "абзацу", "абзацем", "абзаце", "абзацы", "абзацев", "абзацам", "абзацами",
             "абзацах"}

FULL_FORMS: dict[str, int] = {}
for _forms, _lvl in ((_ART_FULL, ART), (_PART_FULL, PART), (_POINT_FULL, POINT), (_SUB_FULL, SUB),
                     (_ABZ_FULL, ABZ)):
    for _f in _forms:
        FULL_FORMS[_f] = _lvl

SHORT_FORMS = {
    "ст": ART, "стат": ART, "стст": ART, "статья": ART,
    "ч": PART, "чч": PART, "част": PART,
    "п": POINT, "пунк": POINT, "пп": SUB, "подп": SUB, "подпп": SUB, "ппп": SUB,
    "абз": ABZ,
    "поддп": SUB, "пдп": SUB, "пддп": SUB, "подпунк": SUB, "подпун": SUB, "подпт": SUB,
}
PLURAL_FULL = {"статьи", "статей", "статьям", "статьями", "статьях", "части", "частей", "частям", "частями",
               "частях", "пункты", "пунктов", "пунктам", "пунктами", "пунктах", "подпункты", "подпунктов",
               "подпунктам", "подпунктами", "подпунктах"}

_FUZZY_TARGETS = sorted(FULL_FORMS)

CONNECTORS = {",", "и", "или", "либо", ";"}
PREPS = {"в", "во", "из", "по", "к"}
LETTERS = set("абвгдежзийклмнопрстуфхцчшщэюя")

_ORD = [
    ("одиннадцат", 11), ("двенадцат", 12), ("тринадцат", 13), ("четырнадцат", 14), ("пятнадцат", 15),
    ("шестнадцат", 16), ("семнадцат", 17), ("восемнадцат", 18), ("девятнадцат", 19), ("двадцат", 20),
    ("тридцат", 30), ("перв", 1), ("втор", 2), ("трет", 3), ("четверт", 4), ("пят", 5), ("шест", 6),
    ("седьм", 7), ("восьм", 8), ("девят", 9), ("десят", 10),
]
_ORD_ENDS = ("ая", "ой", "ую", "ый", "ого", "ому", "ым", "ом", "ые", "ых", "ий", "ья", "ье", "ьей", "ьего",
             "его", "ей", "ему", "ем", "ьему", "ьим", "ьем", "ью", "ьи", "ьих", "ыми", "ими", "ым", "ую", "ое",
             "ии", "и", "ая", "ое")
_ORD_TENS_UNITS = {"перв": 1, "втор": 2, "трет": 3, "четверт": 4, "пят": 5, "шест": 6, "седьм": 7, "восьм": 8,
                   "девят": 9}
_CARD_TENS = {"двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50}


def ordinal_value(word: str) -> Optional[int]:
    for st, v in _ORD:
        if word.startswith(st):
            rest = word[len(st):]
            if rest in _ORD_ENDS or rest == "":
                if rest == "" and st not in ("перв",):
                    return None
                return v
    return None


@dataclass
class Elem:
    level: int
    values: list
    start: int          # индекс токена маркера
    end: int            # индекс токена после последнего значения
    ambig: bool = False  # "п.п." - пункт или подпункт
    sep: str = "none"   # разделитель перед элементом: none / comma / conj / prep


@dataclass
class Chain:
    elems: dict = field(default_factory=dict)  # level -> Elem
    start: int = 0
    end: int = 0
    order: list = field(default_factory=list)

    def levels(self):
        return set(self.elems)

    def add(self, e: Elem):
        self.elems[e.level] = e
        self.order.append(e.level)
        self.end = max(self.end, e.end)
        self.start = min(self.start, e.start) if self.elems else e.start


class RefParser:
    def __init__(self, toks: list[Tok], covered: list[bool], mention_starts: set[int]):
        self.toks = toks
        self.n = len(toks)
        self.covered = covered
        self.mention_starts = mention_starts

    def w(self, i: int) -> str:
        return self.toks[i].text if 0 <= i < self.n and not self.covered[i] else ""

    # ----------------------------------------------------------- markers
    def marker_at(self, i: int):
        """(level, ambig, j_after) для маркера в позиции i."""
        if i >= self.n or self.covered[i]:
            return None
        t = self.toks[i]
        if t.kind != W:
            return None
        w = t.text
        j = i + 1
        # "п.п.", "п. п.", "ст.ст.", "ч.ч."
        if w in ("п", "ст", "ч") and self.w(j) == "." and self.w(j + 1) == w:
            lvl = SHORT_FORMS[w]
            k = j + 2
            if self.w(k) == ".":
                k += 1
            return lvl, w == "п", k
        if w == "п" and self.w(j) == "/" and self.w(j + 1) == "п":
            k = j + 2
            if self.w(k) == ".":
                k += 1
            return SUB, False, k
        if w in SHORT_FORMS and w not in FULL_FORMS:
            lvl = SHORT_FORMS[w]
            if self.w(j) == ".":
                j += 1
            elif w in ("п", "ч", "ст", "пп") and not self._value_follows(j):
                return None
            return lvl, False, j
        if w in FULL_FORMS:
            return FULL_FORMS[w], False, j
        return None

    def fuzzy_marker_at(self, i: int):
        if i >= self.n or self.covered[i]:
            return None
        t = self.toks[i]
        if t.kind != W or len(t.text) < 4:
            return None
        if not self._value_follows(i + 1):
            return None
        w = t.text
        lim = 1 if len(w) <= 6 else 2
        best = None
        for f in _FUZZY_TARGETS:
            if abs(len(f) - len(w)) > lim:
                continue
            d = OSA.distance(w, f, score_cutoff=lim)
            if d <= lim and (best is None or d < best[0]):
                best = (d, f)
        if best:
            return FULL_FORMS[best[1]], False, i + 1
        # склеенное слово: "согласностатье 16"
        for f in ("статье", "статьи", "статья", "статью", "статьей", "пункта", "пункту", "пункте", "пунктом",
                  "части", "частью", "подпункта", "подпункту"):
            if w.endswith(f) and len(w) > len(f) + 1:
                return FULL_FORMS[f], False, i + 1
        return None

    def glued_short_marker(self, i: int):
        """"дляст. 105", "сост. 5" - короткий маркер приклеен к слову."""
        t = self.toks[i]
        if t.kind != W or self.covered[i]:
            return None
        w = t.text
        for pre in ("согласно", "для", "по", "со", "на", "из", "см", "в", "во", "при", "от", "и", "к"):
            if w.startswith(pre) and len(w) > len(pre):
                rest = w[len(pre):]
                if rest in ("ст", "пп", "п", "ч") and self.w(i + 1) == "." and self._value_follows(i + 2):
                    return SHORT_FORMS[rest], False, i + 2
        return None

    def _value_follows(self, j: int) -> bool:
        k = j
        if self.w(k) == ".":
            k += 1
        if self.w(k) in ("№", "n"):
            k += 1
        if self.w(k) in ('"', "'", "("):
            k += 1
        if k >= self.n or self.covered[k]:
            return False
        t = self.toks[k]
        if t.kind == NUM:
            return True
        if t.kind == W and (len(t.text) == 1 and t.text in LETTERS or ordinal_value(t.text) is not None):
            return True
        return False

    # ----------------------------------------------------------- values
    def _follower_ok(self, k: int) -> bool:
        """Что может стоять после значения в перечне."""
        if k >= self.n:
            return True
        if self.covered[k] or k in self.mention_starts:
            return True
        t = self.toks[k]
        if t.kind == P:
            return True
        w = t.text
        if w in ("и", "или", "либо", "а", "также", "рф", "настоящего", "указанного", "того", "этого",
                 "данного", "названного"):
            return True
        if self.marker_at(k) or self.fuzzy_marker_at(k):
            return True
        if w in PREPS and (self.marker_at(k + 1) or self.fuzzy_marker_at(k + 1)):
            return True
        return False

    def _one_value(self, j: int, level: int, first: bool):
        """Одно значение (или диапазон) в позиции j: (list[str], j_after) или None."""
        if j >= self.n or self.covered[j]:
            return None
        k = j
        quoted = False
        if self.w(k) in ('"', "'"):
            quoted = True
            k += 1
        if k >= self.n or self.covered[k]:
            return None
        t = self.toks[k]
        vals = None
        if t.kind == NUM:
            v = t.text.rstrip(".")
            if "." in v and level != ART and False:
                pass
            vals = [v]
            k += 1
            # порядковый суффикс: 1-й, 2-я
            if self.w(k) == "-" and k + 1 < self.n and self.toks[k + 1].kind == W and \
                    self.toks[k + 1].text in ("й", "я", "го", "ой", "ая", "ий", "ей", "ом", "ую", "е", "ым", "му"):
                k += 2
            # диапазон
            elif self.w(k) == "-" and k + 1 < self.n and self.toks[k + 1].kind == NUM and not self.covered[k + 1]:
                rng = expand_range(v, self.toks[k + 1].text.rstrip("."))
                if rng:
                    vals = rng
                    k += 2
        elif t.kind == W and len(t.text) == 1 and t.text in LETTERS and level != ART:
            v = t.text
            nk = k + 1
            if self.w(nk) in ('"', "'") and quoted:
                nk += 1
            elif self.w(nk) == ")":
                nk += 1
            # диапазон в кавычках: «а» - «в»
            if quoted and self.w(nk) == "-" and self.w(nk + 1) in ('"', "'") and nk + 2 < self.n \
                    and self.toks[nk + 2].kind == W and len(self.toks[nk + 2].text) == 1:
                rng = expand_letter_range(v, self.toks[nk + 2].text)
                if rng:
                    nk2 = nk + 3
                    if self.w(nk2) in ('"', "'"):
                        nk2 += 1
                    if self._follower_ok(nk2):
                        return rng, nk2
            # диапазон букв а-в
            if self.w(nk) == "-" and nk + 1 < self.n and self.toks[nk + 1].kind == W and len(self.toks[nk + 1].text) == 1:
                rng = expand_letter_range(v, self.toks[nk + 1].text)
                if rng:
                    nk2 = nk + 2
                    if self.w(nk2) in ('"', "'", ")"):
                        nk2 += 1
                    if self._follower_ok(nk2):
                        return rng, nk2
            if not quoted and not self._follower_ok(nk) and not (self.w(nk) == "," and quoted):
                return None
            if not quoted and v in ("и", "а", "в", "с", "к", "о", "у") and self.w(k + 1) != ")" \
                    and not self._letter_ctx_ok(nk) \
                    and not (not first and self.w(nk) in (",", "и", "или", "либо", ";")):
                return None
            return [v], nk
        elif t.kind == W:
            ov = ordinal_value(t.text)
            if ov is not None:
                vals = [str(ov)]
                k += 1
                # "двадцать первая"
            elif t.text in _CARD_TENS and k + 1 < self.n:
                ov2 = ordinal_value(self.toks[k + 1].text)
                if ov2 is not None and ov2 < 10:
                    vals = [str(_CARD_TENS[t.text] + ov2)]
                    k += 2
        if vals is None:
            return None
        if quoted and self.w(k) in ('"', "'"):
            k += 1
        return vals, k

    def _letter_ctx_ok(self, k: int) -> bool:
        """Буква-предлог считается значением, только если за ней маркер, закон или знак."""
        if k >= self.n:
            return True
        if self.covered[k] or k in self.mention_starts:
            return True
        t = self.toks[k]
        if t.kind == P:
            return True
        return bool(self.marker_at(k) or self.fuzzy_marker_at(k))

    def values_at(self, j: int, level: int):
        """Список значений после маркера: (values, j_after)."""
        k = j
        if self.w(k) in ("№", "n"):
            k += 1
        first = self._one_value(k, level, True)
        if first is None:
            return None
        vals, k = first
        out = list(vals)
        is_letter = not vals[0][0].isdigit()
        while k < self.n:
            # разделитель
            m = k
            seen_conn = False
            seen_word = False
            while m < self.n and not self.covered[m] and self.toks[m].text in (",", "и", "или", "либо", ";", "\n"):
                tx = self.toks[m].text
                if tx == "и" and seen_conn and is_letter:
                    break  # "«л» и и", "д, и и «е»": "и" после связки - буква-значение
                if tx != "\n":
                    seen_conn = True
                if tx in ("и", "или", "либо"):
                    seen_word = True
                m += 1
            if not seen_conn:
                # "«а,в,г,з»" - перечень внутри кавычек
                break
            # после разделителя должен идти не маркер, а значение
            if self.marker_at(m) or (m < self.n and m in self.mention_starts):
                break
            nxt = self._one_value(m, level, False)
            if nxt is None:
                break
            nv, nk = nxt
            if (not nv[0][0].isdigit()) != is_letter:
                break
            # проверка того, что идёт после значения
            if nv[0][0].isdigit() and not self._follower_ok(nk):
                break
            out.extend(nv)
            k = nk
        # закрывающая кавычка общего перечня «а,в,г,з»
        if self.w(k) in ('"', "'") and k > 0 and self.toks[k - 1].kind == W:
            k += 1
        return out, k

    # ----------------------------------------------------------- elements
    def elements(self) -> list[Elem]:
        out: list[Elem] = []
        i = 0
        n = self.n
        while i < n:
            if self.covered[i]:
                i += 1
                continue
            mk = self.marker_at(i) or self.glued_short_marker(i)
            if mk is None and self.toks[i].kind == W and self.toks[i].text not in FULL_FORMS:
                mk = self.fuzzy_marker_at(i)
            if mk is None:
                # порядковое числительное перед маркером: "в третьем пункте"
                if self.toks[i].kind == W:
                    ov = ordinal_value(self.toks[i].text)
                    if ov is not None:
                        mk2 = self.marker_at(i + 1)
                        if mk2 and mk2[0] != ART and not self._value_follows(mk2[2]):
                            out.append(Elem(mk2[0], [str(ov)], i, mk2[2], mk2[1]))
                            i = mk2[2]
                            continue
                i += 1
                continue
            level, ambig, j = mk
            # "ст. ст. 15" - повтор маркера
            mk_rep = self.marker_at(j)
            if mk_rep and mk_rep[0] == level:
                j = mk_rep[2]
            vs = self.values_at(j, level)
            if vs is None:
                i += 1
                continue
            vals, k = vs
            # обратный порядок: "части первой" - уже учтено (значение после маркера)
            out.append(Elem(level, vals, i, k, ambig))
            i = k
        # "п.п. «а», «б» п. 1 ст. 5": "п.п." перед пунктом/частью - это подпункты
        for a, b in zip(out, out[1:]):
            if a.ambig and a.level == POINT and b.level in (POINT, PART) and b.start - a.end <= 1:
                a.level = SUB
                a.ambig = False
        return out


def expand_range(a: str, b: str) -> Optional[list[str]]:
    """"43.2-6" -> 43.2..43.6; "167-170" -> 167..170; "30.1-30.3"."""
    try:
        if "." in a:
            pa = a.split(".")
            if "." in b:
                pb = b.split(".")
                if pa[:-1] != pb[:-1]:
                    return [f"{a}-{b}"]
                lo, hi = int(pa[-1]), int(pb[-1])
            else:
                lo, hi = int(pa[-1]), int(b)
            prefix = ".".join(pa[:-1]) + "."
            if hi <= lo or hi - lo > 60:
                return [f"{a}-{b}"]
            return [f"{prefix}{x}" for x in range(lo, hi + 1)]
        if "." in b:
            return [f"{a}-{b}"]
        lo, hi = int(a), int(b)
        if hi <= lo or hi - lo > 60:
            return [f"{a}-{b}"]
        return [str(x) for x in range(lo, hi + 1)]
    except ValueError:
        return None


_ALPHA = "абвгдежзиклмнопрстуфхцчшщэюя"


def expand_letter_range(a: str, b: str) -> Optional[list[str]]:
    if a in _ALPHA and b in _ALPHA:
        ia, ib = _ALPHA.index(a), _ALPHA.index(b)
        if ia < ib <= ia + 15:
            return list(_ALPHA[ia:ib + 1])
    return None


# ------------------------------------------------------------------ grouping

def group_runs(elems: list[Elem], toks: list[Tok], covered: list[bool]) -> list[list[Elem]]:
    """Делит элементы на серии: соседние элементы, между которыми только связки."""
    runs: list[list[Elem]] = []
    cur: list[Elem] = []
    for e in elems:
        if cur:
            gap = range(cur[-1].end, e.start)
            sep = "none"
            ok = True
            for g in gap:
                if covered[g]:
                    ok = False
                    break
                tx = toks[g].text
                if tx in (",", ";"):
                    sep = "comma" if sep == "none" else sep
                elif tx in ("и", "или", "либо", "а", "также"):
                    sep = "conj"
                elif tx in ("в", "во", "из", "к"):
                    sep = "prep" if sep == "none" else sep
                elif tx in ("\n", "(", ")", "-", "."):
                    pass
                else:
                    ok = False
                    break
            if ok and len(gap) <= 5:
                e.sep = sep
                cur.append(e)
                continue
            runs.append(cur)
            cur = []
        e.sep = "none"
        cur.append(e)
    if cur:
        runs.append(cur)
    return runs


def build_chains(run: list[Elem]) -> list[tuple[Chain, str]]:
    """Группирует элементы серии в цепочки. Возвращает (цепочка, разделитель перед ней).

    Правила:
    - элементы без знаков между ними ("п. 2 пп. 2 ст. 54.1", "ст. 105 ч. 2 п. а") - одна цепочка
      в любом порядке, если уровень ещё не занят;
    - через запятую: нисходящая запись продолжается ("часть 3, ст. 30.1"), восходящая - только
      если после элемента не начинается своя нисходящая цепочка ("статья 5, пункт 2, подпункт б");
    - через "и"/"или" - всегда новая цепочка;
    - "п. 1, п. 2 ст. 5" - перечень одного уровня.
    """
    chains: list[tuple[Chain, str]] = []
    cur: Optional[Chain] = None
    cur_sep = "none"
    asc = False
    for idx, e in enumerate(run):
        if cur is None:
            cur = Chain(start=e.start, end=e.end)
            cur.add(e)
            cur_sep = e.sep
            asc = False
            continue
        lv = cur.levels()
        top, bottom = max(lv), min(lv)
        nxt = run[idx + 1] if idx + 1 < len(run) else None
        # после e начинается своя нисходящая цепочка: "статей 1, 2, п. 10 ст. 373"
        starts_desc = nxt is not None and e.level < nxt.level and nxt.sep in ("none", "prep")
        joined = False
        if e.level not in lv and e.sep != "conj":
            if e.level > top:
                # нисходящая запись: "п. 1 ст. 5", "часть 3, ст. 30.1"
                if not (asc and e.sep == "comma"):
                    joined = True
            elif e.level < bottom:
                if not starts_desc and (e.sep in ("none", "prep") or
                                        (e.sep == "comma" and (ART in lv or asc or len(lv) == 1))):
                    joined = True
                    asc = True
            elif e.sep in ("none", "prep"):
                joined = True
        if joined:
            cur.add(e)
            continue
        if len(lv) == 1 and e.level in lv and e.sep in ("comma", "conj"):
            # "п. 1, п. 2 ст. 5" - перечень одним уровнем
            cur.elems[e.level].values.extend(e.values)
            cur.end = e.end
            continue
        chains.append((cur, cur_sep))
        cur = Chain(start=e.start, end=e.end)
        cur.add(e)
        cur_sep = e.sep
        asc = False
    if cur is not None:
        chains.append((cur, cur_sep))
    # наследование статьи: "пп. а п. 1 и пп. б п. 2 ст. 5"
    for k in range(len(chains) - 2, -1, -1):
        c, _ = chains[k]
        nxt_c, sep = chains[k + 1]
        if ART not in c.elems and ART in nxt_c.elems and max(c.levels()) < ART and sep in ("comma", "conj"):
            top_c = max(c.levels())
            for lvl, el in nxt_c.elems.items():
                if lvl > top_c and lvl not in c.elems:
                    c.elems[lvl] = el
    return chains


def chain_tuples(chain: Chain) -> list[tuple]:
    """Декартово произведение значений -> (article, point, subpoint)."""
    el = dict(chain.elems)
    el.pop(ABZ, None)
    art = el.get(ART)
    part = el.get(PART)
    point = el.get(POINT)
    sub = el.get(SUB)
    if point is not None and point.ambig and sub is None and part is None:
        pass  # "п.п. 1 ст. 5" - пункты
    elif point is not None and point.ambig and sub is None and part is not None:
        pass
    if part is not None and point is not None and sub is not None:
        p_el, s_el = point, sub    # "пп. г п. 6 ч. 1 ст. 81": часть отбрасывается
    elif part is not None and point is not None:
        p_el, s_el = part, point   # "п. 6 ч. 1 ст. 24.5": часть -> пункт, пункт -> подпункт
    elif part is not None:
        p_el, s_el = part, sub
    else:
        p_el, s_el = point, sub
    arts = art.values if art else [None]
    pts = p_el.values if p_el else [None]
    subs = s_el.values if s_el else [None]
    out = []
    for a in arts:
        for p in pts:
            for s in subs:
                out.append((a, p, s))
    return out
