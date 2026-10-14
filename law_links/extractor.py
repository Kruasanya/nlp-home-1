"""Извлечение юридических ссылок: законы + цепочки -> LawLink."""
from __future__ import annotations

from typing import Optional

from .laws import CODE, LawIndex, Mention, MentionFinder
from .normalize import NUM, P, W, normalize, tokenize
from .refs import RefParser, build_chains, chain_tuples, group_runs

# слова, которые можно пропустить между цепочкой и названием закона
FILLER_RIGHT = {
    "рф", "в", "ред", "редакции", "действующей", "действующего", "действующему", "нового", "новой",
    "текущей", "соответствующего", "и", "а", "также", "далее", "так", "называемого", "ныне", "др",
}
GAP_PUNCT = {",", "(", ")", ":", "-", "\n", ";"}
LEFT_GAP = {",", "(", ")", ":", "-", "\n", "и", "а", "также", "в", "во", "его", "ее", "их", "именно", "согласно",
            "по", "—"}


class Extractor:
    def __init__(self, index: LawIndex):
        self.ix = index

    def extract(self, text: str) -> list[tuple]:
        norm = normalize(text)
        toks = tokenize(norm)
        if not toks:
            return []
        finder = MentionFinder(self.ix, toks, norm)
        mentions = finder.find()
        n = len(toks)
        covered = [False] * n
        start_of: dict[int, Mention] = {}
        for m in mentions:
            for k in range(m.start, min(m.end, n)):
                covered[k] = True
            start_of[m.start] = m
        parser = RefParser(toks, covered, set(start_of))
        elems = parser.elements()
        runs = group_runs(elems, toks, covered)

        results: list[tuple] = []
        prev_law: Optional[Mention] = None
        prev_run_end = -1
        for run in runs:
            run_end = run[-1].end
            law = self._right_law(toks, covered, start_of, run_end, n)
            if law is False:
                prev_law = None
                continue
            if law is None:
                law = self._left_law(toks, mentions, run[0].start)
                if law is not None and law.kind not in ("ukaz", "rasp", "std") \
                        and not any(e.level == 4 for e in run):
                    law = None
            if law is None or law.law_id is None:
                continue
            for chain, _ in build_chains(run):
                for a, p, s in chain_tuples(chain):
                    if a is None and p is None and s is None:
                        continue
                    results.append((law.law_id, a, p, s))
            prev_law = law
            prev_run_end = run_end
        # без дублей, порядок как в тексте
        seen = set()
        out = []
        for r in results:
            if r not in seen:
                seen.add(r)
                out.append(r)
        return out

    def _right_law(self, toks, covered, start_of, j: int, n: int):
        """Закон справа от серии. None - не найден, False - явно другой документ."""
        k = j
        words = 0
        while k < n:
            if k in start_of:
                m = start_of[k]
                if m.kind == "unknown" or m.law_id is None:
                    return False
                return m
            t = toks[k]
            if t.kind == P:
                if t.text in GAP_PUNCT or t.text in ('"', "'"):
                    k += 1
                    continue
                if t.text == "." and k > 0 and toks[k - 1].text == "др":
                    k += 1
                    continue
                if t.text == ".":
                    # "ст. 19.5. КоАП" - точка после номера, дальше закон с маленькой буквы
                    # или сокращение кодекса
                    nk = k + 1
                    if nk in start_of:
                        m = start_of[nk]
                        if (m.kind == CODE and m.end - m.start <= 3) or not toks[nk].cap:
                            k = nk
                            continue
                    return None
                return None
            if t.kind == W and t.text in FILLER_RIGHT and words < 3:
                words += 1
                k += 1
                continue
            return None
        return None

    def _left_law(self, toks, mentions, i: int) -> Optional[Mention]:
        """Закон слева в том же предложении: "НК РФ ст. 5", "ГК РФ и ст. 6"."""
        best = None
        for m in mentions:
            if m.end <= i:
                best = m
            else:
                break
        if best is None or best.kind == "unknown" or best.law_id is None:
            return None
        gap = toks[best.end:i]
        if len(gap) > 4:
            return None
        for t in gap:
            if t.text not in LEFT_GAP:
                return None
        return best
