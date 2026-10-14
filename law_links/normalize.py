"""Нормализация текста без изменения длины и токенизация.

Длина строки сохраняется, поэтому позиции токенов совпадают с исходным текстом.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import snowballstemmer

# Латиница, похожая на кириллицу.
_LAT2CYR = {
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у", "k": "к",
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М", "O": "О",
    "P": "Р", "T": "Т", "X": "Х", "Y": "У",
}
_LAT_SET = set(_LAT2CYR)

_CHAR_MAP = {
    "ё": "е", "Ё": "Е",
    "«": '"', "»": '"', "„": '"', "“": '"', "”": '"', "‟": '"', "″": '"', "＂": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "`": "'", "´": "'",
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-", "﹣": "-", "－": "-",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    "　": " ", "\t": " ", "\r": " ", "\x0b": " ", "\x0c": " ",
    "​": " ", "‌": " ", "‍": " ", "﻿": " ",
    "Ν": "N",  # греческая N
    "І": "I", "і": "i",  # украинская i (в номерах законов 2300-I)
    "¹": "1", "²": "2", "³": "3",
}
_TRANS = str.maketrans(_CHAR_MAP)

_WORD_RE = re.compile(r"[A-Za-zА-Яа-я]+")
_CYR_RE = re.compile(r"[А-Яа-я]")


def _fix_homoglyphs(text: str) -> str:
    out = list(text)
    for m in _WORD_RE.finditer(text):
        w = m.group()
        has_lat = any(ch in _LAT_SET for ch in w)
        if not has_lat:
            continue
        if _CYR_RE.search(w):
            conv = True
        else:
            # Чисто латинское слово: переводим только короткие, целиком из двойников
            # (CT, HK, YK, TK, PФ...), длинные (FIFA, UEFA) не трогаем.
            conv = len(w) <= 4 and all(ch in _LAT_SET for ch in w)
        if conv:
            for i, ch in enumerate(w, m.start()):
                if ch in _LAT2CYR:
                    out[i] = _LAT2CYR[ch]
    return "".join(out)


def normalize(text: str) -> str:
    """Нормализованный текст той же длины (регистр сохраняется)."""
    t = text.translate(_TRANS)
    t = _fix_homoglyphs(t)
    assert len(t) == len(text)
    return t


def lower_same_len(text: str) -> str:
    low = text.lower()
    if len(low) == len(text):
        return low
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in text)


# --- токены ---------------------------------------------------------------

W, NUM, P = "W", "N", "P"

_TOKEN_RE = re.compile(r"[а-яa-z]+|\d+(?:\.\d+)*|[^\s]")


@dataclass(slots=True)
class Tok:
    text: str      # в нижнем регистре
    start: int
    end: int
    kind: str      # W / N / P
    upper: bool    # в исходнике слово написано заглавными (для аббревиатур)
    cap: bool      # с заглавной буквы
    space_before: bool


def tokenize(norm: str) -> list[Tok]:
    low = lower_same_len(norm)
    toks: list[Tok] = []
    prev_end = 0
    for m in _TOKEN_RE.finditer(low):
        s, e = m.start(), m.end()
        t = m.group()
        if t[0].isalpha():
            kind = W
        elif t[0].isdigit():
            kind = NUM
        else:
            kind = P
        orig = norm[s:e]
        toks.append(Tok(t, s, e, kind, orig.isupper() and len(orig) > 1 if kind == W else False,
                        orig[:1].isupper(), s > prev_end))
        prev_end = e
    return toks


_stemmer = snowballstemmer.stemmer("russian")


@lru_cache(maxsize=200_000)
def stem(word: str) -> str:
    if not word or not word[0].isalpha():
        return word
    return _stemmer.stemWord(word)
