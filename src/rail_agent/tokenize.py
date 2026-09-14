from __future__ import annotations

import logging
import re
import unicodedata

import jieba


jieba.setLogLevel(logging.ERROR)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._/+\-][a-z0-9]+)*|[\u3400-\u9fff]+", re.I)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    tokens: list[str] = []
    for unit in TOKEN_RE.findall(normalized):
        if re.fullmatch(r"[\u3400-\u9fff]+", unit):
            tokens.extend(token.strip() for token in jieba.cut_for_search(unit) if token.strip())
        else:
            tokens.append(unit)
    return tokens
