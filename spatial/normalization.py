"""Scoring only. Never use answer normalization for entity identity resolution."""
import re
import unicodedata
from .schema import TypedAnswer


def whitespace(text):
    return ' '.join(text.split()) if isinstance(text, str) else None


def integer(text):
    if isinstance(text, bool):
        return None
    if isinstance(text, int):
        return text
    if not isinstance(text, str) or not re.fullmatch(r'[+-]?\d+', text.strip()):
        return None
    return int(text.strip())


def yes_no(text):
    if isinstance(text, bool):
        return text
    if not isinstance(text, str):
        return None
    value = ''.join(c for c in unicodedata.normalize('NFC', text)
                    if unicodedata.category(c) != 'Mn' and c != 'ـ').strip()
    return {'نعم': True, 'لا': False}.get(value)


def answer_matches(answer: TypedAnswer, gold: str) -> bool:
    if answer.kind == 'boolean':
        parsed = yes_no(gold)
        return parsed is not None and answer.value == parsed
    if answer.kind == 'integer':
        parsed = integer(gold)
        return parsed is not None and answer.value == parsed
    if answer.kind == 'direction':
        return answer.text == gold
    left, right = whitespace(answer.text), whitespace(gold)
    return left is not None and right is not None and left == right
