import re
from typing import Tuple

TR_PLATE_RE = re.compile(r"^(0[1-9]|[1-7][0-9]|8[0-1])[A-Z]{1,3}[0-9]{2,5}$")

# Common OCR confusions. We'll apply them selectively.
LETTER_TO_DIGIT = str.maketrans({"O": "0", "I": "1", "S": "5", "B": "8", "Z": "2"})
DIGIT_TO_LETTER = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z"})


def normalize_plate(raw: str) -> str:
    # Uppercase, keep only A-Z0-9
    s = raw.upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def is_valid_tr_plate(s: str) -> bool:
    return bool(TR_PLATE_RE.match(s))


def try_fix_tr_plate(s: str) -> Tuple[str, bool]:
    """
    Try minimal edits to satisfy TR format.
    Strategy:
    - Normalize
    - If valid => return
    - Else try a couple targeted substitutions and keep first valid
    """
    s = normalize_plate(s)
    if is_valid_tr_plate(s):
        return s, True

    # Try converting ambiguous chars globally (letters->digits)
    cand1 = s.translate(LETTER_TO_DIGIT)
    if is_valid_tr_plate(cand1):
        return cand1, True

    # Try digits->letters globally
    cand2 = s.translate(DIGIT_TO_LETTER)
    if is_valid_tr_plate(cand2):
        return cand2, True

    # Try hybrid: first 2 chars are province digits; force digits there
    if len(s) >= 2:
        head = s[:2].translate(LETTER_TO_DIGIT)
        tail = s[2:]
        cand3 = head + tail
        if is_valid_tr_plate(cand3):
            return cand3, True

    return s, False
