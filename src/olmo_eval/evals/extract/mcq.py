"""Multiple-choice answer extraction from free-form model output.

Tries patterns in order of specificity:
    1. ``ANSWER: X``  — explicit instruction-following format
    2. ``\\boxed{X}``  — LaTeX (common with thinking-mode models)
    3. ``(X)``         — parenthesized letter (most common overall)
"""

import re

# "ANSWER: X" or "ANSWER: (X)" — case-insensitive
_ANSWER_PATTERN = re.compile(r"ANSWER\s*:\s*\(?([A-Z])\)?", re.IGNORECASE)

# \boxed{X} or \boxed{\text{X}}
_BOXED_PATTERN = re.compile(r"\\boxed\{(?:\\text\{)?([A-Z])")

# Standalone (X)
_PAREN_LETTER = re.compile(r"\(([A-Z])\)")


def extract_mcq_answer(text: str) -> str | None:
    """Extract an MCQ letter from model output.

    Returns the last match from the highest-priority pattern, or None.
    """
    matches = list(_ANSWER_PATTERN.finditer(text))
    if matches:
        return matches[-1].group(1).upper()

    matches = list(_BOXED_PATTERN.finditer(text))
    if matches:
        return matches[-1].group(1).upper()

    matches = list(_PAREN_LETTER.finditer(text))
    if matches:
        return matches[-1].group(1).upper()

    return None
