"""String normalisation shared by the scorers."""

import re


def clean_string(s) -> str:
    """Normalise a station / route string for comparison.

    Same rules as the original ``tools.clean_string``: drop "Metro", commas, asterisks,
    newlines and any parenthesised part, then remove whitespace / dashes and lower-case.
    """
    if s is None:
        return ""
    s = str(s)
    s = s.replace("Metro", "").replace("metro", "").replace(",", "").replace("*", "").replace("\n", "")
    s = re.sub(r"[\(\（][^\)\）]*[\)\）]", "", s)
    return re.sub(r"[\s\-]+", "", s).lower()
