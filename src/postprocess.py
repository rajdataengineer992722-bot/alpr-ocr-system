"""OCR text post-processing and validation for ALPR."""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from src.config import get_settings
from src.logger import logger


def clean_text(text: str | None) -> str:
    """Normalize OCR text into uppercase alphanumeric form."""

    if not text:
        return ""
    normalized = str(text).upper().strip()
    normalized = normalized.replace(" ", "").replace("\n", "").replace("\t", "")
    normalized = re.sub(r"[^A-Z0-9]", "", normalized)
    return normalized


def _safe_patterns(patterns: Sequence[str] | None = None) -> List[re.Pattern[str]]:
    """Compile validation patterns from config or provided values."""

    source_patterns = patterns if patterns is not None else get_settings().plate_regex_patterns
    return [re.compile(pattern) for pattern in source_patterns]


def validate_plate(text: str | None, patterns: Sequence[str] | None = None) -> bool:
    """Validate a cleaned plate string against configured regex rules."""

    normalized = clean_text(text)
    if not normalized:
        return False
    compiled_patterns = _safe_patterns(patterns)
    return any(pattern.fullmatch(normalized) for pattern in compiled_patterns)


def correct_common_ocr_errors(text: str | None) -> str:
    """Apply targeted OCR confusion corrections without blindly rewriting everything."""

    normalized = clean_text(text)
    if not normalized:
        return ""

    settings = get_settings()
    chars = list(normalized)

    # Conservative heuristic:
    # Plate prefixes are commonly alphabetic, mid sections often numeric,
    # trailing blocks may mix but frequently end with digits.
    for index, char in enumerate(chars):
        # Early prefix positions are more likely letters.
        if index < 2:
            chars[index] = settings.confusion_map_digit_to_alpha.get(char, char)
            continue

        # Common state/region number positions tend to be numeric.
        if index in (2, 3):
            chars[index] = settings.confusion_map_alpha_to_digit.get(char, char)
            continue

        # Stronger digit bias at the end of typical plate strings.
        if index >= max(0, len(chars) - 4):
            chars[index] = settings.confusion_map_alpha_to_digit.get(char, char)
            continue

        # For middle positions, keep original character unless it is obviously ambiguous.
        if char in settings.confusion_map_alpha_to_digit and char.isalpha():
            if char in {"O", "I", "Z", "S", "B"}:
                # Leave ambiguous mid-section characters unchanged unless validation later prefers a corrected form.
                chars[index] = char

    return "".join(chars)


def _generate_candidates(text: str, patterns: Sequence[str] | None = None) -> List[str]:
    """Generate a small set of plausible corrected candidates for validation."""

    normalized = clean_text(text)
    if not normalized:
        return [""]

    settings = get_settings()
    ordered_candidates = [normalized]
    corrected = correct_common_ocr_errors(normalized)
    if corrected not in ordered_candidates:
        ordered_candidates.append(corrected)

    if len(normalized) >= 2:
        chars = list(normalized)
        for index, char in enumerate(chars):
            if char in settings.confusion_map_alpha_to_digit:
                mutated = chars.copy()
                mutated[index] = settings.confusion_map_alpha_to_digit[char]
                candidate = "".join(mutated)
                if candidate not in ordered_candidates:
                    ordered_candidates.append(candidate)
            if char in settings.confusion_map_digit_to_alpha:
                mutated = chars.copy()
                mutated[index] = settings.confusion_map_digit_to_alpha[char]
                candidate = "".join(mutated)
                if candidate not in ordered_candidates:
                    ordered_candidates.append(candidate)

    valid_candidates = [candidate for candidate in ordered_candidates if validate_plate(candidate, patterns)]
    if normalized in valid_candidates:
        return [normalized, *[candidate for candidate in ordered_candidates if candidate != normalized]]
    if corrected in valid_candidates:
        return [corrected, *[candidate for candidate in ordered_candidates if candidate != corrected]]
    if valid_candidates:
        best_valid = valid_candidates[0]
        return [best_valid, *[candidate for candidate in ordered_candidates if candidate != best_valid]]
    return ordered_candidates


def postprocess_text(text: str | None, patterns: Sequence[str] | None = None) -> Tuple[str, str, bool]:
    """Clean OCR text, apply conservative correction, and validate the result.

    Returns:
        cleaned_text,
        corrected_text,
        is_valid
    """

    cleaned = clean_text(text)
    if not cleaned:
        logger.debug("Postprocessing received empty OCR text.")
        return "", "", False

    candidates = _generate_candidates(cleaned, patterns)
    corrected = candidates[0] if candidates else cleaned
    is_valid = validate_plate(corrected, patterns)

    logger.debug(
        "Postprocessed OCR text raw='{}' cleaned='{}' corrected='{}' valid={}",
        text,
        cleaned,
        corrected,
        is_valid,
    )
    return cleaned, corrected, is_valid


class PlatePostProcessor:
    """Object-oriented wrapper used by the ALPR pipeline."""

    def __init__(self, patterns: Sequence[str] | None = None) -> None:
        self.patterns = list(patterns) if patterns is not None else list(get_settings().plate_regex_patterns)

    def clean_text(self, text: str | None) -> str:
        return clean_text(text)

    def correct_confusions(self, text: str | None) -> str:
        return correct_common_ocr_errors(text)

    def validate_plate(self, text: str | None) -> bool:
        return validate_plate(text, self.patterns)

    def process(self, text: str | None) -> Tuple[str, bool, List[str]]:
        """Return final text, validity flag, and evaluated candidates."""

        cleaned, corrected, is_valid = postprocess_text(text, self.patterns)
        candidates = _generate_candidates(cleaned, self.patterns) if cleaned else [""]
        final_text = corrected if corrected else cleaned
        return final_text, is_valid, candidates
