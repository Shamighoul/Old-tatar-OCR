import unicodedata

import pytest

from text.arabic_normalization import (
    PROFILES, RULES, describe_profiles, is_arabic_letter, normalize_for_evaluation as norm,
    source_preserving_text,
)


def rule(rule_id, s):
    return RULES[rule_id].apply(s)


# --- source preserving ---------------------------------------------------------------

def test_source_preserving_keeps_letters_and_marks():
    s = "كتبخانه‌سی ـ ی ك ١٢"
    assert source_preserving_text(s) == s


def test_source_preserving_only_transport_cleanup():
    assert source_preserving_text("﻿  بر\r\nایکی\r ") == "بر\nایکی"
    assert source_preserving_text(None) == ""


# --- individual rules ------------------------------------------------------------------

def test_n1_nfc_composes_kraken_nfd_output():
    nfd = unicodedata.normalize("NFD", "آ أ ؤ ئ")
    assert nfd != "آ أ ؤ ئ"
    assert rule("N1", nfd) == "آ أ ؤ ئ"


def test_p1_presentation_forms_to_base_letters():
    assert rule("P1", "ﺍﺑﻠ") == "ابل"  # isolated alef, initial beh, medial lam
    assert rule("P1", "ﻻ") == "لا"             # lam-alef ligature
    assert rule("P1", "abc") == "abc"


def test_z1_removes_zero_width_and_bidi_controls():
    assert rule("Z1", "كتبخانه‌سی‍‎‏؜﻿") == "كتبخانهسی"


def test_t1_removes_tatweel():
    assert rule("T1", "بـــر") == "بر"


def test_y1_yeh_variants():
    assert rule("Y1", "یىےي") == "ي" * 4


def test_k1_kaf_variants_but_not_gaf_or_ng():
    assert rule("K1", "کڪك") == "ك" * 3
    assert rule("K1", "گ ڭ") == "گ ڭ"


def test_k2_ng_variants():
    assert rule("K2", "ݣڱڭ") == "ڭ" * 3


def test_h1_heh_variants():
    assert rule("H1", "ہھەه") == "ه" * 4


def test_d1_digits_to_ascii():
    assert rule("D1", "١٩١٥ ۱۹۲۵ 1927") == "1915 1925 1927"


def test_q1_punctuation_variants():
    assert rule("Q1", "،؛؟۔«»“”„″〃") == ',;?."""""""'


def test_s1_whitespace():
    assert rule("S1", "  بر\n\nایکی  اوچ\t") == "بر ایکی اوچ"


def test_di1_removes_diacritics():
    assert rule("DI1", "بِسْمِ اللّٰهِ") == "بسم الله"


def test_a1_alef_variants():
    assert rule("A1", "آ أ إ ٱ ا") == "ا ا ا ا ا"


def test_pu1_removes_punctuation():
    assert rule("PU1", "بر، ایکی. (اوچ)!") == "بر ایکی اوچ"


def test_ar1_keeps_only_arabic_letters():
    assert rule("AR1", "ئات Лошадь 12 ،").split() == ["ئات"]
    assert is_arabic_letter("ࢭ") and not is_arabic_letter("Л")


# --- profiles ----------------------------------------------------------------------------

def test_raw_profile_is_identity():
    s = "  ی‌ "
    assert norm(s, "raw") == s


def test_nfc_profile_keeps_letter_distinctions():
    assert norm("ی ي ک ك ە ه", "nfc") == "ی ي ک ك ە ه"


def test_standard_profile_equates_conventional_variants():
    a = "كتبخانه‌سی  ١٩١٥ «سوز»ـ"
    b = "کتبخانهسي 1915 \"سوز\""
    assert norm(a, "standard") == norm(b, "standard")
    assert norm(a, "nfc") != norm(b, "nfc")


def test_standard_profile_keeps_tatar_letters_distinct():
    out = norm("ڭ گ ۋ ۇ و ق", "standard")
    assert out == "ڭ گ ۋ ۇ و ق"


def test_loose_ignores_diacritics_hamza_alef_and_punctuation():
    assert norm("أُوچ، اللّٰه!", "loose") == norm("اوچ الله", "loose")


def test_arabic_only_on_mixed_line():
    assert norm("ئات Лошадь. Бросай", "arabic_only") == "ئات"


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        norm("x", "nope")


def test_every_profile_rule_exists_and_docs_render():
    for rules in PROFILES.values():
        assert all(r in RULES for r in rules)
    md = describe_profiles()
    assert all(r in md for r in RULES)


def test_normalization_is_idempotent():
    s = "حاضرگه بزنڭ مطبوعات عالمیمزده ١) كتبخانه‌سی، «سوز»"
    for p in PROFILES:
        once = norm(s, p)
        assert norm(once, p) == once
