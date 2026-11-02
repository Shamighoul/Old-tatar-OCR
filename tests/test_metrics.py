import pytest

from evaluation import metrics as M


def test_edit_distance_matches_python_fallback():
    pairs = [("", ""), ("abc", ""), ("", "abc"), ("kitten", "sitting"), ("بزنڭ", "بزنك"), ("abc", "acb")]
    for a, b in pairs:
        assert M.edit_distance(a, b) == M._levenshtein_python(a, b)
    assert M.edit_distance(["a", "b"], ["a", "c", "b"]) == 1


def test_cer_basic():
    assert M.cer("abcd", "abcd") == 0.0
    assert M.cer("abcd", "abxd") == pytest.approx(0.25)
    assert M.cer("abcd", "") == 1.0
    assert M.cer("ab", "abcdef") == pytest.approx(2.0)  # insertions are not clipped


def test_cer_empty_reference():
    assert M.cer("", "") == 0.0
    assert M.cer("", "x") == 1.0


def test_wer_basic():
    assert M.wer("بر ایکی اوچ", "بر ایکی اوچ") == 0.0
    assert M.wer("بر ایکی اوچ", "بر ایكی اوچ") == pytest.approx(1 / 3)
    assert M.wer("بر ایکی", "") == 1.0


def test_counts_breakdown():
    c = M.char_counts("abcd", "axcde")
    assert (c.edits, c.substitutions, c.insertions, c.deletions) == (2, 1, 1, 0)
    assert c.ref_len == 4 and c.hyp_len == 5


def test_corpus_rate_is_micro_average():
    counts = [M.char_counts("aaaa", "aaab"), M.char_counts("aaaaaaaaaaaaaaaa", "aaaaaaaaaaaaaaaa")]
    assert M.corpus_rate(counts) == pytest.approx(1 / 20)
    assert (counts[0] + counts[1]).ref_len == 20


def test_exact_line_accuracy():
    assert M.exact_line_accuracy(["a", "b", "c"], ["a", "x", "c"]) == pytest.approx(2 / 3)
    assert M.exact_line_accuracy(["a", "b"], ["a"]) == 0.5
    assert M.exact_line_accuracy([], ["a"]) is None


def test_bag_of_words_ignores_order():
    bow = M.bag_of_words("بر ایکی اوچ", "اوچ بر ایکی")
    assert bow["f1"] == 1.0
    assert M.wer("بر ایکی اوچ", "اوچ بر ایکی") > 0


def test_char_confusions():
    conf = M.char_confusions("بزنڭ", "بزنك")
    assert conf[("ڭ", "ك")] == 1
    conf = M.char_confusions("ab", "a")
    assert conf[("b", "")] == 1


def test_letter_recall_counts_aligned_identical_letters():
    from evaluation.error_analysis import letter_recall

    total, correct = letter_recall("بزنڭ ڭ", "بزنك ڭ")
    assert total["ڭ"] == 2 and correct["ڭ"] == 1 and correct["ب"] == 1
