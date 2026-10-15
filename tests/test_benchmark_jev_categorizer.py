"""Pure-helper tests for the Jev-vs-Haiku benchmark: question building, response
parsing, judge de-randomisation and the agreement maths. No network, no DB."""

import json

from pipeline.categorizer import CATEGORIES, SUBCATEGORIES
from scripts.benchmark_jev_categorizer import (
    CATEGORY_CRITERIA,
    QUALITY_LEVELS,
    agreement,
    arm_input,
    build_jev_questions,
    derandomize_verdict,
    order_flip_block,
    parse_haiku_response,
    parse_jev_answers,
    parse_judge_verdict,
    select_judge_cases,
)


def test_category_criteria_cover_exactly_categories():
    assert set(CATEGORY_CRITERIA) == set(CATEGORIES)


def test_questions_cover_all_subcategories_plus_none_last():
    q = build_jev_questions()
    total = 0
    for cat in CATEGORIES:
        crit = q[f"sub_{cat}"]["criteria"]
        assert set(crit) == set(SUBCATEGORIES[cat]) | {"none"}
        assert list(crit)[-1] == "none"
        total += len(SUBCATEGORIES[cat])
    assert total == 49


def test_questions_shape_and_json_roundtrip():
    q = build_jev_questions()
    assert len(q) == 1 + len(CATEGORIES) + 2
    assert q["category"]["type"] == "choice"
    assert q["is_real_event"]["type"] == "noul"
    assert q["quality"]["type"] == "score"
    assert len(q["quality"]["criteria"]) == len(QUALITY_LEVELS) == 3
    assert json.loads(json.dumps(q)) == q


def test_reversed_questions_reverse_category_order_only():
    fwd, rev = build_jev_questions(), build_jev_questions(reverse_categories=True)
    assert list(fwd["category"]["criteria"]) == CATEGORIES
    assert list(rev["category"]["criteria"]) == CATEGORIES[::-1]
    for cat in CATEGORIES:
        assert fwd[f"sub_{cat}"] == rev[f"sub_{cat}"]


def _answers(cat="music", sub="jazz-blues", noul=0.9, score=2):
    return {
        "category": {
            "type": "choice",
            "choice": cat,
            "confidence": 0.8,
            "probabilities": {c: (0.8 if c == cat else 0.025) for c in CATEGORIES},
        },
        f"sub_{cat}": {
            "type": "choice",
            "choice": sub,
            "confidence": 0.6,
            "probabilities": {sub: 0.6},
        },
        "is_real_event": {"type": "noul", "noul": noul},
        "quality": {
            "type": "score",
            "score": score,
            "confidence": 0.7,
            "legend": {str(i): lvl for i, (lvl, _) in enumerate(QUALITY_LEVELS)},
        },
    }


def test_parse_jev_answers_happy_path():
    out = parse_jev_answers(_answers())
    assert out["category"] == "music"
    assert out["subcategory"] == "jazz-blues"
    assert out["category_confidence"] == 0.8
    assert out["is_real_event"] is True
    assert out["quality_level"] == 2 and out["quality_score"] == 1.0
    assert out["error"] is None


def test_parse_jev_answers_none_invalid_and_missing():
    assert parse_jev_answers(_answers(sub="none"))["subcategory"] is None
    bad = parse_jev_answers(_answers(sub="club-night"))  # valid slug, wrong parent
    assert bad["subcategory"] is None and "bad_subcategory" in bad["error"]
    bad_cat = parse_jev_answers(_answers(cat="sports"))
    assert bad_cat["category"] is None and "bad_category" in bad_cat["error"]
    missing = parse_jev_answers({})
    assert missing["category"] is None and "missing:category" in missing["error"]
    # legend-text score shape
    a = _answers()
    a["quality"]["score"] = QUALITY_LEVELS[1][0]
    assert parse_jev_answers(a)["quality_score"] == 0.5
    assert parse_jev_answers(_answers(noul=0.2))["is_real_event"] is False


def test_parse_haiku_response_strips_fences_and_wraps_single():
    assert parse_haiku_response('```json\n[{"category": "music"}]\n```') == [{"category": "music"}]
    assert parse_haiku_response('{"category": "music"}') == [{"category": "music"}]


def test_arm_input_clears_leak_fields_and_caps_description():
    row = {
        "id": "x",
        "title": "t",
        "venue_name": "v",
        "description": "d" * 1000,
        "price": None,
        "source": "luma",
        "category": "music",
        "tags": ["jazz"],
        "source_tags": ["Jazz"],
    }
    p = arm_input(row)
    assert p["existing_category"] is None and p["existing_tags"] is None
    assert len(p["description"]) == 800
    assert p["source_tags"] == ["Jazz"]


def test_derandomize_verdict():
    assert derandomize_verdict("A", "haiku") == "haiku"
    assert derandomize_verdict("B", "haiku") == "jev"
    assert derandomize_verdict("A", "jev") == "jev"
    assert derandomize_verdict("B", "jev") == "haiku"
    assert derandomize_verdict("both", "jev") == "both"
    assert derandomize_verdict("neither", "haiku") == "neither"


def test_parse_judge_verdict_validates_and_normalises():
    v = parse_judge_verdict(
        'Sure:\n```json\n{"winner":"A","correct_category":"music",'
        '"correct_subcategory":"Jazz","reason":"r"}\n```'
    )
    assert v["error"] is None and v["winner"] == "A" and v["correct_subcategory"] == "jazz-blues"
    assert parse_judge_verdict('{"winner":"C"}')["error"].startswith("bad_winner")
    assert parse_judge_verdict("no json here")["error"] == "no_json"


def test_agreement_math():
    a = {"1": "music", "2": None, "3": "food"}
    b = {"1": "music", "2": None, "4": "food"}
    out = agreement(a, b, ["1", "2", "3", "4"])
    assert out == {"n": 2, "agree": 2, "frac": 1.0}


def test_select_judge_cases_is_deterministic_and_splits():
    rows = [{"id": str(i)} for i in range(10)]
    haiku = {str(i): {"category": "music", "subcategory": None} for i in range(10)}
    jev = {
        str(i): {"category": "music" if i < 7 else "food", "subcategory": None} for i in range(10)
    }
    c1 = select_judge_cases(rows, haiku, jev, seed=1, control_n=3)
    c2 = select_judge_cases(rows, haiku, jev, seed=1, control_n=3)
    assert c1 == c2
    dis = {c["id"] for c in c1 if c["kind"] == "disagree"}
    ctl = {c["id"] for c in c1 if c["kind"] == "control"}
    assert dis == {"7", "8", "9"}
    assert len(ctl) == 3 and ctl <= {str(i) for i in range(7)}
    assert {c["a_arm"] for c in c1} <= {"haiku", "jev"}


def test_order_flip_block():
    jev = {
        str(i): {"category": "music", "category_confidence": 0.8, "category_probs": {"music": 0.8}}
        for i in range(4)
    }
    rev = {
        "0": {"category": "music", "category_confidence": 0.7, "category_probs": {"music": 0.7}},
        "1": {"category": "food", "category_confidence": 0.5, "category_probs": {"food": 0.5}},
        "2": {"category": "music", "category_confidence": 0.8, "category_probs": {"music": 0.8}},
        "3": {"category": "food", "category_confidence": 0.6, "category_probs": {"food": 0.6}},
    }
    out = order_flip_block(jev, rev)
    assert out["n"] == 4 and out["flips"] == 2 and out["category_flip_rate"] == 0.5
    assert out["identical_probs"] == 1
