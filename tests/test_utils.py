import pytest

from trm.utils import parse_reward_json, reward_from_score


def test_parse_plain_json():
    score, data = parse_reward_json('{"final_score": 8.5, "score_reason": "ok"}')
    assert score == 8.5
    assert data["score_reason"] == "ok"


def test_parse_fenced_json():
    score, _ = parse_reward_json('```json\n{"final_score": 11}\n```')
    assert score == 10.0


def test_parse_embedded_json():
    score, _ = parse_reward_json('result:\n{"final_score": -1}\nend')
    assert score == 0.0


def test_parse_think_block():
    score, _ = parse_reward_json('<think>draft</think>{"final_score": 6.25}')
    assert score == 6.25


def test_reward_from_score():
    assert reward_from_score(7.0) == pytest.approx(0.7)
