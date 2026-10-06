from app.commercial import commercial_review

from test_music import load_golden


def test_commercial_review_scores_clean_draft() -> None:
    composition = load_golden()
    composition.agent_trace = ["Coordinator Agent planned the draft."]
    report = commercial_review(composition)
    assert report["score"] >= 80
    assert report["checklist"]


def test_commercial_review_flags_direct_artist_imitation() -> None:
    composition = load_golden()
    composition.agent_trace = ["Coordinator Agent planned the draft."]
    composition.style_notes = ["Make it sound like Taylor Swift"]
    report = commercial_review(composition)
    assert report["score"] < 100
    assert report["warnings"]
