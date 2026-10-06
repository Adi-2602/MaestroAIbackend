# pyrefly: ignore [missing-import]
from app.workflow import (
    build_reference_prompt,
    build_theorist_prompt,
    build_composer_prompt,
    build_lyricist_prompt,
    build_improvisor_prompt,
    build_director_prompt,
    workflow_app,
)
from app.schemas import ComposeRequest, Composition
import json
from pathlib import Path


def load_golden_test() -> Composition:
    payload = json.loads(Path(__file__).parent.joinpath("golden_composition.json").read_text())
    return Composition.model_validate(payload)


def test_langgraph_workflow_compiles() -> None:
    # Ensure the workflow_app compiles successfully and contains the correct nodes
    assert workflow_app is not None
    # Get node names from graph
    nodes = workflow_app.nodes
    assert "reference" in nodes
    assert "theorist" in nodes
    assert "composer" in nodes
    assert "lyricist" in nodes
    assert "sentiment" in nodes
    assert "improvisor" in nodes
    assert "director" in nodes


def test_reference_prompt_builder() -> None:
    req = ComposeRequest(
        style="EDM",
        mood="Energetic",
        theme="electric fields",
        key="A minor",
        tempo_bpm=128,
        bars=8,
    )
    prompt = build_reference_prompt(req)
    assert "EDM" in prompt
    assert "electric fields" in prompt
    assert "sections" in prompt
    assert "bars" in prompt


def test_theorist_prompt_builder() -> None:
    req = ComposeRequest(style="Jazz", mood="Dreamy", theme="clouds", key="C major", bars=4)
    state = {
        "reference_patterns": {"key_profile": "C major scale"},
        "plan": {"sections": [{"name": "Head", "bars": 4}]},
    }
    prompt = build_theorist_prompt(req, state)
    assert "Theorist Agent" in prompt
    assert "Jazz" in prompt
    assert "C major" in prompt
    assert "Head" in prompt


def test_composer_prompt_builder() -> None:
    req = ComposeRequest(style="Folk", mood="Hopeful", theme="mountains", key="G major", bars=8)
    state = {
        "plan": {"sections": [{"name": "Verse", "bars": 8}]},
        "chords": {"sections": [{"name": "Verse", "chords": ["G", "C", "D", "G"]}]},
    }
    prompt = build_composer_prompt(req, state)
    assert "Composer Agent" in prompt
    assert "melody" in prompt
    assert "G" in prompt


def test_lyricist_prompt_builder() -> None:
    req = ComposeRequest(style="Lo-fi", mood="Relaxed", theme="rain", bars=4)
    state = {"plan": {"sections": [{"name": "Hook", "bars": 4}]}}
    prompt = build_lyricist_prompt(req, state)
    assert "Lyricist Agent" in prompt
    assert "rain" in prompt
    assert "Hook" in prompt


def test_improvisor_prompt_builder() -> None:
    req = ComposeRequest(style="Rock", mood="Dark", theme="shadows", bars=8)
    state = {
        "plan": {"sections": [{"name": "Chorus", "bars": 8}]},
        "chords": {"sections": [{"name": "Chorus", "chords": ["Am", "F", "C", "G"]}]},
        "melody": {"sections": [{"name": "Chorus", "melody": [{"pitch": "A4", "duration_beats": 1}]}]},
    }
    prompt = build_improvisor_prompt(req, state)
    assert "Improvisor Agent" in prompt
    assert "embellish" in prompt
    assert "drum_pattern" in prompt
    assert "bassline" in prompt


def test_director_prompt_builder() -> None:
    req = ComposeRequest(style="Pop", mood="Sad", theme="blue skies", key="F major", bars=4)
    state = {
        "plan": {"sections": [{"name": "Chorus", "bars": 4}]},
        "chords": {"sections": [{"name": "Chorus", "chords": ["F", "Bb", "C", "Dm"]}]},
        "melody": {"sections": [{"name": "Chorus", "melody": [{"pitch": "F4", "duration_beats": 2}]}]},
        "lyrics": {"lyrics": ["sky is blue", "feel so new"], "sections": [{"name": "Chorus", "lyric_lines": ["sky is blue"]}]},
        "arrangement": {
            "sections": [{"name": "Chorus", "improvised_chords": ["Fmaj7", "Bbmaj7"]}],
            "drum_pattern": ["pop pattern"],
            "bassline": ["pop bass"],
            "style_notes": ["piano notes"],
            "originality_notes": ["original pop melody"],
        },
    }
    prompt = build_director_prompt(req, state, "safe", state["lyrics"])
    assert "Director Agent" in prompt
    assert "SAFE" in prompt
    assert "Fmaj7" in prompt
    assert "pop pattern" in prompt
