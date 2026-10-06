import json
from pathlib import Path

from app.nim_client import (
    _normalize_plan,
    build_arrangement_agent_prompt,
    build_bass_agent_prompt,
    build_chord_agent_prompt,
    build_coordinator_prompt,
    build_critic_agent_prompt,
    build_drum_agent_prompt,
    build_lyrics_agent_prompt,
    build_melody_agent_prompt,
    build_mix_agent_prompt,
    build_safety_agent_prompt,
    build_user_prompt,
)
from app.schemas import ComposeRequest, Composition


def load_prompt_golden() -> Composition:
    payload = json.loads(Path(__file__).with_name("golden_composition.json").read_text())
    return Composition.model_validate(payload)


def test_rock_prompt_has_style_specific_directions() -> None:
    prompt = build_user_prompt(
        ComposeRequest(
            style="Rock",
            mood="Energetic",
            theme="city lights",
            key="E minor",
            tempo_bpm=128,
            bars=8,
            instrumentation="driven electric guitars, live drums, bass riff",
        )
    )

    assert "driven guitars" in prompt
    assert "power-chord-friendly" in prompt
    assert "Avoid defaulting to C-G-Am-F or Am-F-C-G" in prompt
    assert '"chords": [string]' in prompt


def test_lofi_and_rock_prompts_use_different_blueprints() -> None:
    base = {
        "mood": "Relaxed",
        "theme": "rainy night",
        "key": "A minor",
        "tempo_bpm": 90,
        "bars": 8,
    }
    lofi_prompt = build_user_prompt(ComposeRequest(style="Lo-fi", **base))
    rock_prompt = build_user_prompt(ComposeRequest(style="Rock", **base))

    assert "dusty keys" in lofi_prompt
    assert "driven guitars" in rock_prompt
    assert lofi_prompt != rock_prompt


def test_multi_agent_prompt_builders_name_distinct_agents() -> None:
    request = ComposeRequest(
        style="Rock",
        mood="Energetic",
        theme="open highway",
        key="E minor",
        tempo_bpm=128,
        bars=8,
        instrumentation="driven electric guitars, live drums, bass riff",
    )
    plan = {
        "title": "Highway Static",
        "brief": "Rock song plan",
        "sections": [{"name": "Verse", "bars": 4}, {"name": "Chorus", "bars": 4}],
    }
    chord_doc = {
        "sections": [
            {"name": "Verse", "bars": 4, "chords": ["Em", "G", "D", "A"]},
            {"name": "Chorus", "bars": 4, "chords": ["G", "D", "A", "Em"]},
        ]
    }

    prompts = [
        build_coordinator_prompt(request),
        build_chord_agent_prompt(request, plan),
        build_melody_agent_prompt(request, plan, chord_doc),
        build_lyrics_agent_prompt(request, plan),
        build_arrangement_agent_prompt(request, plan),
        build_drum_agent_prompt(request, plan),
        build_bass_agent_prompt(request, plan, chord_doc),
        build_critic_agent_prompt(load_prompt_golden()),
        build_mix_agent_prompt(request, load_prompt_golden()),
        build_safety_agent_prompt(request, load_prompt_golden()),
    ]

    assert "AGENT: Coordinator Agent" in prompts[0]
    assert "AGENT: Chord Agent" in prompts[1]
    assert "AGENT: Melody Agent" in prompts[2]
    assert "AGENT: Lyrics Agent" in prompts[3]
    assert "AGENT: Arrangement Agent" in prompts[4]
    assert "AGENT: Drum Agent" in prompts[5]
    assert "AGENT: Bass Agent" in prompts[6]
    assert "AGENT: Critic Agent" in prompts[7]
    assert "AGENT: Mix Agent" in prompts[8]
    assert "AGENT: Commercial Safety Agent" in prompts[9]


def test_normalize_plan_preserves_requested_total_bars() -> None:
    request = ComposeRequest(style="Rock", mood="Energetic", theme="storm", bars=12)
    plan = {
        "title": "Storm Wire",
        "sections": [
            {"name": "Intro", "bars": 1},
            {"name": "Verse", "bars": 1},
            {"name": "Chorus", "bars": 1},
        ],
    }

    normalized = _normalize_plan(plan, request)

    assert sum(section["bars"] for section in normalized["sections"]) == 12
    assert normalized["title"] == "Storm Wire"
