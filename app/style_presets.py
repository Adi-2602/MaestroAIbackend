from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class StylePreset:
    name: str
    bpm_min: int
    bpm_max: int
    preferred_chord_tokens: List[str]
    section_hints: List[str]
    arrangement_hints: List[str]


STYLE_PRESETS: Dict[str, StylePreset] = {
    "lo-fi": StylePreset(
        name="Lo-fi",
        bpm_min=65,
        bpm_max=92,
        preferred_chord_tokens=["maj7", "m7", "7", "add9"],
        section_hints=["Intro", "Verse", "Hook", "Outro"],
        arrangement_hints=["warm", "soft", "dusty", "vinyl", "piano", "pad", "drums"],
    ),
    "pop": StylePreset(
        name="Pop",
        bpm_min=80,
        bpm_max=128,
        preferred_chord_tokens=["", "m", "sus", "add9"],
        section_hints=["Verse", "Pre-Chorus", "Chorus", "Bridge"],
        arrangement_hints=["hook", "bright", "singable", "chorus", "beat"],
    ),
    "rock": StylePreset(
        name="Rock",
        bpm_min=90,
        bpm_max=160,
        preferred_chord_tokens=["", "m", "5", "sus"],
        section_hints=["Intro", "Verse", "Chorus", "Bridge"],
        arrangement_hints=["guitar", "drive", "bass", "drums", "riff"],
    ),
    "edm": StylePreset(
        name="EDM",
        bpm_min=118,
        bpm_max=145,
        preferred_chord_tokens=["m", "", "sus", "add9"],
        section_hints=["Intro", "Build", "Drop", "Break"],
        arrangement_hints=["kick", "drop", "build", "sidechain", "synth", "hook"],
    ),
    "jazz": StylePreset(
        name="Jazz",
        bpm_min=70,
        bpm_max=180,
        preferred_chord_tokens=["maj7", "m7", "7", "dim", "9", "13"],
        section_hints=["Head", "A Section", "B Section", "Solo"],
        arrangement_hints=["swing", "walking", "brushes", "extended", "voicing"],
    ),
    "r&b": StylePreset(
        name="R&B",
        bpm_min=65,
        bpm_max=110,
        preferred_chord_tokens=["maj7", "m7", "7", "9", "add9"],
        section_hints=["Verse", "Pre-Chorus", "Hook", "Bridge"],
        arrangement_hints=["smooth", "groove", "bass", "vocal", "pad"],
    ),
    "folk": StylePreset(
        name="Folk",
        bpm_min=70,
        bpm_max=125,
        preferred_chord_tokens=["", "m", "sus"],
        section_hints=["Verse", "Chorus", "Bridge", "Outro"],
        arrangement_hints=["acoustic", "story", "strum", "simple", "warm"],
    ),
    "cinematic": StylePreset(
        name="Cinematic",
        bpm_min=55,
        bpm_max=120,
        preferred_chord_tokens=["m", "maj7", "sus", "add9"],
        section_hints=["Intro", "Theme", "Rise", "Finale"],
        arrangement_hints=["strings", "pad", "tension", "wide", "dramatic"],
    ),
}


def get_style_preset(style: str) -> StylePreset | None:
    return STYLE_PRESETS.get(style.strip().lower())
