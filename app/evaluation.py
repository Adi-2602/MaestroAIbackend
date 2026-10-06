from typing import Dict, List

from music21 import note

from .commercial import commercial_review
from .music import _key_pitch_classes, validate_composition
from .schemas import Composition
from .style_presets import get_style_preset


def _percent(numerator: float, denominator: float) -> int:
    if denominator <= 0:
        return 100
    return max(0, min(100, round((numerator / denominator) * 100)))


def _melody_key_fit(composition: Composition) -> int:
    key_classes = _key_pitch_classes(composition.key)
    if not key_classes:
        return 100
    total = 0
    in_key = 0
    for section in composition.sections:
        for melody_note in section.melody:
            if melody_note.pitch == "rest":
                continue
            try:
                total += 1
                if note.Note(melody_note.pitch).pitch.pitchClass in key_classes:
                    in_key += 1
            except Exception:
                total += 1
    return _percent(in_key, total)


def _duration_fit(composition: Composition) -> int:
    total_score = 0
    sections = 0
    numerator, denominator = composition.time_signature.split("/")
    beats_per_bar = int(numerator) * (4 / int(denominator))
    for section in composition.sections:
        target = section.bars * beats_per_bar
        used = sum(float(item.duration_beats) for item in section.melody)
        if target <= 0:
            continue
        total_score += max(0, 100 - abs(target - used) / target * 100)
        sections += 1
    return round(total_score / sections) if sections else 100


def _style_adherence(composition: Composition) -> int:
    preset = get_style_preset(composition.style)
    if not preset:
        return 70
    score = 0
    checks = 0

    checks += 1
    if preset.bpm_min <= composition.tempo_bpm <= preset.bpm_max:
        score += 1

    all_chords = [chord for section in composition.sections for chord in section.chords]
    if all_chords:
        checks += 1
        preferred = 0
        for chord in all_chords:
            if any(token and token in chord for token in preset.preferred_chord_tokens):
                preferred += 1
            elif "" in preset.preferred_chord_tokens:
                preferred += 1
        if preferred / len(all_chords) >= 0.35:
            score += 1

    section_names = " ".join(section.name.lower() for section in composition.sections)
    checks += 1
    if any(hint.lower() in section_names for hint in preset.section_hints):
        score += 1

    notes = " ".join(composition.style_notes).lower()
    checks += 1
    if any(hint.lower() in notes for hint in preset.arrangement_hints):
        score += 1

    return _percent(score, checks)


def evaluate_composition(composition: Composition) -> Dict[str, object]:
    warnings = validate_composition(composition)
    blocking_warnings = [warning for warning in warnings if warning.startswith("Invalid")]
    melody_key_fit = _melody_key_fit(composition)
    duration_fit = _duration_fit(composition)
    style_adherence = _style_adherence(composition)
    chord_validity = 100 if not blocking_warnings else 60
    lyrics_score = 100 if composition.lyrics else 0
    disclaimer_score = 100 if "commercial use" in composition.disclaimer.lower() else 60
    export_readiness = 100 if not blocking_warnings else 0
    commercial_safety = int(commercial_review(composition)["score"])
    overall = round(
        (
            chord_validity
            + melody_key_fit
            + duration_fit
            + style_adherence
            + lyrics_score
            + disclaimer_score
            + export_readiness
            + commercial_safety
        )
        / 8
    )
    recommendations: List[str] = []
    if melody_key_fit < 80:
        recommendations.append("Regenerate melody or transpose out-of-key notes.")
    if duration_fit < 80:
        recommendations.append("Adjust melody durations so each section fills its bars.")
    if style_adherence < 75:
        recommendations.append("Regenerate arrangement with stronger style-specific instructions.")
    if warnings:
        recommendations.append("Review validation warnings before commercial export.")
    if commercial_safety < 85:
        recommendations.append("Resolve commercial-safety warnings before monetized release.")
    if not recommendations:
        recommendations.append("Draft is ready for MIDI/MusicXML export and human review.")

    return {
        "overall_score": overall,
        "chord_validity": chord_validity,
        "melody_key_fit": melody_key_fit,
        "duration_fit": duration_fit,
        "style_adherence": style_adherence,
        "lyrics_score": lyrics_score,
        "disclaimer_score": disclaimer_score,
        "export_readiness": export_readiness,
        "commercial_safety": commercial_safety,
        "warnings": warnings,
        "recommendations": recommendations,
    }
