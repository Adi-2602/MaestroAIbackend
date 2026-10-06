import math
import shutil
import struct
import subprocess
import tempfile
import wave
from io import BytesIO
from pathlib import Path
import re
from typing import Iterable, List, Tuple

from music21 import chord as m21_chord
from music21 import duration, instrument, meter, note, stream, tempo
from music21.metadata import Metadata

from .schemas import Composition, MelodyNote


CHORD_ROOTS = {
    "C",
    "C#",
    "Db",
    "D",
    "D#",
    "Eb",
    "E",
    "F",
    "F#",
    "Gb",
    "G",
    "G#",
    "Ab",
    "A",
    "A#",
    "Bb",
    "B",
}

MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}
ROOT_TO_PC = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}


def normalize_chord_symbol(symbol: str) -> str:
    cleaned = symbol.strip().replace("min", "m").replace(" ", "")
    match = re.match(r"^([A-G](?:#|b)?)(.*)$", cleaned)
    if not match:
        raise ValueError(f"Invalid chord symbol: {symbol}")
    root, quality = match.groups()
    if root not in CHORD_ROOTS:
        raise ValueError(f"Invalid chord root: {symbol}")
    return f"{root}{quality}"


def _beats_per_bar(time_signature: str) -> float:
    numerator, denominator = time_signature.split("/")
    return int(numerator) * (4 / int(denominator))


def _raw_chord_event_beats(section: object) -> float:
    return sum(
        max(0.0, float(event.duration_beats))
        for event in getattr(section, "chord_events", [])
        if getattr(event, "chord", "").strip()
    )


def _effective_section_beats(section: object, beats_per_bar: float) -> float:
    bar_capacity = float(section.bars) * beats_per_bar
    event_capacity = _raw_chord_event_beats(section)
    return max(bar_capacity, event_capacity)


def _effective_section_bars(section: object, beats_per_bar: float) -> int:
    return max(1, math.ceil(_effective_section_beats(section, beats_per_bar) / beats_per_bar))


def _lyric_chord_tags(section: object) -> list[str]:
    tags: list[str] = []
    for line in getattr(section, "lyric_chord_lines", []):
        tags.extend(tag.strip() for tag in re.findall(r"\[([^\]]+)\]", line) if tag.strip())
    return tags


def _chord_spans(section: object, beats_per_bar: float) -> list[tuple[str, float, float]]:
    target_beats = _effective_section_beats(section, beats_per_bar)
    source: list[tuple[str, float]] = [
        (event.chord.strip(), max(0.0, float(event.duration_beats)))
        for event in getattr(section, "chord_events", [])
        if event.chord.strip() and float(event.duration_beats) > 0
    ]
    lyric_tags = _lyric_chord_tags(section)
    if len(lyric_tags) > len(source):
        duration_beats = target_beats / len(lyric_tags)
        source = [(tag, duration_beats) for tag in lyric_tags]
    if not source:
        source = [(chord.strip(), beats_per_bar) for chord in section.chords if chord.strip()]
    if not source:
        source = [("C", target_beats)]

    spans: list[tuple[str, float, float]] = []
    cursor = 0.0
    while cursor < target_beats - 1e-6:
        for symbol, duration_beats in source:
            if cursor >= target_beats - 1e-6:
                break
            actual_duration = min(duration_beats, target_beats - cursor)
            if actual_duration <= 0:
                continue
            spans.append((symbol, cursor, actual_duration))
            cursor += actual_duration
    return spans


def _key_pitch_classes(key_name: str) -> set[int]:
    parts = key_name.strip().replace("minor", "minor").replace("major", "major").split()
    if not parts:
        return set()
    root = parts[0]
    if root not in ROOT_TO_PC:
        return set()
    mode = "minor" if any(part.lower().startswith("min") for part in parts[1:]) else "major"
    scale = MINOR_SCALE if mode == "minor" else MAJOR_SCALE
    return {(ROOT_TO_PC[root] + interval) % 12 for interval in scale}


def validate_composition(composition: Composition) -> List[str]:
    warnings: List[str] = []
    key_pitch_classes = _key_pitch_classes(composition.key)
    beats_per_bar = _beats_per_bar(composition.time_signature)
    for section in composition.sections:
        if not section.chord_events and len(section.chords) != section.bars:
            warnings.append(
                f"{section.name} has {len(section.chords)} chords for {section.bars} bars; chords will be looped."
            )
        event_beats = _raw_chord_event_beats(section)
        if section.chord_events and event_beats > section.bars * beats_per_bar:
            warnings.append(
                f"{section.name} chord events cover {event_beats:g} beats; preview/export will use the longer timed progression."
            )
        chord_symbols = [symbol for symbol, _, _ in _chord_spans(section, beats_per_bar)]
        for chord_symbol in chord_symbols:
            try:
                normalize_chord_symbol(chord_symbol)
            except ValueError as exc:
                warnings.append(str(exc))
        for melody_note in section.melody:
            if melody_note.pitch != "rest":
                try:
                    parsed_note = note.Note(melody_note.pitch)
                    if key_pitch_classes and parsed_note.pitch.pitchClass not in key_pitch_classes:
                        warnings.append(
                            f"{section.name} melody pitch {melody_note.pitch} is outside {composition.key}; keep if intentional."
                        )
                except Exception:
                    warnings.append(f"Invalid melody pitch: {melody_note.pitch}")
        melody_beats = sum(float(melody_note.duration_beats) for melody_note in section.melody)
        target_beats = section.bars * beats_per_bar
        if melody_beats > target_beats:
            warnings.append(
                f"{section.name} melody is {melody_beats:g} beats but section capacity is {target_beats:g}; export may overflow."
            )
        elif melody_beats < target_beats * 0.75:
            warnings.append(
                f"{section.name} melody covers {melody_beats:g}/{target_beats:g} beats; consider adding rests or notes."
            )
    return warnings


def _nearest_pitch_class(source: int, allowed: set[int]) -> int:
    if source in allowed:
        return source
    best = source
    best_distance = 99
    best_shift = 99
    for candidate in sorted(allowed):
        upward = (candidate - source) % 12
        downward = -((source - candidate) % 12)
        shift = upward if upward <= abs(downward) else downward
        distance = abs(shift)
        if distance < best_distance or (distance == best_distance and shift > best_shift):
            best = candidate
            best_distance = distance
            best_shift = shift
    return best


def _snap_note_to_key(pitch_name: str, key_pitch_classes: set[int]) -> str:
    if pitch_name == "rest" or not key_pitch_classes:
        return pitch_name
    parsed = note.Note(pitch_name)
    target_pc = _nearest_pitch_class(parsed.pitch.pitchClass, key_pitch_classes)
    shift = (target_pc - parsed.pitch.pitchClass) % 12
    if shift > 6:
        shift -= 12
    return parsed.transpose(shift).nameWithOctave


def _repair_melody_pitch(pitch_name: str, fallback_octave: int = 4) -> str:
    cleaned = pitch_name.strip()
    if cleaned.lower() == "rest":
        return "rest"
    try:
        return note.Note(cleaned).nameWithOctave
    except Exception:
        pass

    match = re.match(r"^([A-G](?:#|b)?)(?:m|min|maj|dim|aug|sus|add|\d|#|b)*(?:(-?\d))?$", cleaned, re.IGNORECASE)
    if match:
        root, octave = match.groups()
        return f"{root}{octave or fallback_octave}"
    return "rest"


def optimize_generated_composition(composition: Composition) -> Composition:
    optimized = composition.model_copy(deep=True)
    key_pitch_classes = _key_pitch_classes(optimized.key)
    beats_per_bar = _beats_per_bar(optimized.time_signature)
    for section in optimized.sections:
        event_beats = _raw_chord_event_beats(section)
        if event_beats > section.bars * beats_per_bar:
            section.bars = min(16, max(section.bars, math.ceil(event_beats / beats_per_bar)))
        if section.chord_events:
            section.chords = [symbol for symbol, _, _ in _chord_spans(section, beats_per_bar)][:32]
        elif section.chords:
            section.chords = [section.chords[index % len(section.chords)] for index in range(section.bars)]
        for melody_note in section.melody:
            if melody_note.pitch != "rest":
                melody_note.pitch = _repair_melody_pitch(melody_note.pitch)
            if melody_note.pitch != "rest":
                try:
                    melody_note.pitch = _snap_note_to_key(melody_note.pitch, key_pitch_classes)
                except Exception:
                    melody_note.pitch = "rest"
        target_beats = section.bars * beats_per_bar
        trimmed_melody = []
        used_beats = 0.0
        for melody_note in section.melody:
            if used_beats >= target_beats:
                break
            note_duration = float(melody_note.duration_beats)
            if used_beats + note_duration > target_beats:
                note_duration = target_beats - used_beats
                if note_duration not in {0.25, 0.5, 1, 1.5, 2, 3, 4}:
                    break
                melody_note = melody_note.model_copy(update={"duration_beats": note_duration})
            trimmed_melody.append(melody_note)
            used_beats += note_duration
        section.melody = trimmed_melody or [MelodyNote(pitch="rest", duration_beats=target_beats if target_beats in {0.25, 0.5, 1, 1.5, 2, 3, 4} else 4)]
        used_beats = sum(float(melody_note.duration_beats) for melody_note in section.melody)
        remaining = target_beats - used_beats
        while remaining > 0:
            rest_duration = 4 if remaining >= 4 else 2 if remaining >= 2 else 1 if remaining >= 1 else 0.5
            if rest_duration > remaining:
                rest_duration = remaining
            section.melody.append(MelodyNote(pitch="rest", duration_beats=rest_duration))
            remaining -= rest_duration
    return optimized


def composition_validation_report(composition: Composition) -> dict:
    return {
        "warnings": validate_composition(composition),
        "total_sections": len(composition.sections),
        "total_bars": sum(section.bars for section in composition.sections),
        "total_chords": sum(len(section.chords) for section in composition.sections),
        "total_melody_notes": sum(len(section.melody) for section in composition.sections),
        "lyric_lines": len(composition.lyrics),
    }


def _chord_pitches(symbol: str) -> List[str]:
    normalized = normalize_chord_symbol(symbol)
    root_match = re.match(r"^([A-G](?:#|b)?)(.*)$", normalized)
    if not root_match:
        return ["C3", "E3", "G3"]
    root, quality = root_match.groups()
    quality_lower = quality.lower()
    intervals = [0, 4, 7]
    if quality_lower.startswith("m") and not quality_lower.startswith("maj"):
        intervals = [0, 3, 7]
    if "sus2" in quality_lower:
        intervals = [0, 2, 7]
    if "sus4" in quality_lower or quality_lower.startswith("sus"):
        intervals = [0, 5, 7]
    if "dim" in quality_lower:
        intervals = [0, 3, 6]
    if "aug" in quality_lower:
        intervals = [0, 4, 8]
    if "b5" in quality_lower and intervals[-1] == 7:
        intervals[-1] = 6
    if "#5" in quality_lower and intervals[-1] == 7:
        intervals[-1] = 8
    if "maj7" in quality_lower or "ma7" in quality_lower:
        intervals.append(11)
    elif "7" in quality_lower:
        intervals.append(10)
    elif "6" in quality_lower:
        intervals.append(9)
    if "9" in quality_lower:
        intervals.append(14)
    if "11" in quality_lower:
        intervals.append(17)
    if "13" in quality_lower:
        intervals.append(21)
    intervals = list(dict.fromkeys(intervals))[:6]
    base = note.Note(f"{root}3")
    return [base.transpose(interval).nameWithOctave for interval in intervals]


def _chord_root_pitch(symbol: str, octave: int = 2) -> str:
    normalized = normalize_chord_symbol(symbol)
    root_match = re.match(r"^([A-G](?:#|b)?)(.*)$", normalized)
    if not root_match:
        return f"C{octave}"
    return f"{root_match.group(1)}{octave}"


def _power_chord_pitches(symbol: str) -> List[str]:
    normalized = normalize_chord_symbol(symbol)
    root_match = re.match(r"^([A-G](?:#|b)?)(.*)$", normalized)
    if not root_match:
        return ["C3", "G3", "C4"]
    root = root_match.group(1)
    base = note.Note(f"{root}3")
    return [base.nameWithOctave, base.transpose(7).nameWithOctave, base.transpose(12).nameWithOctave]


def _melody_elements(notes: Iterable[MelodyNote]) -> List[note.NotRest]:
    elements: List[note.NotRest] = []
    for melody_note in notes:
        if melody_note.pitch == "rest":
            item = note.Rest()
        else:
            item = note.Note(melody_note.pitch)
            item.volume.velocity = 100
        item.duration = duration.Duration(float(melody_note.duration_beats))
        elements.append(item)
    return elements


TICKS_PER_BEAT = 480
DRUM_CHANNEL = 9

STYLE_PROGRAMS = {
    "rock": {"harmony": 30, "bass": 33, "melody": 29, "pad": 30},
    "edm": {"harmony": 90, "bass": 38, "melody": 81, "pad": 88},
    "jazz": {"harmony": 0, "bass": 32, "melody": 65, "pad": 4},
    "folk": {"harmony": 25, "bass": 32, "melody": 22, "pad": 25},
    "cinematic": {"harmony": 48, "bass": 43, "melody": 50, "pad": 89},
    "r&b": {"harmony": 4, "bass": 38, "melody": 54, "pad": 89},
    "lo-fi": {"harmony": 4, "bass": 33, "melody": 11, "pad": 89},
    "pop": {"harmony": 81, "bass": 38, "melody": 80, "pad": 88},
}

STYLE_VOLUME = {
    "rock": {"harmony": 104, "bass": 106, "melody": 104, "pad": 74},
    "edm": {"harmony": 100, "bass": 112, "melody": 108, "pad": 82},
    "jazz": {"harmony": 92, "bass": 104, "melody": 98, "pad": 74},
    "folk": {"harmony": 96, "bass": 92, "melody": 96, "pad": 70},
    "cinematic": {"harmony": 94, "bass": 108, "melody": 96, "pad": 102},
    "r&b": {"harmony": 92, "bass": 106, "melody": 104, "pad": 88},
    "lo-fi": {"harmony": 90, "bass": 96, "melody": 92, "pad": 82},
    "pop": {"harmony": 96, "bass": 104, "melody": 110, "pad": 84},
}

STYLE_INSTRUMENTS = {
    "rock": {
        "drums": "acoustic_drums",
        "bass": "electric_bass",
        "harmony": "distorted_guitar",
        "melody": "electric_guitar",
    },
    "edm": {
        "drums": "electronic_drums",
        "bass": "synth_bass",
        "harmony": "synth_pad",
        "melody": "piano_lead",
    },
    "jazz": {
        "drums": "brush_drums",
        "bass": "acoustic_bass",
        "harmony": "piano",
        "melody": "piano_lead",
    },
    "folk": {
        "drums": "acoustic_drums",
        "bass": "acoustic_bass",
        "harmony": "acoustic_guitar",
        "melody": "violin_lead",
    },
    "cinematic": {
        "drums": "orchestral_timpani",
        "bass": "acoustic_bass",
        "harmony": "synth_pad",
        "melody": "violin_lead",
    },
    "r&b": {
        "drums": "808_drums",
        "bass": "electric_bass",
        "harmony": "electric_piano",
        "melody": "vocal_lead",
    },
    "lo-fi": {
        "drums": "808_drums",
        "bass": "sub_bass",
        "harmony": "electric_piano",
        "melody": "piano_lead",
    },
    "pop": {
        "drums": "electronic_drums",
        "bass": "electric_bass",
        "harmony": "piano",
        "melody": "piano_lead",
    },
}

INSTRUMENT_MIDI_PROGRAMS = {
    # Harmony
    "piano": 0,            # Acoustic Grand Piano
    "acoustic_piano": 0,   # Acoustic Grand Piano
    "electric_piano": 4,   # Rhodes Electric Piano
    "acoustic_guitar": 24, # Acoustic Guitar (nylon)
    "electric_guitar": 27, # Electric Guitar (clean)
    "distorted_guitar": 30, # Distortion Guitar
    "synth_pad": 89,       # Pad 2 (warm)
    "church_organ": 19,    # Church Organ
    "flute": 73,           # Flute (harmony)
    
    # Bass
    "acoustic_bass": 32,   # Acoustic Bass
    "electric_bass": 33,   # Electric Bass (finger)
    "slap_bass": 36,       # Slap Bass 1
    "synth_bass": 38,      # Synth Bass 1
    "sub_bass": 39,        # Synth Bass 2
    
    # Melody
    "piano_lead": 0,       # Acoustic Grand Piano
    "sine_lead": 80,       # Lead 1 (square/sine)
    "saw_lead": 81,        # Lead 2 (saw)
    "square_lead": 80,     # Lead 1 (square)
    "vocal_lead": 54,      # Voice Oohs
    "flute_lead": 73,      # Flute
    "violin_lead": 40,     # Violin
}

DRUM_MIDI_PROGRAMS = {
    "acoustic_drums": 0,      # Standard Kit (Channel 10)
    "electric_drums": 24,     # Electronic Kit
    "808_drums": 25,          # TR-808 Kit
    "brush_drums": 40,        # Brush Kit
    "orchestral_timpani": 48, # Orchestral Kit
    "tabla_drums": 32,        # Jazz Kit (closest to tabla timbre in GM)
    "bongo_drums": 48,        # Orchestral Kit (used for bongo feel)
}


def _varlen(value: int) -> bytes:
    value = max(0, int(value))
    buffer = value & 0x7F
    value >>= 7
    while value:
        buffer <<= 8
        buffer |= (value & 0x7F) | 0x80
        value >>= 7
    output = bytearray()
    while True:
        output.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(output)


def _track_chunk(events: list[tuple[int, int, bytes]]) -> bytes:
    data = bytearray()
    previous_tick = 0
    for tick, _order, payload in sorted(events, key=lambda item: (item[0], item[1])):
        data.extend(_varlen(max(0, tick - previous_tick)))
        data.extend(payload)
        previous_tick = tick
    data.extend(_varlen(0))
    data.extend(b"\xff\x2f\x00")
    return b"MTrk" + struct.pack(">I", len(data)) + bytes(data)


def _meta_track(composition: Composition) -> bytes:
    tempo_microseconds = int(60_000_000 / max(1, composition.tempo_bpm))
    numerator, denominator = composition.time_signature.split("/")
    denominator_power = int(math.log2(max(1, int(denominator))))
    events = [
        (0, 0, b"\xff\x03" + bytes([len(composition.title[:48])]) + composition.title[:48].encode("ascii", "ignore")),
        (0, 1, b"\xff\x51\x03" + tempo_microseconds.to_bytes(3, "big")),
        (0, 2, b"\xff\x58\x04" + bytes([int(numerator), denominator_power, 24, 8])),
    ]
    return _track_chunk(events)


def _midi_bytes_from_tracks(composition: Composition, tracks: list[bytes]) -> bytes:
    chunks = [_meta_track(composition), *tracks]
    header = b"MThd" + struct.pack(">IHHH", 6, 1, len(chunks), TICKS_PER_BEAT)
    return header + b"".join(chunks)


def _pitch_to_midi(pitch_name: str) -> int | None:
    if pitch_name == "rest":
        return None
    try:
        return int(note.Note(pitch_name).pitch.midi)
    except Exception:
        return None


def _pitch_name_at_octave(pitch_name: str, octave: int) -> str:
    parsed = note.Note(pitch_name)
    parsed.octave = octave
    return parsed.nameWithOctave


def _chord_midis(symbol: str, octave: int = 3, power: bool = False) -> list[int]:
    pitch_names = _power_chord_pitches(symbol) if power else _chord_pitches(symbol)
    notes = []
    for pitch_name in pitch_names:
        try:
            notes.append(_pitch_to_midi(_pitch_name_at_octave(pitch_name, octave)))
        except Exception:
            continue
    return [item for item in notes if item is not None]


def _root_midi(symbol: str, octave: int = 2) -> int:
    return _pitch_to_midi(_chord_root_pitch(symbol, octave=octave)) or 36


def _add_note(
    events: list[tuple[int, int, bytes]],
    *,
    channel: int,
    start_tick: int,
    duration_ticks: int,
    pitch: int | None,
    velocity: int,
    order: int = 20,
) -> None:
    if pitch is None or duration_ticks <= 0:
        return
    pitch = max(0, min(127, int(pitch)))
    velocity = max(1, min(127, int(velocity)))
    start_tick = max(0, int(start_tick))
    duration_ticks = max(1, int(duration_ticks))
    events.append((start_tick, order, bytes([0x90 + channel, pitch, velocity])))
    events.append((start_tick + duration_ticks, order + 1, bytes([0x80 + channel, pitch, 0])))


def _add_chord(
    events: list[tuple[int, int, bytes]],
    *,
    channel: int,
    start_tick: int,
    duration_ticks: int,
    pitches: list[int],
    velocity: int,
    strum_ticks: int = 0,
) -> None:
    for index, pitch in enumerate(pitches):
        _add_note(
            events,
            channel=channel,
            start_tick=start_tick + index * strum_ticks,
            duration_ticks=max(1, duration_ticks - index * strum_ticks),
            pitch=pitch,
            velocity=max(1, velocity - index * 3),
            order=20 + index,
        )


def _control_events(channel: int, *, program: int, volume: int, pan: int) -> list[tuple[int, int, bytes]]:
    return [
        (0, 0, bytes([0xC0 + channel, max(0, min(127, program))])),
        (0, 1, bytes([0xB0 + channel, 7, max(0, min(127, volume))])),
        (0, 2, bytes([0xB0 + channel, 10, max(0, min(127, pan))])),
        (0, 3, bytes([0xB0 + channel, 91, 24])),
    ]


def _drum(events: list[tuple[int, int, bytes]], start_tick: int, beats: float, drum_note: int, velocity: int) -> None:
    _add_note(
        events,
        channel=DRUM_CHANNEL,
        start_tick=start_tick,
        duration_ticks=max(24, int(beats * TICKS_PER_BEAT)),
        pitch=drum_note,
        velocity=velocity,
        order=30,
    )


def _section_offsets(composition: Composition) -> list[tuple[int, object]]:
    offsets = []
    cursor = 0
    beats_per_bar = _beats_per_bar(composition.time_signature)
    for section in composition.sections:
        offsets.append((cursor, section))
        cursor += int(_effective_section_beats(section, beats_per_bar) * TICKS_PER_BEAT)
    return offsets


def _style_programs(family: str) -> dict[str, int]:
    return STYLE_PROGRAMS.get(family, STYLE_PROGRAMS["lo-fi"])


def _style_volumes(family: str) -> dict[str, int]:
    return STYLE_VOLUME.get(family, STYLE_VOLUME["lo-fi"])


def _arrange_harmony_and_bass(
    composition: Composition,
    *,
    family: str,
    harmony_events: list[tuple[int, int, bytes]],
    bass_events: list[tuple[int, int, bytes]],
    pad_events: list[tuple[int, int, bytes]],
) -> None:
    beats_per_bar = _beats_per_bar(composition.time_signature)
    bar_ticks = int(beats_per_bar * TICKS_PER_BEAT)
    half = TICKS_PER_BEAT // 2
    quarter = TICKS_PER_BEAT

    for section_tick, section in _section_offsets(composition):
        for symbol, start_beats, duration_beats in _chord_spans(section, beats_per_bar):
            bar_tick = section_tick + int(start_beats * TICKS_PER_BEAT)
            duration_ticks = max(1, int(duration_beats * TICKS_PER_BEAT))
            beat_count = max(1, math.ceil(duration_beats))
            chord_full = _chord_midis(symbol, octave=3, power=False)
            chord_high = _chord_midis(symbol, octave=4, power=False)
            chord_power = _chord_midis(symbol, octave=3, power=True)
            root = _root_midi(symbol, octave=2)
            fifth = root + 7
            octave = root + 12

            if family == "rock":
                for step in range(max(1, int(duration_beats * 2))):
                    tick = bar_tick + step * half
                    velocity = 106 if step in {0, 4} else 84
                    _add_chord(harmony_events, channel=0, start_tick=tick, duration_ticks=int(half * 0.82), pitches=chord_power, velocity=velocity, strum_ticks=8)
                    _add_note(bass_events, channel=1, start_tick=tick, duration_ticks=int(half * 0.88), pitch=root if step % 4 else octave, velocity=102)
                continue

            if family == "edm":
                for beat in range(beat_count):
                    _add_note(bass_events, channel=1, start_tick=bar_tick + beat * quarter, duration_ticks=int(quarter * 0.72), pitch=root if beat % 2 == 0 else octave, velocity=112)
                for beat in (0, 2):
                    if beat < duration_beats:
                        _add_chord(harmony_events, channel=0, start_tick=bar_tick + beat * quarter, duration_ticks=int(quarter * 0.72), pitches=chord_high[:4], velocity=94, strum_ticks=0)
                _add_chord(pad_events, channel=3, start_tick=bar_tick, duration_ticks=int(duration_ticks * 0.96), pitches=chord_full[:4], velocity=54)
                continue

            if family == "jazz":
                walking = [root, root + 4, fifth, root + 10]
                if "m" in symbol.lower() and "maj" not in symbol.lower():
                    walking[1] = root + 3
                for beat, pitch in enumerate(walking[:beat_count]):
                    _add_note(bass_events, channel=1, start_tick=bar_tick + beat * quarter, duration_ticks=int(quarter * 0.88), pitch=pitch, velocity=92)
                for beat in (0, 2):
                    if beat < duration_beats:
                        _add_chord(harmony_events, channel=0, start_tick=bar_tick + beat * quarter + 28, duration_ticks=int(quarter * 1.32), pitches=chord_full[:5], velocity=78, strum_ticks=3)
                continue

            if family == "folk":
                arpeggio = (chord_full + [root + 12, fifth + 12])[:6] or [root, fifth, octave]
                for step in range(max(1, int(duration_beats * 2))):
                    pitch = arpeggio[step % len(arpeggio)]
                    _add_note(harmony_events, channel=0, start_tick=bar_tick + step * half, duration_ticks=int(half * 0.74), pitch=pitch, velocity=82 if step % 2 == 0 else 68)
                _add_note(bass_events, channel=1, start_tick=bar_tick, duration_ticks=int(quarter * 0.9), pitch=root, velocity=76)
                if duration_beats > 2:
                    _add_note(bass_events, channel=1, start_tick=bar_tick + 2 * quarter, duration_ticks=int(quarter * 0.9), pitch=fifth, velocity=72)
                continue

            if family == "cinematic":
                _add_chord(harmony_events, channel=0, start_tick=bar_tick, duration_ticks=int(duration_ticks * 0.98), pitches=chord_full[:5], velocity=82, strum_ticks=18)
                _add_chord(pad_events, channel=3, start_tick=bar_tick, duration_ticks=int(duration_ticks * 0.98), pitches=[pitch + 12 for pitch in chord_full[:4]], velocity=70, strum_ticks=22)
                for beat in (0, 2):
                    if beat < duration_beats:
                        _add_note(bass_events, channel=1, start_tick=bar_tick + beat * quarter, duration_ticks=int(quarter * 1.75), pitch=root - 12, velocity=104)
                continue

            # Lo-fi and R&B lean on softer electric keys plus more space.
            strum = 18 if family == "lo-fi" else 6
            _add_chord(harmony_events, channel=0, start_tick=bar_tick + 18, duration_ticks=int(duration_ticks * 0.82), pitches=chord_full[:5], velocity=76 if family == "lo-fi" else 86, strum_ticks=strum)
            if family == "r&b":
                for beat, pitch in [(0, root), (1.5, octave), (2.5, fifth)]:
                    if beat < duration_beats:
                        _add_note(bass_events, channel=1, start_tick=bar_tick + int(beat * quarter), duration_ticks=int(quarter * 0.82), pitch=pitch, velocity=98)
            else:
                _add_note(bass_events, channel=1, start_tick=bar_tick, duration_ticks=int(quarter * 1.2), pitch=root, velocity=84)
                if duration_beats > 2.5:
                    _add_note(bass_events, channel=1, start_tick=bar_tick + int(2.5 * quarter), duration_ticks=int(quarter * 0.72), pitch=fifth, velocity=74)


def _arrange_drums(composition: Composition, *, family: str, drum_events: list[tuple[int, int, bytes]]) -> None:
    beats_per_bar = _beats_per_bar(composition.time_signature)
    bar_ticks = int(beats_per_bar * TICKS_PER_BEAT)
    quarter = TICKS_PER_BEAT
    eighth = TICKS_PER_BEAT // 2

    for section_tick, section in _section_offsets(composition):
        # Read the mixer drum instrument once — checked before style-family patterns
        drum_inst = (composition.mixer.drums.instrument if composition.mixer and composition.mixer.drums else "")

        for bar_index in range(_effective_section_bars(section, beats_per_bar)):
            bar_tick = section_tick + bar_index * bar_ticks
            if bar_index == 0:
                _drum(drum_events, bar_tick, 0.5, 49 if family in {"rock", "edm"} else 51, 84)

            # --- Custom drum instruments take priority over style-family patterns ---

            if drum_inst == "tabla_drums":
                # Teentaal-inspired pattern using GM percussion notes:
                # 41=low floor tom (bayan bass), 45=low tom (bayan open),
                # 47=mid tom (tun), 77=high wood block (na/tip), 76=low wood block (te/slap)
                _drum(drum_events, bar_tick, 0.15, 41, 104)                       # Dha  (bass stroke)
                _drum(drum_events, bar_tick + int(0.5 * eighth), 0.08, 77, 72)    # ge   (finger)
                _drum(drum_events, bar_tick + eighth, 0.10, 76, 88)               # Ti
                _drum(drum_events, bar_tick + int(1.5 * eighth), 0.08, 77, 70)    # Ra
                _drum(drum_events, bar_tick + 2 * eighth, 0.14, 45, 96)           # Dhin (bayan open)
                _drum(drum_events, bar_tick + int(2.5 * eighth), 0.08, 47, 68)    # tun  (resonant)
                _drum(drum_events, bar_tick + 3 * eighth, 0.10, 76, 80)           # Ta
                if beats_per_bar > 4:
                    _drum(drum_events, bar_tick + int(3.5 * eighth), 0.08, 77, 64)  # Na
                continue

            if drum_inst == "bongo_drums":
                # 60=hi bongo, 61=lo bongo, 62=mute hi conga, 63=open hi conga
                for step in range(int(beats_per_bar * 2)):
                    pitch = 60 if step % 2 == 0 else 61
                    velocity = 96 if step % 4 == 0 else 72
                    _drum(drum_events, bar_tick + step * eighth, 0.09, pitch, velocity)
                _drum(drum_events, bar_tick + 1 * quarter, 0.12, 63, 100)   # open hi conga accent
                _drum(drum_events, bar_tick + 3 * quarter, 0.12, 62, 88)    # mute hi conga accent
                continue

            # --- Style-family patterns (only reached when no custom instrument selected) ---

            if family == "rock":
                for beat in (0, 2):
                    _drum(drum_events, bar_tick + beat * quarter, 0.18, 36, 118)
                for beat in (1, 3):
                    if beat < beats_per_bar:
                        _drum(drum_events, bar_tick + beat * quarter, 0.12, 38, 112)
                for step in range(int(beats_per_bar * 2)):
                    _drum(drum_events, bar_tick + step * eighth, 0.06, 42, 70 if step % 2 else 86)
                continue

            if family == "edm":
                for beat in range(int(beats_per_bar)):
                    _drum(drum_events, bar_tick + beat * quarter, 0.16, 36, 124)
                for beat in (1, 3):
                    if beat < beats_per_bar:
                        _drum(drum_events, bar_tick + beat * quarter, 0.1, 39, 104)
                for step in range(int(beats_per_bar * 2)):
                    _drum(drum_events, bar_tick + int((step + 0.5) * eighth), 0.04, 42 if step % 2 else 46, 68)
                continue

            if family == "jazz":
                for beat in range(int(beats_per_bar)):
                    _drum(drum_events, bar_tick + beat * quarter, 0.08, 51, 66 if beat % 2 else 78)
                _drum(drum_events, bar_tick + 2 * quarter, 0.12, 38, 52)
                continue

            if family == "folk":
                _drum(drum_events, bar_tick, 0.14, 36, 76)
                if beats_per_bar > 2:
                    _drum(drum_events, bar_tick + 2 * quarter, 0.12, 38, 64)
                for step in range(int(beats_per_bar)):
                    _drum(drum_events, bar_tick + step * quarter, 0.04, 42, 42)
                continue

            if family == "cinematic":
                _drum(drum_events, bar_tick, 0.24, 41, 98)
                if bar_index % 2 == 1:
                    _drum(drum_events, bar_tick + 3 * quarter, 0.18, 45, 86)
                continue

            if family == "r&b":
                _drum(drum_events, bar_tick, 0.14, 36, 96)
                _drum(drum_events, bar_tick + int(1.75 * quarter), 0.1, 38, 74)
                _drum(drum_events, bar_tick + 2 * quarter, 0.14, 36, 82)
                _drum(drum_events, bar_tick + 3 * quarter, 0.1, 38, 92)
                for step in range(int(beats_per_bar * 2)):
                    _drum(drum_events, bar_tick + step * eighth, 0.035, 42, 48 if step % 2 else 58)
                continue

            # Default fallback
            _drum(drum_events, bar_tick, 0.14, 36, 78)
            if beats_per_bar > 2:
                _drum(drum_events, bar_tick + 2 * quarter, 0.1, 38, 68)
            _drum(drum_events, bar_tick + int(2.75 * quarter), 0.08, 36, 62)
            for step in range(int(beats_per_bar * 2)):
                if step % 2 == 1:
                    _drum(drum_events, bar_tick + step * eighth, 0.035, 42, 38)


def _arrange_melody(composition: Composition, *, family: str, melody_events: list[tuple[int, int, bytes]]) -> None:
    channel = 2
    beat_ticks = TICKS_PER_BEAT
    for section_tick, section in _section_offsets(composition):
        cursor = section_tick
        for melody_note in section.melody:
            duration_ticks = int(float(melody_note.duration_beats) * beat_ticks)
            pitch = _pitch_to_midi(melody_note.pitch)
            if pitch is not None:
                if family in {"rock", "edm"}:
                    pitch += 12 if pitch < 72 else 0
                velocity = 102 if family in {"rock", "edm", "r&b"} else 88
                _add_note(
                    melody_events,
                    channel=channel,
                    start_tick=cursor,
                    duration_ticks=int(duration_ticks * 0.9),
                    pitch=pitch,
                    velocity=velocity,
                )
            cursor += duration_ticks


def _composition_to_arranged_midi_bytes(composition: Composition) -> bytes:
    family = _style_family(composition)
    programs = _style_programs(family)
    volumes = _style_volumes(family)

    def parse_pan(pan_str: str) -> int:
        if not pan_str:
            return 64
        pan_str = str(pan_str).strip().upper()
        if pan_str == "C":
            return 64
        match = re.match(r"^([LR])(\d+)$", pan_str)
        if match:
            direction, val_str = match.groups()
            val = int(val_str)
            offset = int(val * 6.3)
            if direction == "L":
                return max(0, 64 - offset)
            else:
                return min(127, 64 + offset)
        return 64

    m = composition.mixer
    if m:
        v_harm = int((m.harmony.volume if m.harmony else 80) * 1.27)
        p_harm = parse_pan(m.harmony.pan if m.harmony else "C")
        prog_harm = INSTRUMENT_MIDI_PROGRAMS.get(m.harmony.instrument if m.harmony else "piano", 0)
        
        v_bass = int((m.bass.volume if m.bass else 80) * 1.27)
        p_bass = parse_pan(m.bass.pan if m.bass else "L8")
        prog_bass = INSTRUMENT_MIDI_PROGRAMS.get(m.bass.instrument if m.bass else "electric_bass", 33)
        
        v_mel = int((m.melody.volume if m.melody else 80) * 1.27)
        p_mel = parse_pan(m.melody.pan if m.melody else "C")
        prog_mel = INSTRUMENT_MIDI_PROGRAMS.get(m.melody.instrument if m.melody else "piano_lead", 0)
        
        v_pad = int((m.harmony.volume if m.harmony else 80) * 0.9 * 1.27)
        p_pad = parse_pan(m.harmony.pan if m.harmony else "C")
        prog_pad = 89
        
        v_drum = int((m.drums.volume if m.drums else 80) * 1.27)
        p_drum = parse_pan(m.drums.pan if m.drums else "C")
        prog_drum = DRUM_MIDI_PROGRAMS.get(m.drums.instrument if m.drums else "acoustic_drums", 0)
    else:
        v_harm = volumes["harmony"]
        p_harm = 48
        prog_harm = programs["harmony"]
        
        v_bass = volumes["bass"]
        p_bass = 42
        prog_bass = programs["bass"]
        
        v_mel = volumes["melody"]
        p_mel = 74
        prog_mel = programs["melody"]
        
        v_pad = volumes["pad"]
        p_pad = 64
        prog_pad = programs["pad"]
        
        v_drum = 110
        p_drum = 64
        prog_drum = 0

    harmony_events = _control_events(0, program=prog_harm, volume=v_harm, pan=p_harm)
    bass_events = _control_events(1, program=prog_bass, volume=v_bass, pan=p_bass)
    melody_events = _control_events(2, program=prog_mel, volume=v_mel, pan=p_mel)
    pad_events = _control_events(3, program=prog_pad, volume=v_pad, pan=p_pad)
    
    # Send program change on Channel 10 for custom drum kits
    drum_events = [
        (0, 0, bytes([0xB0 + DRUM_CHANNEL, 7, v_drum])), 
        (0, 1, bytes([0xB0 + DRUM_CHANNEL, 10, p_drum])),
        (0, 2, bytes([0xC0 + DRUM_CHANNEL, prog_drum]))
    ]

    _arrange_harmony_and_bass(
        composition,
        family=family,
        harmony_events=harmony_events,
        bass_events=bass_events,
        pad_events=pad_events,
    )
    _arrange_drums(composition, family=family, drum_events=drum_events)
    _arrange_melody(composition, family=family, melody_events=melody_events)

    tracks = [
        _track_chunk(harmony_events),
        _track_chunk(bass_events),
        _track_chunk(melody_events),
        _track_chunk(pad_events),
        _track_chunk(drum_events),
    ]
    return _midi_bytes_from_tracks(composition, tracks)


def build_score(composition: Composition) -> stream.Score:
    score = stream.Score()
    score.metadata = Metadata()
    score.metadata.title = composition.title
    score.insert(0, tempo.MetronomeMark(number=composition.tempo_bpm))
    score.insert(0, meter.TimeSignature(composition.time_signature))

    family = _style_family(composition)
    m = composition.mixer
    chord_part = stream.Part(id="chords")
    chord_inst = m.harmony.instrument if (m and m.harmony) else "piano"
    if chord_inst == "electric_piano":
        chord_part.insert(0, instrument.ElectricPiano())
    elif chord_inst == "acoustic_guitar":
        chord_part.insert(0, instrument.AcousticGuitar())
    elif chord_inst in {"electric_guitar", "distorted_guitar"}:
        chord_part.insert(0, instrument.ElectricGuitar())
    elif chord_inst == "church_organ":
        chord_part.insert(0, instrument.ChurchOrgan())
    elif chord_inst == "flute":
        chord_part.insert(0, instrument.Flute())
    else:
        chord_part.insert(0, instrument.Piano())

    bass_part = stream.Part(id="bass")
    bass_inst = m.bass.instrument if (m and m.bass) else "electric_bass"
    if bass_inst == "acoustic_bass":
        bass_part.insert(0, instrument.AcousticBass())
    else:
        bass_part.insert(0, instrument.ElectricBass())

    melody_part = stream.Part(id="melody")
    melody_inst = m.melody.instrument if (m and m.melody) else "piano_lead"
    if melody_inst == "flute_lead":
        melody_part.insert(0, instrument.Flute())
    elif melody_inst == "violin_lead":
        melody_part.insert(0, instrument.Violin())
    else:
        melody_part.insert(0, instrument.Vocalist())

    drum_part = stream.Part(id="drums")
    drum_part.insert(0, instrument.Percussion())

    beats_per_bar = _beats_per_bar(composition.time_signature)
    for section in composition.sections:
        for event_index, (symbol, _start_beats, duration_beats) in enumerate(_chord_spans(section, beats_per_bar)):
            chord_obj = m21_chord.Chord(_power_chord_pitches(symbol) if family == "rock" else _chord_pitches(symbol))
            chord_obj.duration = duration.Duration(duration_beats)
            chord_obj.volume.velocity = 76
            chord_obj.addLyric(section.name if event_index == 0 else "")
            chord_part.append(chord_obj)

            bass_note = note.Note(_chord_root_pitch(symbol))
            bass_note.duration = duration.Duration(duration_beats)
            bass_note.volume.velocity = 92
            bass_part.append(bass_note)

        for _bar_index in range(_effective_section_bars(section, beats_per_bar)):
            drum_events = [("C2", 1), ("D2", 1), ("C2", 1), ("D2", 1)] if family == "rock" else [("C2", 2), ("D2", 2)]
            for drum_pitch, drum_duration in drum_events:
                drum_note = note.Note(drum_pitch)
                drum_note.duration = duration.Duration(drum_duration)
                drum_note.volume.velocity = 96 if drum_pitch == "C2" else 82
                drum_part.append(drum_note)
        for item in _melody_elements(section.melody):
            melody_part.append(item)

    score.append(chord_part)
    score.append(bass_part)
    score.append(drum_part)
    score.append(melody_part)
    return score


def composition_to_midi_bytes(composition: Composition) -> bytes:
    return _composition_to_arranged_midi_bytes(composition)


def composition_to_soundfont_wav_bytes(
    composition: Composition,
    *,
    fluidsynth_path: str,
    soundfont_path: str,
    sample_rate: int = 44100,
) -> bytes:
    renderer = fluidsynth_path or shutil.which("fluidsynth")
    if not renderer:
        raise RuntimeError("FluidSynth is not configured.")
    renderer_path = Path(renderer)
    soundfont = Path(soundfont_path)
    if not renderer_path.exists() and not shutil.which(renderer):
        raise RuntimeError(f"FluidSynth executable not found: {renderer}")
    if not soundfont.exists():
        raise RuntimeError(f"SoundFont not found: {soundfont}")

    temp_dir = Path(tempfile.mkdtemp())
    try:
        midi_path = temp_dir / "composition.mid"
        wav_path = temp_dir / "composition.wav"
        midi_path.write_bytes(composition_to_midi_bytes(composition))
        subprocess.run(
            [
                str(renderer_path if renderer_path.exists() else renderer),
                "-ni",
                "-g",
                "1.4",
                "-F",
                str(wav_path),
                "-r",
                str(sample_rate),
                str(soundfont),
                str(midi_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return wav_path.read_bytes()
    finally:
        for child in temp_dir.glob("*"):
            child.unlink(missing_ok=True)
        temp_dir.rmdir()


def _pitch_frequency(pitch_name: str) -> float | None:
    if pitch_name == "rest":
        return None
    try:
        return float(note.Note(pitch_name).pitch.frequency)
    except Exception:
        return None


def _triangle_wave(phase: float) -> float:
    return (2 / math.pi) * math.asin(math.sin(phase))


def _saw_wave(phase: float) -> float:
    cycle = (phase / (2 * math.pi)) % 1
    return 2 * cycle - 1


def _mix_tone(
    samples: List[float],
    *,
    frequency: float | None,
    start_seconds: float,
    duration_seconds: float,
    sample_rate: int,
    amplitude: float,
    waveform: str = "sine",
) -> None:
    if not frequency or duration_seconds <= 0:
        return

    start = max(0, int(start_seconds * sample_rate))
    end = min(len(samples), int((start_seconds + duration_seconds) * sample_rate))
    if end <= start:
        return

    attack = min(int(0.025 * sample_rate), max(1, (end - start) // 4))
    release = min(int(0.08 * sample_rate), max(1, (end - start) // 3))
    
    # Custom attack/release envelopes for specific instruments
    if waveform in {"piano", "acoustic_piano", "piano_lead"}:
        attack = min(int(0.01 * sample_rate), max(1, (end - start) // 10))
    elif waveform in {"synth_pad"}:
        attack = min(int(0.35 * sample_rate), max(1, (end - start) // 3))
        release = min(int(0.42 * sample_rate), max(1, (end - start) // 2))
    elif waveform in {"violin_lead", "violin"}:
        attack = min(int(0.15 * sample_rate), max(1, (end - start) // 4))

    two_pi_frequency = 2 * math.pi * frequency
    last_val = 0.0

    for index in range(start, end):
        local = index - start
        remaining = end - index
        envelope = 1.0
        if local < attack:
            envelope = local / attack
        if remaining < release:
            envelope = min(envelope, remaining / release)

        # Decay envelope for pluck/piano instruments
        t_local = local / sample_rate
        if waveform in {"piano", "acoustic_piano", "piano_lead"}:
            envelope *= math.exp(-2.2 * t_local)
        elif waveform in {"electric_piano"}:
            envelope *= math.exp(-1.8 * t_local)
        elif waveform in {"acoustic_guitar"}:
            envelope *= math.exp(-3.5 * t_local)
        elif waveform in {"slap_bass"}:
            envelope *= math.exp(-4.2 * t_local)
        elif waveform in {"electric_bass"}:
            envelope *= math.exp(-2.8 * t_local)
        elif waveform in {"sub_bass"}:
            envelope *= math.exp(-1.5 * t_local)

        t = index / sample_rate
        phase = two_pi_frequency * t
        
        # Pitch vibrato (LFO) for expressive leads
        if waveform in {"violin_lead", "vocal_lead", "vocal"}:
            phase += 0.05 * math.sin(2 * math.pi * 6.0 * t)
        elif waveform in {"flute_lead", "flute"}:
            phase += 0.02 * math.sin(2 * math.pi * 5.5 * t)

        # Instrument Synthesizer Generators
        if waveform == "triangle":
            value = _triangle_wave(phase)
        elif waveform == "saw":
            value = _saw_wave(phase)
        elif waveform == "square":
            value = 1.0 if math.sin(phase) >= 0 else -1.0
        elif waveform == "distorted":
            raw = math.sin(phase) + 0.42 * math.sin(phase * 2) + 0.18 * math.sin(phase * 3)
            value = math.tanh(raw * 2.2)
        elif waveform in {"piano", "acoustic_piano", "piano_lead"}:
            value = (
                math.sin(phase) 
                + 0.45 * math.sin(phase * 2) 
                + 0.22 * math.sin(phase * 3) 
                + 0.1 * math.sin(phase * 4)
            )
        elif waveform == "electric_piano":
            tremolo = 1.0 + 0.18 * math.sin(2 * math.pi * 5.8 * t)
            value = (math.sin(phase) + 0.35 * _triangle_wave(phase * 2)) * tremolo
        elif waveform == "acoustic_guitar":
            value = _triangle_wave(phase) + 0.25 * _saw_wave(phase * 2)
        elif waveform in {"electric_guitar", "distorted_guitar"}:
            raw = _saw_wave(phase) + 0.3 * _saw_wave(phase * 2)
            gain = 4.8 if waveform == "distorted_guitar" else 2.2
            raw_clipped = math.tanh(raw * gain)
            value = 0.35 * raw_clipped + 0.65 * last_val
            last_val = value
        elif waveform == "synth_pad":
            phase_detune1 = 2 * math.pi * (frequency * 1.006) * t
            phase_detune2 = 2 * math.pi * (frequency * 0.994) * t
            value = 0.5 * _triangle_wave(phase) + 0.25 * _saw_wave(phase_detune1) + 0.25 * _saw_wave(phase_detune2)
        elif waveform == "church_organ":
            value = (
                0.5 * math.sin(phase) 
                + 0.35 * math.sin(phase * 2) 
                + 0.25 * math.sin(phase * 3) 
                + 0.4 * math.sin(phase * 0.5)
            )
        elif waveform == "flute_lead":
            breath = _noise_value(index, 42) * 0.12 * envelope
            value = math.sin(phase) + breath
        elif waveform == "flute":
            # Harmony flute: pure sine with gentle breath noise and vibrato
            breath = _noise_value(index, 17) * 0.08 * envelope
            value = math.sin(phase) + 0.10 * math.sin(phase * 2) + breath
        elif waveform == "violin_lead":
            value = _saw_wave(phase) + 0.15 * _triangle_wave(phase * 2)
        elif waveform == "vocal_lead":
            value = _triangle_wave(phase) + 0.4 * math.sin(phase * 2)
        elif waveform == "sub_bass":
            value = math.sin(phase)
        elif waveform == "synth_bass":
            value = 0.7 * (1.0 if math.sin(phase) >= 0 else -1.0) + 0.3 * _saw_wave(phase)
        elif waveform == "electric_bass":
            value = _saw_wave(phase)
        elif waveform == "acoustic_bass":
            value = _triangle_wave(phase)
        elif waveform == "slap_bass":
            click = (1.0 if math.sin(phase * 3) >= 0 else -1.0) * math.exp(-30.0 * t_local) * 0.45
            value = _saw_wave(phase) + click
        else:
            value = math.sin(phase) + 0.22 * math.sin(phase * 2)

        samples[index] += value * amplitude * envelope


def _noise_value(index: int, seed: int = 0) -> float:
    value = math.sin((index + 1 + seed * 977) * 12.9898) * 43758.5453
    return (value - math.floor(value)) * 2 - 1


def _mix_noise(
    samples: List[float],
    *,
    start_seconds: float,
    duration_seconds: float,
    sample_rate: int,
    amplitude: float,
    seed: int = 0,
    body_frequency: float | None = None,
) -> None:
    start = max(0, int(start_seconds * sample_rate))
    end = min(len(samples), int((start_seconds + duration_seconds) * sample_rate))
    if end <= start:
        return

    length = end - start
    for index in range(start, end):
        progress = (index - start) / max(1, length)
        envelope = (1 - progress) ** 2
        noise_value = _noise_value(index, seed)
        body = 0.0
        if body_frequency:
            local_seconds = (index - start) / sample_rate
            body = math.sin(2 * math.pi * body_frequency * local_seconds) * 0.45
        samples[index] += (noise_value + body) * amplitude * envelope


def _mix_kick(
    samples: List[float],
    *,
    start_seconds: float,
    duration_seconds: float,
    sample_rate: int,
    amplitude: float,
) -> None:
    start = max(0, int(start_seconds * sample_rate))
    end = min(len(samples), int((start_seconds + duration_seconds) * sample_rate))
    if end <= start:
        return

    length = end - start
    phase = 0.0
    for index in range(start, end):
        progress = (index - start) / max(1, length)
        frequency = 115 - 70 * progress
        phase += 2 * math.pi * frequency / sample_rate
        envelope = (1 - progress) ** 3
        click = _noise_value(index, 31) * max(0.0, 1 - progress * 22) * 0.15
        samples[index] += (math.sin(phase) + click) * amplitude * envelope


def _window_rms(values: List[float], start: int, end: int) -> float:
    if end <= start:
        return 0.0
    total = sum(value * value for value in values[start:end])
    return math.sqrt(total / (end - start))


def _level_full_mix_samples(samples: List[float], sample_rate: int) -> None:
    window = max(1, int(sample_rate * 0.7))
    rms_values = []
    for start in range(0, len(samples), window):
        rms = _window_rms(samples, start, min(len(samples), start + window))
        if rms > 0.01:
            rms_values.append(rms)
    if not rms_values:
        return

    rms_values.sort()
    target = min(0.22, max(0.12, rms_values[len(rms_values) // 2]))
    previous_gain = 1.0
    for start in range(0, len(samples), window):
        end = min(len(samples), start + window)
        rms = _window_rms(samples, start, end)
        if rms <= 0.01:
            desired_gain = 1.0
        else:
            desired_gain = max(0.72, min(3.5, target / rms))

        for index in range(start, end):
            progress = (index - start) / max(1, end - start)
            gain = previous_gain + (desired_gain - previous_gain) * progress
            samples[index] *= gain
        previous_gain = desired_gain


def _style_family(composition: Composition) -> str:
    style = composition.style.strip().lower()
    # Exact match first
    if style in {"rock", "edm", "jazz", "folk", "cinematic", "r&b", "lo-fi"}:
        return style
    # Common aliases
    alias_map = {
        "lofi": "lo-fi",
        "lo fi": "lo-fi",
        "lo-fi hip hop": "lo-fi",
        "chillhop": "lo-fi",
        "rnb": "r&b",
        "r and b": "r&b",
        "rhythm and blues": "r&b",
        "soul": "r&b",
        "hip hop": "r&b",
        "hip-hop": "r&b",
        "trap": "r&b",
        "rap": "r&b",
        "hood": "r&b",
        "pop": "pop",
        "classical": "cinematic",
        "orchestral": "cinematic",
        "country": "folk",
        "americana": "folk",
        "bluegrass": "folk",
        "blues": "rock",
        "punk": "rock",
        "metal": "rock",
        "classic rock": "rock",
        "hard rock": "rock",
        "alternative": "rock",
        "grunge": "rock",
        "indie rock": "rock",
        "indie": "rock",
        "electronica": "edm",
        "house": "edm",
        "techno": "edm",
        "trance": "edm",
        "dubstep": "edm",
        "dance": "edm",
    }
    if style in alias_map:
        return alias_map[style]
    # Keyword scan as last resort
    text = " ".join([
        composition.style,
        composition.mood,
        " ".join(composition.style_notes),
    ]).lower()
    if any(w in text for w in ("rock", "guitar", "riff", "punk", "metal", "grunge")):
        return "rock"
    if any(w in text for w in ("edm", "drop", "sidechain", "house", "techno", "rave")):
        return "edm"
    if any(w in text for w in ("jazz", "swing", "walking bass", "bebop")):
        return "jazz"
    if any(w in text for w in ("folk", "acoustic", "country", "bluegrass")):
        return "folk"
    if any(w in text for w in ("cinematic", "strings", "orchestral", "epic")):
        return "cinematic"
    if any(w in text for w in ("r&b", "groove", "soul", "hip hop", "trap", "rnb")):
        return "r&b"
    if any(w in text for w in ("chill", "lofi", "lo-fi", "vinyl", "mellow", "relaxed")):
        return "lo-fi"
    if any(w in text for w in ("pop", "radio", "catchy", "hook")):
        return "pop"
    # No match — return the style as-is, let the caller use the lo-fi fallback in STYLE_PROGRAMS
    return "lo-fi"


def _mix_drum_pattern(
    samples: List[float],
    *,
    instrument: str,
    bar_start: float,
    beats_per_bar: float,
    beat_seconds: float,
    sample_rate: int,
    amplitude_scale: float = 1.0,
) -> None:
    beat_count = max(1, int(beats_per_bar))

    if instrument == "electric_drums":
        for beat in range(beat_count):
            _mix_kick(samples, start_seconds=bar_start + beat * beat_seconds, duration_seconds=0.20, sample_rate=sample_rate, amplitude=0.22 * amplitude_scale)
        for beat in (1, 3):
            if beat < beat_count:
                _mix_noise(samples, start_seconds=bar_start + beat * beat_seconds, duration_seconds=0.15, sample_rate=sample_rate, amplitude=0.13 * amplitude_scale, seed=beat, body_frequency=230)
        for step in range(beat_count * 2):
            _mix_noise(samples, start_seconds=bar_start + (step + 0.5) * beat_seconds / 2, duration_seconds=0.04, sample_rate=sample_rate, amplitude=0.035 * amplitude_scale, seed=step + 50)
        return

    if instrument == "808_drums":
        for beat in range(beat_count):
            start_idx = max(0, int((bar_start + beat * beat_seconds) * sample_rate))
            end_idx = min(len(samples), int((bar_start + beat * beat_seconds + 0.45) * sample_rate))
            length = end_idx - start_idx
            phase = 0.0
            for index in range(start_idx, end_idx):
                progress = (index - start_idx) / max(1, length)
                frequency = 85 - 50 * progress
                phase += 2 * math.pi * frequency / sample_rate
                envelope = (1 - progress) ** 1.8
                samples[index] += math.sin(phase) * 0.24 * amplitude_scale * envelope
        for beat in (1, 3):
            if beat < beat_count:
                clap_start = bar_start + beat * beat_seconds
                for trigger in range(3):
                    _mix_noise(samples, start_seconds=clap_start + trigger * 0.012, duration_seconds=0.015, sample_rate=sample_rate, amplitude=0.14 * amplitude_scale, seed=beat + trigger, body_frequency=180)
                _mix_noise(samples, start_seconds=clap_start + 0.036, duration_seconds=0.16, sample_rate=sample_rate, amplitude=0.11 * amplitude_scale, seed=beat + 10, body_frequency=180)
        for step in range(beat_count * 2):
            _mix_noise(samples, start_seconds=bar_start + step * beat_seconds / 2, duration_seconds=0.018, sample_rate=sample_rate, amplitude=0.04 * amplitude_scale, seed=step + 200)
        return

    if instrument == "brush_drums":
        for beat in range(beat_count):
            if beat % 2 == 0:
                _mix_kick(samples, start_seconds=bar_start + beat * beat_seconds, duration_seconds=0.14, sample_rate=sample_rate, amplitude=0.08 * amplitude_scale)
        for beat in (1, 3):
            if beat < beat_count:
                _mix_noise(samples, start_seconds=bar_start + beat * beat_seconds, duration_seconds=0.25, sample_rate=sample_rate, amplitude=0.04 * amplitude_scale, seed=beat + 30, body_frequency=120)
        for step in range(beat_count * 2):
            _mix_noise(samples, start_seconds=bar_start + step * beat_seconds / 2, duration_seconds=0.15, sample_rate=sample_rate, amplitude=0.012 * amplitude_scale, seed=step + 150)
        return

    if instrument == "orchestral_timpani":
        timpani_start = bar_start
        start_idx = max(0, int(timpani_start * sample_rate))
        end_idx = min(len(samples), int((timpani_start + 0.8) * sample_rate))
        length = end_idx - start_idx
        phase = 0.0
        for index in range(start_idx, end_idx):
            progress = (index - start_idx) / max(1, length)
            frequency = 75 - 30 * progress
            phase += 2 * math.pi * frequency / sample_rate
            envelope = (1 - progress) ** 2
            samples[index] += _triangle_wave(phase) * 0.22 * amplitude_scale * envelope
        _mix_noise(samples, start_seconds=timpani_start, duration_seconds=1.4, sample_rate=sample_rate, amplitude=0.15 * amplitude_scale, seed=99, body_frequency=300)
        return

    if instrument == "tabla_drums":
        # Tabla Teentaal-inspired pattern: Dha-ge-Ti-Ra-Dhin-tun-Ta pattern
        # Dha: low bayan bass stroke (low body thud)
        _mix_kick(samples, start_seconds=bar_start, duration_seconds=0.12, sample_rate=sample_rate, amplitude=0.16 * amplitude_scale)
        # ge: light finger resonance
        _mix_noise(samples, start_seconds=bar_start + beat_seconds * 0.25, duration_seconds=0.06, sample_rate=sample_rate, amplitude=0.04 * amplitude_scale, seed=1, body_frequency=420)
        # Ti: sharp tabla treble slap (Na/tip)
        _mix_noise(samples, start_seconds=bar_start + beat_seconds * 0.5, duration_seconds=0.07, sample_rate=sample_rate, amplitude=0.10 * amplitude_scale, seed=2, body_frequency=680)
        # Ra: resonant ring
        _mix_noise(samples, start_seconds=bar_start + beat_seconds * 0.75, duration_seconds=0.08, sample_rate=sample_rate, amplitude=0.06 * amplitude_scale, seed=3, body_frequency=540)
        # Dhin: open bayan (mid-beat)
        _mix_kick(samples, start_seconds=bar_start + beat_seconds, duration_seconds=0.10, sample_rate=sample_rate, amplitude=0.12 * amplitude_scale)
        _mix_noise(samples, start_seconds=bar_start + beat_seconds, duration_seconds=0.15, sample_rate=sample_rate, amplitude=0.07 * amplitude_scale, seed=4, body_frequency=360)
        # tun: resonant sustain
        _mix_noise(samples, start_seconds=bar_start + beat_seconds * 1.5, duration_seconds=0.12, sample_rate=sample_rate, amplitude=0.05 * amplitude_scale, seed=5, body_frequency=460)
        # Ta: sharp treble on beat 3 (if 4/4)
        if beat_count >= 3:
            _mix_noise(samples, start_seconds=bar_start + beat_seconds * 2, duration_seconds=0.07, sample_rate=sample_rate, amplitude=0.11 * amplitude_scale, seed=6, body_frequency=700)
            _mix_noise(samples, start_seconds=bar_start + beat_seconds * 2.5, duration_seconds=0.06, sample_rate=sample_rate, amplitude=0.05 * amplitude_scale, seed=7, body_frequency=500)
        if beat_count >= 4:
            _mix_kick(samples, start_seconds=bar_start + beat_seconds * 3, duration_seconds=0.10, sample_rate=sample_rate, amplitude=0.13 * amplitude_scale)
            _mix_noise(samples, start_seconds=bar_start + beat_seconds * 3.5, duration_seconds=0.07, sample_rate=sample_rate, amplitude=0.06 * amplitude_scale, seed=8, body_frequency=620)
        return

    if instrument == "bongo_drums":
        # Bongo pattern: alternating hi/lo bongo with conga accents
        for step in range(beat_count * 2):
            step_start = bar_start + step * beat_seconds / 2
            # Alternate hi-bongo (higher freq) and lo-bongo (lower freq)
            freq = 520 if step % 2 == 0 else 320
            amp = 0.13 * amplitude_scale if step % 4 == 0 else 0.08 * amplitude_scale
            _mix_noise(samples, start_seconds=step_start, duration_seconds=0.09, sample_rate=sample_rate, amplitude=amp, seed=step + 60, body_frequency=freq)
        # Open hi-conga accent on beat 2
        if beat_count > 1:
            _mix_noise(samples, start_seconds=bar_start + beat_seconds, duration_seconds=0.14, sample_rate=sample_rate, amplitude=0.14 * amplitude_scale, seed=20, body_frequency=600)
        # Mute hi-conga accent on beat 4
        if beat_count > 3:
            _mix_noise(samples, start_seconds=bar_start + beat_seconds * 3, duration_seconds=0.10, sample_rate=sample_rate, amplitude=0.11 * amplitude_scale, seed=21, body_frequency=450)
        return

    # Default to acoustic_drums
    for beat in (0, 2):
        if beat < beat_count:
            _mix_kick(samples, start_seconds=bar_start + beat * beat_seconds, duration_seconds=0.2, sample_rate=sample_rate, amplitude=0.2 * amplitude_scale)
    for beat in (1, 3):
        if beat < beat_count:
            _mix_noise(samples, start_seconds=bar_start + beat * beat_seconds, duration_seconds=0.18, sample_rate=sample_rate, amplitude=0.15 * amplitude_scale, seed=beat + 10, body_frequency=210)
    for step in range(beat_count * 2):
        _mix_noise(samples, start_seconds=bar_start + step * beat_seconds / 2, duration_seconds=0.035, sample_rate=sample_rate, amplitude=0.028 * amplitude_scale, seed=step + 90)


def composition_to_wav_bytes(composition: Composition, sample_rate: int = 44100, stem: str = "full") -> bytes:
    beat_seconds = 60 / composition.tempo_bpm
    beats_per_bar = _beats_per_bar(composition.time_signature)
    family = _style_family(composition)
    include_drums = stem in {"full", "rhythm", "drums"}
    include_bass = stem in {"full", "rhythm", "bass"}
    include_harmony = stem in {"full", "harmony"}
    include_melody = stem in {"full", "melody"}
    total_seconds = sum(_effective_section_beats(section, beats_per_bar) * beat_seconds for section in composition.sections)
    total_samples = max(1, int((total_seconds + 0.25) * sample_rate))
    samples: List[float] = [0.0] * total_samples

    m = composition.mixer
    vol_drums = (m.drums.volume / 74.0) if (m and m.drums) else 1.0
    vol_bass = (m.bass.volume / 82.0) if (m and m.bass) else 1.0
    vol_harmony = (m.harmony.volume / 68.0) if (m and m.harmony) else 1.0
    vol_melody = (m.melody.volume / 88.0) if (m and m.melody) else 1.0
    vol_master = (m.master.volume / 78.0) if (m and m.master) else 1.0

    family_insts = STYLE_INSTRUMENTS.get(family, STYLE_INSTRUMENTS["lo-fi"])
    inst_drums = m.drums.instrument if (m and m.drums) else family_insts["drums"]
    inst_bass = m.bass.instrument if (m and m.bass) else family_insts["bass"]
    inst_harmony = m.harmony.instrument if (m and m.harmony) else family_insts["harmony"]
    inst_melody = m.melody.instrument if (m and m.melody) else family_insts["melody"]

    cursor = 0.0
    for section in composition.sections:
        section_start = cursor
        bar_seconds = beats_per_bar * beat_seconds

        for bar_index in range(_effective_section_bars(section, beats_per_bar)):
            bar_start = section_start + bar_index * bar_seconds
            if include_drums:
                _mix_drum_pattern(
                    samples,
                    instrument=inst_drums,
                    bar_start=bar_start,
                    beats_per_bar=beats_per_bar,
                    beat_seconds=beat_seconds,
                    sample_rate=sample_rate,
                    amplitude_scale=vol_drums,
                )

        for symbol, start_beats, duration_beats in _chord_spans(section, beats_per_bar):
            chord_start = section_start + start_beats * beat_seconds
            chord_seconds = duration_beats * beat_seconds
            if include_bass:
                if inst_bass == "electric_bass" and family == "rock":
                    num_steps = int(duration_beats * 2)
                    for step in range(num_steps):
                        step_start = chord_start + step * 0.5 * beat_seconds
                        step_duration = 0.44 * beat_seconds
                        # Alternate octave on offbeats to make it driving!
                        pitch_octave = 2 if step % 4 else 3
                        _mix_tone(
                            samples,
                            frequency=_pitch_frequency(_chord_root_pitch(symbol, octave=pitch_octave)),
                            start_seconds=step_start,
                            duration_seconds=step_duration,
                            sample_rate=sample_rate,
                            amplitude=0.13 * vol_bass,
                            waveform=inst_bass,
                        )
                else:
                    _mix_tone(
                        samples,
                        frequency=_pitch_frequency(_chord_root_pitch(symbol, octave=2)),
                        start_seconds=chord_start,
                        duration_seconds=chord_seconds,
                        sample_rate=sample_rate,
                        amplitude=(0.18 if inst_bass in {"synth_bass", "sub_bass"} else 0.12) * vol_bass,
                        waveform=inst_bass,
                    )

            chord_pitches = _power_chord_pitches(symbol) if inst_harmony == "distorted_guitar" else _chord_pitches(symbol)
            chord_amplitude = 0.05 if inst_harmony in {"distorted_guitar", "electric_guitar"} else 0.045 if inst_harmony == "synth_pad" else 0.035
            if include_harmony:
                if inst_harmony in {"distorted_guitar", "electric_guitar"} and family == "rock":
                    num_steps = int(duration_beats * 2)
                    for step in range(num_steps):
                        step_start = chord_start + step * 0.5 * beat_seconds
                        # Strum pattern: DUN chug-chug DUN DUN chug DUN chug
                        is_chug = step % 8 in {1, 5, 6}
                        step_duration = (0.12 if is_chug else 0.44) * beat_seconds
                        step_amp = chord_amplitude * (0.65 if is_chug else 1.0)
                        waveform_type = "slap_bass" if is_chug else inst_harmony
                        
                        for chord_index, chord_pitch in enumerate(chord_pitches):
                            strum_offset = chord_index * 0.015 if not is_chug else 0.0
                            _mix_tone(
                                samples,
                                frequency=_pitch_frequency(chord_pitch),
                                start_seconds=step_start + strum_offset,
                                duration_seconds=step_duration - strum_offset,
                                sample_rate=sample_rate,
                                amplitude=step_amp * vol_harmony,
                                waveform=waveform_type,
                            )
                else:
                    for chord_index, chord_pitch in enumerate(chord_pitches):
                        strum_offset = chord_index * 0.018 if inst_harmony in {"distorted_guitar", "acoustic_guitar"} else 0.0
                        _mix_tone(
                            samples,
                            frequency=_pitch_frequency(chord_pitch),
                            start_seconds=chord_start + strum_offset,
                            duration_seconds=chord_seconds * 0.96,
                            sample_rate=sample_rate,
                            amplitude=chord_amplitude * vol_harmony,
                            waveform=inst_harmony,
                        )

        melody_cursor = section_start
        for melody_note in section.melody:
            note_seconds = float(melody_note.duration_beats) * beat_seconds
            if include_melody:
                _mix_tone(
                    samples,
                    frequency=_pitch_frequency(melody_note.pitch),
                    start_seconds=melody_cursor,
                    duration_seconds=note_seconds * 0.94,
                    sample_rate=sample_rate,
                    amplitude=0.18 * vol_melody,
                    waveform=inst_melody,
                )
            melody_cursor += note_seconds

        cursor = section_start + _effective_section_beats(section, beats_per_bar) * beat_seconds

    if stem == "full":
        _level_full_mix_samples(samples, sample_rate)

    peak = max(max(abs(value) for value in samples), 0.001)
    gain = (0.95 / peak) * vol_master
    pcm = bytearray()
    for value in samples:
        clipped = max(-1.0, min(1.0, value * gain))
        pcm.extend(struct.pack("<h", int(clipped * 32767)))

    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(pcm))
    return output.getvalue()


def composition_to_musicxml_bytes(composition: Composition) -> bytes:
    score = build_score(composition)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        score.write("musicxml", fp=str(temp_path))
        return temp_path.read_bytes()
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def composition_to_notation_text(composition: Composition) -> str:
    lines = [
        f"Title: {composition.title}",
        f"Style: {composition.style}",
        f"Mood: {composition.mood}",
        f"Key: {composition.key}",
        f"Tempo: {composition.tempo_bpm} BPM",
        f"Time: {composition.time_signature}",
        "",
    ]
    beats_per_bar = _beats_per_bar(composition.time_signature)
    for section in composition.sections:
        lines.append(f"[{section.name}]")
        lines.append("Chords: " + " | ".join(symbol for symbol, _, _ in _chord_spans(section, beats_per_bar)))
        melody = " ".join(f"{item.pitch}:{item.duration_beats}" for item in section.melody)
        lines.append("Melody: " + melody)
        if section.lyric_lines:
            lines.append("Lyrics:")
            lines.extend(section.lyric_lines)
        lines.append("")
    lines.append("Disclaimer: " + composition.disclaimer)
    return "\n".join(lines)


def flatten_chords(composition: Composition) -> List[Tuple[str, str]]:
    beats_per_bar = _beats_per_bar(composition.time_signature)
    return [
        (section.name, chord_symbol)
        for section in composition.sections
        for chord_symbol, _, _ in _chord_spans(section, beats_per_bar)
    ]
