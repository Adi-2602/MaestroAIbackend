import json
import wave
from io import BytesIO
from pathlib import Path
from struct import unpack
from zipfile import ZipFile

from app.evaluation import evaluate_composition
from app.music import (
    composition_validation_report,
    composition_to_midi_bytes,
    composition_to_musicxml_bytes,
    composition_to_notation_text,
    composition_to_soundfont_wav_bytes,
    composition_to_wav_bytes,
    normalize_chord_symbol,
    optimize_generated_composition,
    validate_composition,
)
from app.schemas import ChordEvent, Composition


def load_golden() -> Composition:
    payload = json.loads(Path(__file__).with_name("golden_composition.json").read_text())
    return Composition.model_validate(payload)


def test_normalize_chord_symbol_accepts_common_symbols() -> None:
    assert normalize_chord_symbol(" Am ") == "Am"
    assert normalize_chord_symbol("Fmaj7") == "Fmaj7"
    assert normalize_chord_symbol("Bbmin7") == "Bbm7"


def test_golden_composition_validates_cleanly() -> None:
    composition = load_golden()
    assert validate_composition(composition) == []


def test_notation_export_contains_core_sections() -> None:
    composition = load_golden()
    notation = composition_to_notation_text(composition)
    assert "Title: Rain Trace" in notation
    assert "[Verse]" in notation
    assert "Chords: Am | Fmaj7 | C | G" in notation
    assert "Disclaimer:" in notation


def test_midi_export_produces_bytes() -> None:
    composition = load_golden()
    midi = composition_to_midi_bytes(composition)
    assert midi[:4] == b"MThd"
    assert len(midi) > 100


def test_wav_export_produces_playable_audio_bytes() -> None:
    composition = load_golden()
    wav = composition_to_wav_bytes(composition)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert len(wav) > 100_000
    with wave.open(BytesIO(wav), "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
    sample_count = len(frames) // 2
    samples = unpack(f"<{sample_count}h", frames)
    assert max(abs(sample) for sample in samples) > 1_000


def _window_rms_int16(samples: tuple[int, ...], sample_rate: int, start_seconds: float, duration_seconds: float) -> float:
    start = int(start_seconds * sample_rate)
    end = min(len(samples), int((start_seconds + duration_seconds) * sample_rate))
    if end <= start:
        return 0.0
    return (sum(sample * sample for sample in samples[start:end]) / (end - start)) ** 0.5


def test_full_wav_preview_keeps_middle_loudness_stable() -> None:
    composition = load_golden()
    composition.tempo_bpm = 120
    composition.sections = [composition.sections[0].model_copy(deep=True)]
    section = composition.sections[0]
    section.bars = 6
    section.chord_events = [
        ChordEvent(chord="Am", duration_beats=8),
        ChordEvent(chord="F", duration_beats=8),
        ChordEvent(chord="Am", duration_beats=8),
    ]
    section.chords = ["Am", "F", "Am"]
    section.melody = [
        section.melody[0].model_copy(update={"pitch": "A4", "duration_beats": 8}),
        section.melody[0].model_copy(update={"pitch": "rest", "duration_beats": 8}),
        section.melody[0].model_copy(update={"pitch": "A4", "duration_beats": 8}),
    ]

    wav = composition_to_wav_bytes(composition)

    with wave.open(BytesIO(wav), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    sample_count = len(frames) // 2
    samples = unpack(f"<{sample_count}h", frames)
    first = _window_rms_int16(samples, sample_rate, 1.0, 2.0)
    middle = _window_rms_int16(samples, sample_rate, 5.0, 2.0)
    last = _window_rms_int16(samples, sample_rate, 9.0, 2.0)

    assert middle / max(first, last) > 0.55


def test_wav_export_uses_repeated_timed_chord_events() -> None:
    composition = load_golden()
    composition.tempo_bpm = 120
    section = composition.sections[0]
    section.bars = 1
    section.chords = ["Am"]
    section.chord_events = [
        ChordEvent(chord="Am", duration_beats=4),
        ChordEvent(chord="Am", duration_beats=4),
    ]
    section.melody = [section.melody[0].model_copy(update={"pitch": "rest", "duration_beats": 4})]

    wav = composition_to_wav_bytes(composition)

    with wave.open(BytesIO(wav), "rb") as wav_file:
        duration_seconds = wav_file.getnframes() / wav_file.getframerate()
    assert duration_seconds >= 4.0
    assert "Chords: Am | Am" in composition_to_notation_text(composition)


def test_preview_uses_denser_lyric_chord_tags() -> None:
    composition = load_golden()
    section = composition.sections[0]
    section.bars = 1
    section.chords = ["Am"]
    section.chord_events = [ChordEvent(chord="Am", duration_beats=4)]
    section.lyric_chord_lines = ["[Am]Rain [Am]keeps [Am]falling"]

    notation = composition_to_notation_text(composition)
    wav = composition_to_wav_bytes(composition)

    assert "Chords: Am | Am | Am" in notation
    with wave.open(BytesIO(wav), "rb") as wav_file:
        assert wav_file.getnframes() > 100_000


def test_rock_wav_uses_different_rendering_path() -> None:
    lofi = load_golden()
    rock = lofi.model_copy(deep=True)
    rock.style = "Rock"
    rock.mood = "Energetic"
    rock.tempo_bpm = 128
    rock.style_notes = ["driven electric guitars", "live drums", "bass riff"]

    lofi_wav = composition_to_wav_bytes(lofi)
    rock_wav = composition_to_wav_bytes(rock)

    assert rock_wav[:4] == b"RIFF"
    assert rock_wav != lofi_wav


def test_arranged_midi_uses_style_specific_programs_and_drum_channel() -> None:
    lofi = load_golden()
    rock = lofi.model_copy(deep=True)
    rock.style = "Rock"
    rock.mood = "Energetic"
    rock.style_notes = ["driven electric guitars", "live drums", "bass riff"]

    lofi_midi = composition_to_midi_bytes(lofi)
    rock_midi = composition_to_midi_bytes(rock)

    assert b"\xc0\x04" in lofi_midi  # Electric piano for lo-fi harmony.
    assert b"\xc0\x1e" in rock_midi  # Distortion guitar for rock harmony.
    assert b"\xc1\x21" in rock_midi  # Electric bass for rock bassline.
    assert b"\x99" in rock_midi  # General MIDI drum channel.
    assert rock_midi != lofi_midi


def test_wav_stems_are_exportable_and_distinct() -> None:
    composition = load_golden()
    full = composition_to_wav_bytes(composition)
    melody = composition_to_wav_bytes(composition, stem="melody")
    harmony = composition_to_wav_bytes(composition, stem="harmony")
    rhythm = composition_to_wav_bytes(composition, stem="rhythm")

    assert melody[:4] == b"RIFF"
    assert harmony[:4] == b"RIFF"
    assert rhythm[:4] == b"RIFF"
    assert len({full, melody, harmony, rhythm}) == 4


def test_soundfont_wav_export_when_renderer_is_bundled() -> None:
    root = Path(__file__).resolve().parents[1]
    fluidsynth = root / "tools" / "fluidsynth" / "dist" / "fluidsynth-v2.5.4-win10-x64-cpp11" / "bin" / "fluidsynth.exe"
    soundfont = root / "assets" / "soundfonts" / "MuseScore_General.sf3"
    if not fluidsynth.exists() or not soundfont.exists():
        return

    wav = composition_to_soundfont_wav_bytes(
        load_golden(),
        fluidsynth_path=str(fluidsynth),
        soundfont_path=str(soundfont),
    )

    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert len(wav) > 500_000


def test_musicxml_export_produces_xml() -> None:
    composition = load_golden()
    musicxml = composition_to_musicxml_bytes(composition)
    assert b"score-partwise" in musicxml
    assert b"Rain Trace" in musicxml


def test_validation_report_summarizes_composition() -> None:
    composition = load_golden()
    report = composition_validation_report(composition)
    assert report["total_sections"] == 2
    assert report["total_bars"] == 8
    assert report["total_chords"] == 8
    assert report["lyric_lines"] == 4


def test_evaluation_scores_golden_composition() -> None:
    composition = load_golden()
    report = evaluate_composition(composition)
    assert report["overall_score"] >= 80
    assert report["export_readiness"] == 100
    assert report["lyrics_score"] == 100


def test_zip_package_shape() -> None:
    composition = load_golden()
    package = BytesIO()
    with ZipFile(package, "w") as zip_file:
        zip_file.writestr("composition.json", composition.model_dump_json())
        zip_file.writestr("composition.mid", composition_to_midi_bytes(composition))
        zip_file.writestr("composition.musicxml", composition_to_musicxml_bytes(composition))
    package.seek(0)
    with ZipFile(package) as zip_file:
        names = set(zip_file.namelist())
    assert {"composition.json", "composition.mid", "composition.musicxml"}.issubset(names)


def test_optimize_generated_composition_snaps_and_pads_melody() -> None:
    composition = load_golden()
    composition.sections[0].chords = ["F", "C", "Am"]
    composition.sections[0].chord_events = []
    composition.sections[0].melody = composition.sections[0].melody[:1]
    composition.sections[0].melody[0].pitch = "D#4"
    optimized = optimize_generated_composition(composition)
    assert optimized.sections[0].chords == ["F", "C", "Am", "F"]
    assert optimized.sections[0].melody[0].pitch == "E4"
    assert sum(float(item.duration_beats) for item in optimized.sections[0].melody) == 16
    assert validate_composition(optimized) == []


def test_optimize_generated_composition_repairs_chord_like_melody_pitch() -> None:
    composition = load_golden()
    composition.sections[0].melody = composition.sections[0].melody[:1]
    composition.sections[0].melody[0].pitch = "Am4"

    optimized = optimize_generated_composition(composition)

    assert optimized.sections[0].melody[0].pitch == "A4"
    assert "Invalid melody pitch: Am4" not in validate_composition(optimized)


def test_optimize_generated_composition_trims_overflowing_melody() -> None:
    composition = load_golden()
    composition.sections[0].bars = 1
    composition.sections[0].chords = ["Em"]
    composition.sections[0].chord_events = []
    composition.sections[0].melody = composition.sections[0].melody[:2]

    optimized = optimize_generated_composition(composition)

    assert sum(float(item.duration_beats) for item in optimized.sections[0].melody) == 4
    assert not any("overflow" in warning for warning in validate_composition(optimized))


def test_custom_mixer_instruments_are_synthesized_and_midi_mapped() -> None:
    composition = load_golden()
    from app.schemas import MixerSettings
    composition.mixer = MixerSettings()
    # Modify mixer with custom instruments
    composition.mixer.drums.instrument = "808_drums"
    composition.mixer.bass.instrument = "sub_bass"
    composition.mixer.harmony.instrument = "church_organ"
    composition.mixer.melody.instrument = "flute_lead"
    
    # 1. MIDI generation
    midi = composition_to_midi_bytes(composition)
    assert midi[:4] == b"MThd"
    # Channel 0 program change to 19 (church organ) -> \xc0\x13
    assert b"\xc0\x13" in midi
    # Channel 1 program change to 39 (sub bass / synth bass 2) -> \xc1\x27
    assert b"\xc1\x27" in midi
    # Channel 2 program change to 73 (flute) -> \xc2\x49
    assert b"\xc2\x49" in midi
    # Drum track (channel 10, i.e., index 9) program change to 25 (TR-808 kit) -> \xc9\x19
    assert b"\xc9\x19" in midi

    # 2. WAV generation
    wav = composition_to_wav_bytes(composition)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert len(wav) > 100_000
