from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


Pitch = str
Duration = float  # Snapped to nearest valid value by MelodyNote validator
_VALID_DURATIONS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
_VALID_CHORD_DURATIONS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]


class ComposeRequest(BaseModel):
    style: str = Field(..., min_length=2, max_length=80)
    mood: str = Field(..., min_length=2, max_length=80)
    theme: str = Field(..., min_length=2, max_length=160)
    key: str = Field(default="C major", max_length=40)
    tempo_bpm: int = Field(default=90, ge=45, le=220)
    bars: int = Field(default=8, ge=4, le=32)
    time_signature: str = Field(default="4/4", pattern=r"^\d+/\d+$")
    creativity: float = Field(default=0.75, ge=0.0, le=1.0)
    instrumentation: Optional[str] = Field(default="piano, bass, light drums", max_length=120)
    custom_lyrics: Optional[str] = Field(default=None, max_length=3000)


class MelodyNote(BaseModel):
    pitch: Pitch = Field(..., min_length=1, max_length=8)
    duration_beats: Duration = 1.0
    lyric_syllable: Optional[str] = Field(default=None, max_length=24)

    @field_validator("duration_beats", mode="before")
    @classmethod
    def snap_duration(cls, value) -> float:
        """Snap any AI-generated float to the nearest valid musical duration."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 1.0
        return min(_VALID_DURATIONS, key=lambda d: abs(d - v))

    @field_validator("pitch")
    @classmethod
    def clean_pitch(cls, value: str) -> str:
        import re
        value = value.strip()
        if value.upper() == "REST":
            return "rest"
        # If the LLM returns a bare note letter (e.g. 'F', 'Bb', 'C#') without
        # an octave number, append octave 4 (middle range) automatically.
        if re.fullmatch(r"[A-Ga-g](#|b|##|bb)?", value):
            value = value.capitalize() + "4"
        return value


class ChordEvent(BaseModel):
    """A chord with a precise beat duration — enables Chordify-style timeline display."""
    chord: str = Field(..., min_length=1, max_length=20)
    duration_beats: float = Field(default=4.0, gt=0, le=32.0)

    @field_validator("chord")
    @classmethod
    def clean_chord(cls, value: str) -> str:
        return value.strip()

    @field_validator("duration_beats", mode="before")
    @classmethod
    def snap_chord_duration(cls, value) -> float:
        """Snap any AI float to nearest musical chord duration."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 4.0
        return min(_VALID_CHORD_DURATIONS, key=lambda d: abs(d - v))


class SongSection(BaseModel):
    name: str = Field(..., min_length=2, max_length=40)
    bars: int = Field(..., ge=1, le=16)
    # chord_events is the primary timed representation (Chordify-style)
    chord_events: List[ChordEvent] = Field(default_factory=list, max_length=32)
    # chords is the flat list for audio engine & backward-compat
    chords: List[str] = Field(default_factory=list, max_length=32)
    melody: List[MelodyNote] = Field(..., min_length=1, max_length=128)
    lyric_lines: List[str] = Field(default_factory=list, max_length=12)
    # lyric_chord_lines: each entry uses [Chord]word notation for chord-over-lyrics display
    # Example: "[Am]Zindagi chalti [G]jaaye, [F]gham bhi [C]hote hain saath"
    lyric_chord_lines: List[str] = Field(default_factory=list, max_length=12)

    @field_validator("chords")
    @classmethod
    def clean_chords(cls, value: List[str]) -> List[str]:
        return [chord.strip() for chord in value if chord.strip()]

    @model_validator(mode="after")
    def sync_chord_representations(self) -> "SongSection":
        """Keep chord_events and chords in sync with each other."""
        if self.chord_events and not self.chords:
            # Derive flat list from timed events
            self.chords = [e.chord for e in self.chord_events]
        elif self.chords and not self.chord_events:
            # Derive timed events from flat list (assume 4 beats each = 1 bar in 4/4)
            self.chord_events = [ChordEvent(chord=c, duration_beats=4.0) for c in self.chords]
        return self


class MixerChannel(BaseModel):
    volume: int = Field(default=80, ge=0, le=100)
    pan: str = Field(default="C", max_length=8)
    instrument: str = Field(default="piano", max_length=40)


class MixerSettings(BaseModel):
    drums: MixerChannel = Field(default_factory=lambda: MixerChannel(volume=74, pan="C", instrument="acoustic_drums"))
    bass: MixerChannel = Field(default_factory=lambda: MixerChannel(volume=82, pan="L8", instrument="electric_bass"))
    harmony: MixerChannel = Field(default_factory=lambda: MixerChannel(volume=68, pan="R6", instrument="piano"))
    melody: MixerChannel = Field(default_factory=lambda: MixerChannel(volume=88, pan="C", instrument="piano_lead"))
    master: MixerChannel = Field(default_factory=lambda: MixerChannel(volume=78, pan="C", instrument="master"))


class Composition(BaseModel):
    title: str = Field(..., min_length=2, max_length=80)
    style: str = Field(..., min_length=2, max_length=80)
    mood: str = Field(..., min_length=2, max_length=80)
    key: str = Field(..., min_length=1, max_length=40)
    tempo_bpm: int = Field(..., ge=45, le=220)
    time_signature: str = Field(..., pattern=r"^\d+/\d+$")
    sections: List[SongSection] = Field(..., min_length=1, max_length=8)
    mixer: Optional[MixerSettings] = None
    lyrics: List[str] = Field(default_factory=list, max_length=80)
    style_notes: List[str] = Field(default_factory=list, max_length=12)
    originality_notes: List[str] = Field(default_factory=list, max_length=12)
    drum_pattern: List[str] = Field(default_factory=list, max_length=16)
    bassline: List[str] = Field(default_factory=list, max_length=16)
    mix_notes: List[str] = Field(default_factory=list, max_length=16)
    commercial_notes: List[str] = Field(default_factory=list, max_length=16)
    agent_trace: List[str] = Field(default_factory=list, max_length=16)
    disclaimer: str = Field(
        default="Generated music may resemble existing works. Review and clear rights before commercial use."
    )

    @model_validator(mode="after")
    def ensure_lyrics(self) -> "Composition":
        if not self.lyrics:
            self.lyrics = [line for section in self.sections for line in section.lyric_lines]
        return self


class ComposeResponse(BaseModel):
    draft_id: str
    composition: Composition
    warnings: List[str] = Field(default_factory=list)
    versions: Optional[Dict[str, Composition]] = None


class RefineRequest(BaseModel):
    target: Literal["chords", "melody", "lyrics", "arrangement"] = "arrangement"
    composition: Composition
    instructions: Optional[str] = Field(default=None, max_length=240)


class ValidationReport(BaseModel):
    warnings: List[str] = Field(default_factory=list)
    total_sections: int
    total_bars: int
    total_chords: int
    total_melody_notes: int
    lyric_lines: int


class EvaluationReport(BaseModel):
    overall_score: int
    chord_validity: int
    melody_key_fit: int
    duration_fit: int
    style_adherence: int
    lyrics_score: int
    disclaimer_score: int
    export_readiness: int
    commercial_safety: int
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class CommercialReview(BaseModel):
    score: int
    warnings: List[str] = Field(default_factory=list)
    checklist: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class DraftSummary(BaseModel):
    draft_id: str
    title: str
    style: str
    mood: str
    updated_at: str


class DraftRecord(BaseModel):
    draft_id: str
    composition: Composition
    created_at: str
    updated_at: str


class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=80)
    email: str = Field(..., min_length=5, max_length=160)
    password: str = Field(..., min_length=6, max_length=120)


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=160)
    password: str = Field(..., min_length=6, max_length=120)


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=160)


class ResetPasswordRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=160)
    otp: str = Field(..., min_length=8, max_length=8)
    new_password: str = Field(..., min_length=6, max_length=120)


class UserPublic(BaseModel):
    user_id: str
    name: str
    email: str
    created_at: str


class AuthSession(BaseModel):
    token: str
    user: UserPublic


class WorkspaceCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=80)


class WorkspaceRecord(BaseModel):
    workspace_id: str
    user_id: str
    name: str
    created_at: str
    updated_at: str


class ProjectCreate(BaseModel):
    title: str = Field(..., min_length=2, max_length=80)


class ProjectUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=2, max_length=80)
    draft_id: Optional[str] = Field(default=None, max_length=80)


class ProjectRecord(BaseModel):
    project_id: str
    workspace_id: str
    user_id: str
    title: str
    draft_id: Optional[str] = None
    created_at: str
    updated_at: str


class ProviderInfo(BaseModel):
    provider: Literal["nvidia_nim"] = "nvidia_nim"
    base_url: str
    model: str
    api_key_configured: bool
    architecture: str = "multi_agent_nim"
    audio_engine: str = "procedural"
