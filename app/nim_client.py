import json
import json_repair
import logging
import asyncio
from typing import Any, Dict, AsyncGenerator

import httpx
from pydantic import ValidationError

from .config import Settings
from .schemas import ComposeRequest, Composition


logger = logging.getLogger("music-composer")


class NimTimeoutError(RuntimeError):
    pass


SYSTEM_PROMPT = """
You are an expert music composition agent. Generate original, editable symbolic song drafts.
Return JSON only. Do not include markdown, commentary, or code fences.
The composition must be practical for MIDI export and human editing.
Avoid quoting existing lyrics or recognizable melodies.
Do not imitate a living artist or copyrighted song. Use broad genre language only.
""".strip()


STYLE_RULES = {
    "lo-fi": "Use relaxed BPM, seventh chords, short motifs, warm nostalgia, and soft imagery.",
    "pop": "Use clear verse/hook contrast, singable melody, simple emotional language, and stable progressions.",
    "rock": "Use direct language, strong chord roots, compact hooks, and guitar-friendly progressions.",
    "edm": "Use repetitive hooks, build/drop section logic, short lyric phrases, and energetic contour.",
    "jazz": "Use extended chords, smoother voice-leading, swing-friendly rhythm, and sophisticated but editable lines.",
    "r&b": "Use minor/major seventh color, melismatic-friendly melody, and intimate lyrics.",
    "folk": "Use simple diatonic chords, narrative lyrics, and acoustic-friendly melody.",
    "cinematic": "Use dramatic section labels, wider contour, sparse lyrics, and arrangement notes.",
}


STYLE_BLUEPRINTS = {
    "lo-fi": {
        "sections": "Intro, Verse, Hook, Outro",
        "chords": "minor/major seventh color, add9 chords, soft borrowed chords; avoid power-chord language",
        "melody": "short relaxed motifs with rests, stepwise movement, medium-low contour",
        "lyrics": "soft visual images, late-night reflection, understated phrasing",
        "arrangement": "dusty keys, vinyl texture, warm pad, soft bass, brushed or muted drums",
    },
    "pop": {
        "sections": "Verse, Pre-Chorus, Chorus, Bridge",
        "chords": "simple diatonic triads, sus/add9 color, clear tension into chorus",
        "melody": "singable hook, repeated chorus motif, clear phrase endings",
        "lyrics": "direct emotional language, memorable chorus line, conversational imagery",
        "arrangement": "bright drums, polished bass, layered hook, vocal doubles",
    },
    "rock": {
        "sections": "Intro, Verse, Chorus, Bridge",
        "chords": "root-heavy triads, power-chord-friendly movement, sus chords, strong cadences",
        "melody": "short punchy phrases, wider leaps on chorus, riff-like repetition",
        "lyrics": "direct physical language, urgency, grit, compact hook lines",
        "arrangement": "driven guitars, live drums, bass riff, crash accents, amp energy",
    },
    "edm": {
        "sections": "Intro, Build, Drop, Break",
        "chords": "loopable minor/major triads, sus tension, add9 color for synth pads",
        "melody": "repetitive topline, short rhythmic hook, drop-friendly motif",
        "lyrics": "few chantable phrases, motion, lights, release, crowd energy",
        "arrangement": "four-on-floor kick, sidechain synth, risers, drop bass, chopped hook",
    },
    "jazz": {
        "sections": "Head, A Section, B Section, Solo",
        "chords": "maj7, m7, dominant 7, 9, 13, ii-V color, smoother voice-leading",
        "melody": "swing-friendly syncopation, chromatic approach tones, conversational contour",
        "lyrics": "wry or intimate phrasing, smoky imagery, spacious lines",
        "arrangement": "walking bass, brushed drums, piano voicings, horn-like melody",
    },
    "r&b": {
        "sections": "Verse, Pre-Chorus, Hook, Bridge",
        "chords": "maj7, m7, 9, add9, smooth minor color and passing tension",
        "melody": "fluid phrases, held notes, call-and-response hook shape",
        "lyrics": "intimate direct address, late-night emotion, smooth repetition",
        "arrangement": "deep groove, soft electric keys, sub bass, vocal stacks, pad",
    },
    "folk": {
        "sections": "Verse, Chorus, Bridge, Outro",
        "chords": "plain acoustic triads, sus color, simple cadence, easy strumming",
        "melody": "narrative melody, stepwise motion, natural breathing spaces",
        "lyrics": "storytelling, concrete places, human details, plainspoken images",
        "arrangement": "acoustic guitar, light percussion, warm bass, simple harmony",
    },
    "cinematic": {
        "sections": "Intro, Theme, Rise, Finale",
        "chords": "minor movement, suspended harmony, add9 color, dramatic pedal tones",
        "melody": "wide emotional contour, sparse theme, rising final phrase",
        "lyrics": "minimal lyrical fragments, scale, memory, tension, release",
        "arrangement": "strings, low pulse, wide pads, percussion swells, dramatic dynamics",
    },
}


def _style_blueprint(style: str) -> dict[str, str]:
    return STYLE_BLUEPRINTS.get(style.strip().lower(), {
        "sections": "Intro, Verse, Hook, Bridge, Outro",
        "chords": "choose chord qualities that clearly fit the requested genre",
        "melody": "make the contour and rhythm match the requested genre",
        "lyrics": "match the requested genre, mood, and theme",
        "arrangement": "include arrangement details that make the genre obvious",
    })


def build_user_prompt(request: ComposeRequest) -> str:
    style_rule = STYLE_RULES.get(request.style.lower(), "Follow the requested style using broad genre traits.")
    blueprint = _style_blueprint(request.style)
    return f"""
Create a concise song draft as valid JSON only. Use this schema exactly:
{{
  "title": string,
  "style": "{request.style}",
  "mood": "{request.mood}",
  "key": "{request.key}",
  "tempo_bpm": {request.tempo_bpm},
  "time_signature": "{request.time_signature}",
  "sections": [
    {{
      "name": string,
      "bars": integer,
      "chords": [string],
      "melody": [
        {{"pitch": string, "duration_beats": number, "lyric_syllable": string or null}}
      ],
      "lyric_lines": [string]
    }}
  ],
  "lyrics": [string],
  "style_notes": [string],
  "originality_notes": [string],
  "disclaimer": "Generated music may resemble existing works. Review and clear rights before commercial use."
}}

Rules:
- Total section bars must equal {request.bars} if possible.
- Prefer 2-4 sections. Style-appropriate section names: {blueprint["sections"]}.
- Use only parseable chord symbols like C, G, Am, Fmaj7, Dm7, G7.
- Use melody pitches like C4, D#4, Bb4, or "rest".
- Use duration_beats only from 0.25, 0.5, 1, 1.5, 2, 3, 4.
- For each section, melody duration should cover most of the section capacity.
- Make chords, melody rhythm, lyrics, and style_notes noticeably different for each style selection.
- Do not reuse the same chord progression across styles. Avoid defaulting to C-G-Am-F or Am-F-C-G unless the style/theme makes it clearly necessary.
- Chord direction for {request.style}: {blueprint["chords"]}.
- Melody direction for {request.style}: {blueprint["melody"]}.
- Lyric direction for {request.style}: {blueprint["lyrics"]}.
- Arrangement direction for {request.style}: {blueprint["arrangement"]}.
- Make lyrics match style "{request.style}", mood "{request.mood}", and theme "{request.theme}".
- Instrumentation direction: {request.instrumentation}.
- Style rule: {style_rule}
- Add useful style_notes and originality_notes that name the selected style's musical traits.
- Keep output compact enough for a first draft.
""".strip()


def build_coordinator_prompt(request: ComposeRequest) -> str:
    blueprint = _style_blueprint(request.style)
    return f"""
AGENT: Coordinator Agent.
Create the song plan only. Return JSON only:
{{
  "title": string,
  "brief": string,
  "sections": [
    {{"name": string, "bars": integer, "purpose": string}}
  ]
}}

Requirements:
- Style: {request.style}
- Mood: {request.mood}
- Theme: {request.theme}
- Key: {request.key}
- Tempo: {request.tempo_bpm}
- Total bars must equal {request.bars}.
- Use 2-4 sections, max 16 bars each.
- Style section vocabulary: {blueprint["sections"]}.
- The plan must make the style obvious before any other agent writes music.
""".strip()


def build_chord_agent_prompt(request: ComposeRequest, plan: Dict[str, Any], sentiment_info: Dict[str, Any] = None) -> str:
    blueprint = _style_blueprint(request.style)
    sentiment_guide = ""
    if sentiment_info:
        label = sentiment_info.get("label", "neutral")
        score = sentiment_info.get("score", 1.0)
        sentiment_guide = f"\n- Predicted Lyrics Sentiment: {label} (confidence: {score:.2f})\n"
        if label == "negative":
            sentiment_guide += "- PREFER minor chord colors, slow or heavy harmonic pacing, and tense progressions to match this sentiment.\n"
        elif label == "positive":
            sentiment_guide += "- PREFER major chord colors, brighter or energetic chord voicing shifts, and optimistic cadences.\n"
            
    return f"""
AGENT: Chord Agent.
Write only chord progressions for the coordinator plan. Return JSON only:
{{
  "sections": [
    {{"name": string, "bars": integer, "chords": [string]}}
  ],
  "chord_notes": [string]
}}

Context:
{json.dumps(plan)}

Rules:
- Style: {request.style}; Key: {request.key}; Mood: {request.mood}.
{sentiment_guide}- Chord direction: {blueprint["chords"]}.
- Each section's chords array must contain exactly one chord per bar.
- Use parseable chord symbols only, such as C, G, Am, Fmaj7, Dm7, G7, Esus4, A5.
- Avoid defaulting to C-G-Am-F or Am-F-C-G.
""".strip()


def build_melody_agent_prompt(request: ComposeRequest, plan: Dict[str, Any], chord_doc: Dict[str, Any], sentiment_info: Dict[str, Any] = None) -> str:
    blueprint = _style_blueprint(request.style)
    sentiment_guide = ""
    if sentiment_info:
        label = sentiment_info.get("label", "neutral")
        score = sentiment_info.get("score", 1.0)
        sentiment_guide = f"\n- Predicted Lyrics Sentiment: {label} (confidence: {score:.2f})\n"
        if label == "negative":
            sentiment_guide += "- PREFER lower melodic contour, stepwise descents, expressive pauses, and somber intervals to match this sentiment.\n"
        elif label == "positive":
            sentiment_guide += "- PREFER rising melodic phrases, wider soaring leaps on chorus, and upbeat syncopation.\n"

    return f"""
AGENT: Melody Agent.
Write symbolic melody notes for each section. Return JSON only:
{{
  "sections": [
    {{
      "name": string,
      "melody": [
        {{"pitch": string, "duration_beats": number, "lyric_syllable": string or null}}
      ]
    }}
  ],
  "melody_notes": [string]
}}

Coordinator plan:
{json.dumps(plan)}

Chord Agent output:
{json.dumps(chord_doc)}

Rules:
- Style: {request.style}; Key: {request.key}; Tempo: {request.tempo_bpm}; Time: {request.time_signature}.
{sentiment_guide}- Melody direction: {blueprint["melody"]}.
- Use pitch names like C4, D#4, Bb4, or "rest".
- Use duration_beats only from 0.25, 0.5, 1, 1.5, 2, 3, 4.
- Each section melody should cover most of that section's bars.
""".strip()


def build_lyrics_agent_prompt(request: ComposeRequest, plan: Dict[str, Any]) -> str:
    blueprint = _style_blueprint(request.style)
    return f"""
AGENT: Lyrics Agent.
Write editable lyric lines for each section. Return JSON only:
{{
  "sections": [
    {{"name": string, "lyric_lines": [string]}}
  ],
  "lyrics": [string],
  "lyric_notes": [string]
}}

Coordinator plan:
{json.dumps(plan)}

Rules:
- Style: {request.style}; Mood: {request.mood}; Theme: {request.theme}.
- Lyric direction: {blueprint["lyrics"]}.
- Avoid quoting existing songs or recognizable lyric lines.
- Keep lines short enough for a first draft.
""".strip()


def build_arrangement_agent_prompt(request: ComposeRequest, plan: Dict[str, Any]) -> str:
    blueprint = _style_blueprint(request.style)
    return f"""
AGENT: Arrangement Agent.
Write arrangement and originality notes. Return JSON only:
{{
  "style_notes": [string],
  "originality_notes": [string],
  "instrumentation_notes": [string]
}}

Coordinator plan:
{json.dumps(plan)}

Rules:
- Style: {request.style}; Mood: {request.mood}; Instrumentation: {request.instrumentation}.
- Arrangement direction: {blueprint["arrangement"]}.
- Name concrete sonic choices that match the selected style.
- Include clear human-review/originality guidance.
""".strip()


def build_drum_agent_prompt(request: ComposeRequest, plan: Dict[str, Any]) -> str:
    blueprint = _style_blueprint(request.style)
    return f"""
AGENT: Drum Agent.
Design explicit drum/percussion patterns for the song. Return JSON only:
{{
  "drum_pattern": [string],
  "section_drums": [
    {{"name": string, "pattern": string}}
  ]
}}

Coordinator plan:
{json.dumps(plan)}

Rules:
- Style: {request.style}; Tempo: {request.tempo_bpm}; Mood: {request.mood}.
- Arrangement direction: {blueprint["arrangement"]}.
- Describe kick/snare/hat or percussion behavior in practical production language.
- Keep patterns concise and editable.
""".strip()


def build_bass_agent_prompt(request: ComposeRequest, plan: Dict[str, Any], chord_doc: Dict[str, Any]) -> str:
    blueprint = _style_blueprint(request.style)
    return f"""
AGENT: Bass Agent.
Design bassline guidance from the chord progression. Return JSON only:
{{
  "bassline": [string],
  "section_bass": [
    {{"name": string, "pattern": string}}
  ]
}}

Coordinator plan:
{json.dumps(plan)}

Chord Agent output:
{json.dumps(chord_doc)}

Rules:
- Style: {request.style}; Key: {request.key}; Tempo: {request.tempo_bpm}.
- Chord direction: {blueprint["chords"]}.
- Mention roots, passing tones, rhythmic feel, and section contrast.
- Keep it practical for MIDI/audio rendering.
""".strip()


def build_mix_agent_prompt(request: ComposeRequest, composition: Composition) -> str:
    return f"""
AGENT: Mix Agent.
Create lightweight mix/master notes for this draft. Return JSON only:
{{
  "mix_notes": [string],
  "stem_plan": [string]
}}

Composition:
{composition.model_dump_json()}

Rules:
- Style: {request.style}; Instrumentation: {request.instrumentation}.
- Include balance, panning, ambience, dynamics, and export/stem advice.
- Keep notes short and actionable.
""".strip()


def build_safety_agent_prompt(request: ComposeRequest, composition: Composition) -> str:
    return f"""
AGENT: Commercial Safety Agent.
Review prompt and composition for commercial release risk. Return JSON only:
{{
  "commercial_notes": [string],
  "risk_warnings": [string],
  "approved_for_demo": boolean
}}

User request:
{request.model_dump_json()}

Composition:
{composition.model_dump_json()}

Rules:
- Flag direct living-artist imitation, copied lyrics, voice cloning, or claims of guaranteed uniqueness.
- Recommend broad genre language and human rights review.
- Do not give legal advice; provide practical product-safety notes.
""".strip()


def build_critic_agent_prompt(composition: Composition) -> str:
    return f"""
AGENT: Critic Agent.
Review this generated composition for style adherence, editability, and licensing safety.
Return JSON only:
{{
  "accepted": boolean,
  "warnings": [string],
  "improvements": [string]
}}

Composition:
{composition.model_dump_json()}

Rules:
- Do not rewrite the song.
- Flag only important issues.
- Confirm whether chords, melody, lyrics, and arrangement clearly match {composition.style}.
""".strip()


def build_refine_prompt(target: str, composition: Composition, instructions: str | None) -> str:
    payload = composition.model_dump()
    return f"""
Revise only the requested musical component: {target}.
User instructions: {instructions or "Improve quality while preserving the song identity."}

Return the full composition JSON again with the same schema.
Keep title, style, mood, key, tempo_bpm, time_signature, disclaimer, and section count unless the target requires a small change.
If target is chords, improve chord progressions and keep bars valid.
If target is melody, improve notes and durations to fit the chords/key.
If target is lyrics, improve lyric_lines and lyrics while avoiding existing copyrighted lines.
If target is arrangement, improve style_notes, structure, and all weak areas lightly.

Current composition JSON:
{json.dumps(payload)}
""".strip()


def strip_reasoning(text: str) -> str:
    """Drop the <think>...</think> block that reasoning models put before their answer.

    Some models omit the opening tag, so everything up to the last closing tag is dropped.
    """
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return text


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = strip_reasoning(text).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("NIM response did not contain a JSON object.")
    json_str = cleaned[start : end + 1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        repaired = json_repair.repair_json(json_str, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
        raise ValueError("Failed to repair JSON object.")


ALLOWED_DURATIONS = [0.25, 0.5, 1, 1.5, 2, 3, 4]


def _nearest_duration(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 1
    return min(ALLOWED_DURATIONS, key=lambda duration: abs(duration - numeric))


def _fallback_sections(request: ComposeRequest) -> list[dict[str, Any]]:
    section_names = _style_blueprint(request.style)["sections"].split(", ")
    section_count = min(4, max(2, request.bars // 4 if request.bars >= 8 else 2))
    base_bars = max(1, request.bars // section_count)
    sections: list[dict[str, Any]] = []
    remaining = request.bars
    for index in range(section_count):
        bars = remaining if index == section_count - 1 else min(16, base_bars)
        remaining -= bars
        sections.append({"name": section_names[index % len(section_names)], "bars": max(1, bars), "purpose": "Draft section"})
    return sections


def _normalize_plan(plan: Dict[str, Any], request: ComposeRequest) -> Dict[str, Any]:
    sections = plan.get("sections")
    if not isinstance(sections, list) or not sections:
        sections = _fallback_sections(request)

    normalized: list[dict[str, Any]] = []
    for index, section in enumerate(sections[:4]):
        if not isinstance(section, dict):
            continue
        name = str(section.get("name") or f"Section {index + 1}")[:40]
        try:
            bars = int(section.get("bars") or 4)
        except (TypeError, ValueError):
            bars = 4
        normalized.append({
            "name": name,
            "bars": max(1, min(16, bars)),
            "purpose": str(section.get("purpose") or "Draft section")[:120],
        })

    if not normalized:
        normalized = _fallback_sections(request)

    while sum(section["bars"] for section in normalized) < request.bars:
        grew = False
        for section in normalized:
            if sum(item["bars"] for item in normalized) >= request.bars:
                break
            if section["bars"] < 16:
                section["bars"] += 1
                grew = True
        if not grew:
            break

    while sum(section["bars"] for section in normalized) > request.bars:
        shrank = False
        for section in reversed(normalized):
            if sum(item["bars"] for item in normalized) <= request.bars:
                break
            if section["bars"] > 1:
                section["bars"] -= 1
                shrank = True
        if not shrank:
            break

    return {
        "title": str(plan.get("title") or f"{request.style} Draft")[:80],
        "brief": str(plan.get("brief") or f"{request.style} {request.mood} song about {request.theme}")[:240],
        "sections": normalized,
    }


def _sections_by_name(doc: Dict[str, Any]) -> dict[str, dict[str, Any]]:
    sections = doc.get("sections")
    if not isinstance(sections, list):
        return {}
    output: dict[str, dict[str, Any]] = {}
    for section in sections:
        if isinstance(section, dict) and section.get("name"):
            output[str(section["name"]).strip().lower()] = section
    return output


def _fallback_chords(style: str, key: str) -> list[str]:
    style_key = style.strip().lower()
    key_root = key.split()[0] if key else "C"
    if style_key == "rock":
        return [f"{key_root}5", "G5", "D5", "A5"]
    if style_key == "jazz":
        return [f"{key_root}m7", "D7", "Gmaj7", "Cmaj7"]
    if style_key == "edm":
        return [f"{key_root}m", "F", "C", "G"]
    if style_key == "folk":
        return [key_root, "G", "Am", "F"]
    return [f"{key_root}m7", "Fmaj7", "Cmaj7", "G7"]


def _normalize_chords(value: Any, bars: int, request: ComposeRequest) -> list[str]:
    chords = [str(chord).strip() for chord in value] if isinstance(value, list) else []
    chords = [chord for chord in chords if chord]
    if not chords:
        chords = _fallback_chords(request.style, request.key)
    return [chords[index % len(chords)] for index in range(bars)]


def _normalize_melody(value: Any, bars: int) -> list[dict[str, Any]]:
    melody = value if isinstance(value, list) else []
    cleaned: list[dict[str, Any]] = []
    for item in melody[:128]:
        if not isinstance(item, dict):
            continue
        pitch = str(item.get("pitch") or "rest").strip() or "rest"
        cleaned.append({
            "pitch": pitch,
            "duration_beats": _nearest_duration(item.get("duration_beats")),
            "lyric_syllable": item.get("lyric_syllable"),
        })
    if not cleaned:
        cleaned = [{"pitch": "rest", "duration_beats": 4, "lyric_syllable": None} for _ in range(bars)]
    return cleaned


def _normalize_lines(value: Any, fallback: str) -> list[str]:
    if isinstance(value, list):
        lines = [str(line).strip() for line in value if str(line).strip()]
        if lines:
            return lines[:12]
    return [fallback]


class NimClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def _chat(self, user_prompt: str, temperature: float = 0.75, max_tokens: int = 2400) -> str:
        if not self.settings.nim_api_key:
            raise RuntimeError("NVIDIA_API_KEY is not configured.")

        payload = {
            "model": self.settings.nim_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "top_p": 0.9,
            "max_tokens": max_tokens,
            # Reasoning models are too slow for a multi-agent pipeline; answer directly.
            "chat_template_kwargs": {"enable_thinking": False},
        }

        timeout = httpx.Timeout(
            connect=20,
            read=self.settings.nim_timeout_seconds,
            write=30,
            pool=20,
        )
        attempts = self.settings.nim_retries + 1
        last_timeout: httpx.TimeoutException | None = None

        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(1, attempts + 1):
                try:
                    response = await client.post(
                        f"{self.settings.normalized_nim_base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.settings.nim_api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                    break
                except httpx.TimeoutException as exc:
                    last_timeout = exc
                    logger.warning("NVIDIA NIM request timed out on attempt %s/%s", attempt, attempts)
                    if attempt == attempts:
                        raise NimTimeoutError(
                            f"NVIDIA NIM timed out after {self.settings.nim_timeout_seconds:g} seconds. "
                            "Try Generate again, reduce bars, or use a faster NIM model."
                        ) from exc
                except httpx.HTTPStatusError as exc:
                    detail = exc.response.text[:500]
                    raise RuntimeError(f"NVIDIA NIM API error {exc.response.status_code}: {detail}") from exc
            else:
                raise NimTimeoutError("NVIDIA NIM timed out before returning a response.") from last_timeout

        return data["choices"][0]["message"]["content"]

    async def _parse_or_repair(self, content: str) -> Composition:
        try:
            parsed = extract_json_object(content)
            return Composition.model_validate(parsed)
        except (json.JSONDecodeError, ValueError, ValidationError) as exc:
            repair_prompt = f"""
Repair this model output into valid JSON matching the required composition schema.
Return JSON only. Keep all usable musical content. Fix missing fields, invalid durations, and malformed arrays.

Invalid output:
{content}

Error:
{exc}
""".strip()
            repaired_content = await self._chat(repair_prompt, temperature=0.1, max_tokens=2400)
            parsed = extract_json_object(repaired_content)
            try:
                return Composition.model_validate(parsed)
            except ValidationError as repair_exc:
                raise ValueError(f"NIM returned invalid composition JSON after repair: {repair_exc}") from repair_exc

    async def _parse_json_or_repair(self, content: str, repair_goal: str) -> Dict[str, Any]:
        try:
            return extract_json_object(content)
        except (json.JSONDecodeError, ValueError) as exc:
            repair_prompt = f"""
Repair this output into valid JSON for this goal: {repair_goal}.
Return JSON only. Keep all usable fields.

Invalid output:
{content}

Error:
{exc}
""".strip()
            repaired_content = await self._chat(repair_prompt, temperature=0.1, max_tokens=1200)
            return extract_json_object(repaired_content)

    async def _chat_json(self, prompt: str, repair_goal: str, temperature: float = 0.6, max_tokens: int = 1600) -> Dict[str, Any]:
        content = await self._chat(prompt, temperature=temperature, max_tokens=max_tokens)
        return await self._parse_json_or_repair(content, repair_goal)

    def _merge_multi_agent_outputs(
        self,
        request: ComposeRequest,
        plan: Dict[str, Any],
        chord_doc: Dict[str, Any],
        melody_doc: Dict[str, Any],
        lyrics_doc: Dict[str, Any],
        arrangement_doc: Dict[str, Any],
        drum_doc: Dict[str, Any],
        bass_doc: Dict[str, Any],
    ) -> Composition:
        chord_sections = _sections_by_name(chord_doc)
        melody_sections = _sections_by_name(melody_doc)
        lyric_sections = _sections_by_name(lyrics_doc)

        sections = []
        for section in plan["sections"]:
            name = section["name"]
            section_key = name.strip().lower()
            bars = section["bars"]
            chord_section = chord_sections.get(section_key, {})
            melody_section = melody_sections.get(section_key, {})
            lyric_section = lyric_sections.get(section_key, {})
            sections.append({
                "name": name,
                "bars": bars,
                "chords": _normalize_chords(chord_section.get("chords"), bars, request),
                "melody": _normalize_melody(melody_section.get("melody"), bars),
                "lyric_lines": _normalize_lines(
                    lyric_section.get("lyric_lines"),
                    f"{request.theme} moves through the {request.mood.lower()} night",
                ),
            })

        lyrics = _normalize_lines(lyrics_doc.get("lyrics"), sections[0]["lyric_lines"][0])
        style_notes = _normalize_lines(arrangement_doc.get("style_notes"), f"{request.style} arrangement for {request.instrumentation}")
        instrumentation_notes = _normalize_lines(arrangement_doc.get("instrumentation_notes"), str(request.instrumentation or "editable band arrangement"))
        drum_pattern = _normalize_lines(drum_doc.get("drum_pattern"), f"{request.style} drum groove matching {request.tempo_bpm} BPM")
        bassline = _normalize_lines(bass_doc.get("bassline"), f"{request.style} bass follows chord roots with section contrast")
        originality_notes = _normalize_lines(
            arrangement_doc.get("originality_notes"),
            "Original draft uses broad genre traits and should be reviewed before release.",
        )

        return Composition.model_validate({
            "title": plan["title"],
            "style": request.style,
            "mood": request.mood,
            "key": request.key,
            "tempo_bpm": request.tempo_bpm,
            "time_signature": request.time_signature,
            "sections": sections,
            "lyrics": lyrics,
            "style_notes": (style_notes + instrumentation_notes)[:12],
            "originality_notes": originality_notes[:12],
            "drum_pattern": drum_pattern[:16],
            "bassline": bassline[:16],
            "mix_notes": [],
            "commercial_notes": [],
            "agent_trace": [
                "Coordinator Agent planned title, sections, and bar counts.",
                "Chord Agent generated section-level chord progressions.",
                "Lyrics Agent generated editable section lyrics.",
                "Arrangement Agent generated instrumentation and originality notes.",
                "Drum Agent generated percussion patterns.",
                "Bass Agent generated bassline guidance from chords.",
                "Melody Agent generated symbolic melody after chords.",
            ],
            "disclaimer": "Generated music may resemble existing works. Review and clear rights before commercial use.",
        })

    async def compose_multi_agent(self, request: ComposeRequest) -> Composition:
        logger.info("Running multi-agent composition pipeline for style=%s", request.style)
        plan_raw = await self._chat_json(
            build_coordinator_prompt(request),
            repair_goal="coordinator song plan",
            temperature=max(0.2, request.creativity - 0.15),
            max_tokens=2048,
        )
        plan = _normalize_plan(plan_raw, request)

        # Generate lyrics first to get sentiment context
        lyrics_task = self._chat_json(
            build_lyrics_agent_prompt(request, plan),
            repair_goal="lyrics agent output",
            temperature=request.creativity,
            max_tokens=2048,
        )
        arrangement_task = self._chat_json(
            build_arrangement_agent_prompt(request, plan),
            repair_goal="arrangement agent output",
            temperature=max(0.4, request.creativity - 0.1),
            max_tokens=2048,
        )
        drum_task = self._chat_json(
            build_drum_agent_prompt(request, plan),
            repair_goal="drum agent output",
            temperature=max(0.4, request.creativity - 0.05),
            max_tokens=2048,
        )
        lyrics_doc, arrangement_doc, drum_doc = await asyncio.gather(lyrics_task, arrangement_task, drum_task)

        # Perform sentiment analysis on the generated lyrics
        lyrics_list = lyrics_doc.get("lyrics", [])
        if not lyrics_list and request.custom_lyrics:
            lyrics_list = [request.custom_lyrics]
        from .sentiment import analyze_lyrics_sentiment
        sentiment_res = analyze_lyrics_sentiment(lyrics_list)
        logger.info("Lyrics sentiment analysis result (compose_multi_agent): %s", sentiment_res)

        # Now generate chords incorporating the lyrics sentiment
        chord_doc = await self._chat_json(
            build_chord_agent_prompt(request, plan, sentiment_info=sentiment_res),
            repair_goal="chord agent output",
            temperature=request.creativity,
            max_tokens=2048,
        )

        # Generate melody and bass in parallel, passing sentiment to melody
        melody_task = self._chat_json(
            build_melody_agent_prompt(request, plan, chord_doc, sentiment_info=sentiment_res),
            repair_goal="melody agent output",
            temperature=request.creativity,
            max_tokens=3000,
        )
        bass_task = self._chat_json(
            build_bass_agent_prompt(request, plan, chord_doc),
            repair_goal="bass agent output",
            temperature=max(0.4, request.creativity - 0.05),
            max_tokens=2048,
        )
        melody_doc, bass_doc = await asyncio.gather(melody_task, bass_task)
        composition = self._merge_multi_agent_outputs(request, plan, chord_doc, melody_doc, lyrics_doc, arrangement_doc, drum_doc, bass_doc)

        critic_doc = await self._chat_json(
            build_critic_agent_prompt(composition),
            repair_goal="critic agent output",
            temperature=0.2,
            max_tokens=1500,
        )
        mix_doc = await self._chat_json(
            build_mix_agent_prompt(request, composition),
            repair_goal="mix agent output",
            temperature=0.35,
            max_tokens=1500,
        )
        safety_doc = await self._chat_json(
            build_safety_agent_prompt(request, composition),
            repair_goal="commercial safety agent output",
            temperature=0.2,
            max_tokens=1500,
        )
        critic_notes = _normalize_lines(critic_doc.get("improvements"), "Critic agent accepted the editable draft.")
        if critic_doc.get("warnings"):
            critic_notes.extend(_normalize_lines(critic_doc.get("warnings"), "Review warnings before commercial release."))
        composition.originality_notes = (composition.originality_notes + critic_notes)[:12]
        composition.mix_notes = _normalize_lines(mix_doc.get("mix_notes"), "Balance drums, bass, harmony, and melody before export.")[:16]
        composition.commercial_notes = (
            _normalize_lines(safety_doc.get("commercial_notes"), "Use human review before commercial release.")
            + _normalize_lines(safety_doc.get("risk_warnings"), "No major commercial safety warnings returned.")
        )[:16]
        composition.agent_trace = (
            composition.agent_trace
            + [
                "Critic Agent reviewed style adherence and editability.",
                "Mix Agent generated production and stem guidance.",
                "Commercial Safety Agent reviewed release risks.",
            ]
        )[:16]
        return composition

    async def stream_compose_langgraph(self, request: ComposeRequest) -> AsyncGenerator[Dict[str, Any], None]:
        from .workflow import stream_composition_workflow
        async for event in stream_composition_workflow(request):
            yield event


    async def refine(self, target: str, composition: Composition, instructions: str | None = None) -> Composition:
        content = await self._chat(build_refine_prompt(target, composition, instructions), temperature=0.6)
        return await self._parse_or_repair(content)
