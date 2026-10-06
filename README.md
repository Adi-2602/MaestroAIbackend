# Maestro AI — Backend

The API behind **Maestro AI**, an AI music composition studio. A multi-agent **LangGraph** pipeline, running on **NVIDIA NIM** LLMs, turns a short brief (style, mood, theme, key, tempo) into a full composition: song structure, chords, melody and lyrics. The backend then validates it against music theory, scores it, and renders it to MIDI, audio and notation.

The web app lives in the `MaestroAI` repo.

## How it works

```
reference → lyricist → sentiment → theorist → composer → improvisor → director
```

| Agent        | Role                                                                    |
| ------------ | ----------------------------------------------------------------------- |
| Reference    | Builds a style blueprint and song plan from the brief                   |
| Lyricist     | Writes lyrics, or adapts the user's own lyrics                          |
| Sentiment    | Checks that the lyrics' mood matches the brief (lexical; optional BERT) |
| Theorist     | Picks idiomatic chord progressions for the style                        |
| Composer     | Writes the melody over the chords                                       |
| Improvisor   | Adds variation and musical interest to the melody                        |
| Director     | Assembles three creative tiers (safe, balanced, wild) with validation and review |

`/compose` streams each agent's progress to the client over **Server-Sent Events**.

## Features

- Multi-agent composition and targeted AI refinement (`/refine`)
- Music-theory validation, an evaluation score and a commercial-readiness review
- Exports: **MIDI**, **WAV**, **MP3**, **stems**, **MusicXML**, notation text and a ZIP package
- Real-instrument audio via **FluidSynth** + SoundFont, with a built-in procedural synth as fallback
- Accounts with PBKDF2-hashed passwords, bearer-token sessions, and password reset by emailed OTP
- Workspaces → projects → drafts, stored in **SQLite** (default) or **PostgreSQL**

## Tech stack

Python 3.11 · FastAPI · LangGraph / LangChain · NVIDIA NIM · music21 · FluidSynth · ffmpeg · SQLite / PostgreSQL

## Getting started

Requirements: Python 3.11+, an [NVIDIA NIM API key](https://build.nvidia.com/). For real-instrument audio you also need `fluidsynth` and `ffmpeg` (optional).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env               # add your NVIDIA_API_KEY
python app.py                      # http://localhost:5050  (docs at /docs)
```

### Audio (optional)

- **Linux / macOS**: install `fluidsynth` and `ffmpeg` with your package manager; they are found on `PATH` automatically.
- **Windows**: download FluidSynth from its [releases page](https://github.com/FluidSynth/fluidsynth/releases), extract it under `tools/fluidsynth/`, and set `FLUIDSYNTH_PATH` in `.env`.
- Put a SoundFont (e.g. `MuseScore_General.sf3` or `FluidR3_GM.sf2`) in `assets/soundfonts/`, or set `SOUNDFONT_PATH`.

Without these, WAV export uses the built-in procedural engine.

### Key environment variables

| Variable                       | Description                                                          |
| ------------------------------ | -------------------------------------------------------------------- |
| `NVIDIA_API_KEY`               | NVIDIA NIM API key (required for composing)                          |
| `NVIDIA_NIM_MODEL`             | Model id, default `meta/llama-3.1-8b-instruct`                       |
| `CORS_ORIGIN`                  | Comma-separated list of allowed frontend origins                     |
| `DATABASE_URL`                 | Postgres URL; leave empty to use SQLite at `APP_DATABASE_PATH`       |
| `SMTP_HOST` / `SMTP_USER` / …  | SMTP settings for password-reset emails                              |
| `NODE_ENV`                     | `development` prints reset codes to the console when SMTP is unset; use `production` when deployed |

See [`.env.example`](.env.example) for the full list.

## API overview

All routes are under `/api/v1`. Interactive docs are served at `/docs`.

| Area        | Endpoints                                                                                     |
| ----------- | --------------------------------------------------------------------------------------------- |
| Auth        | `POST /auth/register` · `/auth/login` · `/auth/logout` · `/auth/forgot-password` · `/auth/reset-password` · `GET /auth/me` |
| Compose     | `POST /compose` (SSE) · `/refine` · `/validate` · `/evaluate` · `/commercial-review`          |
| Library     | `GET/POST /workspaces` · `GET/POST /workspaces/{id}/projects` · `PUT /projects/{id}` · `GET/POST /drafts` · `GET/PUT /drafts/{id}` |
| Export      | `POST /export/midi` · `/wav` · `/mp3` · `/stems` · `/musicxml` · `/notation` · `/package`      |
| Health      | `GET /health` · `GET /api/v1/provider`                                                        |

## Tests

```bash
pytest -q
```

## Docker

The image installs FluidSynth and ffmpeg and downloads a General MIDI SoundFont:

```bash
docker build -t maestro-ai-api .
docker run -p 5050:5050 --env-file .env maestro-ai-api
```

The image sets `NODE_ENV=production`, so configure SMTP if you want password reset to work in a deployment.

## Project structure

```
app/
├── main.py           # FastAPI app, auth and all routes
├── workflow.py       # LangGraph multi-agent pipeline
├── nim_client.py     # NVIDIA NIM client and prompts
├── music.py          # Validation, optimisation, MIDI/WAV/MusicXML rendering
├── evaluation.py     # Composition scoring
├── commercial.py     # Commercial-readiness review
├── sentiment.py      # Lyrics sentiment analysis
├── storage.py        # SQLite and PostgreSQL stores
├── schemas.py        # Pydantic models
└── config.py         # Settings from environment
tests/                # pytest suite
```

## License

[MIT](LICENSE)
