FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NODE_ENV=production \
    FLUIDSYNTH_PATH=/usr/bin/fluidsynth \
    SOUNDFONT_PATH=/app/assets/soundfonts/FluidR3_GM.sf2 \
    APP_DATABASE_PATH=/tmp/drafts.db

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl fluidsynth ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the SoundFont used by FluidSynth for WAV previews/exports.
# The download is non-fatal (|| true): if GitHub CDN is unavailable during build,
# the app falls back to its built-in procedural WAV engine automatically.
RUN mkdir -p /app/assets/soundfonts && \
    (curl --fail --location --retry 3 --retry-delay 5 \
    --output /app/assets/soundfonts/FluidR3_GM.sf2 \
    "https://github.com/urish/caniuse-midi/raw/master/public/soundfonts/FluidR3_GM.sf2" \
    || echo "WARNING: SoundFont download failed. Procedural audio fallback will be used.") && \
    fluidsynth --version && \
    ffmpeg -version

COPY app ./app

EXPOSE 5050

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-5050}"]
