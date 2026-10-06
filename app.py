from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _restart_with_repo_venv() -> None:
    """Let `py app.py` use the project virtual environment when it exists."""
    backend_dir = Path(__file__).resolve().parent
    repo_dir = backend_dir.parent
    candidates = [
        repo_dir / ".venv" / "Scripts" / "python.exe",
        repo_dir / ".venv" / "bin" / "python",
    ]
    current = Path(sys.executable).resolve()
    for candidate in candidates:
        if candidate.exists() and candidate.resolve() != current:
            command = [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]]
            raise SystemExit(subprocess.call(command))


def main() -> None:
    _restart_with_repo_venv()

    from dotenv import load_dotenv
    import uvicorn

    load_dotenv(Path(__file__).resolve().parent / ".env")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5050"))
    print(f"Starting backend on {host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()
