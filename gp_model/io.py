from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import json
import platform
import sys
import hashlib
import yaml 

@dataclass(frozen=True)
class RunMetadata:
    """Minimal run metadata for reproducibility."""
    run_id: str
    created_utc: str
    python_version: str
    platform: str
    command: str
    git_hint: str | None = None  # optional: user can fill later


def make_run_id(prefix: str = "run") -> str:
    # Short, readable, collision-resistant enough for local experiments
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_metadata(path: Path, run_id: str, git_hint: str | None = None) -> None:
    meta = RunMetadata(
        run_id=run_id,
        created_utc=datetime.now(timezone.utc).isoformat(),
        python_version=sys.version.replace("\n", " "),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        command=" ".join(sys.argv),
        git_hint=git_hint,
    )
    write_json(path, asdict(meta))
    
    
def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML config file and return a dict.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
