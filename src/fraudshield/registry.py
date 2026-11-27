"""Minimal file-based model registry: versioned joblib artefacts + metadata JSON + a LATEST pointer."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import joblib

from .model import FraudModel

_VERSION = re.compile(r"^v(\d{4})$")


class ModelRegistry:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def versions(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and _VERSION.match(p.name))

    def latest_version(self) -> str | None:
        pointer = self.root / "LATEST"
        if pointer.exists():
            v = pointer.read_text().strip()
            if (self.root / v).is_dir():
                return v
        versions = self.versions()
        return versions[-1] if versions else None

    def _next_version(self) -> str:
        versions = self.versions()
        n = int(_VERSION.match(versions[-1]).group(1)) + 1 if versions else 1
        return f"v{n:04d}"

    def register(self, model: FraudModel, metadata: dict, reference_profile: dict | None = None) -> str:
        version = self._next_version()
        vdir = self.root / version
        vdir.mkdir(parents=True, exist_ok=False)
        joblib.dump(model, vdir / "model.joblib", compress=3)
        meta = {"version": version, "registered_at": datetime.now(UTC).isoformat(timespec="seconds"), **metadata}
        (vdir / "metadata.json").write_text(json.dumps(meta, indent=2, default=float))
        if reference_profile is not None:
            (vdir / "reference_profile.json").write_text(json.dumps(reference_profile, default=float))
        (self.root / "LATEST").write_text(version)
        return version

    def load(self, version: str | None = None) -> tuple[FraudModel, dict]:
        version = version or self.latest_version()
        if version is None:
            raise FileNotFoundError(f"no model registered under {self.root}")
        vdir = self.root / version
        model = joblib.load(vdir / "model.joblib")
        meta = json.loads((vdir / "metadata.json").read_text())
        return model, meta

    def load_reference_profile(self, version: str | None = None) -> dict | None:
        version = version or self.latest_version()
        if version is None:
            return None
        path = self.root / version / "reference_profile.json"
        return json.loads(path.read_text()) if path.exists() else None
