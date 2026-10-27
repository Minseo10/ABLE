"""Discover the public model in this account, without private hard-coded paths."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def resolve_tidybot_model(configured: str | None, repo: Path) -> Path:
    """Honor an explicit path and otherwise search installed/sibling KINDER."""
    explicit = configured or os.environ.get("SPROUT_TIDYBOT_MODEL")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ValueError(f"TidyBot model not found: {path}. Update scene.json.")
        return path.resolve()
    relative = Path("envs/dynamic3d/models/stanford_tidybot/tidybot.xml")
    spec = importlib.util.find_spec("kinder")
    roots = list(spec.submodule_search_locations or []) if spec else []
    roots.append(str(repo.parent / "kindergarden/src/kinder"))
    for root in roots:
        path = Path(root) / relative
        if path.is_file():
            return path.resolve()
    raise ValueError(
        "Public TidyBot model not found. Install KINDER in the workstation "
        "environment or set SPROUT_TIDYBOT_MODEL to stanford_tidybot/tidybot.xml."
    )
