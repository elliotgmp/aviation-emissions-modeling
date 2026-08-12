"""Typed configuration loader.

One YAML file, one entry point, no module-level constants scattered across the
codebase. Attribute access with dotted paths keeps call sites readable
(``cfg.get("models.validation.n_splits")``) without paying for a full schema
library.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

__all__ = ["Config", "load_config", "setup_logging", "project_root"]


def project_root(start: Path | None = None) -> Path:
    """Walk up from this file until a directory containing ``configs/`` is hit.

    Makes every script runnable from anywhere -- repo root, ``scripts/``, or a
    notebook two levels down -- without ``sys.path`` surgery or brittle
    ``../..`` relative paths.
    """
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "configs").is_dir() and (parent / "src").is_dir():
            return parent
    return Path.cwd()


@dataclass
class Config:
    """Thin, dotted-path wrapper over the parsed YAML."""

    data: dict[str, Any] = field(default_factory=dict)
    root: Path = field(default_factory=project_root)

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.data
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def __getitem__(self, path: str) -> Any:
        value = self.get(path, _MISSING)
        if value is _MISSING:
            raise KeyError(path)
        return value

    def path(self, key: str) -> Path:
        """Resolve a ``paths.*`` entry against the project root."""
        raw = self.get(f"paths.{key}")
        if raw is None:
            raise KeyError(f"paths.{key} not defined in config")
        p = Path(raw)
        return p if p.is_absolute() else self.root / p

    def raw_file(self, key: str) -> Path:
        """Resolve a ``data.*_file`` entry inside the raw directory."""
        name = self.get(f"data.{key}")
        if name is None:
            raise KeyError(f"data.{key} not defined in config")
        return self.path("raw_dir") / name

    def ensure_dirs(self) -> None:
        for key in ("interim_dir", "processed_dir", "figures_dir", "results_dir"):
            self.path(key).mkdir(parents=True, exist_ok=True)


_MISSING = object()


def load_config(path: str | Path | None = None) -> Config:
    """Load ``configs/config.yaml`` (or an explicit path).

    Environment override: ``AEM_CONFIG=/path/to/other.yaml``. Useful for running
    the same code against a sampled dev extract and the full production file
    without touching the repo.
    """
    root = project_root()
    path = Path(os.environ.get("AEM_CONFIG", path or root / "configs" / "config.yaml"))
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cfg = Config(data=data, root=root)
    logger.debug("config loaded from %s (root=%s)", path, root)
    return cfg


def load_reference_results(path: str | Path | None = None) -> dict:
    """Load the frozen legacy results used by the regression tests."""
    root = project_root()
    path = Path(path or root / "configs" / "reference_results.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def setup_logging(level: str = "INFO") -> None:
    """Consistent logging across scripts. Called once, at the entry point."""
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)-38s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
