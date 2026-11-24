"""Restore archived import paths in an isolated, generated work directory."""

from pathlib import Path
from shutil import copyfile
import json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = ROOT / ".reproduction" / "workspace"


def source_paths():
    with (ROOT / "docs/frozen_workspace.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def prepare_workspace(workspace):
    destination = workspace.resolve()
    if destination == ROOT or ROOT.is_relative_to(destination):
        raise ValueError("Use a separate runtime directory, such as .reproduction/workspace.")
    if destination.is_relative_to(ROOT) and not destination.is_relative_to(ROOT / ".reproduction"):
        raise ValueError("Use .reproduction/ for runtime workspaces inside the repository.")

    missing = []
    conflicts = []
    for old, new in source_paths().items():
        source, target = ROOT / new, destination / old
        if not target.resolve().is_relative_to(destination):
            raise ValueError(f"Source destination points outside the runtime workspace: {old}")
        if target.is_file() and target.read_bytes() == source.read_bytes():
            continue
        if target.exists() or target.is_symlink():
            conflicts.append(old)
        else:
            missing.append((source, target))
    if conflicts:
        names = "\n  ".join(conflicts)
        raise ValueError(f"Different source files already exist in {destination}:\n  {names}")
    for source, target in missing:
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, target)
    return destination
