from __future__ import annotations

import hashlib
import shutil
import tomllib
from pathlib import Path
from typing import Any

from .config import harness_definition_dir, harness_dir

BASE_IMAGE = "cybergym-harness-base:1"


def read_manifest(directory: Path) -> dict[str, Any]:
    path = directory / "harness.toml"
    if not path.is_file():
        raise ValueError(f"Harness definition not found: {path}")
    manifest = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest.get("entry"), list) or not manifest["entry"]:
        raise ValueError(f"{path} must set entry, e.g. entry = [\"python3\", \"/harness/main.py\"]")
    if "agent_timeout_seconds" in manifest:
        raise ValueError(f"{path}: time limits belong in the eval config (agent_timeout_seconds), not in a harness")
    manifest.setdefault("image", BASE_IMAGE)
    manifest.setdefault("settings", {})
    return manifest


def _tree_hash(roots: list[Path]) -> str:
    digest = hashlib.sha256()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            digest.update(f"{root.name}/{path.relative_to(root)}".encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _code_dir(config: dict[str, Any], harness: dict[str, Any]) -> Path:
    return harness_dir(config, harness) / "code"


def _record_image(manifest: dict[str, Any], target: Path) -> str:
    import docker
    from docker.errors import ImageNotFound

    path = target / "image.id"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    try:
        image_id = docker.from_env().images.get(manifest["image"]).id
    except ImageNotFound:
        raise ValueError(f"Harness image {manifest['image']} is not built; run ~/cybergym-harnesses/build.sh") from None
    path.write_text(image_id + "\n", encoding="utf-8")
    return image_id


def snapshot_code(config: dict[str, Any], harness: dict[str, Any]) -> str:
    source = harness_definition_dir(config, harness)
    common = Path(config["harness_repo"]) / "common"
    manifest = read_manifest(source)
    current = _tree_hash([source, common])
    target = _code_dir(config, harness)
    stamp = target / "code.sha256"
    if stamp.exists():
        if stamp.read_text(encoding="utf-8").strip() != current:
            raise ValueError(
                f"The code of harness {harness['name']!r} changed since label {harness['label']!r} was frozen. "
                "Use a new label for the new code."
            )
        _record_image(manifest, target)
        return current
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(source, target / "harness", ignore=shutil.ignore_patterns("__pycache__"))
    if common.exists():
        shutil.copytree(common, target / "common", ignore=shutil.ignore_patterns("__pycache__"))
    else:
        (target / "common").mkdir(parents=True)
    stamp.write_text(current + "\n", encoding="utf-8")
    _record_image(manifest, target)
    return current


def frozen_code(config: dict[str, Any], harness: dict[str, Any]) -> dict[str, Any]:
    target = _code_dir(config, harness)
    stamp = target / "code.sha256"
    if not stamp.exists():
        raise ValueError(f"Harness {harness['label']!r} has no frozen code; run prepare first")
    manifest = read_manifest(target / "harness")
    image_file = target / "image.id"
    return {
        "image_id": image_file.read_text(encoding="utf-8").strip() if image_file.exists() else None,
        "harness_dir": target / "harness",
        "common_dir": target / "common",
        "manifest": manifest,
        "settings": {**manifest["settings"], **harness["settings"]},
        "code_sha256": stamp.read_text(encoding="utf-8").strip(),
    }
