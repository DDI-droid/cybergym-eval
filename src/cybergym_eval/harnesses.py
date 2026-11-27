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
    models = manifest.get("models")
    if models is not None and (
        not isinstance(models, list) or not models or not all(isinstance(m, str) and m for m in models) or len(set(models)) != len(models)
    ):
        raise ValueError(f'{path}: models must be a list of model names, e.g. models = ["deepseek.v3.2"]')
    manifest.setdefault("image", BASE_IMAGE)
    manifest.setdefault("settings", {})
    return manifest


def harness_models(config: dict[str, Any], harness: dict[str, Any]) -> list[str]:
    """The models a harness says it uses, from its live harness.toml; prepare requires them in the models table."""
    models = read_manifest(harness_definition_dir(config, harness)).get("models")
    if not models:
        raise ValueError(f'Harness {harness["name"]!r} lists no models; add e.g. models = ["deepseek.v3.2"] to its harness.toml')
    return models


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


def _frozen_manifest(target: Path) -> dict[str, Any]:
    """The manifest is kept beside the mounted folder (code/harness.toml), so the agent never sees it. Labels frozen
    by the older code kept it inside code/harness/."""
    return read_manifest(target if (target / "harness.toml").is_file() else target / "harness")


def _current_image_id(manifest: dict[str, Any]) -> str:
    """The id the manifest's image tag points at right now."""
    import docker
    from docker.errors import ImageNotFound

    try:
        return docker.from_env().images.get(manifest["image"]).id
    except ImageNotFound:
        raise ValueError(f"Harness image {manifest['image']} is not built; run ~/cybergym-harnesses/build.sh") from None


def _record_image(manifest: dict[str, Any], target: Path) -> str:
    path = target / "image.id"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    image_id = _current_image_id(manifest)
    path.write_text(image_id + "\n", encoding="utf-8")
    return image_id


def check_code(config: dict[str, Any], harness: dict[str, Any]) -> None:
    """Refuse if a label's frozen code or image no longer matches what is on this machine now. Writes nothing.

    prepare calls this for every harness before it freezes anything. run does not repeat it: containers start
    from the frozen copy and the frozen image id, so later edits or rebuilds cannot reach a run in progress.
    """
    source = harness_definition_dir(config, harness)
    target = _code_dir(config, harness)
    stamp = target / "code.sha256"
    if not stamp.exists():
        _current_image_id(read_manifest(source))  # a first freeze only needs the image to exist
        return
    if stamp.read_text(encoding="utf-8").strip() != _tree_hash([source, Path(config["harness_repo"]) / "common"]):
        raise ValueError(
            f"The code of harness {harness['name']!r} changed since label {harness['label']!r} was frozen. "
            "Use a new label for the new code."
        )
    manifest = _frozen_manifest(target)
    recorded = target / "image.id"
    now = _current_image_id(manifest)
    if recorded.exists() and recorded.read_text(encoding="utf-8").strip() != now:
        then = recorded.read_text(encoding="utf-8").strip()
        raise ValueError(
            f"Harness image {manifest['image']} changed since label {harness['label']!r} was frozen "
            f"({then[7:19]} then, {now[7:19]} now). Use a new label for the new image."
        )


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
    # The folder that gets mounted at /harness holds the harness's files but not its manifest: the runner reads
    # harness.toml on the host, and the agent has no use for it. The hash above still covers it.
    shutil.copytree(source, target / "harness", ignore=shutil.ignore_patterns("__pycache__", "harness.toml"))
    shutil.copy2(source / "harness.toml", target / "harness.toml")
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
    manifest = _frozen_manifest(target)
    image_file = target / "image.id"
    return {
        "image_id": image_file.read_text(encoding="utf-8").strip() if image_file.exists() else None,
        "harness_dir": target / "harness",
        "common_dir": target / "common",
        "manifest": manifest,
        "settings": {**manifest["settings"], **harness["settings"]},
        "code_sha256": stamp.read_text(encoding="utf-8").strip(),
    }
