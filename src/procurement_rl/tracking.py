"""Small, dependency-free experiment records and immutable data checks."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def json_digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def verified_dataset(data_dir):
    """Fail closed if any frozen corpus/task bytes differ from their manifest."""
    root = Path(data_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for split in ("train", "val", "test"):
        for name in ("tasks.jsonl", "corpus.jsonl"):
            relative = f"{split}/{name}"
            expected = manifest["files"][relative]
            if file_digest(root / relative) != expected:
                raise ValueError(f"Dataset hash mismatch: {relative}")
    return manifest


def runtime_manifest(kind, config):
    root = Path(__file__).resolve().parents[2]
    package_dir = Path(__file__).resolve().parent
    versions = {}
    for package in ("procurement-search-rl", "torch", "transformers", "vllm", "ray", "pandas", "pyarrow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                           stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    upstream_path = root / "upstream.lock.json"
    return {
        "schema_version": 1, "kind": kind, "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": versions, "git_head": git_head,
        "source_sha256": {str(path.relative_to(package_dir)): file_digest(path)
                          for path in sorted(package_dir.rglob("*.py"))},
        "upstream_lock": json.loads(upstream_path.read_text(encoding="utf-8")) if upstream_path.exists() else None,
        "cost": {"currency": "CNY", "hourly_instance_rate": None, "elapsed_seconds": None,
                 "estimated_compute_cost": None,
                 "basis": "elapsed wall time times supplied whole-instance hourly rate; billing rounding and idle time excluded"},
    }


def finish_manifest(manifest, elapsed_seconds, hourly_cost_cny=None, error=None):
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "failed" if error is not None else "completed"
    if error is not None:
        manifest["error"] = f"{type(error).__name__}: {error}"
    manifest["cost"].update(
        elapsed_seconds=elapsed_seconds, hourly_instance_rate=hourly_cost_cny,
        estimated_compute_cost=None if hourly_cost_cny is None else elapsed_seconds / 3600 * hourly_cost_cny,
    )
    return manifest
