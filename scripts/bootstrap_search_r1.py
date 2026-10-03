#!/usr/bin/env python3
"""Fetch the pinned upstream without modifying an existing checkout."""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(directory, *args):
    return subprocess.check_output(
        ["git", "-c", "core.fsmonitor=false", "-C", str(directory), *args], text=True
    ).strip()


def verify_checkout(directory, expected):
    if not (directory / ".git").exists():
        raise RuntimeError(f"Existing path is not an upstream Git checkout: {directory}")
    actual = git(directory, "rev-parse", "HEAD")
    if actual != expected:
        raise RuntimeError(f"Refusing to overwrite checkout at {actual}; expected {expected}. Use a separate path.")
    if git(directory, "status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("Upstream checkout has local changes; preserve them and use a clean separate checkout.")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify only; no network or writes.")
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    lock = json.loads((ROOT / "upstream.lock.json").read_text(encoding="utf-8"))
    destination = (args.destination or ROOT / lock["path"]).resolve()
    if destination.exists():
        verify_checkout(destination, lock["commit"])
    elif args.check:
        raise SystemExit(f"Missing {destination}; run bootstrap_search_r1.py first.")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", lock["url"], str(destination)], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", lock["commit"]], check=True)
        verify_checkout(destination, lock["commit"])
    print(f"Verified Search-R1 {lock['commit']} at {destination}")


if __name__ == "__main__":
    main()
