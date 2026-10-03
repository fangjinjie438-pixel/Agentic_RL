# Project handoff

This is an implementation project. The user explicitly authorized implementation on 2026-10-03, superseding the older read-only teaching instruction in `D:/AGENTS.md`.

Start by reading `docs/STATUS.md`, then `docs/FILE_MAP.md`. Update STATUS after each meaningful milestone and before ending a turn. Keep factual progress separate from plans and unverified GPU behavior.

## Requirements

- Simulated enterprise procurement, based on **Search-R1 bundled veRL**, not modern upstream veRL APIs.
- School smoke tests: Qwen2.5-1.5B-Instruct; AutoDL formal results: **at least 3B**, default Qwen2.5-3B-Instruct. Never silently downgrade a formal run.
- Research main line: outcome GRPO / trajectory-broadcast information gain / turn-level IGPO v2. Same data, initialization and outcome verifier.
- Frozen simulated facts, integer money, family-disjoint splits, true retrieved evidence provenance. Gold labels must not enter actor prompts or retrieval results.
- CPU tests do not prove GPU training works. Report model evaluation separately from deterministic scripted environment checks.
- Track method sources/decisions in `docs/research_log.md`; write each experiment's real configuration, versions, hashes and cost. Never invent improvement numbers.
- Upstream is pinned in `upstream.lock.json`. Use project adapters; retain the upstream license and identify any patches explicitly.

## Working conventions

Use package code under `src/procurement_rl`, tests under `tests`, Linux entry scripts under `scripts`, profiles under `configs`. The local development machine is Windows; training targets Linux NVIDIA GPU machines. Keep shell scripts LF. Avoid modifying the user's global Python or CUDA installations.

When referencing modern veRL in explanations, label it `[Modern veRL]`; current implementation is `[Search-R1 bundled veRL]`.
