#!/usr/bin/env python3
"""Profile-driven Pre-PR Checklist and evidence validator for GitHub Pull Requests.

[Role]
Validates that a Pull Request description completes required profile-specific
machine-stable checklist IDs and supplies concrete, non-placeholder evidence
plus Summary and Verification sections.

[Invariants]
- Pure stdlib (`< 150` lines); zero third-party or network dependencies.
- Strips HTML comments and fenced code blocks before parsing headings/items.
- Rejects unchecked boxes, duplicate IDs, missing profile IDs, template/angle-bracket
  placeholders, and `not-applicable: docs-only` when runtime paths are modified.

[Timeouts & Locks]
- Pure in-memory regex parsing (`< 50ms`); acquires no locks.

[Side-Effect Policy]
- Strictly read-only over `--body-file`, `PR_BODY`, or `GITHUB_EVENT_PATH`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

PROFILES: dict[str, tuple[str, ...]] = {
    "hub": ("HUB-ROUTE", "HUB-DOC", "HUB-RELEASE", "HUB-GATE"),
    "harness": ("HAR-ABI", "HAR-UPGRADE", "HAR-PROC", "HAR-STATE"),
    "host-plugin": ("HOST-NATIVE", "HOST-WIRE", "HOST-ADMISSION", "HOST-UPGRADE"),
    "control-plane": ("CTRL-LEASE-OR-AUTH", "CTRL-HANDLE", "CTRL-STREAM-IO", "CTRL-RECEIPT"),
    "state-service": ("SVC-CONSENT", "SVC-STATE-COMPAT", "SVC-LIFETIME", "SVC-EXPORT"),
    "device-tool": ("DEV-ZERO-CTX", "DEV-READONLY-PATH", "DEV-TOPOLOGY", "DEV-BOUNDED-MEM"),
    "bench": ("BENCH-ORACLE", "BENCH-VERDICT", "BENCH-PROVENANCE", "BENCH-ISOLATION"),
    "corpus": ("CORPUS-SCHEMA", "CORPUS-EVIDENCE", "CORPUS-REDACT", "CORPUS-REVISION"),
}
DEFAULT_PROFILE = "device-tool"
PLACEHOLDERS = frozenset({"todo", "tbd", "n/a", "na", "none", "null", "-", "无", "略", "待补充"})
COMMENT_OR_FENCE_RE = re.compile(r"<!--.*?-->|^\s*(?:```|~~~).*?^\s*(?:```|~~~)[^\n]*", re.DOTALL | re.MULTILINE)
HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
BOX_RE = re.compile(r"^\s*-\s*\[(?P<mark>[ xX])\]\s+(?:\*\*)?(?P<id>[A-Z]+(?:-[A-Z0-9]+)+)?(?:\*\*)?(?P<rest>.*)$", re.MULTILINE)
EVIDENCE_RE = re.compile(r"^\s*(?:[-*+]\s+)?(?:\*\*)?(?P<id>[A-Z]+(?:-[A-Z0-9]+)+)(?:\*\*)?\s*[：:]\s*(?P<val>.+?)\s*$", re.MULTILINE)
DOC_PATH_RE = re.compile(r"^(?:docs/|README(?:\.[a-zA-Z-]+)?\.md$|AGENTS\.md$|CONTRIBUTING(?:\.[a-zA-Z-]+)?\.md$|LICENSE$|\.github/pull_request_template\.md$)")


def _find_section(sections: dict[str, str], keywords: tuple[str, ...]) -> str | None:
    for title, content in sections.items():
        if any(k.lower() in title.lower() for k in keywords):
            return content
    return None


def _substantive_text(section: str) -> str:
    kept: list[str] = []
    for raw in section.splitlines():
        line = re.sub(r"^[-*+>]\s*", "", raw.strip()).strip("` ")
        if not line or line.lower() in PLACEHOLDERS or (line.startswith("<") and line.endswith(">")):
            continue
        kept.append(line)
    return " ".join(kept)


def validate_pr_body(raw_body: str, profile: str = DEFAULT_PROFILE, changed_files: list[str] | None = None) -> list[str]:
    if profile not in PROFILES:
        return [f"Unknown profile '{profile}'; expected one of {sorted(PROFILES)}."]
    cleaned = COMMENT_OR_FENCE_RE.sub("", raw_body or "")
    if not cleaned.strip():
        return ["PR description is empty; fill out .github/pull_request_template.md."]
    matches = list(HEADING_RE.finditer(cleaned))
    sections = {m.group(1).strip(): cleaned[m.end() : (matches[i + 1].start() if i + 1 < len(matches) else len(cleaned))].strip() for i, m in enumerate(matches)}
    errors: list[str] = []
    summary = _find_section(sections, ("变更摘要", "Summary"))
    if summary is None or len(_substantive_text(summary)) < 10:
        errors.append("Section '## 变更摘要 (Summary)' is missing or lacks substantive description (>= 10 chars).")
    review = _find_section(sections, ("设计原则 Review", "Pre-PR Checklist", "Review"))
    if review is None:
        errors.append("Section '## 设计原则 Review (Pre-PR Checklist)' is missing.")
    else:
        checked_ids: list[str] = []
        for m in BOX_RE.finditer(review):
            cid = m.group("id") or m.group("rest").strip()[:24]
            if m.group("mark") == " ":
                errors.append(f"Unchecked checklist item: '{cid}'.")
            elif m.group("id"):
                if m.group("id") in checked_ids:
                    errors.append(f"Duplicate checklist ID '{m.group('id')}'.")
                checked_ids.append(m.group("id"))
        for req in PROFILES[profile]:
            if req not in checked_ids:
                errors.append(f"Missing checked required ID '{req}' for profile '{profile}'.")
    evidence_sec = _find_section(sections, ("证据记录", "Evidence")) or (review or "")
    ev_map: dict[str, str] = {}
    for m in EVIDENCE_RE.finditer(evidence_sec):
        eid, val = m.group("id"), m.group("val").strip().strip("`")
        if eid in ev_map:
            errors.append(f"Duplicate evidence entry for '{eid}'.")
        ev_map[eid] = val
    docs_only = bool(changed_files) and all(DOC_PATH_RE.search(p.replace("\\", "/")) for p in changed_files)
    has_runtime_changes = bool(changed_files) and not docs_only
    for req in PROFILES[profile]:
        val = ev_map.get(req, "")
        low = val.lower()
        if not val or low in PLACEHOLDERS or (val.startswith("<") and val.endswith(">")):
            errors.append(f"Missing or placeholder evidence for '{req}'.")
        elif low.startswith("not-applicable"):
            reason = val.split(":", 1)[1].strip() if ":" in val else ""
            if len(reason) < 4 or (reason.startswith("<") and reason.endswith(">")):
                errors.append(f"'{req}' uses 'not-applicable' without a concrete reason ('not-applicable: <reason>').")
            elif has_runtime_changes and "docs-only" in reason.lower():
                errors.append(f"'{req}' cannot claim 'not-applicable: docs-only' when non-doc files changed.")
        elif len(val) < 8:
            errors.append(f"Evidence for '{req}' is too brief (< 8 chars); record command/fixture and outcome.")
    verif = _find_section(sections, ("验证结果", "Verification"))
    if verif is None or len(_substantive_text(verif)) < 10:
        errors.append("Section '## 验证结果 (Verification)' is missing or lacks concrete test outcomes (>= 10 chars).")
    return errors


def _load_body(body_file: str | None) -> str:
    if body_file:
        return Path(body_file).read_text(encoding="utf-8")
    if "PR_BODY" in os.environ:
        return os.environ["PR_BODY"]
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path and Path(event_path).is_file():
        payload = json.loads(Path(event_path).read_text(encoding="utf-8"))
        return str((payload.get("pull_request") or {}).get("body") or "")
    raise SystemExit("No PR body source provided (use --body-file, PR_BODY, or GITHUB_EVENT_PATH).")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate profile-driven Pre-PR Checklist and evidence.")
    parser.add_argument("--profile", default=os.environ.get("PR_CHECKLIST_PROFILE", DEFAULT_PROFILE), choices=sorted(PROFILES))
    parser.add_argument("--body-file", help="Path to markdown file containing PR body.")
    parser.add_argument("--changed-file", action="append", default=[], help="Changed repo-relative path (repeatable).")
    args = parser.parse_args(argv)
    errors = validate_pr_body(_load_body(args.body_file), profile=args.profile, changed_files=args.changed_file)
    if errors:
        print(f"Pre-PR Checklist validation failed (profile={args.profile}):\n" + "\n".join(f"  - {e}" for e in errors), file=sys.stderr)
        return 1
    print(f"Pre-PR Checklist validation passed (profile={args.profile}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
