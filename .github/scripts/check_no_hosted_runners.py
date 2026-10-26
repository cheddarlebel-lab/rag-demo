#!/usr/bin/env python3
"""Fail if any job in any workflow can run on a GitHub-hosted runner.

Leo's rule (2026-10-26): zero GitHub-hosted runner minutes in every
cheddarlebel-lab repo. Every job runs on our own self-hosted runners; GitHub
only stores the code and dispatches.

This reads every workflow under .github/workflows and exits 1 when ANY job
could land on a hosted runner: a hosted image name (ubuntu-*, windows-*,
macos-*) anywhere in its runs-on, a runs-on that is not a self-hosted label
list, a fromJSON(vars.X) with no self-hosted fallback (or a hosted one), a
runs-on taken from the matrix, or a job with no runs-on at all.

Stdlib only, so it runs anywhere python3 does:
    python3 .github/scripts/check_no_hosted_runners.py              check this repo
    python3 .github/scripts/check_no_hosted_runners.py --self-test  prove it can fail
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HOSTED_IMAGE = re.compile(r"\b(ubuntu|windows|macos)-(latest|\d[\w.-]*)\b", re.I)
FROMJSON_VAR = re.compile(r"fromJSON\(\s*vars\.(\w+)\s*(?:\|\|\s*'([^']*)')?\s*\)")


def _jobs(text: str) -> dict[str, list[str]]:
    jobs: dict[str, list[str]] = {}
    in_jobs, job_indent, current = False, None, None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            if current is not None:
                jobs[current].append(line)
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            in_jobs = line.rstrip() == "jobs:"
            current, job_indent = None, None
            continue
        if not in_jobs:
            continue
        if job_indent is None:
            job_indent = indent
        if indent == job_indent:
            m = re.match(r"\s*([\w-]+):\s*$", line)
            if not m:
                raise ValueError(f"unexpected line under jobs: {line!r}")
            current = m.group(1)
            jobs[current] = []
        elif current is not None:
            jobs[current].append(line)
    return jobs


def _runs_on(body: list[str]) -> str | None:
    key_indent = None
    for i, line in enumerate(body):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if key_indent is None:
            key_indent = indent
        if indent != key_indent:
            continue
        m = re.match(r"\s*runs-on:\s*(.*)$", line)
        if not m:
            continue
        value = m.group(1).split(" #")[0].strip()
        if value:
            return value
        items = []
        for nxt in body[i + 1:]:
            if not nxt.strip():
                continue
            if len(nxt) - len(nxt.lstrip(" ")) <= key_indent:
                break
            items.append(nxt.strip())
        return " ".join(items)
    return None


def _self_hosted_list(literal: str) -> bool:
    try:
        value = json.loads(literal)
    except ValueError:
        return False
    return isinstance(value, list) and "self-hosted" in value and all(
        isinstance(x, str) and not HOSTED_IMAGE.search(x) for x in value)


def problems(text: str) -> list[str]:
    out = []
    jobs = _jobs(text)
    if not jobs:
        out.append("no jobs found (the reader did not understand this workflow)")
    for job, body in jobs.items():
        if any(re.match(r"\s*uses:\s*\S+\.ya?ml", ln) for ln in body):
            out.append(f"{job}: calls a reusable workflow; its runner is not checked here")
            continue
        value = _runs_on(body)
        if value is None:
            out.append(f"{job}: no runs-on")
            continue
        if HOSTED_IMAGE.search(value):
            out.append(f"{job}: names a GitHub-hosted image: {value}")
            continue
        if "${{" in value:
            if "matrix." in value.replace("matrix.os == ", "") and "fromJSON" not in value:
                out.append(f"{job}: runs-on comes from the matrix: {value}")
                continue
            calls = FROMJSON_VAR.findall(value)
            if not calls:
                out.append(f"{job}: runs-on is an expression with no fromJSON(vars.X || '[...]'): {value}")
            for var, fallback in calls:
                if not fallback:
                    out.append(f"{job}: fromJSON(vars.{var}) has no self-hosted fallback")
                elif not _self_hosted_list(fallback):
                    out.append(f"{job}: vars.{var} falls back to {fallback}, not a self-hosted label list")
            for lit in re.findall(r"'([^']*)'", FROMJSON_VAR.sub("", value)):
                if lit.startswith(("[", '"')) and not _self_hosted_list(lit):
                    out.append(f"{job}: runs-on can resolve to {lit}")
            continue
        labels = value.strip("[]").replace('"', "").replace("'", "")
        labels = [x.strip(" -") for x in re.split(r"[,\s]+", labels) if x.strip(" -")]
        if "self-hosted" not in labels:
            out.append(f"{job}: runs-on {value} is not a self-hosted label list")
    return out


GOOD = '    runs-on: [self-hosted, Linux, X64, ci-linux]\n'
PLANTED = {
    "plain image": "    runs-on: ubuntu-latest\n",
    "pinned image": "    runs-on: ubuntu-24.04\n",
    "windows image": "    runs-on: windows-latest\n",
    "mac image": "    runs-on: macos-14\n",
    "image in a list": "    runs-on: [self-hosted, ubuntu-latest]\n",
    "hosted fallback": "    runs-on: ${{ fromJSON(vars.RUNS_ON || '\"ubuntu-latest\"') }}\n",
    "no fallback": "    runs-on: ${{ fromJSON(vars.RUNS_ON) }}\n",
    "matrix": "    runs-on: ${{ matrix.os }}\n",
    "not self-hosted": "    runs-on: [ci-linux]\n",
    "missing": "    timeout-minutes: 5\n",
}


def self_test() -> int:
    def wf(bad: str) -> str:
        return ("name: x\non: push\njobs:\n  ok:\n" + GOOD + "    steps:\n      - run: true\n"
                "  bad:\n" + bad + "    steps:\n      - run: true\n")
    failed = 0
    for name, snippet in PLANTED.items():
        found = problems(wf(snippet))
        if not found or not all(p.startswith("bad:") for p in found):
            print(f"SELF-TEST FAIL: planted '{name}' not caught: {found}")
            failed += 1
    good = ("name: x\non: push\njobs:\n  a:\n" + GOOD +
            "  b:\n    runs-on:\n      - self-hosted\n      - macmini\n"
            "  c:\n    runs-on: ${{ fromJSON(vars.RUNS_ON || '[\"self-hosted\",\"ci-linux\"]') }}\n")
    if problems(good):
        print(f"SELF-TEST FAIL: good forms flagged: {problems(good)}")
        failed += 1
    print(f"self-test: {len(PLANTED)} planted hosted runners, {len(PLANTED) - failed if failed <= len(PLANTED) else 0} caught; good forms {'pass' if not problems(good) else 'FLAGGED'}")
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    root = Path(__file__).resolve().parents[2]
    wfs = sorted((root / ".github" / "workflows").glob("*.y*ml"))
    if not wfs:
        print("no workflows found")
        return 1
    bad = 0
    for p in wfs:
        found = problems(p.read_text(encoding="utf-8"))
        for f in found:
            print(f"{p.name}: {f}")
        bad += len(found)
    if bad:
        print(f"FAIL: {bad} job(s) could run on a GitHub-hosted runner")
        return 1
    print(f"ok: every job in {len(wfs)} workflow(s) runs on a self-hosted runner")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
