"""CLI: python -m rag_demo {crawl,index,ask,eval,export,anythingllm}"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .log import configure


def positive_int(s: str) -> int:
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {s!r}") from None
    if v < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {v}")
    return v


def int_list(s: str) -> list[int]:
    out = [positive_int(x) for x in s.split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return out


def _print(obj: Any) -> None:
    json.dump(obj, sys.stdout, indent=2, ensure_ascii=False)
    print()


def build_parser() -> argparse.ArgumentParser:
    from . import config

    p = argparse.ArgumentParser(prog="rag_demo")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("crawl", help="polite crawl of the corpus into data/raw")
    c.add_argument("--max-pages", type=positive_int)
    c.add_argument("--refresh", action="store_true", help="ignore the on-disk cache")

    sub.add_parser("index", help="chunk + embed cached pages into data/index (atomic)")

    a = sub.add_parser("ask", help="retrieve + analyze one question, print JSON")
    a.add_argument("question")
    a.add_argument("-k", type=positive_int)
    a.add_argument("--mode", choices=config.RETRIEVERS)
    a.add_argument("--retrieve-only", action="store_true")

    e = sub.add_parser("eval", help="hit-rate@k / MRR@depth over eval/questions.jsonl")
    e.add_argument("--ks", type=int_list, default=[1, 3, 5, 10])
    e.add_argument("--depth", type=positive_int, help=f"candidate depth (default {config.CANDIDATE_DEPTH})")
    e.add_argument("--out", default="eval/results.json")
    e.add_argument("--answers", action="store_true",
                   help="also run the end-to-end LLM grounding eval (slow on CPU/laptop)")

    x = sub.add_parser("export", help="precompute example answers for the static site")
    x.add_argument("--out", default="site/data.json")

    al = sub.add_parser("anythingllm", help="AnythingLLM workspace: ingest, ask, eval")
    alsub = al.add_subparsers(dest="alcmd", required=True)
    for sp in (alsub.add_parser("ingest", help="upload chunks as raw-text docs and embed them in the workspace"),
               alsub.add_parser("ask", help="query-mode chat; its sources go to this repo's LLM layer"),
               alsub.add_parser("eval", help="page/section hit@k via workspace vector-search")):
        sp.add_argument("--workspace", default=config.ANYTHINGLLM_WORKSPACE)
    ing = alsub.choices["ingest"]
    ing.add_argument("--limit", type=positive_int, help="ingest only the first N chunks (documented subset)")
    ing.add_argument("--dry-run", action="store_true", help="show the plan without uploading")
    ing.add_argument("--no-prune", action="store_true", help="keep rag-demo docs for chunks that no longer exist")
    ask = alsub.choices["ask"]
    ask.add_argument("question")
    ask.add_argument("--format", choices=("json", "markdown"), default="json")
    ask.add_argument("--save", help="also write the JSON result to this path")
    alsub.choices["eval"].add_argument("--out", default="eval/anythingllm_results.json")
    return p


def main(argv: list[str] | None = None) -> int:
    configure()
    args = build_parser().parse_args(argv)
    from .index import IndexValidationError
    from .llm import LLMOutputError

    try:
        return _run(args)
    except (IndexValidationError, LLMOutputError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:  # AnythingLLMError, CrawlRefused, CrawlStopped, IncompatibleArtifacts
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    if args.cmd == "crawl":
        from . import config
        from .crawl import crawl
        crawl(max_pages=args.max_pages or config.MAX_PAGES, refresh=args.refresh)
    elif args.cmd == "index":
        from .index import build_index
        build_index()
    elif args.cmd == "ask":
        from .pipeline import answer, retrieve_only
        _print(retrieve_only(args.question, k=args.k, mode=args.mode) if args.retrieve_only
               else answer(args.question, k=args.k, mode=args.mode))
    elif args.cmd == "eval":
        from .evaluate import run_eval
        res = run_eval(ks=args.ks, out=args.out, depth=args.depth)
        _print({k: res[k] for k in ("dense_nomic", "bm25_baseline", "hybrid_rrf")})
        if args.answers:
            from .answer_eval import run_answer_eval
            _print(run_answer_eval())
    elif args.cmd == "export":
        from .export import export_site
        export_site(out=args.out)
    elif args.cmd == "anythingllm":
        from . import anythingllm as al
        client = al.AnythingLLMClient()
        if args.alcmd == "ingest":
            _print(al.ingest(client, args.workspace, limit=args.limit, dry_run=args.dry_run,
                             prune=not args.no_prune))
        elif args.alcmd == "ask":
            r = al.ask(client, args.question, args.workspace)
            if args.save:
                from pathlib import Path
                Path(args.save).write_text(json.dumps(r, indent=2, ensure_ascii=False))
            if args.format == "markdown":
                sys.stdout.write(al.to_markdown(r))
            else:
                _print(r)
        elif args.alcmd == "eval":
            res = al.run_eval(client, args.workspace, out=args.out)
            _print({k: v for k, v in res.items() if k != "per_question"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
