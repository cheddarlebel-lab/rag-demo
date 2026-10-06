"""CLI: python -m rag_demo {crawl,index,ask,eval,export}"""
import argparse
import json
import sys


def main(argv=None):
    p = argparse.ArgumentParser(prog="rag_demo")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("crawl", help="polite crawl of the corpus into data/raw")
    c.add_argument("--max-pages", type=int)
    c.add_argument("--refresh", action="store_true", help="ignore the on-disk cache")

    sub.add_parser("index", help="chunk + embed cached pages into data/index")

    a = sub.add_parser("ask", help="retrieve + analyze one question, print JSON")
    a.add_argument("question")
    a.add_argument("-k", type=int)
    a.add_argument("--retrieve-only", action="store_true")

    e = sub.add_parser("eval", help="hit-rate@k / MRR over eval/questions.jsonl")
    e.add_argument("--ks", default="1,3,5,10")
    e.add_argument("--out", default="eval/results.json")
    e.add_argument("--answers", action="store_true",
                   help="also run the end-to-end LLM grounding eval (slow on CPU/laptop)")

    x = sub.add_parser("export", help="precompute example answers for the static site")
    x.add_argument("--out", default="site/data.json")

    args = p.parse_args(argv)
    if args.cmd == "crawl":
        from . import config
        from .crawl import crawl
        crawl(max_pages=args.max_pages or config.MAX_PAGES, refresh=args.refresh)
    elif args.cmd == "index":
        from .index import build_index
        build_index()
    elif args.cmd == "ask":
        from .pipeline import answer, retrieve_only
        out = retrieve_only(args.question, k=args.k) if args.retrieve_only else answer(args.question, k=args.k)
        json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
        print()
    elif args.cmd == "eval":
        from .evaluate import run_eval
        run_eval(ks=[int(k) for k in args.ks.split(",")], out=args.out)
        if args.answers:
            from .answer_eval import run_answer_eval
            run_answer_eval()
    elif args.cmd == "export":
        from .export import export_site
        export_site(out=args.out)


if __name__ == "__main__":
    main()
