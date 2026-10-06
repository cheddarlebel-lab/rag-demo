# Security

This is a demo pipeline, not a hosted service. These notes cover how it handles secrets and
untrusted input, and what it does not protect against. Each control listed here has a test in
`tests/` unless marked otherwise.

## Reporting

Open a GitHub issue that describes the affected area without exploit details, and ask for a private
contact channel.

## Secrets

| Secret | Where it comes from | Where it never goes |
|---|---|---|
| `ANYTHINGLLM_API_KEY` | Environment only | Logs, artifacts, error messages, the repo. The client strips the key from every error message (`test_partial_failure_raises_keeps_old_doc_and_redacts_key`), and the ingest manifest records `<ANYTHINGLLM_BASE_URL>` instead of the server address. |
| `ANTHROPIC_API_KEY` | Environment only, read by the official SDK | Nothing in this repo reads or logs it. |

- `.env` and `.env.*` are git-ignored. Keep AnythingLLM storage, its admin password and API keys outside the
  repository.
- No secret is needed to run the default pipeline, the tests or CI. Tests patch `requests` so any real network
  call fails the test (`tests/conftest.py`).
- The published `site/data.json` contains upstream Flask documentation examples such as placeholder
  `SECRET_KEY` values. They are documentation excerpts, not credentials of this project.

## Threat notes

### Prompt injection via retrieved content

Crawled pages are untrusted. A page can contain text such as "ignore previous instructions".

- The system prompt says that passages are untrusted data and that instructions inside them must not be followed.
- Passages are wrapped in `<passage id="N">...</passage>` delimiters. Passage text that tries to close or forge
  a delimiter is escaped (`test_prompt_injection_boundary`).
- Model output must pass JSON-schema validation (`test_schema_validation_errors_are_clear`). Every citation is
  checked against the passages that were actually supplied, and an empty quote never counts as a match
  (`test_analyze_verifies_citations`).
- Limits: these measures reduce the risk but do not eliminate it. A model can still be steered by
  well-crafted content. The output is advisory text with checked citations. Never wire it to actions such as
  tool calls, shell commands or writes without a separate authorization layer.

### SSRF and scope escape in the crawler

The crawler takes its seed and prefix from configuration. Do not expose it as a service that accepts
arbitrary URLs from untrusted users.

- The seed and every URL are compared with the configured prefix after parsing, in the canonical form that
  `requests` will actually send. Scheme, host, port and path prefix must match. URLs with userinfo, dot
  segments (including encoded ones) or backslashes are rejected. Lookalikes such as `https://host.evil`,
  `/en/stable-old/` and `/en/stable/../../private` fail the check (`test_in_scope`,
  `test_scope_judges_the_url_requests_will_send`).
- Redirects are never followed automatically (`allow_redirects=False`). Each hop is checked against the
  scope and robots.txt and is paced, with at most 5 hops (`test_redirect_*`).
- robots.txt returning 401 or 403 refuses the crawl. A 429 or 5xx backs off and then stops. A redirect that
  leaves the site refuses the crawl (`test_robots_*`).
- Every request, including the robots.txt fetch, waits at least 1 s after the previous one. The floor cannot
  be lowered through configuration (`test_delay_floor_cannot_be_zero`, `test_crawl_paces_every_request...`).
- Not covered: DNS rebinding and private-IP blocking. The prefix host is trusted configuration. If you accept
  untrusted prefixes, also resolve the host and block private and link-local ranges.

### Data integrity

- Each index build writes a new generation directory. It is published with one atomic symlink switch, so an
  interrupted build leaves the previous index in place (`test_build_publishes_generations_atomically`). The
  one-time migration of a pre-0.2 plain index directory is the exception. If it is interrupted, the index
  path can be missing until the next load or build re-links the newest complete generation
  (`test_interrupted_legacy_migration_recovers`).
- Loading checks the chunks fingerprint, the embeddings checksum, the embedding model, row counts, dimensions
  and that every vector is finite (`test_index_validation`, `test_same_row_count_corruption_is_detected`).
- Exports refuse to combine artifacts that came from a different corpus fingerprint, embedding model,
  retriever or LLM (`test_require_compatible`).

### AnythingLLM

- The adapter only calls the developer API routes listed in `rag_demo/anythingllm.py`, and it never follows
  redirects while carrying the API key (`test_redirects_are_refused`).
- It only modifies documents whose `rag-demo:<workspace>:...` marker names the workspace being ingested.
  It replaces a document only after the new copy is embedded, and it never deletes a document that another
  workspace still embeds (`test_ownership_is_per_workspace_and_shared_docs_survive`,
  `test_partial_failure_raises_keeps_old_doc_and_redacts_key`).
- Bind AnythingLLM to `127.0.0.1` or put it behind authentication whenever it can be reached from another
  machine. Its developer API key grants full document and workspace control.
