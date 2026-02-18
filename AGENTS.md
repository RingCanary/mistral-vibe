## Repo Guidelines

- Start each fresh turn by checking README and the latest 5 WORKLOG entries.
- Check git status before edits; do not revert user changes you did not make.
- Before any write/build/exec work, add a WORKLOG entry with `[epoch:<unix_seconds>]`.

## Vendor and Model Agnostic Rules

- Treat provider and model as configuration, never as hardcoded constants.
- Do not hardcode API base URLs, model IDs, auth headers, or vendor env var names in core flows.
- Resolve runtime selection through config aliases (`active_model`, `models`, `providers`).
- Keep vendor-specific logic only in backend adapters/protocol layers (`api_style`, `backend`).
- Prefer provider registries and OpenAI-compatible paths before adding custom code paths.
- Use capability checks before assuming support for tools, streaming, or reasoning features.
- Never silently fall back to a different provider/model; surface fallback behavior in user-facing output.

## Credentials and Security

- Load secrets from env vars or `.env`; never commit real keys/tokens.
- Keep sample configs placeholder-only; rotate/remove exposed credentials immediately.
- Keep MCP/provider authentication declarative in config, not hardcoded in runtime branches.

## Python Implementation Conventions

- Use Python 3.12+ style: `match-case` when applicable, guard clauses, modern type hints (`|`, built-in generics).
- Prefer Pydantic v2 validation (`model_validate`, validators) over ad-hoc parsing.
- Prefer `pathlib.Path` methods over `os.path`.
- Avoid inline suppressions (`# type: ignore`, `# noqa`) unless hidden behind a narrow typed wrapper.
- Run Python tooling via `uv` (`uv run`, `uv add`, `uv sync`), not bare `python`/`pip`.

## Change Hygiene

- If provider/model behavior changes, update README/docs examples in the same task.
- Add or update tests for each touched provider adapter/capability path.
- Keep this file concise and capped at 50 lines.


