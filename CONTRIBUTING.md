# Contributing

Thanks for working on EchOnyx. The repo is moving toward a 1.0 line, so keep changes focused, tested, and aligned with the support matrix in [docs/1.0-READINESS.md](docs/1.0-READINESS.md).

## Development Setup

### Backend

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload
```

Run a worker in another shell:

```bash
cd backend
uv run celery -A app.workers.celery_app worker --loglevel=info -Q video_processing,batch_processing,default
```

Apple Silicon host runs should use:

```bash
uv run celery -A app.workers.celery_app worker --pool=solo --concurrency=1 --loglevel=info
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Canonical Checks

Backend:

```bash
cd backend
uv run pytest tests
```

Frontend:

```bash
cd frontend
npm run lint && npx tsc --noEmit && npm run test && npm run build
```

`npm run test` is included here because an in-flight task is adding the script. Until that lands, call out that the test script is unavailable rather than silently skipping frontend tests.

## Acceptance Script

Use the repository acceptance script for end-to-end API checks:

```bash
ECHONYX_PASSWORD='<admin-password>' \
scripts/acceptance.sh \
  --base-url http://127.0.0.1:8000 \
  --primary-fixture /path/to/fixture-one.mp4 \
  --secondary-fixture /path/to/fixture-two.mp4 \
  --search-query "budget review" \
  --ask-question "When is the budget review due?" \
  --ask-expects "Friday" \
  --run-batch
```

For secured deployments, use `ECHONYX_PASSWORD` or `--password`. Do not hardcode secrets in repo files or paste real tokens into logs, issues, PRs, or screenshots.

## CI Expectations

CI is expected to run backend tests, backend lint gates, frontend lint/typecheck/build, frontend Docker build smoke, and Compose config validation on pushes and pull requests to `main`. If a change affects runtime behavior, settings, queueing, ranking, auth, or public API contracts, add or update tests in the same change.

When a check is environment-blocked, say exactly what was blocked and why. Do not hide failed attempts.

## Commit Style

Recent history uses conventional-ish subjects:

```text
feat: add guided model downloads
fix: repair setup script defaults
docs: add readiness audit
chore: refresh nvidia vision baseline
ci: add GitHub Actions pipeline
```

Use `type: summary`, keep the first line concise, and keep one behavior change per commit when practical.

## Code Style

- Backend Python targets Python 3.11.
- Ruff is configured in `backend/pyproject.toml` with line length `100` and lint families `E`, `F`, `I`, `N`, `W`, and `UP`.
- Prefer existing FastAPI route, Pydantic schema, SQLAlchemy model, and Celery task patterns over introducing a new style.
- Keep settings changes wired through `.env.example`, Settings APIs, docs, and tests when behavior changes.
- Frontend code uses the existing Next.js, React, Tailwind, TanStack Query, and local component patterns.
- Keep frontend controls accessible, responsive, and consistent with the existing design primitives.

## Documentation

Update docs when you change startup, deployment assumptions, worker behavior, settings, acceptance checks, operator workflow, or user-visible features. The main docs are:

- [README.md](README.md)
- [backend/README.md](backend/README.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/1.0-READINESS.md](docs/1.0-READINESS.md)
- [CHANGELOG.md](CHANGELOG.md)
- [ROADMAP.md](ROADMAP.md)
