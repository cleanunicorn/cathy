# cathy — Agent guide

Guidance for AI agents (and humans) contributing to **cathy**.

cathy turns any text file into narrated audio, **fully locally** — no cloud, no
API keys; synthesis runs on your own GPU (or CPU). It is a Python 3.12 CLI
(`uv`-managed) with Kokoro-82M as the default engine and three optional
engines (qwen, chatterbox, fish), each living in its own delegated environment.

A change is "done" when the pytest suite passes. This project follows GitHub
flow: `main` is always releasable, all work happens on short-lived branches,
and every change lands through a pull request. **Merging to `main` ships
nothing** — releases are cut by pushing a version tag (see
[Release automation](#release-automation)).

Read [README.md](README.md) for setup and usage. This file is the operational
checklist for *how to work* here.

## Prerequisites

- **Python 3.12** — pinned by `.python-version`
  (`requires-python = ">=3.12,<3.14"`; the torch cap has no 3.14 wheels).
- **[uv](https://docs.astral.sh/uv/)** — owns the whole dev flow.
- **`gh` CLI** — for opening pull requests.
- Only to actually *narrate* (never needed for tests): `espeak-ng`
  (phonemizer backend), `ffmpeg` (non-wav output), `sox` (qwen engine), and an
  NVIDIA GPU is recommended (~2–3 GB VRAM for Kokoro; falls back to CPU).

## Commands

There is no Makefile; `uv` is the task runner.

- **Install / bootstrap:** `uv sync` — creates `.venv` with the dev group
  (pytest). Note this installs the full dependency set **including torch**
  (a multi-GB download).
- **Run locally:** `uv run cathy book.txt`
- **Test (all):** `uv run pytest -q` — 57 tests, under a second, no GPU and
  no network needed.
- **Test (single file):** `uv run pytest tests/test_normalize.py`
- **Test (focused):** `uv run pytest -k <pattern>`
- **Build:** `uv build` — sdist + wheel (what
  [publish.yml](.github/workflows/publish.yml) runs).
- **Lint / format / type-check:** none are configured. The pytest suite is
  the entire gate — CI runs exactly it.

The torch-free test run (what CI does — useful to skip the multi-GB `uv
sync`):

```bash
uv venv && uv pip install pytest beautifulsoup4 ebooklib numpy soundfile tqdm
PYTHONPATH=src uv run --no-project pytest -q
```

Always run the test suite before opening a PR.

## Golden rules

1. **Never commit directly to `main`.** Always branch, always PR. `main` is
   not protected on GitHub, but every change so far has landed through one.
2. **Never force-push a shared branch.**
3. **Keep `main` green.** Run the tests locally before opening a PR (see
   [Run the checks locally](#5-run-the-checks-locally)).
4. **Use `uv`.** `uv sync` / `uv run` are the single source of the dev flow —
   don't hand-roll pip or venvs.
5. **Never disable, skip, or delete a test to make a build pass.** If a test
   is wrong, say so and propose the fix.
6. **cathy stays fully local.** No cloud services, no API keys, no telemetry.
   The only network traffic cathy may make is first-run downloads of models
   and engine environments (Hugging Face, PyPI, pinned git refs). Never add a
   network-dependent synthesis path. `HF_TOKEN` exists solely so users can
   *download* the gated fish model — it is not a service credential.
7. **Tests stay lightweight.** CI installs only pytest and the small
   text-plumbing deps (see [ci.yml](.github/workflows/ci.yml)), so tests must
   never import torch, engines' heavy modules, or download models.
8. **Dependency pins are deliberate.** torch is capped `<2.9` everywhere
   (newer wheels bundle cu130 → cuDNN mismatch), and the engine extras are
   mutually exclusive environments (`tool.uv.conflicts`). Read the comments
   in [pyproject.toml](pyproject.toml) before touching pins.

## Communication

- Always explain the reasoning behind decisions and approaches.
- When claiming something works or is fixed, prove it — a passing test, a
  script that validates the behavior, or a clear explanation of why. Don't just
  assert.
- When uncertain, say so rather than presenting a guess as fact.
- End each response with a confidence indicator: 🟢 High | 🟡 Medium | 🔴 Low

## The GitHub flow, step by step

### 1. Start from an up-to-date `main`

```bash
git checkout main
git pull origin main
```

### 2. Create a branch

Branch names are short, lowercase, and hyphenated, prefixed by intent where it
helps. Match the Conventional Commits type you expect the PR to use (see
[Commit](#4-commit)):

```
feat/<short-description>      # new feature
fix/<short-description>       # bug fix
refactor/<short-description>  # internal change, no behavior change
perf/<short-description>      # performance work
docs/<short-description>      # documentation only
chore/<short-description>     # tooling, deps, housekeeping
```

Examples: `fix/chunk-cap-gaps`, `feat/folder-convert`. (Existing branches like
`cap-chunk-size` skip the prefix — either style is fine; keep it short and
descriptive.)

### 3. Make focused changes

- One logical change per PR — and a whole feature *is* one logical change.
  Ship its code, tests, and docs together; don't split it across a chain of
  dependent PRs. Don't bundle an unrelated refactor into a fix either.
- Match the surrounding style: type hints throughout, `typing.NamedTuple`
  records, lazy imports for anything heavy, comments that explain *why*.
- Keep diffs focused: everything in the diff should serve that one change.

### 4. Commit

Commits follow [Conventional Commits](https://www.conventionalcommits.org):

```
type(optional-scope): short imperative description
```

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`,
`build`, `ci`, `chore`, `revert`. Add `!` before the colon for a breaking
change.

```
fix: keep checkpoint hashes stable across speed changes
feat(convert): accept a folder of chapter files
```

Write in the imperative mood ("add", not "added"). Keep the subject under ~72
characters and explain the *why* in the body when it isn't obvious. Nothing
enforces this — older history predates it ("Add a Development section to the
README") — but new commits should follow it.

### 5. Run the checks locally

Do not open a PR with these failing — they mirror what CI runs:

```bash
uv run pytest -q
```

That's the whole gate. There is no lint, format, or type-check step.

### 6. Push and open a PR

```bash
git push -u origin <branch>
gh pr create --base main --fill
```

Target **`main`**.

## PR titles

The PR title follows the same Conventional Commits format as commits:

```
type(optional scope)!: description
```

No workflow enforces it — CI (`.github/workflows/`) only runs the pytest
suite, and PR titles drive no automation. Follow the format anyway for a
readable history.

## PR description

Keep it short and useful:

- **What** changed and **why** (the motivation/problem).
- **How to test** / what you ran (`uv run pytest -q`, plus any manual
  narration check that mattered).
- **Linked issues**: `Closes #123` when it resolves one.

## After opening the PR

- Make sure **CI is green** — a single job that runs the pytest suite on
  Linux with the lightweight dependencies.
- Address review feedback by pushing more commits to the same branch.

## Release automation

Merging to `main` ships nothing — it only runs CI. A release goes out when a
version tag is pushed:

- **Is `main` protected?** No.
- **What does merging trigger?** The test suite, nothing else.
- **How a release ships:** bump `version` in
  [pyproject.toml](pyproject.toml), push a `v*` tag (e.g. `v0.3.0`).
  [publish.yml](.github/workflows/publish.yml) then runs `uv build` +
  `uv publish` to PyPI via trusted publishing (environment `pypi`).
- **Does the PR title decide the bump?** No — the version is bumped manually
  in `pyproject.toml`; no automation derives it from PR titles.

## Project map (where things live)

```
src/cathy/
  __init__.py     package docstring; exports main
  cli.py          the bulk: argparse CLI, ebook unpacking, chapter detection,
                  chunking, graded pauses, ffmpeg/convert plumbing,
                  checkpointing (<output>.partial/ + cathy-chapters.json)
  engines.py      per-engine synthesis backends, voice/speaker tables,
                  availability detection, sentence splitting
  normalize.py    text cleanup: front/back-matter skip lists, footnote
                  markers, URLs, glyphs, page numbers
tests/            pytest suite — no GPU, no network
.github/workflows/  ci.yml (pytest on PRs + push to main),
                    publish.yml (PyPI on v* tags)
```

Layering: `cli.py` consumes `engines.py` and `normalize.py`. Each engine
synthesizes one paragraph at a time and exposes a sample rate; everything else
(pauses, chapters, output formats) is shared in the CLI. Heavy imports are
lazy — inside functions — so importing `cathy` stays cheap and the tests can
run without torch.

## Conventions

- **Naming:** modules and functions `snake_case`; records are
  `typing.NamedTuple` (`BookInfo`) or documented tuple aliases
  (`Chapter = tuple[str, list[str]]`).
- **Error handling:** user-facing failures `sys.exit("error: <friendly,
  actionable message>")` — say what's wrong and how to fix it (see the
  existing `--chapters` and ffmpeg messages).
- **Configuration:** module-level constants at the top of `cli.py` (pause
  lengths, format tuples, `MANIFEST_NAME`). Environment variables:
  `CATHY_SOURCE` (where delegated engine environments install cathy from —
  point it at your clone to test local changes), `HF_TOKEN` (gated fish model
  download only), `CATHY_DELEGATED` (internal reentry guard — never set it).
- **Registering engines:** an engine joins `ENGINES` and `ENGINE_IMPORTS`
  (plus its voice tables) in `engines.py`; the CLI discovers everything from
  there.
- **Comments explain why, not what** — the pin rationale in `pyproject.toml`
  is the model.

## Testing

- **Framework / runner:** pytest ≥ 8 (uv dev group).
- **Location & naming:** `tests/test_cli.py`, `tests/test_normalize.py`;
  `TestX` classes with `test_*` methods.
- **What to cover:** text extraction, normalization, chunking, and audio
  plumbing — happy path plus edge cases. The bar: pure-Python, no torch, no
  model imports, no network (CI enforces this by installing only the
  lightweight deps). Build inputs in-test (`tmp_path`, hand-made epub/wav
  fixtures) — nothing real is needed.
- A full run is 57 tests in under a second; the README's "49 tests" is stale —
  don't trust hardcoded counts, run the suite.

## Security

- Never commit secrets, API keys, credentials, or sensitive data. The only
  credential the app knows is `HF_TOKEN`, which lives in the user's
  environment, never in git.
- cathy is local-only: users' books and documents flow through it. Never add
  telemetry, uploads, or any network path beyond first-run downloads.
- Input files (epubs, mobi, html) are untrusted data, parsed in memory —
  keep new parsing defensive.

## Gotchas

- **`uv sync` downloads torch** (multi-GB). For a quick check, mirror CI with
  the torch-free run in [Commands](#commands) instead.
- **The engine extras are mutually exclusive** (`tool.uv.conflicts` — their
  pins genuinely conflict). Run one at a time: `uv run --extra qwen cathy …`;
  never try to install two engines into one environment.
- **Delegated engine runs ignore local changes** unless `CATHY_SOURCE` points
  at the clone: `CATHY_SOURCE=$PWD cathy book.txt -e qwen`. Otherwise the
  engine environment fetches cathy from GitHub.
- **Don't bump torch past the `<2.9` cap** casually — newer PyPI wheels bundle
  cu130 and hit a cuDNN mismatch at inference; repo development additionally
  pins 2.8.0 via `tool.uv.override-dependencies`. On Windows, torch routes to
  the `pytorch-cu128` index (PyPI's win32 wheels are CPU-only) — keep that
  `tool.uv.sources` mapping intact.
- **Python is pinned to 3.12** — the `<3.14` ceiling exists because the torch
  cap has no 3.14 wheels.
- **First narration run downloads ~330 MB** (Kokoro) from Hugging Face;
  everything after that is offline. Tests never trigger this.

## Start here

The fastest path to understanding this codebase:

[README.md](README.md) → [src/cathy/cli.py](src/cathy/cli.py) →
[src/cathy/engines.py](src/cathy/engines.py) →
[src/cathy/normalize.py](src/cathy/normalize.py) →
[tests/test_cli.py](tests/test_cli.py)
