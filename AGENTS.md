# Repository Guidelines

## Project Structure & Module Organization
This repository is currently documentation-first. The main source file is [`ChatGPT-DetectorClaw 智能体设计.md`](/mnt/c/Songtan/DetectorClaw/ChatGPT-DetectorClaw%20%E6%99%BA%E8%83%BD%E4%BD%93%E8%AE%BE%E8%AE%A1.md), which captures the exported design discussion for the DetectorClaw concept. Keep top-level files limited to core reference documents. If the repo grows, place supporting notes in `docs/`, reusable figures in `assets/`, and helper scripts in `scripts/`.

## Build, Test, and Development Commands
No build system or runnable application is configured in this workspace. Use lightweight document checks instead:

- `ls -la` to confirm the current file set.
- `rg --files` to list tracked content quickly.
- `sed -n '1,120p' "ChatGPT-DetectorClaw 智能体设计.md"` to review document sections from the terminal.
- `wc -w AGENTS.md` to keep contributor docs concise.

Before submitting changes, preview Markdown in your editor and verify headings, links, and code fences render correctly.

## Coding Style & Naming Conventions
Write in clear, instructional Markdown with short sections and descriptive headings. Prefer ASCII file names in `kebab-case` for new files, even though the existing exported source uses mixed English/Chinese naming. Keep examples concrete, use backticks for commands and paths, and avoid overly long paragraphs. When adding folders, use singular, purpose-based names such as `docs/`, `assets/`, and `scripts/`.

## Testing Guidelines
There is no automated test suite yet. Treat documentation review as the validation step: check Markdown rendering, confirm links open correctly, and make sure any example commands match the current repository layout. If you add automation later, document the exact command here and keep test files near the code or script they validate.

## Commit & Pull Request Guidelines
Git history is not available in this exported workspace, so no repository-specific commit convention can be inferred. Use short, imperative commit messages such as `docs: add contributor guide` or `docs: refine detector notes`. Pull requests should include a brief summary, the files changed, and any context needed to review exported chat content or research notes. Add screenshots only when rendered diagrams or layouts change.

## Security & Content Handling
This repository may contain exported conversation material and personal metadata. Redact private details, API keys, and unpublished research data before committing. Prefer sanitized excerpts over raw exports when sharing externally.
