# 2026-03-25 Playwright Session Broker

## Summary

Project-specific browser control was formalized around a fixed `playwright-cli` session and persisted session records.

## Main changes

- Added a CLI-first browser broker around `playwright-cli`.
- Persisted active browser session metadata to `.detectorclaw/browser_session.json`.
- Added append-only browser history and doctor/status commands.
- Standardized the project on the `rcf-live` named browser session.

## Verification

- `live-open`, `live-shot`, `live-attach`, `live-session`, and `live-doctor` were exercised locally.
- The browser could be reattached across turns using the stored active session metadata.

## Notes

- The skill-level guidance was updated to match the project session-management pattern.
