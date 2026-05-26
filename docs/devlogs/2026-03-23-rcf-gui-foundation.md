# 2026-03-23 RCF GUI Foundation

## Summary

The first usable RCF GUI workflow was brought into the repository and connected to the existing preprocessing pipeline.

## Main changes

- Added the local FastAPI GUI entrypoint and session loading flow.
- Connected GUI state to RCF scan loading, patch review, and export.
- Established the split between workflow view and expert editing behavior.

## Verification

- Local GUI route served successfully.
- Session state and patch geometry updates persisted to `gui_session.json`.

## Notes

- This established the project pattern later extended into versioned sessions and browser-assisted review.
