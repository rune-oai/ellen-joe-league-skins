# Contributing

Keep contributions focused on Ellen Joe skins for League of Legends.

## Report a bug

Use the bug report template. Include champion, package version, League patch, mod manager/version, reproduction steps, and expected versus actual behavior. Screenshots or a short clip help with animation issues. Distinguish in-game observations from offline inspection.

## Submit a skin or fix

1. Open an issue describing the champion and intended change.
2. Keep original creator credits and identify the source and permissions for new assets.
3. Store packages under `skins/<champion>/<version>/`, with metadata and SHA-256 checksums.
4. Update the README catalog, changelog, and known issues with measured validation results.
5. For changes to the existing Talon package, run `python3 scripts/verify_package.py`.
6. Submit a pull request describing the change and the checks actually performed.

Do not include local game installations, credentials, unrelated character skins, or uncredited source assets. Preserve earlier release downloads when publishing a new version.
