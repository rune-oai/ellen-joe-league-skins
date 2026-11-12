# Installation

## Download and import

Download the `.fantome` asset from the [Talon v2.4 prerelease](https://github.com/runefoo/ellen-joe-league-skins/releases/tag/talon-v2.4.0). GitHub's source-code ZIP is the repository snapshot, not the installable mod.

[LTK Manager](https://github.com/LeagueToolkit/ltk-manager) supports importing Fantome packages; consult its [official documentation](https://github.com/LeagueToolkit/docs/blob/main/docs/LTK-Manager.md) for current setup and import instructions. [CSLoL Manager](https://github.com/LeagueToolkit/cslol-manager) is a legacy Fantome-compatible manager whose upstream now directs development to LTK Manager. This skin has not been runtime-tested with either manager in this repository.

Import the package using your manager's import control, enable it, disable other Talon replacements, and follow the manager's launch instructions. Choose **Talon's base skin**. Start in Practice Tool and follow the [acceptance checklist](KNOWN_ISSUES.md).

## Verify the download

Download `SHA256SUMS` alongside the package. In the folder containing both files:

```sh
# Linux
sha256sum -c SHA256SUMS
# macOS
shasum -a 256 -c SHA256SUMS
```

PowerShell:

```powershell
Get-FileHash .\Ellen_Joe_Talon_V2_4.fantome -Algorithm SHA256
```

Compare the result with `SHA256SUMS`. A matching checksum confirms file integrity; it does not establish client compatibility.

## Troubleshooting

- **No model change:** confirm base Talon is selected, the package is enabled, and no other Talon mod is active.
- **Import failure:** download the `.fantome` release asset again and verify its checksum.
- **Broken animation or recall effects:** record the League patch, manager version, reproduction steps, and a short clip in a [bug report](https://github.com/runefoo/ellen-joe-league-skins/issues/new?template=bug_report.yml).
- **Failure after a League update:** disable the package and report the patch; no current-patch compatibility guarantee is recorded.
