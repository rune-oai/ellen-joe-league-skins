# Known issues and runtime acceptance

Status: **offline validated; in-game acceptance pending**. No League patch is recorded as runtime-verified for v2.4.0.

## Retained limitations

- The recall scissors float off the back to clear the tail, plant with some between-key wobble, and return to idle rather than reproducing Samira's full departure.
- Source weapon effects may depict ghost guns.
- Recall draw contact, shallow ground dips, action-blend penetrations, and vault/roll tail near-contact remain possible in-game concerns.
- The tail uses keyed animation rather than dynamically following turns.
- The inherited voice-over bank is absent. Existing reports describe retained unused/dead resources.
- Nearest-vertex clearance checks are not proofs of triangle-level nonintersection.

## In-game acceptance checklist

Test the exact package whose checksum is recorded in `skins/talon/2.4.0/SHA256SUMS`. Report the League patch, manager version, and result for each item:

- [ ] Import and load base Talon successfully.
- [ ] Inspect idle, attacks, Q/W/E/R, emotes, death, wall vaults, and action transitions.
- [ ] Check normal running and homeguard, including homeguard activation mid-stride.
- [ ] Complete recall and inspect particle timing, appearance, scale, and attachment.
- [ ] Cancel recall early, midway, and late; check cleanup and repeated recalls.
- [ ] Confirm whether recall events at 8.000, 8.100, and 8.267 seconds actually fire during Talon's recall lifecycle.
- [ ] Check scissors/tail ground clearance and collisions in motion.

The original [validation evidence](validation/v2.4/) was supplied with the build. Repository CI checks archive integrity and metadata only; it does not run League, render particles, or rerun the full historical motion audit.
