# Build source and reproducibility

`tools/talon/` preserves all 33 Python scripts from the supplied `V2_4_Review/build_source.zip`, unchanged. The snapshot identified its original source commit as `0063c217dff0145994caa3c01fc7b957b6759da4`. That commit belongs to the original build workspace; it is not a commit in this repository.

This is a source snapshot, **not a complete reproducible cold build**. The extracted input trees, hash tables, downloaded manifests, recall dependencies, intermediate files, and external tooling from the original workspace are not included. Installing Python dependencies alone will not make it build.

## Legacy entry points

- `build_rig.py S`: reads extracted Aemeath Talon and Ellen Samira trees; emits skeleton, mesh, and rig data.
- `build_anim.py S [output-folder]`: retargets and repairs animations using the prepared rig/intermediates.
- `fetch_recall_assets.py S`: retrieves dependencies using previously prepared manifests and CDTB tooling.
- `build_pack.py S OUT.fantome [build-folder]`: assembles the prepared assets and metadata into the package.
- `validate.py S FANTOME [output-folder]`: runs the original package/motion checks against a prepared workspace.

`S` is a separate prepared asset workspace, not this repository root. Scripts assume directories such as `w/`, `x/`, `hashes/`, `dl/`, and `build/`. Imports include NumPy, pyritofile, pyzstd, xxhash, and requests; dependency versions and complete setup were not captured. The scripts are retained for recovery and inspection rather than offered as a supported build command.

Some tools overwrite output folders. In particular, `validate.py` deletes and recreates its selected output directory. Inspect these legacy scripts before running them and use a disposable build workspace.

## Checks available from a fresh clone

```sh
python3 scripts/verify_package.py
```

This standard-library check verifies the recorded SHA-256, archive CRCs, expected Fantome members, PNG signature, WAD signature, and Talon version metadata. Historical deep checks are recorded under `docs/validation/v2.4/`; CI does not claim to reproduce them.
