# Automatic pipeline resume

Run the same command for new and interrupted samples:

```bash
python -u scripts/run_inference.py --config config/EGFR/2/config_EGFR_t70.json
```

`start_from_stage` is removed from the sample configs. The entry point discovers
expected tiles from image folders, checks completed artifacts, and computes missing
work. If the image root is unavailable it can discover already recorded raw tiles;
this fallback cannot discover tiles which never produced a CSV. Image-dependent
stages still require their image folders. `stop_after_step` selects a named boundary or stage number;
`null` completes the pipeline. Use `"detection"` for raw detection or
`"tile_positions"` for enabled tile solving and XML publication. See
[all stop boundaries](alignment_qc.md).

## Inputs and storage

For `pre_align`, ordinary channels go from `1_2d_raw` to `2_2d_filtered` to
`3_2d_aligned`, then directly into global stitching. They no longer create duplicate
CSVs in `4_2d_filtered`. Double-exposure channels use `3_2d_aligned_fusion` followed
by final filtering in `4_2d_filtered`. Post-align channels still use `4_2d_filtered`.
Mixed-exposure runs route each channel independently. Saved-result viewers and QC
use the same routing; old directories are not deleted.

## Checkpoints

| Stage | Unit and completion evidence | Recovery |
|---|---|---|
| Detection | Tile/channel CSV and existing input manifest | Skip valid results; detect missing channels; report corrupted completed artifacts |
| Filter | Tile/channel filter manifest | Compare only that channel's filter parameters and source content; preserve saved alignment on conflict |
| Alignment | Tile alignment manifest, offsets, CSV row counts and frame | Skip matching results; complete missing groups; stop on changed complete artifacts/settings |
| Fusion/final filter | Tile/channel manifests | Reuse or rebuild the affected channel |
| Tile positions | `_solver_manifest.json` | Keep solved positions if XML generation fails |
| XML | `_pipeline_manifest.json` | Regenerate XML without rerunning a valid solver checkpoint |
| Global 2D | `<channel>_global_manifest.json` | Commit before Z-Link; resume per channel |
| Z-Link | `<channel>_tracks_manifest.json` | Reuse global 2D when linking fails or link settings change |
| Colocalization | `_provenance_manifest.json` | Validate upstream stages first, then reuse or rebuild |
| Exports | `_exports_manifest.json` | Rebuild missing centroids/reports from saved colocalization |

Files with hashes compare by content, so timestamp changes and mount-point changes
alone do not invalidate them. Complete alignment results are protected because
they may have been reviewed manually: conflicting data/settings produce a named
error, not an automatic overwrite. Archive the affected stage and its dependents
before intentionally changing reviewed alignment. This is artifact protection,
not an assertion that the pipeline has performed a human accuracy review.

Filter settings are excluded from raw inference comparisons. Alignment's temporary
linking parameters live in `pre_align_params.z_linker`; downstream tracking and
colocalization retain the top-level `z_linker`. EGFR and TSC configs explicitly
preserve their existing alignment values in the new nested field. Legacy configs
without that field retain the old fallback to top-level values.

## Configuration history and legacy results

When `runtime_config.json` is missing, the current `--config` supplies a new runtime
snapshot. If present, `alignment_runtime_config.json` supplies historical alignment
context. `resume_bootstrap_config.json` records the adoption and explicitly marks
historical raw origin as unverified; it does not invent the original model hash.
Each invocation also saves a resolved snapshot under `run_configs/`. Existing stage
manifests remain the authority for their own outputs. Complete legacy combined
Stage 3 manifests can be split without recalculating their valid 2D or 3D outputs.

Updates to alignment checkpoint handling in this release explicitly recognize the
previous aligner fingerprint because its numerical algorithm is unchanged. Future
numerical changes must update compatibility/version checks, not silently accept
old algorithm fingerprints.

## Windows PC: resume T70

From the project root in PowerShell, run:

```powershell
cmd /d /c scripts\inference_t70_local.cmd
```

The CMD wrapper selects this PC's `%LOCALAPPDATA%\miniconda3\envs\brain_detector\python.exe`.
Alternatively, activate the `brain_detector` environment and run
`python -u scripts/inference_t70_local.py` directly. `/d` skips CMD AutoRun hooks.

The launcher reads `config/EGFR/2/config_EGFR_t70.json`, maps the four image
folders and `detection_results3` to the accessible Windows sample folder, and
writes the resolved config to `detection_results3/local_config.json` (replaced
on each launch). Console logs remain under `scripts/logs/t70_local_*.log`.
It retains the existing inference/filter/alignment parameters and uses the normal
automatic resume entry point. Missing runtime metadata is handled by that entry
point. By default alignment/tile solving use four CPU workers and Stage 3 runs
one channel at a time to reduce peak memory.

```powershell
# Check paths and generate the local config without starting the pipeline:
cmd /d /c scripts\inference_t70_local.cmd --prepare-only
# Adjust CPU workers; Stage 3 remains one channel at a time:
cmd /d /c scripts\inference_t70_local.cmd --workers 4 --stage3-workers 1
# Optional alternate sample storage (must contain all images and results):
cmd /d /c scripts\inference_t70_local.cmd --sample-dir D:\EGFR_brain\T70
```

Computation runs on the PC; the default sample folders still reside on the Y:
network share, which must stay reachable. Keep the console open during the run.
