# Alignment sensitivity QC and stopping the pipeline

Alignment QC is optional and runs after Stage 2.5 frame validation. It evaluates
saved offsets without fitting or publishing offsets, detections, or image XMLs.
The module works on CPU and needs numpy, pandas, scipy, scikit-image, tifffile,
and matplotlib from the existing brain_detector environment.

## Stop after any processing step

Set `stop_after_step` in an inference config, or override it with
`--stop-after-step`. JSON `null` and CLI `none` complete the pipeline. The old
`stop_after_detection` and `stop_before_stitching` options have been removed;
configs containing them are rejected with a migration message.

| Name | Number | Completed work |
| --- | --- | --- |
| setup | 1 | Configuration, tile discovery and runtime provenance |
| detection | 2 | Raw tile detections and input checks |
| filtering | 2.25 | Unaligned pre-align filtering |
| alignment | 2.5 | Saved channel offsets, aligned CSVs and frame validation |
| alignment_qc | 2.55 | Optional alignment sensitivity QC |
| fusion | 2.6 | Optional dual-exposure fusion |
| publish_filtered | 2.75 | Final tile filtering/publication |
| histograms | 2.8 | Optional raw histograms |
| tile_positions | 2.9 | Optional tile geometry and published XMLs |
| image_stitching | 2.95 | Optional direct image stitching |
| z_linking | 3 | Global channel stitching and per-channel Z linking |
| colocalization | 3.5 | Cross-channel matching and source exports |
| exports | 4 | Centroids and optional statistics |

Names are recommended; numeric JSON values and numeric CLI strings also work.
A stop applies after checkpoint reuse as well as newly computed work. Optional
steps must be enabled and applicable to the pipeline mode; this is checked
before processing. Increase the boundary or set null to resume using checkpoints.
Stage 3 performs channel stitching and Z linking together; it stops before any
cross-channel colocalization. `setup` still creates output directories and saves
runtime provenance, but does not infer detections.

```bash
python scripts/run_inference.py --config config/EGFR/2/config_EGFR_t70.json --stop-after-step alignment
python scripts/run_inference.py --config config/EGFR/2/config_EGFR_t70.json --alignment-qc --stop-after-step alignment_qc
python scripts/run_inference.py --config config/EGFR/2/config_EGFR_t70.json --stop-after-step none
```

## Running QC

Set `qc.alignment.enabled: true` in an inference config. `--alignment-qc` and
`--no-alignment-qc` override the config for that invocation. Disabled QC imports
no image-processing modules of its own and does not create QC outputs.
Standalone execution runs the requested QC even when `enabled` is false:

```bash
python -m src.qc.alignment --sample /path/to/detection_results3 --qc-config config_example/alignment_qc.example.json
python -m src.qc.alignment --sample /path/to/detection_results3 --tiles 305500_319100 --workers 2
```

The standalone command prefers runtime_config.json and saved _align_settings.json.
`--config` supplies the original inference config only when the runtime snapshot
is absent. `--qc-config` accepts either the standalone options template or an
inference config containing qc.alignment. Current model/filter/Z-link options do
not override saved alignment settings. `path_mappings` maps original path
prefixes to the current machine; missing image roots also try the same channel
basename beside the result directory. TIFF indices follow acquisition-anchor
slice names, including missing-channel slices. Downsampled detections are rejected.

The reference defaults to the saved measurement reference. Moving soma and TF
channels default to active channels of those types. Set reference_channel,
soma_channels and tf_channels to choose explicit comparisons. No marker IDs or
sample names are hard-coded. The reference must be a soma channel.

## Measurement and sampling

Moving soma channels use box-volume voxel IoU against the fixed soma channel,
including the unaligned baseline. TF channels use median 3D physical distance
between fixed bright-region centers. The feature definition is frozen from the
October 15 T70 experiment: Gaussian XY sigma=1 pixel, no Z smoothing; largest
3D Otsu component within observed boxes; P10-background subtraction; the intensity
centroid of the upper-decile connected component containing its peak. Flat,
nonfinite, saturated, ambiguous and edge-truncated objects are excluded.

Unique baseline containment pairs are frozen across every displacement. Fresh
cohorts use whole observed tracks within the sampling ROI and outside the fit
window plus guard. Translated landmarks may leave that ROI: images are never
resampled and pairs are never reselected during a sweep. This avoids changing
the cohort with the scan radius. Historical T70 runs retain their exact original
cohorts, which had additional common-feature and narrow-sweep eligibility rules.

Defaults: all requested tiles, deterministic seed 42, one 10-slice held-out ROI,
up to 50 candidate windows, target 10 unique pairs per TF channel. Sampling uses
baseline containment counts and records sparse fallback windows when necessary;
feature exclusions can reduce final counts further. No held-out window is an
execution failure, never a silent in-fit fallback. IoU uses a common source-valid
footprint for the entire sweep and the inferred-volume guard. Source windows are
checked against the original alignment fit window in raw coordinates.

Defaults scan X/Y +/-20 um and Z +/-40 um. X/Y use one-pixel steps plus exact
physical endpoints; Z uses one-slice steps and requires an integer-slice radius.
At T70 resolution (0.65, 0.65, 8 um), there are 63 X, 63 Y and 11 Z positions.
Positive extra displacement increases the moving channel coordinate. Each curve
varies one axis while the other two remain at their saved offsets. Overview
curves average per-tile values with equal tile weight; they do not pool nuclei.

## Outputs, resume and failure behavior

Default output: `<results>/qc/alignment/`. `--out-dir` overrides it. Protected
pipeline stage directories and historical alignment_qc directories are rejected
as output targets. Files include per-tile 2-by-3 PNG/PDF plots, overview plots,
curves.csv, summary.csv, sampling.json, qc_report.md and run_details.json. Each
tile also saves fixed pairs/feature coordinates and a provenance manifest.

summary.csv includes zero value, best value, signed extra displacement in native
units and um, boundary-extremum flags and pair counts. Missing data and low pair
counts are explicit statuses. A completed run means computation succeeded; it is
not a biological accuracy PASS. Containment selection uses the saved alignment
and may favor zero. Sharp minima are sensitivity evidence, not an independent
registration accuracy test or a statistical significance test.

Completed tiles resume only when inputs, options, settings, code and outputs
match. Detection/offset inputs use content hashes; consumed TIFFs use size/mtime.
`--force` recomputes QC outputs. Results are saved per tile, so interrupted or
failed tiles can be rerun. Avoid simultaneous jobs targeting the same QC folder.
Standalone failures return a nonzero exit code. Integrated QC logs failures and
continues by default; `fail_on_error: true` aborts. An explicit stop at
alignment_qc returns failure if QC did not complete successfully.

## Reproduce the historical T70 cohort

Use `--previous-landmarks` with the expanded October 15 run_details.json. Saved
settings, offsets and source fingerprints must match. Only the bright-region
coordinates (third historical feature) are scored; fixed pair membership is
unchanged. IoU is recomputed with the historical held-out Z window and the
intersection of its ROI with the expanded valid footprint.

```bash
python -m src.qc.alignment --sample Y:/Fengyi/EGFR_brain/T70/detection_results3 --config config/EGFR/2/alignment_qc_t70_local.json --qc-config config_example/alignment_qc.example.json --previous-landmarks Y:/Fengyi/EGFR_brain/T70/detection_results3/alignment_qc_candidate_metrics_xy20um_z40um_20261015/run_details.json --out-dir qc/alignment_replay
```

`--roi-manifest` can instead reuse sampling.json while extracting fresh features.
It validates that each requested ROI remains valid for the current scan. A
previous-landmark run takes precedence over a separately supplied ROI manifest.
