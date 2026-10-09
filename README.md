# brain_detector

A tile-based light-sheet microscopy pipeline for cell detection, channel alignment, 3D Z linking, and multi-channel colocalization. The primary workflow detects cells on raw channel images (`pre_align`) and estimates alignment from detections. [`solve_tile_positions.py`](src/core/solve_tile_positions.py) can also estimate tile positions from cells shared by neighboring tiles, without running TeraStitcher.

In `pre_align`, set `tile_position_params.enabled: true` to solve global tile positions from filtered detections inside the pipeline. The solver exports a frame XML for Stage 3 and one `xml_merging.xml` into each original channel image directory. A structural `xml_import.xml` or existing `xml_merging.xml` is required in each channel directory; the first publication preserves it as `xml_merging.original.xml`.

## Start here

1. Copy [`config_example/detection_config.example.json`](config_example/detection_config.example.json) to `config/config.json` and [`config_example/visualization_config.example.json`](config_example/visualization_config.example.json) to `config/vis_config.json`.
2. Replace the example image, model, and result paths. `config/` is local and ignored by Git; the files in `config_example/` are generic templates. Configuration files accept `//` comments.
3. Set `pipeline_mode: "pre_align"` and `tile_position_params.enabled: true` to solve tile positions from detections. Set `stop_before_stitching: true` to stop after the XMLs are written, before Stage 3.
4. Run detection and inspect the aligned tile results:

```bash
python scripts/run_inference.py --config config/config.json
python src/utils/visualize.py --config config/vis_config.json --mode 2d --2d-source filtered
```

The viewer's `2d` mode reads `4_2d_filtered/` and shifts the displayed raw images into the same tile-local frame. It does not load XML.

The expected image layout is a channel directory containing row directories and tile directories, for example `<channel>/305500/305500_319100/*.tif`. Active channel IDs, their types, models, and directory keys are defined in `channels_routing`.

## Processing stages

| Stage | What it does | Main output |
| --- | --- | --- |
| 2 | Detect 2D soma or TF boxes independently in each raw channel and tile | `1_2d_raw/<tile>_<channel>_result.csv` |
| 2.25 | Filter raw per-tile detections before channel alignment in pre_align | `2_2d_filtered/` |
| 2.5 | In `pre_align`, estimate per-tile channel shifts from filtered detections and write shifted CSVs | `3_2d_aligned/<tile>_offsets.json` and aligned CSVs |
| 2.6 | Optionally fuse two exposures of one logical channel, one Z slice at a time | `3_2d_aligned_fusion/` |
| 2.75 | Publish final filtered tile CSVs for Stage 3 | `4_2d_filtered/` |
| 2.8 | Optionally save area and intensity histograms | `1_2d_raw/histograms/` |
| 2.9 | Optionally solve tile positions and publish channel XMLs | Positions in `5_2d_global/tile_positions/`; one `xml_merging.xml` in each channel image directory |
| 3 | Place filtered boxes in global coordinates, link detections across Z, and colocalize channels | `5_2d_global/`, `6_3d_global/`, `7_colocalization/` |
| 4 | Save cell centroids and optional summary statistics | `7_colocalization/cell_centroids/` |

`stop_after_detection: true` exits after Stage 2. With the tile solver enabled, `stop_before_stitching: true` exits after Stage 2.9 has written the XMLs. Without it, the stop point remains after tile filtering. Completed CSV and PKL checkpoints are reused only when their provenance manifests match the source files and settings. Keep `runtime_config.json` with the results: it records the original run, later runs, model and code hashes, and coordinate settings.

Filtering has one path and no `stage_2_75_enabled` switch. In `pre_align`, raw detections are filtered once into `2_2d_filtered/` before channel alignment. The aligned single-channel CSVs are copied from `3_2d_aligned/` into `4_2d_filtered/` without a second filter; fused double-exposure CSVs are filtered when published. In `post_align`, raw detections are filtered directly into `4_2d_filtered/`.

`start_from_stage: 1` scans image directories. Values of 2 or greater read existing raw detection CSVs from `paths.pATHRESULT/1_2d_raw` and skip detection. In `pre_align`, the pipeline filters these CSVs before estimating tile channel shifts; Stage 3 stitches the aligned, filtered CSVs.

## Four-channel alignment in `pre_align`

The typical four-channel sample has two soma channels (`GFP`, `RFP`) and two TF channels (`Sox9`, `Olig2`). Stage 2.5 first Z-links each channel's detections within a tile, then estimates an integer `(dx, dy, dz)` shift for each channel. A positive shift is added to the filtered tile detection coordinates and to the displayed image placement. The shifts are saved per tile, so optical drift can vary across the sample.

`pre_align_params.reference_channel` is the **measurement reference** and must be an active soma channel. The non-reference soma channel is matched to it by voxel IoU. Soma-to-TF alignment is scored by nucleus containment inside soma boxes. `pre_align_params.tf_align_mode` controls how TF shifts are estimated:

| Mode | TF alignment |
| --- | --- |
| `chain` | Align additional TF channels to the first TF by voxel IoU, align that first TF to the reference soma by containment, then add the shifts. |
| `direct` | Align each TF channel independently to the reference soma by containment. This avoids relying on overlap between different TF populations. |
| `sequential_joint` | Requires active `GFP`/`RFP` soma and `Sox9`/`Olig2` TF channels, with `GFP` as the measurement reference. Match RFP to GFP first. Evaluate Sox9 against the combined GFP/RFP somata while preserving a better full-data GFP estimate. Evaluate Olig2 using soma, nearest-cell, and Sox9-derived candidates, then select the shift using joint evidence. Previously solved channel shifts stay fixed. |

The implementation is in [`src/core/point_cloud_aligner.py`](src/core/point_cloud_aligner.py). `3_2d_aligned/<tile>_measured_offsets.json` records the shifts in the measurement frame. The per-exposure aligned CSVs and `<tile>_offsets.json` in that directory use the **final frame** selected by `stitching_reference_channel`.

These two references may differ. If the measured raw-to-reference shift for channel `c` is `t_c` and the chosen final frame is `f`, the written shift is:

```text
s_c = t_c - t_f
s_f = (0, 0, 0)
aligned_detection_c = raw_detection_c + s_c
```

For example, the example detection config measures alignment relative to GFP but selects RFP as the final global frame. When the integrated solver is enabled, `stitching_reference_channel` chooses the tile-position and final global frame; it may differ from `pre_align_params.reference_channel`. Stage 2.5 rebases the measured offsets into the tile-position frame. The first active channel in `channels_routing` supplies the tile-enumeration anchor and, when no XML path is explicit, the XML lookup directory; it does not automatically determine either reference.

For a double-exposure channel, Stage 2 detects both exposures. Stage 2.5 gives the second exposure the primary channel's shift, and Stage 2.6 fuses matching 2D detections before filtering. Subsequent stages treat them as one logical channel.

## Solve tile positions from detections

[`src/core/solve_tile_positions.py`](src/core/solve_tile_positions.py) is the global position solver used by optional pipeline Stage 2.9 and standalone runs. It reads filtered, unaligned `2_2d_filtered/` CSVs in pre_align mode, matches detections from the **same channel** across neighboring tile overlaps, and uses those measured seam displacements to solve tile positions. Tile names provide nominal stage-coordinate priors. The default `joint` model fits shared tile positions plus a smooth channel-dependent displacement field. Cross-channel data estimates constant offsets; local per-tile refinement is enabled by default. This geometry step does not use TeraStitcher displacement estimates.

After Stage 2 raw detections exist, run, for example:

```bash
python src/core/solve_tile_positions.py \
    --sample /path/to/brain_sample \
    --config config/config.json \
    --workers 8
```

The default output directory is `<results_dir>/5_2d_global/tile_positions/`:

| File | Contents |
| --- | --- |
| `seams.csv` | Measured neighboring-tile displacements and match quality |
| `tile_positions.csv` | Shared and per-channel tile positions, alignment fields, refinement, and `s_<channel>_<axis>` shifts into the selected frame |
| `solution.json` | Model coefficients, constant-offset provenance, and diagnostics |
| `report.txt` | Human-readable solver summary |

`--ref` chooses the alignment measurement reference; `--frame` chooses the final coordinate frame. By default these come from `pre_align_params.reference_channel` and `stitching_reference_channel`. `--write-aligned` optionally creates pipeline-style aligned CSVs and offsets in a **separate** `3_2d_aligned_solved/` directory; it does not automatically replace Stage 2.5 output. `--write-xml` writes `xml_merging_<channel>.xml` in the report directory. `--xml-into-channel-dirs` also publishes each result as `xml_merging.xml` in its original channel image directory. Standalone XML export requires an XML template; integrated Stage 2.9 generates XML without one.

In integrated mode, Stage 2.9 reads `2_2d_filtered/` and reuses Stage 2.5 offsets. It writes one `xml_merging.xml` into each original channel image directory; `stacks_dir` and `mdata_bin` use the corresponding `Y:/Fengyi/...` path on the shared filesystem. Stage 3 loads the solved reference-channel XML automatically; `paths.pATHXML` is not needed. Cell coordinates combine frame tile positions with rebased channel shifts. Per-channel image XMLs keep seam-derived geometry plus global channel translation; local cell-alignment residuals are not applied to image tiles. Changed geometry is rejected while global checkpoints exist. The summary report is disabled by default; set `generate_analysis_report: true` to write `7_colocalization/global_summary_statistics.csv`. Cell centroids are written regardless of this setting.

## Global coordinates and deduplication

When integrated solving is enabled, Stage 3 loads the generated frame XML automatically. Otherwise, [`scripts/run_inference.py`](scripts/run_inference.py) currently loads the final-frame geometry from `paths.pATHXML`. If that path is unset, it tries `xml_merging.xml` and then `xml_import.xml` in the first active channel's directory. In manual XML mode, set `paths.pATHXML` to the geometry that matches the run; a stale channel XML changes the global coordinates. A named `xml_merging_<channel>.xml` is checked against `stitching_reference_channel`.

For a tile with XML values `ABS_H`, `ABS_V`, and `ABS_D`, the pipeline normalizes positions as follows. Local detection Z is one-based:

```text
P_x = ABS_H - min_tile(ABS_H)
P_y = ABS_V - min_tile(ABS_V)
P_z = max_tile(ABS_D) - ABS_D

x_global = x_raw + s_x + P_x
y_global = y_raw + s_y + P_y
z_global = z_raw + s_z - P_z
```

All four channels' saved global detections use the **same final frame geometry**. The source image channel used for a whole-brain mosaic does not by itself select the detection coordinate frame. For tile viewing, `CoordinateContext` reverses the corresponding frame position: `x_local = x_global - P_x`, `y_local = y_global - P_y`, and `z_local_0based = z_global + P_z - 1`.

Stage 3 performs several distinct kinds of overlap handling:

1. [`combine_predictions()`](src/core/stitcher.py) converts filtered tile detections to global positions. In tile overlap regions, it removes a later box when an earlier tile has a same-type box on the same Z slice with IoU above 0.4. The rejected ID, winner ID, and IoU are written to the stitch decisions CSV.
2. [`run_z_linker()`](src/core/z_linker.py) links same-channel boxes across Z using one-to-one XY IoU matching. A track becomes one 3D cell, with a representative box at the median Z. `z_linker.soma` and `z_linker.tf` provide separate IoU, minimum-layer, maximum-span, and gap settings. With `min_z_layers: 1`, isolated single-slice detections remain.
3. [`match_soma_3d_iou()`](src/core/stitcher.py) matches soma cells across channels using 3D IoU or IoMin. Each matched pair becomes the axis-aligned union of its source 3D bounding boxes (minimum lower and maximum upper X/Y/Z coordinates); its center is recomputed from that union. A later neuron/glia overlap pass gives glia priority. Sox9 and Olig2 nuclei are then assigned independently to the same merged soma geometry. The final `coloc_result.csv` contains soma cells with their marker combinations and channel positivity status.

`6_3d_global/<channel>_3d_tracked.csv` contains one representative row per Z-linked cell; the adjacent PKL holds its per-Z boxes and 3D extent. `7_colocalization/coloc_result.csv` contains the cross-channel soma result.

**Configuration fields to treat carefully:** `ENABLE_Z_LINKER` and `detection_params.cross_tile_iomin_thresh` are present in some configs, but the current Python pipeline does not read them. Setting either one does not change Stage 3 behavior. The current code also does not read the solver's `tile_positions.csv` for global conversion.

Changing alignment references or stitching frame geometry after global checkpoints exist requires a fresh result directory or regeneration of the affected checkpoints. Changes to filtered CSVs or Z-link settings invalidate the Stage 3 and Stage 4 manifests and trigger their rebuild. The exact stitching XML file is hashed in both manifests. Changing model content or detection settings while raw CSVs exist is rejected because those raw boxes must be recomputed first.

## Visualize results

Run `python src/utils/visualize.py` to open the interactive launcher. Select a
sample and click **Choose** to scan its tiles; no sample is scanned on startup.
Select one or more tiles, adjust Z range, annotation/crop paths, and display
options, then click **Open selected tiles**. Tiles retain their zero-based
list numbers in a spatial row/column grid; click grid numbers or list entries to
select multiple tiles. Drag the divider between the sample/tile column and the
settings panel to adjust their widths; the sample and tile panels share one width.
**Start with images only** is enabled by default: Napari opens TIFF layers
visible and all detection, point, and overlap overlays hidden. Turn result
layers on in Napari as needed. In a global view, image layers from every
selected tile appear together in the layer list. Disable this option to
restore the configured initial visibility of overlays.
The **Full config** tab has an editable parameter tree for every setting, including sample paths,
channel routing, colors, and filters. Its JSON page also allows adding or
removing fields; click **Apply JSON** before selecting tiles after changing a
sample path. **Save config as** writes a new JSON file; opening a viewer uses
a temporary config and leaves the source config untouched.

`src/utils/visualize.py` reads `config/vis_config.json` by default. In
`2d` mode, **2D source** chooses one raw or filtered detection set. In
`prealign` mode, the raw and filtered 2D checkboxes independently control
additional overlays and can both be enabled. The launcher shows these controls
only in their applicable modes. To use the original command-line flow, add
`--direct`. Passing `--mode` or `--2d-source` also keeps the command-line
flow.


| Mode | Displays | Geometry |
| --- | --- | --- |
| 2d | Tile-local saved 2D detections and shifted images | No XML needed |
| prealign | Saved tile 2D, stitched 2D, per-channel 3D, and Stage 4 results | Uses runtime_config.json, saved offsets, and the final-frame XML |
| post | Legacy single-tile saved-result view | Uses the XML selected in visualization config |

The prealign view performs no Z linking, channel matching, or detection
filtering. It reads the run's channel order and final coordinate frame from
the result directory's runtime_config.json. Visualization sample paths locate
the TIFFs and results on the current machine. If the final XML is not
available yet, the local view shows saved tile 2D stages only. Missing later
checkpoints are reported and never calculated by the GUI.

Set view_space to local to view each selected tile in final-frame tile
coordinates. Set it to global to place all selected tiles in one viewer on
the final-frame XML axes. The global view loads selected tile images and
result rows intersecting their footprints and displayed Z range.
global_z_start may set the global zero-based first slice; otherwise it is
derived from the first selected tile and z_start.

The saved Stage 3 CSV contains a summary box at the track's representative
Z. Shift-click its layer to load that channel's saved PKL and show the
track's actual per-Z boxes. The PKL is loaded only on demand and may take
time for dense channels. Stage 4 CSV stores only one representative-Z box
per cell. Stage 4 layers show exact marker combinations; each cell appears
in one combination layer. Clicking a Stage 4 box reports its saved row
without inferring a new pairing.

### Per-channel coordinates in colocalization output

New Stage 4 runs write three linked files under `7_colocalization/`:

- `coloc_result.csv`: one row per soma, with a stable `coloc_id`, merged 3D
  `cx,cy,cz,x1_3d,y1_3d,x2_3d,y2_3d,z_min,z_max,bounds_method`, and a
  `source_3d` JSON object keyed by every matched channel ID. For matched soma
  channels, the merged bounds are their 3D bounding-box union, and the legacy
  `x1,y1,x2,y2` display box is its XY projection at the rounded center Z.
  Single-channel somas retain their representative-Z display box. The CSV also
  records `soma_positive_channels`, `tf_positive_channels` (JSON arrays),
  unique-channel counts, and `soma_status`/`tf_status` as `negative`,
  `single_positive`, `double_positive`, or `multi_positive`. TF channels are
  assigned independently; double positive means both markers were assigned to
  the soma, not necessarily that both were seen in the same nucleus. Each
  channel in `source_3d` contains a list so multiple matched TF tracks remain.
- `coloc_source_3d.csv`: one row per matched source track with its original
  Stage 3 center (`cx,cy,cz`) and 3D bounds
  (`x1_3d,y1_3d,x2_3d,y2_3d,z_min,z_max`).
- `coloc_source_boxes.csv`: each source track's exact per-Z 2D boxes and detection IDs.

Join the flat files to the main result on `coloc_id`; join the two source
files on `coloc_id,source_track`. All coordinates use the saved, aligned
global frame. Z values are one-based, as in the pipeline CSV files. These
coordinates are the individual Stage 3 tracks before colocalization; they
are not raw TIFF tile coordinates. On the next pipeline run, a Stage 4 checkpoint lacking these fields is
rebuilt from the saved channel tracks.

### Provenance files and coordinate definitions

| Stage | Files | Key retained information |
| --- | --- | --- |
| Raw detection | `1_2d_raw/<tile>_<channel>_result.csv`, `_inputs.json` | Stable `detection_id`, original TIFF path, raw slice/Z, score and intensity definitions; TIFF size and modification time, output hash, and processed/missing/unreadable status per slice. |
| YOLO / StarDist detail | `1_2d_raw/*_candidates.csv`, `*_instances.csv` | Every post-model YOLO window candidate with patch origin, kept/suppressed decision, winner ID and IoU; StarDist label, mask area, centroid and polygon vertices when supplied by the model. |
| Filtering | `2_2d_filtered/` and `4_2d_filtered/` `*_result.csv`, `*_rejected.csv`, `*_filter_manifest.json` | Accepted IDs are preserved; rejected rows retain their first failing filter step in `rejection_reason`; containment NMS also records winner ID, IoMin, and threshold. Pre-aligned single channels have an empty Stage 2.75 rejection file because their filtering happened earlier. |
| Alignment and fusion | `3_2d_aligned/*_result.csv`, `*_alignment_manifest.json`; `3_2d_aligned_fusion/*_result.csv`, `*_fusion_manifest.json`, `fusion_summary.csv` | Aligned rows retain raw slice/Z and TIFF path. Fusion rows retain both parent IDs, winning source ID, fusion IoU, and the winning source image. |
| Global stitching | `5_2d_global/<channel>_2d_global.csv`, `*_stitch_decisions.csv` | Accepted detection ID and exact source tile/slice; cross-tile overlap rejections with competing ID and IoU. |
| Z linking | `6_3d_global/<channel>_3d_tracked.csv`, `*_track_members.csv`, `*_track_rejections.csv`, `*_stage3_manifest.json` | Track ID, exact member detections, chosen representative Z, observed XY envelope, and tracks rejected by `min_z_layers`. |
| Colocalization | `7_colocalization/coloc_result.csv`, `coloc_source_3d.csv`, `coloc_source_boxes.csv`, `coloc_match_evidence.csv`, `coloc_decisions.csv`, `_provenance_manifest.json` | Stable cell ID, all individual channel tracks and boxes, accepted match measurements, and cross-class/TF exclusion decisions. |

`detection_id` links raw, filtered, aligned, fused, global and track-member rows. A fused row has its own ID; `parent_detection_ids` links to both exposures. `track_id` links track summaries to their member detections. `coloc_id` links the final cell to every participating channel track. `source_3d` is a JSON map from channel ID to a list, preserving multiple matched TF tracks from one channel. Each saved match measurement appears in `coloc_match_evidence.csv`.

Raw boxes and `raw_z` use the original tile image frame. Aligned tile boxes use the stitching-reference tile frame. Global 2D, 3D and colocalization boxes use the final global frame. Pixel coordinates are zero-origin; slice Z is one-based. `source_trace_status` distinguishes an exact center-Z source member, a representative source member, and a merged soma union whose displayed XY box has no single source detection. The exported physical center coordinates use `cx_um = cx * xy_resolution_um`, `cy_um = cy * xy_resolution_um`, and `cz_um = (cz - 1) * z_resolution_um`. They are relative to the global pixel/slice origin, not an anatomical atlas origin. In individual Stage 3/source-track rows, `x1_3d...y2_3d` are the representative-centered matching volume and `observed_x1...y2` are the measured per-Z envelope. In a merged Stage 4 row, `x1_3d...y2_3d` are the union of the source matching volumes.

Manifests are completion markers with input signatures and output SHA256 hashes. Newly written raw, alignment, Stage 3 and Stage 4 checkpoints are validated on resume. Legacy files without the required source fields can be read, but missing TIFF provenance or original track members cannot be reconstructed from a summary box. New Stage 3 runs rebuild legacy global/track checkpoints from available filtered CSVs. Keep all linked files together when moving result directories; absolute paths in manifests reflect their original location.

    python src/utils/visualize.py --config config/vis_config.json --mode prealign

## Code map

- [`scripts/run_inference.py`](scripts/run_inference.py): pipeline orchestration, checkpoints, and global frame selection.
- [`src/core/point_cloud_aligner.py`](src/core/point_cloud_aligner.py): per-tile pre-alignment and four-channel shift modes.
- [`src/core/solve_tile_positions.py`](src/core/solve_tile_positions.py): detection-based seam measurement and global tile position solving.
- [`src/core/stitch_raw_tiles.py`](src/core/stitch_raw_tiles.py): optional raw-image mosaics on a shared channel canvas.
- [`src/core/align_stitched_channels.py`](src/core/align_stitched_channels.py): align previously stitched channel images.
- [`src/core/channel_stage3.py`](src/core/channel_stage3.py): per-channel global 2D construction and Z-link calls.
- [`src/core/z_linker.py`](src/core/z_linker.py): across-Z tracking.
- [`src/core/stitcher.py`](src/core/stitcher.py): overlap handling, soma matching, and TF annotation.
- [`src/utils/coordinate_context.py`](src/utils/coordinate_context.py): global-to-tile visualization coordinates.
- [`src/utils/visualize.py`](src/utils/visualize.py): napari tile viewer.

Legacy `post_align` and XML comparison tools remain in the repository for older datasets. The workflow above describes the current `pre_align` path and its checked-in integration limits.
