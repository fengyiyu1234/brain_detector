# brain_detector

A tile-based light-sheet microscopy pipeline for cell detection, channel alignment, 3D Z linking, and multi-channel colocalization. The primary workflow detects cells on raw channel images (`pre_align`) and estimates alignment from detections. [`solve_tile_positions.py`](scripts/solve_tile_positions.py) can also estimate tile positions from cells shared by neighboring tiles, without running TeraStitcher.

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

The viewer's `2d` mode reads `0_channel_alignment/` or `1_tile_2d_filtered/` and shifts the displayed raw images into the same tile-local frame. It does not load XML.

The expected image layout is a channel directory containing row directories and tile directories, for example `<channel>/305500/305500_319100/*.tif`. Active channel IDs, their types, models, and directory keys are defined in `channels_routing`.

## Processing stages

| Stage | What it does | Main output |
| --- | --- | --- |
| 2 | Detect 2D soma or TF boxes independently in each raw channel and tile | `1_tile_2d_raw/<tile>_<channel>_result.csv` |
| 2.25 | Filter raw per-tile detections before channel alignment in pre_align | `1_tile_2d_prefiltered/` |
| 2.5 | In `pre_align`, estimate per-tile channel shifts from filtered detections and write shifted CSVs | `0_channel_alignment/<tile>_offsets.json` and aligned CSVs |
| 2.6 | Optionally fuse two exposures of one logical channel, one Z slice at a time | `1_tile_2d_fused/` |
| 2.75 | Publish final filtered tile CSVs for Stage 3 | `1_tile_2d_filtered/` |
| 2.8 | Optionally save area and intensity histograms | `1_tile_2d_histograms/` |
| 2.9 | Optionally solve tile positions and publish frame/channel XMLs | `5_analysis_report/tile_positions/`, each channel's `xml_merging.xml` |
| 3 | Place filtered boxes in global coordinates, link detections across Z, and colocalize channels | `2_global_2d_raw/`, `3_channel_3d/`, `4_colocalization/` |
| Report | Save cell positions and summary statistics | `5_analysis_report/` |

`stop_after_detection: true` exits after Stage 2. With the tile solver enabled, `stop_before_stitching: true` exits after Stage 2.9 has written the XMLs. Without it, the stop point remains after tile filtering. Existing CSV and PKL checkpoints are reused on later runs. Keep the saved `runtime_config.json` with the results: it records the coordinate settings used by that run.

Filtering has one path and no `stage_2_75_enabled` switch. In `pre_align`, raw detections are filtered once into `1_tile_2d_prefiltered/` before channel alignment. The aligned single-channel CSVs are then published to `1_tile_2d_filtered/` without a second filter; fused double-exposure CSVs are filtered when published. In `post_align`, raw detections are filtered directly into `1_tile_2d_filtered/`.

`start_from_stage: 1` scans image directories. Values of 2 or greater read existing raw detection CSVs from `paths.pATHRESULT/1_tile_2d_raw` and skip detection. In `pre_align`, the pipeline filters these CSVs before estimating tile channel shifts; Stage 3 stitches the aligned, filtered CSVs.

## Four-channel alignment in `pre_align`

The typical four-channel sample has two soma channels (`GFP`, `RFP`) and two TF channels (`Sox9`, `Olig2`). Stage 2.5 first Z-links each channel's detections within a tile, then estimates an integer `(dx, dy, dz)` shift for each channel. A positive shift is added to the filtered tile detection coordinates and to the displayed image placement. The shifts are saved per tile, so optical drift can vary across the sample.

`pre_align_params.reference_channel` is the **measurement reference** and must be an active soma channel. The non-reference soma channel is matched to it by voxel IoU. Soma-to-TF alignment is scored by nucleus containment inside soma boxes. `pre_align_params.tf_align_mode` controls how TF shifts are estimated:

| Mode | TF alignment |
| --- | --- |
| `chain` | Align additional TF channels to the first TF by voxel IoU, align that first TF to the reference soma by containment, then add the shifts. |
| `direct` | Align each TF channel independently to the reference soma by containment. This avoids relying on overlap between different TF populations. |
| `sequential_joint` | Requires active `GFP`/`RFP` soma and `Sox9`/`Olig2` TF channels, with `GFP` as the measurement reference. Match RFP to GFP first. Evaluate Sox9 against the combined GFP/RFP somata while preserving a better full-data GFP estimate. Evaluate Olig2 using soma, nearest-cell, and Sox9-derived candidates, then select the shift using joint evidence. Previously solved channel shifts stay fixed. |

The implementation is in [`src/core/point_cloud_aligner.py`](src/core/point_cloud_aligner.py). `0_channel_alignment/<tile>_measured_offsets.json` records the shifts in the measurement frame. The aligned CSVs and `<tile>_offsets.json` use the **final frame** selected by `stitching_reference_channel`.

These two references may differ. If the measured raw-to-reference shift for channel `c` is `t_c` and the chosen final frame is `f`, the written shift is:

```text
s_c = t_c - t_f
s_f = (0, 0, 0)
aligned_detection_c = raw_detection_c + s_c
```

For example, the example detection config measures alignment relative to GFP but selects RFP as the final global frame. When the integrated solver is enabled, `stitching_reference_channel` chooses the tile-position and final global frame; it may differ from `pre_align_params.reference_channel`. Stage 2.5 rebases the measured offsets into the tile-position frame. The first active channel in `channels_routing` supplies the tile-enumeration anchor and, when no XML path is explicit, the XML lookup directory; it does not automatically determine either reference.

For a double-exposure channel, Stage 2 detects both exposures. Stage 2.5 gives the second exposure the primary channel's shift, and Stage 2.6 fuses matching 2D detections before filtering. Subsequent stages treat them as one logical channel.

## Solve tile positions from detections

[`scripts/solve_tile_positions.py`](scripts/solve_tile_positions.py) is the global position solver used by optional pipeline Stage 2.9 and standalone runs. It reads filtered, unaligned `1_tile_2d_prefiltered/` CSVs in pre_align mode, matches detections from the **same channel** across neighboring tile overlaps, and uses those measured seam displacements to solve tile positions. Tile names provide nominal stage-coordinate priors. The default `joint` model fits shared tile positions plus a smooth channel-dependent displacement field. Cross-channel data estimates constant offsets; local per-tile refinement is enabled by default. This geometry step does not use TeraStitcher displacement estimates.

After Stage 2 raw detections exist, run, for example:

```bash
python scripts/solve_tile_positions.py \
    --sample /path/to/brain_sample \
    --config config/config.json \
    --workers 8
```

The default output directory is `<results_dir>/5_analysis_report/tile_positions/`:

| File | Contents |
| --- | --- |
| `seams.csv` | Measured neighboring-tile displacements and match quality |
| `tile_positions.csv` | Shared and per-channel tile positions, alignment fields, refinement, and `s_<channel>_<axis>` shifts into the selected frame |
| `solution.json` | Model coefficients, constant-offset provenance, and diagnostics |
| `report.txt` | Human-readable solver summary |

`--ref` chooses the alignment measurement reference; `--frame` chooses the final coordinate frame. By default these come from `pre_align_params.reference_channel` and `stitching_reference_channel`. `--write-aligned` optionally creates pipeline-style aligned CSVs and offsets in a **separate** `0_channel_alignment_solved/` directory; it does not automatically replace Stage 2.5 output. `--write-xml` writes `xml_merging_<channel>.xml` in the report directory. `--xml-into-channel-dirs` also publishes each result as `xml_merging.xml` in its original channel image directory. Both require an XML template.

In integrated mode, Stage 2.9 reads `1_tile_2d_prefiltered/` and reuses Stage 2.5 offsets. Stage 3 loads the solved frame XML automatically; `paths.pATHXML` is not needed. Cell coordinates combine frame tile positions with rebased channel shifts. Per-channel image XMLs keep seam-derived geometry plus global channel translation; local cell-alignment residuals are not applied to image tiles. Changed geometry is rejected while global checkpoints exist.

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

1. [`combine_predictions()`](src/core/stitcher.py) converts filtered tile detections to global positions and discards boxes whose centers fall in the left or upper neighbor's overlap region. This current cross-tile rule is a geometric mask, not IoU-based union of duplicate boxes.
2. [`run_z_linker()`](src/core/z_linker.py) links same-channel boxes across Z using one-to-one XY IoU matching. A track becomes one 3D cell, with a representative box at the median Z. `z_linker.soma` and `z_linker.tf` provide separate IoU, minimum-layer, maximum-span, and gap settings. With `min_z_layers: 1`, isolated single-slice detections remain.
3. [`match_soma_3d_iou()`](src/core/stitcher.py) matches soma cells across channels using 3D IoU or IoMin and combines their marker labels. A later neuron/glia overlap pass gives glia priority. TF nuclei then annotate containing soma cells; the final `coloc_result.csv` contains soma cells with their marker combinations.

`3_channel_3d/<channel>_3d_tracked.csv` contains one representative row per Z-linked cell; the adjacent PKL holds its per-Z boxes and 3D extent. `4_colocalization/coloc_result.csv` contains the cross-channel soma result.

**Configuration fields to treat carefully:** `ENABLE_Z_LINKER` and `detection_params.cross_tile_iomin_thresh` are present in some configs, but the current Python pipeline does not read them. Setting either one does not change Stage 3 behavior. The current code also does not read the solver's `tile_positions.csv` for global conversion.

Changing alignment references, frame geometry, or `paths.pATHXML` after global checkpoints exist requires a fresh result directory or regeneration of the affected checkpoints. The pipeline checks saved runtime provenance to prevent reuse in a different frame.

## Visualize results

[`src/utils/visualize.py`](src/utils/visualize.py) reads `config/vis_config.json` by default. It supports three modes:

| Mode | Displays | XML needed by current viewer? |
| --- | --- | --- |
| `2d` | Aligned raw or filtered tile-local 2D boxes and shifted images | No |
| `prealign` | Tile alignment QC, optional raw boxes, in-memory Z linking, and colocalization previews | Yes: the current entry point builds `CoordinateContext` |
| `post` | Saved filtered 2D boxes, Z-linked cells, and final colocalization results | Yes: global boxes are mapped back to the selected tile |

```bash
python src/utils/visualize.py --config config/vis_config.json --mode 2d --2d-source raw
python src/utils/visualize.py --config config/vis_config.json --mode 2d --2d-source filtered
python src/utils/visualize.py --config config/vis_config.json --mode post
```

In `prealign`, set `show_coloc: true` and `coloc_source: "preview"` to compute colocalization from the same filtered tile CSVs as the displayed Z-linked cells. Use `coloc_source: "saved"` to inspect the existing Stage 4 CSV. Colocalization layers are visible when loaded. Saved Stage 3/4 cells are shown in every tile their global boxes overlap, even when the CSV `tile_name` names a neighboring tile. Stage 4 draws each cell only on its representative Z slice. `post` always reads the saved Stage 4 CSV.

For `prealign` and `post`, use the same final-frame geometry as the run that created the saved results. Set the visualization sample's `frame_channel` to that frame and `paths.pATHXML` to its XML. The viewer checks frame and tile compatibility and reports the selected coordinate context. A different channel's XML is interchangeable only when its normalized tile positions are identical.

## Code map

- [`scripts/run_inference.py`](scripts/run_inference.py): pipeline orchestration, checkpoints, and global frame selection.
- [`src/core/point_cloud_aligner.py`](src/core/point_cloud_aligner.py): per-tile pre-alignment and four-channel shift modes.
- [`scripts/solve_tile_positions.py`](scripts/solve_tile_positions.py): detection-based seam measurement and global tile position solving.
- [`src/core/channel_stage3.py`](src/core/channel_stage3.py): per-channel global 2D construction and Z-link calls.
- [`src/core/z_linker.py`](src/core/z_linker.py): across-Z tracking.
- [`src/core/stitcher.py`](src/core/stitcher.py): overlap handling, soma matching, and TF annotation.
- [`src/utils/coordinate_context.py`](src/utils/coordinate_context.py): global-to-tile visualization coordinates.
- [`src/utils/visualize.py`](src/utils/visualize.py): napari tile viewer.

Legacy `post_align` and XML comparison tools remain in the repository for older datasets. The workflow above describes the current `pre_align` path and its checked-in integration limits.
