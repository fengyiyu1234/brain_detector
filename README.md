# brain_detector

Light-sheet microscopy pipeline for 3D brain cell detection and multi-channel colocalization.
Designed for 0.65 × 0.65 × 8 µm/pixel tile-based acquisitions (TeraStitcher format).

---

## Project Structure

```text
brain_detector/
├── config/
│   ├── config.json               # Main pipeline config (paths, model params, pipeline mode)
│   └── vis/
│       └── vis_config.json       # Visualization config (napari viewer settings)
├── models/
│   ├── train18_best_0515.pt      # YOLO soma detector
│   └── 2D_versatile_fluo/        # StarDist TF nucleus detector
├── scripts/
│   ├── run_inference.py          # Main pipeline entrypoint
│   ├── validate_align_shifts.py  # Pre-align QC: overlap-vs-offset curves on held-out subvolumes
│   └── compare_stitching.py      # Stitching QC: TeraStitcher self-report, seam residuals, XML cross-check
├── src/
│   ├── config/
│   │   └── loader.py             # JSON config loader (strips // comments)
│   ├── core/
│   │   ├── worker.py             # Per-tile parallel inference (YOLO + StarDist)
│   │   ├── stitcher.py           # Global stitching, soma merge, 3D colocalization
│   │   ├── z_linker.py           # Z-axis tracking (Hungarian matching)
│   │   └── point_cloud_aligner.py# Pre-align mode: point-cloud-based channel alignment
│   └── utils/
│       ├── io.py                 # Tile listing, TeraStitcher XML parsing
│       ├── image.py              # Normalization, patch inference
│       ├── logger.py             # Logging setup
│       ├── visualize.py          # Napari result viewer
│       └── vis_stitched.py       # Stitched volume visualization helpers
└── README.md
```

---

## Pipeline Modes

### `post_align` (default)
Runs on images already aligned by numorph + TeraStitcher. Reads per-channel aligned tile directories → detects → global stitching → Z-linking → colocalization.

### `pre_align`
Runs on raw unaligned images. After per-tile detection, inserts a **Stage 2.5** point-cloud alignment step that computes per-tile XYZ channel offsets (replacing numorph), then continues with the same downstream pipeline.

The reference channel is `pre_align_params.reference_channel` (must be a soma channel; defaults to the first soma channel in `channels_routing`). Other soma channels are aligned to it by voxel IoU. TF channels are aligned according to `pre_align_params.tf_align_mode`:

`"chain"` (default):
```
Step 1a: align other soma channels (RFP)      → reference soma (GFP)
Step 1b: align other TF channels (Olig2)      → first TF (Sox9)
Step 2:  align first TF (Sox9)                → reference soma (GFP)   [containment]

Final offsets:
  GFP:   (0, 0, 0)              ← global reference
  RFP:   step-1a shift
  Sox9:  step-2 shift
  Olig2: step-1b shift + step-2 ← chained
```

`"direct"`: every TF channel is aligned independently to the reference soma by containment (no TF-to-TF step). Use this when the TF markers label different cell populations (e.g. Sox9 vs Olig2), where TF-to-TF overlap is too sparse to align on.
```
  GFP:   (0, 0, 0)              ← global reference
  RFP:   voxel-IoU shift → GFP
  Sox9:  containment shift → GFP
  Olig2: containment shift → GFP
```

Search windows: the coarse step covers ±`xy_search_range_px` / ±`z_search_range_slices`, the fine step ±`xy_fine_search_px` / ±`z_fine_search_slices` around the coarse result, so a shift outside the sum of the two can never be found. There is no cap beyond that — check `validate_align_shifts.py` for shifts that sit at the edge.

The soma↔TF (containment) step finds its coarse candidates from a histogram of soma−TF centroid displacements (`containment_coarse: "displacement_hist"`, default): a nucleus only counts at shift *s* when a soma centroid lies within the gate radius of nucleus + *s*, so the histogram peak is where containment peaks. The previous coarse step (`"fft"`, 3D-FFT correlation of occupancy grids, still used by the intra-soma / intra-TF steps) locked onto spurious peaks for a dense nuclear channel against a GFP soma reference on T4, and the ±8 px fine search never reached the real one. `_align_settings.json` files written before this option existed count as `"fft"`, so re-running a sample aligned with the old code stops with a settings mismatch: delete `0_channel_alignment/` to re-align, or set `"containment_coarse": "fft"` to keep the old offsets.

#### Validating the computed offsets

`scripts/validate_align_shifts.py` answers whether a tile's offsets are a real optimum or just one point in a noise floor: it re-measures cross-channel cell overlap on a **held-out** subvolume while walking the offset away from the stored solution along each axis. A trustworthy offset gives a peak at Δ=0; a flat curve — or a peak several pixels off — means that tile's offset is not supported by data the solver never saw.

```bash
python scripts/validate_align_shifts.py --sample /path/to/sample18 \
    --n-tiles 8 --regions-per-tile 2 --workers 4
```

`--sample` takes the sample directory (or its `detection_results/` directly). Alignment parameters are read from `0_channel_alignment/_align_settings.json` when present, otherwise re-resolved from `runtime_config.json` (override with `--config`).

**Held-out region.** Stage 2.5 solves on the full XY extent of the central z-window, so the only never-used data is z *outside* that window. The script recomputes each tile's `z_center` exactly as Stage 2.5 does, samples a z-slab outside `[z_center ± sample_z_center_count/2]` plus a `--z-guard` margin (default `max_cell_z_span + z_search_range_slices + z_fine_search_slices`, because `build_cell_boxes` keeps cells that merely *overlap* the window and those extend past its edges), then crops a random `--xy-size` square inside it. If a tile is too thin to avoid the window, the region is still used but flagged `held_out=False`.

**Metrics**, swept one axis at a time (Δx varies while Δy/Δz stay at the optimum):

| Metric | Channels | Matches the solver's objective for |
|--------|----------|------------------------------------|
| `voxel_iou` | all | intra-soma / intra-TF alignment |
| `containment` | TF only | soma↔TF alignment — fraction of TF nuclei in the region contained by a reference soma |

Candidate offsets are applied to cell coordinates *before* voxelization (pixel-exact), so `--xy-step` need not be a multiple of `voxel_bin_size_px`.

**Output** → `5_analysis_report/align_validation/`: `<sample>_curves.csv` (one row per tile × region × channel × axis × Δ), `<sample>_summary.csv` (peak location, half-width, edge drop per curve), and a 3-panel PNG per channel × metric (thin lines = individual regions, thick = mean). The console summary's two key columns are `peak_hit` (fraction of regions whose peak lands on Δ=0) and `drop` (relative fall at the sweep edges).

Expect ~2 min/tile, dominated by z-linking dense TF channels — use `--workers`. Pass `--no-plots` where matplotlib is broken; CSVs are written before plotting, so a failure there never costs data.

#### Comparing stitchings / stitching reference ≠ alignment reference

Cell global coordinates are `XML tile position + tile-local coordinate in the alignment reference frame`, which is only right when the XML was stitched on the alignment reference channel. When the two differ (e.g. stitch on 488nm/Olig2 because its signal is dense, align on 640nm/GFP because that is the soma channel), the per-tile channel shift between them has to be accounted for. `scripts/compare_stitching.py` measures how well each candidate stitching — and that chain — actually lines up:

```bash
# before the second stitching exists: self-report + seam residuals of the 640 one
python scripts/compare_stitching.py --sample Y:/Fengyi/EGFR_brain/T4 --xml 640nm:GFP
# both stitchings + GFP cells carried into the 488 frame through the Stage 2.5 offsets
python scripts/compare_stitching.py --sample Y:/Fengyi/EGFR_brain/T4 \
    --xml 640nm:GFP --xml 488nm:Olig2 --also GFP --workers 4
```

`--xml NAME[:CHANNEL][=PATH]`: `NAME` labels the stitching and defaults its directory to `<sample>/<NAME>`; `CHANNEL` is the `channels_routing` id whose raw images were stitched. Each part runs as soon as its inputs exist:

| Part | Needs | Measures |
|------|-------|----------|
| A. self-report | TeraStitcher XMLs | per adjacent pair: fraction of axes replaced by the mechanical default (`xml_displthres`), spread of the per-subblock displacements (`xml_displcomp`), and `placed − pair displacement` in `xml_merging` (loop inconsistency). NCC/reliability values depend on image content, so don't compare their absolute level across channels |
| B. seam residual | + `1_tile_2d_raw` | independent of TeraStitcher: the same cells detected by both tiles of an overlap are z-linked, placed with the XML, and matched; the median `B − A` is that seam's error (ideal 0). `--also CH` first moves `CH` into the XML channel's frame (`raw + s_CH − s_frame`), i.e. the error cells will actually have on that stitched image |
| C. cross-check | two XMLs with channels + Stage 2.5 offsets | `pos_A(t) − pos_B(t)` should equal `s_a(t) − s_b(t)` up to a constant; a residual std well below the position-difference std means stitching and channel alignment corroborate each other, and flagged tiles are where one of them is wrong |

B uses the raw (unshifted) detection CSVs, so it can run while detection is still in progress — seams whose CSVs are missing are reported as `missing_csv`. Output → `5_analysis_report/stitch_compare/` (`<NAME>_pairs.csv`, `seams.csv`, `cross_<A>_vs_<B>.csv`, and grid/scatter PNGs). Plots are drawn only after every CSV is written; use the `antsreg` env for plotting or pass `--no-plots`.

---

## Pipeline Stages

| Stage | Description | Checkpoint (skip if exists) |
|-------|-------------|----------------------------|
| 2 | Per-tile detection (YOLO + StarDist), parallel per GPU | `1_tile_2d_raw/<tile>_<ch>_result.csv` |
| 2.5 | Point-cloud channel alignment *(pre_align only)* | `0_channel_alignment/_align_done.flag` |
| 2.75 | Per-tile bbox size/intensity filtering | `1_tile_2d_filtered/<tile>_<ch>_result.csv` |
| 3 | Global stitching → Z-linking → 3D colocalization | `4_colocalization/coloc_result.csv` |
| 4 | Per-class centroid files + summary statistics | `5_analysis_report/global_summary_statistics.csv` |

Each stage is a **linear checkpoint**: if its output already exists, it is skipped automatically. To re-run a stage, delete its checkpoint file/folder.

To re-run from Stage 3 only (e.g. after changing colocalization parameters), delete `4_colocalization/` and `5_analysis_report/`, then set `"start_from_stage": 3` in config.

---

## Running the Pipeline

```bash
# Default config (config/config.json):
python scripts/run_inference.py

# Custom config:
python scripts/run_inference.py --config /path/to/config.json
```

---

## Configuration (`config/config.json`)

> Config files support `//` line comments.

### `models`
| Key | Description |
|-----|-------------|
| `yolo_path` | YOLO model weight path (relative to project root) |
| `stardist_basedir` | StarDist model root directory (relative to project root) |
| `stardist_name` | StarDist model subdirectory name |

### `model_classes`
Maps YOLO output indices to class names. Currently `{"0": "neuron", "1": "glia"}`.

### `channels_routing`
Array defining each channel's detection strategy. Order matters — the first entry is the anchor channel.

| Field | Values | Description |
|-------|--------|-------------|
| `id` | e.g. `"RFP"` | Channel name, used as label prefix throughout |
| `type` | `"soma"` / `"tf"` | `soma` → YOLO; `tf` → StarDist nucleus detection |
| `model` | `"yolo"` / `"stardist"` | Inference backend |
| `dir_key` | key in `paths` | Points to this channel's tile directory |
| `active` | `true` / `false` | Set `false` to skip a channel entirely |
| `double_exposure` | `true` / `false` | Optional. When `true`, this channel has a second exposure/laser-power image that gets fused in at the raw per-tile 2D level before filtering (Stage 2.6) |
| `second_intensity_id` | e.g. `"GFP_25"` | Required if `double_exposure=true`. Internal id for the second exposure (used for its raw CSV filename and in logs) |
| `second_intensity_dir_key` | key in `paths` | Required if `double_exposure=true`. Points to the second exposure's tile directory |
| `fusion_iou_thresh` | float, default `0.3` | Per-z-slice IoU threshold used to match low/high exposure boxes during fusion |

### `paths`
| Key | Description |
|-----|-------------|
| `rfp_dir`, `gfp_dir`, `sox9_dir`, `olig2_dir` | Per-channel tile root directories |
| `pATHRESULT` | Output root directory |

### `pipeline_mode`
`"post_align"` (default) or `"pre_align"`. See [Pipeline Modes](#pipeline-modes).

### `start_from_stage`
| Value | Behavior |
|-------|----------|
| `1` | Full pipeline from scratch; scans tile directories over the network |
| `2` | Skip network scan; infer tile list from existing CSVs in `1_tile_2d_raw/` |
| `3` | Skip detection and filtering entirely; load directly from `3_channel_3d/` pkl files |

Use `3` to re-run only colocalization and downstream steps without re-running detection.

### `stop_after_detection`
`true` = exit immediately after Stage 2 (tile detection). Useful to run GPU-heavy detection on HPC, then run the CPU-only stages locally.

### `stop_before_stitching`
`true` = run all per-tile stages (detection, 2.5 alignment, 2.6 fusion, 2.75 filtering, 2.8 histograms), then exit before Stage 3. None of these stages need the TeraStitcher XML, so a sample whose stitching isn't finished yet can be processed up to here; set back to `false` once `xml_merging.xml` exists and re-run — finished stages are skipped by their checkpoints. The XML is looked up from `paths.pATHXML` first, then `xml_merging.xml` / `xml_import.xml` in the anchor channel directory.

### `ENABLE_Z_LINKER`
`true` (default) = run Z-axis tracking. `false` = output raw 2D detections only.

### `stage3_n_workers`
Stage 3 stitches and z-links every channel in its own process; this caps how many run at once (default: all channels, limited by `$SLURM_CPUS_PER_TASK`). Dense TF channels hold several GB each while they run, so lower it if the job runs out of memory. Cross-channel colocalization (3A/3B/3C) stays in the main process.

**Z-linker solver.** Each slice's Hungarian matching is solved separately inside every connected group of boxes with IoU > 0 (`run_z_linker(..., solver='sparse')`, the default) instead of on one whole-brain cost matrix. This is the same optimum — the number of forced cross-type pairs doesn't depend on which positive-IoU pairs are chosen — so results only differ where two assignments have exactly equal cost. On sample18, soma channels came out identical and Sox9 differed in 2 of 3.69 M cells; Sox9 z-linking went from 56 min to ~3.5 min. `solver='dense'` keeps the original for comparison.

### `pre_align_params` *(pre_align mode only)*
| Key | Default | Description |
|-----|---------|-------------|
| `reference_channel` | first soma channel | Soma channel every other channel is shifted onto |
| `tf_align_mode` | `"chain"` | `"chain"`: TF-N → first TF → reference; `"direct"`: each TF → reference independently (see [pre_align](#pre_align)) |
| `sample_z_center_count` | 50 | Z slices from tile center used to build alignment point cloud |
| `voxel_bin_size_px` | 4 | Voxel bin size for 3D FFT alignment (px); smaller = more precise but slower |
| `xy_search_range_px` | 30 | FFT coarse-search XY radius (px) |
| `z_search_range_slices` | 5 | FFT coarse-search Z range (±slices); soft cap 5, hard cap 10 |
| `xy_fine_search_px` | 8 | Fine-search XY range around FFT peak (px) |
| `z_fine_search_slices` | 2 | Fine-search Z range around FFT peak (slices) |
| `containment_coarse` | `"displacement_hist"` | Coarse search of the soma↔TF containment step; `"fft"` reproduces results aligned before this option existed (see [pre_align](#pre_align)) |
| `tile_overlap_pct` | 15 | Tile overlap % (fallback grid calculation when TeraStitcher XML is absent) |
| `n_workers` | `$SLURM_CPUS_PER_TASK`, else CPU count | Stage 2.5 CPU processes, one tile each. Stage 2.5 never uses a GPU, so run it in a CPU job (`scripts/inference_cpu.slurm`) rather than holding GPUs. Not part of `_align_settings.json` — changing it never invalidates finished tiles |

Stage 2.5 resumes per tile: a tile counts as finished when its `_offsets.json` exists (written last) and every aligned CSV has as many lines as its raw CSV. A killed or timed-out job just needs resubmitting.

### `z_linker`
Parameters are split by channel type (`soma` / `tf`):

| Key | Description |
|-----|-------------|
| `iou_thresh` | Minimum 2D bbox IoU for cross-z frame matching |
| `min_z_layers` | Minimum z-layers to qualify as a 3D cell |
| `max_cell_z_span` | Maximum z-span per cell (prevents over-merging) |

Additional soma-only keys:

| Key | Description |
|-----|-------------|
| `iou_thresh_3d` | 3D IoU threshold for cross-channel soma matching |
| `z_pad_3d` | One-sided z-gap tolerance (slices) for 3D soma matching; only bridges a real gap between non-overlapping boxes, never inflates boxes that already overlap in z |
| `cross_class_iou_thresh` | neuron–glia overlap threshold; glia takes priority |

Additional tf-only keys:

| Key | Default | Description |
|-----|---------|-------------|
| `gmm_p_thresh` | 0.5 | GMM colocalization probability threshold *(visualization in-memory path only)* |
| `max_center_dist_ratio` | 0.5 | Hard gate for soma–TF colocalization: the TF nucleus centroid must be within `ratio × soma_radius` of the soma centroid. Prevents edge-overlap false positives when soma bboxes are large. |

### `detection_params`

**Physical resolution:**

| Key | Default | Description |
|-----|---------|-------------|
| `xy_resolution_um` | 0.65 | XY pixel size (µm/pixel) |
| `z_resolution_um` | 8 | Z slice spacing (µm) |

**Detection thresholds:**

| Key | Default | Description |
|-----|---------|-------------|
| `conf_thresh` | 0.3 | YOLO confidence threshold |
| `nms_iou` | 0.3 | NMS IoU threshold |

**Inference patch:**

| Key | Default | Description |
|-----|---------|-------------|
| `xsize` / `ysize` | 512 | Inference patch width/height (px) |
| `step` | 384 | Sliding window stride (px); overlap = xsize − step |
| `tILESIZE` | 2048 | TeraStitcher tile edge length (px) |

**Processing range:**

| Key | Default | Description |
|-----|---------|-------------|
| `sTARTID` / `eNDID` | null | Tile index range (`null` = all) |
| `DOWNSAMPLE` | false | Skip-frame mode for fast debug runs |
| `DOWNSAMPLE_Z_STEP` | 41 | Skip interval when `DOWNSAMPLE=true` |

**Image normalization:**

| Key | Default | Description |
|-----|---------|-------------|
| `normalize_PERCENTILE_LOW` | 0.1 | Lower percentile for 16-bit → 8-bit stretch |
| `normalize_PERCENTILE_HIGH` | 99.9 | Upper percentile |

**YOLO-specific filters** (`detection_params.yolo`):

| Key | Default | Description |
|-----|---------|-------------|
| `bbox_min` / `bbox_max` | null | Width/height absolute limits (px); `null` = no filter |
| `bbox_area_pct_min` | 10 | Drop boxes below this area percentile (within-tile) |
| `bbox_mean_pct_min` | null | Drop boxes below this intensity percentile |
| `bbox_mean_min` | 0 | Absolute intensity floor (raw 16-bit value) |

**StarDist-specific filters** (`detection_params.stardist`):

| Key | Default | Description |
|-----|---------|-------------|
| `norm_low` / `norm_high` | 1 / 99.9 | Normalization percentiles for StarDist input |
| `prob_thresh` | 0.5 | Instance probability threshold |
| `nms_thresh` | 0.4 | NMS overlap threshold |
| `n_tiles` | [4, 4] | Inference tiling [Y, X]; larger = lower peak VRAM |
| `bbox_min` / `bbox_max` | 8 / 17 | Width/height limits (px) |
| `bbox_area_pct_min` | 5 | Drop boxes below this area percentile |
| `bbox_mean_pct_min` | null | Drop boxes below this intensity percentile |
| `bbox_mean_min` | 0 | Absolute intensity floor |

---

## Output Structure

```text
pATHRESULT/
├── 0_channel_alignment/         # [pre_align only] per-tile offset JSONs + aligned CSVs
│   ├── _align_settings.json     # resolved alignment settings; a re-run with different settings stops and asks you to delete this folder
│   └── _align_done.flag         # checkpoint: alignment complete (not written if any detection CSV is missing)
├── 1_tile_2d_raw/               # Per-tile 2D detection CSVs (one file per tile×channel)
├── 1_tile_2d_filtered/          # Same CSVs after size/intensity filtering (Stage 2.75 output)
├── 2_global_2d_raw/             # Globally stitched 2D detections (one CSV per channel)
├── 3_channel_3d/                # Per-channel Z-linked 3D cells
│   ├── <ch>_3d_tracked.csv      # Summary (center_z bbox per cell)
│   └── <ch>_3d_tracked.pkl      # Full volumetric vol_list
├── 4_colocalization/            # Colocalization results
│   ├── coloc_result.csv         # All 3D soma cells with colocalized TF class labels
│   └── <class>.csv              # Per-class split of coloc_result.csv
└── 5_analysis_report/
    ├── global_summary_statistics.csv
    ├── align_validation/            # [optional] validate_align_shifts.py: curves CSV + summary + PNGs
    ├── stitch_compare/              # [optional] compare_stitching.py: pair / seam / cross-check CSVs + PNGs
    └── cell_centroids/
        └── <class>_centroids.csv         # Physical centroids (µm) per cell class
```

**Class label convention**: `{soma_type}_{channel}_{TF}`, e.g. `neuron_RFP_Sox9` for an RFP+ neuron colocalized with Sox9. Multi-positive soma channels and TF markers are joined with `_` in sorted order.

---

## Visualization

```bash
python src/utils/visualize.py
python src/utils/visualize.py --config config/vis/vis_config.json
```

Edit `config/vis/vis_config.json` to select mode, tile, Z range, and which napari layers to show. The viewer supports two modes:

- **`post`**: loads saved results from `3_channel_3d/` and `4_colocalization/`. Layers: `[s1]` raw 2D · `[s3]` Z-linked · `[s4]` colocalization.
- **`prealign`**: runs Z-linking and colocalization in memory for a single tile; useful for parameter QC without re-running the full pipeline.

Key `vis_config.json` settings:

| Key | Description |
|-----|-------------|
| `mode` | `"post"` or `"prealign"` |
| `tile` | Tile directory name; `null` = interactive selection |
| `z_range` | `[start, end]` absolute slice indices; `null` = auto-center |
| `stage` | `"all"` / `"s1"` / `"s3"` / `"s4"` — which result layers to load |
| `show_coloc` | Show colocalization layer *(prealign mode)* |
| `filter` | Per-type bbox size/intensity filters applied at display time only |

## Detection CSV filtering

`1_tile_2d_raw/` is the complete detector output: it includes only detector-native
post-processing (YOLO patch stitching/model NMS or StarDist instance NMS), never
configurable size, percentile, intensity, or containment filtering. Stage 2.75 always
reads raw/post-alignment CSVs (or fused CSVs for a double-exposure logical channel),
applies the shared filter once, and atomically writes `1_tile_2d_filtered/`. Stage 3
reads only that filtered directory.

Model defaults live in `detection_params.yolo` / `detection_params.stardist`. A
`channel_filter_overrides.<channel-id>` object overrides only explicitly present keys;
`null` deliberately disables an inherited filter. This permits Olig2-specific tuning
without changing Sox9. After a parameter change, filtered and all downstream stages are
stale; regenerate filtered output before rerunning Stage 3 onward.

To regenerate without loading images or models:

```bash
python scripts/refilter_detections.py --config config/config_EGFR_t4_local_gpu.json --channels Olig2 --dry-run
python scripts/refilter_detections.py --config config/config_EGFR_t4_local_gpu.json --channels Olig2 --overwrite
```

The CLI reads the correct raw/aligned/fused CSV source by pipeline mode, uses the CSV
`mean` column, writes CSVs atomically, and warns instead of deleting downstream output.
The 2-D visualizer uses the same implementation and, when present, resolves parameters
from `runtime_config.json`; its layers distinguish source, preview-kept, and preview-
rejected boxes.