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
│   └── run_inference.py          # Main pipeline entrypoint
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

Two-step alignment strategy:
```
Step 1a: align secondary soma channels (GFP) → reference soma (RFP)
Step 1b: align secondary TF channels (Olig2) → reference TF (Sox9)
Step 2:  align reference TF (Sox9)           → reference soma (RFP)

Final offsets:
  RFP:   (0, 0, 0)              ← global reference
  GFP:   step-1a shift
  Sox9:  step-2 shift
  Olig2: step-1b shift + step-2 ← chained
```

Z-search is soft-capped at ±5 slices (warning if exceeded) and hard-capped at ±10 (forced to 0).

---

## Pipeline Stages

| Stage | Description | Checkpoint (skip if exists) |
|-------|-------------|----------------------------|
| 2 | Per-tile detection (YOLO + StarDist), parallel per GPU | `1_tile_2d_raw/<tile>_<ch>_result.csv` |
| 2.5 | Point-cloud channel alignment *(pre_align only)* | `0_channel_alignment/_align_done.flag` |
| 2.75 | Per-tile bbox size/intensity filtering | `1_tile_2d_filtered/<tile>_<ch>_result.csv` |
| 3 | Global stitching → Z-linking → 3D colocalization | `4_colocalization/coloc_result.csv` |
| 4 | Per-class centroid files + summary statistics | `5_analysis_report/global_summary_statistics.csv` |
| 5 | Colocalization permutation test | `5_analysis_report/colocalization_significance.csv` |

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

### `ENABLE_Z_LINKER`
`true` (default) = run Z-axis tracking. `false` = output raw 2D detections only.

### `pre_align_params` *(pre_align mode only)*
| Key | Default | Description |
|-----|---------|-------------|
| `sample_z_center_count` | 50 | Z slices from tile center used to build alignment point cloud |
| `voxel_bin_size_px` | 4 | Voxel bin size for 3D FFT alignment (px); smaller = more precise but slower |
| `xy_search_range_px` | 30 | FFT coarse-search XY radius (px) |
| `z_search_range_slices` | 5 | FFT coarse-search Z range (±slices); soft cap 5, hard cap 10 |
| `xy_fine_search_px` | 8 | Fine-search XY range around FFT peak (px) |
| `z_fine_search_slices` | 2 | Fine-search Z range around FFT peak (slices) |
| `tile_overlap_pct` | 15 | Tile overlap % (fallback grid calculation when TeraStitcher XML is absent) |

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
| `z_pad_3d` | Z padding (slices) applied during 3D soma matching |
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

**Permutation test:**

| Key | Default | Description |
|-----|---------|-------------|
| `n_permutations` | 100 | Permutation iterations for colocalization significance |
| `max_soma_sample` | 50000 | Max soma count sampled per permutation run |

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
│   └── _align_done.flag         # checkpoint: alignment complete
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
    ├── colocalization_significance.csv   # Permutation test p-values per TF marker
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
