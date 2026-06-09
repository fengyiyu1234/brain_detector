# brain_detector

Light-sheet microscopy pipeline for 3D brain cell detection and multi-channel colocalization.
Designed for 0.65 × 0.65 × 8 µm/pixel tile-based acquisitions.

---

## Project Structure

```text
brain_detector/
├── config/
│   └── config.json               # Global configuration (paths, model params, pipeline mode)
├── models/
│   ├── custom_nuclei_model       # YOLO soma detector
│   └── nuclei_model_0515         # Cellpose TF/nucleus detector
├── scripts/
│   └── run_inference.py          # Main pipeline entrypoint
├── src/
│   ├── config/
│   │   └── loader.py             # JSON config loader (strips comments)
│   ├── core/
│   │   ├── worker.py             # Per-tile parallel inference (YOLO + Cellpose)
│   │   ├── stitcher.py           # Global 2D stitching, soma merge, 3D colocalization
│   │   ├── z_linker.py           # Z-axis deduplication (Hungarian matching)
│   │   └── point_cloud_aligner.py# Pre-align mode: point-cloud-based channel alignment
│   └── utils/
│       ├── io.py                 # Tile listing, TeraStitcher XML parsing
│       ├── image.py              # Normalization, patch inference
│       ├── geometry.py           # IoU, centroid utilities
│       ├── logger.py             # Logging setup
│       ├── visualize.py          # Napari multi-tile result viewer (post-align)
│       └── visualize_prealign.py # Napari single-tile alignment inspector (pre-align)
├── requirements.txt              # Base packages (macOS + HPC shared)
├── requirements-local.txt        # macOS visualization environment
├── requirements-hpc.txt          # HPC Linux GPU inference environment
├── environment.yml               # Conda environment (Python 3.10)
└── README.md
```

---

## Pipeline Modes

### `post_align` (default)
Runs on images already aligned by numorph + TeraStitcher. Reads per-channel aligned directories → detects → global stitching → Z-linking → colocalization.

### `pre_align`
Runs on raw unaligned images. After per-tile detection, inserts a **Phase 2.5** point-cloud alignment step that computes per-tile XYZ channel offsets (replacing numorph), then continues with the same downstream pipeline.

Two-step alignment strategy:
```
Step 1a: align secondary soma channels → reference soma (RFP)
Step 1b: align secondary TF channels  → reference TF (Sox9)
Step 2:  align reference TF (Sox9)    → reference soma (RFP)

Final offsets:
  RFP:   (0, 0, 0)        ← global reference
  GFP:   step-1a shift
  Sox9:  step-2 shift
  Olig2: step-1b + step-2  ← chained
```

Z-search is soft-capped at ±5 slices (warning if exceeded) and hard-capped at ±10 (forced to 0 if exceeded).

---

## Environment Setup

### macOS (visualization + development)
```bash
pip install -r requirements-local.txt
```

### HPC Linux (GPU inference)
```bash
# Check CUDA version first:
nvcc --version    # or: nvidia-smi

# CUDA 11.8:
pip install -r requirements-hpc.txt

# CUDA 12.1:
pip install -r requirements-hpc.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.1.2+cu121 torchvision==0.16.2+cu121

# CUDA 12.4:
pip install -r requirements-hpc.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch==2.1.2+cu124 torchvision==0.16.2+cu124
```

### Conda — macOS local
```bash
conda env create -f environment.yml
conda activate brain_detector
```

### Conda — HPC Linux (CUDA)
```bash
# 1. Check HPC CUDA version:
nvcc --version    # or: nvidia-smi

# 2. Edit environment-hpc.yml — uncomment the block matching your CUDA version
#    (default is CUDA 11.8; change to 12.1 or 12.4 if needed)

# 3. Create the environment:
conda env create -f environment-hpc.yml
conda activate brain_detector
```

---

## Configuration (`config/config.json`)

### `models`
| Key | Description |
|-----|-------------|
| `yolo_path` | YOLO model weight path (relative to project root) |
| `cellpose_path` | Cellpose model path (relative to project root) |

### `channels_routing`
Array defining each channel's detection strategy:
- `id`: channel name (e.g. `"RFP"`, `"Sox9"`)
- `type`: `"soma"` → YOLO detection; `"tf"` → Cellpose nucleus detection
- `model`: `"yolo"` or `"cellpose"`
- `dir_key`: key in `paths` that points to this channel's tile directory
- `active`: set `false` to skip a channel

### `paths`
| Key | Description |
|-----|-------------|
| `rfp_dir`, `gfp_dir`, `sox9_dir`, `olig2_dir` | Per-channel tile root directories |
| `pATHRESULT` | Output root directory |

**HPC Linux paths**: SSH in and run `df -h | grep -i deep` to find the DeepDesign mount point, then use absolute paths (e.g. `/deepdesign/Fengyi/...`).

### `pipeline_mode`
`"post_align"` (default) or `"pre_align"`. See [Pipeline Modes](#pipeline-modes).

### `pre_align_params` (pre_align mode only)
| Key | Default | Description |
|-----|---------|-------------|
| `sample_z_center_count` | 50 | Z slices from tile center used to build alignment point cloud |
| `xy_search_range_px` | 30 | FFT cross-correlation XY search radius (px) |
| `z_search_range_slices` | 5 | Z brute-force search range (±slices); soft cap 5, hard cap 10 |
| `match_distance_px` | 15 | Point-cloud IoU matching threshold (px) |
| `tile_overlap_pct` | 15 | Tile overlap % (used for grid fallback without TeraStitcher XML) |

### `z_linker`
| Group | Key | Description |
|-------|-----|-------------|
| `soma` | `iou_thresh` | Minimum bbox IoU for cross-z soma matching |
| `soma` | `min_z_layers` | Minimum z-layers to qualify as a 3D cell |
| `soma` | `max_cell_z_span` | Maximum z-span per cell (prevents over-merging) |
| `tf` | same keys | Same parameters tuned for smaller TF nuclei |

### `detection_params` (key entries)
| Key | Description |
|-----|-------------|
| `conf_thresh` | YOLO detection confidence threshold |
| `nms_iou` | NMS IoU threshold |
| `xsize` / `ysize` / `step` | Inference patch size and sliding-window stride |
| `tILESIZE` | TeraStitcher tile edge length (px) |
| `sTARTID` / `eNDID` | Tile index range to process (`null` = all) |
| `normalize_PERCENTILE_LOW/HIGH` | 16-bit → 8-bit normalization percentiles |
| `DOWNSAMPLE` / `DOWNSAMPLE_Z_STEP` | Skip-frame mode for faster debug runs |
| `coloc_use_centroid_box` | `true` = centroid-in-bbox colocalization; `false` = sphere-distance |
| `soma_merge_iou_thresh` | Cross-soma-channel merge IoU threshold |
| `n_permutations` | Permutation test iterations for colocalization significance |

---

## Running the Pipeline

```bash
# Default (uses <project_root>/config/config.json):
python scripts/run_inference.py

# Specify a custom config (useful on HPC):
python scripts/run_inference.py --config /path/to/config.json
```

---

## Output Structure

```text
pATHRESULT/
├── 0_channel_alignment/         # [pre_align only] per-tile offset JSONs + aligned CSVs
├── 1_tile_2d_raw/               # Per-tile 2D detection CSVs (one per channel)
├── 2_global_2d_raw/             # Globally stitched 2D detections
├── 3_channel_3d/                # Per-channel Z-linked 3D cell volumes
├── 4_colocalization/            # Soma–TF colocalization results
│   ├── global_bboxes.csv        # All 3D soma cells with colocalized TF labels
│   ├── coloc_result.csv         # Colocalization counts and permutation test p-values
│   └── global_summary_statistics.csv
└── 5_analysis_report/           # Run metadata and timing logs
```

**Colocalization class labels** follow the pattern `neuron_RFP_Sox9_Olig2` for cells positive in multiple channels (each TF channel is annotated independently via its own GMM).

---

## Visualization

### Post-align results viewer
```bash
python src/utils/visualize.py                      # interactive tile selection
python src/utils/visualize.py --tiles 0 1 4        # select tiles by index
python src/utils/visualize.py --list-tiles
python src/utils/visualize.py --no-images          # bounding boxes only
python src/utils/visualize.py --stage s4           # colocalization layer only
python src/utils/visualize.py --config /path/to/config.json
```

Napari layers: `[img]` raw tile images · `[s1]` 2D raw detections · `[s3]` 3D Z-linked cells · `[s4]` colocalized final result.

### Pre-align alignment inspector
```bash
python src/utils/visualize_prealign.py --tile 423800_302400
python src/utils/visualize_prealign.py --tile 423800_302400 --z-range 400 450
python src/utils/visualize_prealign.py --tile 423800_302400 --no-images
python src/utils/visualize_prealign.py --tile 423800_302400 --show-before
python src/utils/visualize_prealign.py --list-tiles
```

Napari layers: `[img]` raw channel images · `[aligned]` post-alignment detections · `[raw]` pre-alignment detections (hidden by default). Console prints per-channel shift and IoU score.
