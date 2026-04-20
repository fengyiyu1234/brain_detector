## Project Structure
```text
MADM_Detection/
├── config/
│   └── config.json                   # Global configuration file
├── scripts/
│   └── run_inference.py              # Main execution script 
├── src/
│   ├── core/                         # Core algorithms (Parallel Workers, Stitcher, Z-Linker)
│   └── utils/                        # Utilities (Logging, I/O, TeraStitcher parsing, image processing)
└── README.md 
```

## Configuration
Before running the script, please ensure the paths and parameters in config/config.json are set correctly.
### Models:
    yolo_path: Pretrained yolo model for neuron and glia detection in single channel
    cellpose_path: Pretrained CellPose-SAM model for neuclei detection in single channel
### channels_routing:
    Define detection pattern for each channel
### Paths:
    channel_dir: Root directory for input data, and .xml files, corresponding to information in channels_routing
    pATHRESULT: Global output directory for analysis results.
### Detection:
    sTARTID / eNDID: Specify the range of Tile indices to process (useful for batch processing).
    tILESIZE: Physical pixel size of a Tile (default is 2048).
    mERGEZ: Enable/disable 3D Z-axis association (default is True).
    conf_thresh: threshold for output detection confidence
    dOWNSAMPLE_PERCENTILE_LOW/HIGH: Percentile information for normalization from 16bit to 8bit
    DOWNSAMPLE: allows detection with skipping, saving detection time. False by default.
    DOWNSAMPLE_Z_STEP: When DOWNSAMPLE is enabled, define detection in every z images.
### Remove Duplicates：
    ENABLE_Z_LINKER: Remove duplicates for a single cell in 3D. Disable this when image step is bigger than cell size.
### Visulaization:
    "vISUALIZATIONSAMPLESTEP": Visualization in every n images in raw data tiles.
    "vISUALIZATIONSAMPLECOUNT": Continuously visualize m images.

## Outputs:
- **global_bboxes.csv:** A comprehensive list of globally deduplicated bounding boxes (including scores, classes, and Z-layers).
- **global_centroids.csv:** The final 3D cell centroid coordinates (x, y, z) after Z-Linker fusion.
- **global_summary_statistics.csv:** A statistical report summarizing cell counts and percentages categorized by class and color.
- **cell_centroids/:** A subdirectory containing individual CSV files for each class (ob_<class_id>.csv).
- **tile_detections/:** Contains raw detection results (_result.csv) for each processed Tile.
- **visualization_tile/:** Visualization of raw detection results of each tile