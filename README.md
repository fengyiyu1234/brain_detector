## Project Structure
```text
MADM_Detection/
├── config/
│   └── config.json                   # Global configuration file
├── scripts/
│   └── run_inference.py              # Main execution script 
├── src/
│   ├── analysis/                     # QC metrics and statistical reporting
│   ├── core/                         # Core algorithms (Parallel Workers, Stitcher, Z-Linker)
│   ├── utils/                        # Utilities (Logging, I/O, TeraStitcher parsing, image processing)
│   └── visualization/                # Matplotlib visualizations and large map generation
└── README.md 
```

## Configuration
Before running the script, please ensure the paths and parameters in config/config.json are set correctly.
### Paths:
    channel1_dir: Root directory for input data (currently RFP channel), and .xml files.
    channel2_dir: Another channel
    pATHRESULT: Global output directory for analysis results.
### Detection:
    sTARTID / eNDID: Specify the range of Tile indices to process (useful for batch processing).
    tILESIZE: Physical pixel size of a Tile (default is 2048).
    mERGEZ: Enable/disable 3D Z-axis association (default is True).
    dOWNSAMPLE_Z_2X: Enable/disable 2x downsampling along the Z-axis (default is false).
    labels_to_names: Mapping dictionary between model class IDs and biological labels (e.g., RFP, GFP).

## Outputs:
- **global_bboxes.csv:** A comprehensive list of globally deduplicated bounding boxes (including scores, classes, and Z-layers).
- **global_centroids.csv:** The final 3D cell centroid coordinates (x, y, z) after Z-Linker fusion.
- **global_summary_statistics.csv:** A statistical report summarizing cell counts and percentages categorized by class and color.
- **cell_centroids/:** A subdirectory containing individual CSV files for each class (ob_<class_id>.csv).
- **tile_detections/:** Contains raw detection results (_result.csv) and quality control metrics (_qc_metrics.csv) for each processed Tile.
- **Visualizations (optional):** Automatically generated large stitched images with bounding box annotations for rapid visual inspection. 