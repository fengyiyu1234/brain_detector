import os
import numpy as np
import xml.etree.ElementTree as ET
import csv
import datetime
import json
import platform
from pathlib import Path

from src.core.provenance import (PROVENANCE_SCHEMA_VERSION, atomic_json,
                                 file_sha256, model_fingerprints)

def loadTeraxml(fxml, tile_size=2048):
    tILESIZE = tile_size
    tree = ET.parse(fxml)
    root = tree.getroot()
    dimensions = root.find('dimensions')
    n_row = int(dimensions.get('stack_rows'))
    n_col = int(dimensions.get('stack_columns'))
    n_slices = int(dimensions.get('stack_slices'))
    dir_dict = {}
    disp_mat = np.full((n_row,n_col,3), None)
    stacks = root.find('STACKS')
    for i in range(len(stacks)):
        stack = stacks[i]
        dir_name = stack.get('DIR_NAME')
        abs_x, abs_y, abs_z = int(stack.get('ABS_H')), int(stack.get('ABS_V')), int(stack.get('ABS_D'))
        row, col = int(stack.get('ROW')), int(stack.get('COL'))
        disp_mat[row, col] = [abs_x, abs_y, abs_z]
        dir_dict[dir_name] = (row, col)
    disp_mat_fin = disp_mat.copy()
    x_min, y_min, z_min = disp_mat_fin[:,:,0].min(), disp_mat_fin[:,:,1].min(), disp_mat_fin[:,:,2].min()
    x_max, y_max, z_max = disp_mat_fin[:,:,0].max(), disp_mat_fin[:,:,1].max(), disp_mat_fin[:,:,2].max()
    W = x_max-x_min+tILESIZE
    H = y_max-y_min+tILESIZE
    Z = n_slices-z_max+z_min
    z_start = z_max
    disp_mat_fin = disp_mat_fin - [x_min,y_min,0]
    return dir_dict, H, W, Z, z_start, disp_mat_fin

# 台面坐标编码单位：tile 目录名里的行/列数字是显微镜台面位置，以 0.1 um 为
# 一个单位（通过对照本项目 sample12_410q 自己的 TeraStitcher xml 输出交叉
# 验证得到：目录名间距 11300 对应 xml 里的真实像素间距 1738px，
# 11300/10/0.65um = 1738.46px，吻合）。
STAGE_UNIT_UM = 0.1


def compute_grid_fallback_offsets(tile_paths, tile_size, overlap_pct, xy_res_um):
    """Parse '<row>_<col>'-style tile directory names into a grid and derive
    per-tile global pixel offsets, used when no TeraStitcher XML exists
    (pre_align mode fallback). Mirrors loadTeraxml's grid outputs:
      dir_dict: tile_name -> (grid_row, grid_col)
      disp_mat_fin: ndarray (n_row, n_col, 3) of [ABS_X, ABS_Y, ABS_Z=0]
    """
    overlap = overlap_pct / 100.0
    step_px = int(tile_size * (1.0 - overlap))
    dir_dict = {}
    raw_entries = []
    for tile_path in sorted(tile_paths):
        tile_name = os.path.split(tile_path)[-1]
        parts = tile_name.split('_')
        try:
            raw_row = int(parts[0]) if len(parts) >= 2 else 0
            raw_col = int(parts[1]) if len(parts) >= 2 else 0
        except ValueError:
            raw_row, raw_col = 0, 0
        raw_entries.append((tile_name, raw_row, raw_col))
    if raw_entries:
        sorted_rows = sorted(set(r for _, r, _ in raw_entries))
        sorted_cols = sorted(set(c for _, _, c in raw_entries))
        row_idx = {v: i for i, v in enumerate(sorted_rows)}
        col_idx = {v: i for i, v in enumerate(sorted_cols)}
        n_row, n_col = len(sorted_rows), len(sorted_cols)
        use_raw_offset = (sorted_rows[-1] > step_px or sorted_cols[-1] > step_px)
        disp_mat_fin = np.zeros((n_row, n_col, 3), dtype=int)
        for tile_name, raw_row, raw_col in raw_entries:
            gi, gj = row_idx[raw_row], col_idx[raw_col]
            dir_dict[tile_name] = (gi, gj)
            if use_raw_offset:
                # 目录名数字是台面坐标（单位 STAGE_UNIT_UM），换算成像素
                ax = round(raw_col * STAGE_UNIT_UM / xy_res_um)
                ay = round(raw_row * STAGE_UNIT_UM / xy_res_um)
            else:
                ax = gj * step_px
                ay = gi * step_px
            disp_mat_fin[gi, gj] = [ax, ay, 0]
        if use_raw_offset:
            # 和 loadTeraxml 一样，归一化到从 0 开始
            x_min = disp_mat_fin[:, :, 0].min()
            y_min = disp_mat_fin[:, :, 1].min()
            disp_mat_fin[:, :, 0] -= x_min
            disp_mat_fin[:, :, 1] -= y_min
    else:
        disp_mat_fin = np.zeros((1, 1, 3), dtype=int)
    return dir_dict, disp_mat_fin

def listFile(path, ext):
    filename_list, filepath_list = [], []
    for r, d, f in os.walk(path):
        for filename in f:
            if ext in filename:
                filename_list.append(filename)
                filepath_list.append(os.path.join(r, filename))
    return sorted(filename_list), sorted(filepath_list)

def listTile(path):
    dir_list = []
    dirname_list = []
    for r, d, f in os.walk(path):
        if not d:
            dir_list.append(r)
            dirname_list.append(os.path.basename(r))
    return sorted(dirname_list), sorted(dir_list)

def listTile_from_local_csvs(det_res_path, anchor_ch_id, anchor_dir):
    """Fast alternative to listTile() when tile detection is already done.
    Scans local 1_2d_raw/ for *_{anchor_ch_id}_result.csv files,
    extracts tile names, and reconstructs full paths by walking anchor_dir
    and matching on leaf-directory basename (handles both flat and nested
    tile directory layouts)."""
    suffix = f"_{anchor_ch_id}_result.csv"
    names = []
    if os.path.isdir(det_res_path):
        for fname in os.listdir(det_res_path):
            if fname.endswith(suffix):
                tile_name = fname[: -len(suffix)]
                if tile_name:
                    names.append(tile_name)
    dirnames = sorted(names)

    path_by_basename = {}
    for r, d, f in os.walk(anchor_dir):
        if not d:
            path_by_basename[os.path.basename(r)] = r

    missing = [name for name in dirnames if name not in path_by_basename]
    if missing:
        raise FileNotFoundError(
            f"❌ 在 {anchor_dir} 下找不到以下 Tile 对应的叶子目录: {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )
    pATHTILE_all = [path_by_basename[name] for name in dirnames]
    return dirnames, pATHTILE_all

def load_cached_detections(csv_path):
    detection_map = {}
    if not os.path.exists(csv_path): return detection_map
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row or len(row) < 9: continue
                try:
                    z_val = int(float(row[8]))
                    x1, y1, x2, y2 = float(row[1]), float(row[2]), float(row[3]), float(row[4])
                    class_name = row[5] # 保持读取为字符串
                    score = float(row[6])
                    
                    bbox = [x1, y1, x2, y2, score, class_name]
                    
                    if z_val not in detection_map:
                        detection_map[z_val] = []
                    detection_map[z_val].append(bbox)
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"Error loading cache {csv_path}: {e}")
        
    return detection_map

def save_run_metadata(cfg, start_time_stamp, model_hashes=None):
    """Record the current run while retaining the first checkpoint's origin."""
    save_path = os.path.join(cfg['paths']['pATHRESULT'], 'runtime_config.json')
    metadata = cfg.copy()
    previous = None
    if os.path.isfile(save_path):
        try:
            with open(save_path, encoding='utf-8') as handle:
                previous = json.load(handle)
        except (OSError, ValueError):
            previous = None

    models_info = {name: os.path.basename(path)
                   for name, path in cfg.get('models', {}).items()}
    run_info = {
        "start_time": datetime.datetime.fromtimestamp(
            start_time_stamp).isoformat(),
        "platform": platform.platform(),
        "models_used": models_info,
    }
    project_root = Path(__file__).resolve().parents[2]
    if model_hashes is None:
        model_hashes = model_fingerprints(cfg.get('models'), project_root)
    code_files = [
        project_root / 'scripts' / 'run_inference.py',
        *(project_root / 'src' / 'core').glob('*.py'),
        *(project_root / 'src' / 'utils').glob('*.py'),
    ]
    metadata['provenance'] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "model_sha256": model_hashes,
        "code_sha256": {
            str(path.relative_to(project_root)): file_sha256(path)
            for path in sorted(code_files) if path.is_file()
        },
    }
    history = list(previous.get('run_history', [])) if previous else []
    if previous and not history and previous.get('run_info'):
        history.append(previous['run_info'])
    history.append(run_info)
    metadata['run_history'] = history
    metadata['origin_config'] = (
        previous.get('origin_config', {k: v for k, v in previous.items()
                                       if k not in ('run_history', 'run_info', 'origin_run_info', 'provenance')})
        if previous else cfg.copy())
    metadata['origin_run_info'] = (
        previous.get('origin_run_info', previous.get('run_info'))
        if previous else run_info)
    metadata['run_info'] = run_info
    atomic_json(save_path, metadata)
