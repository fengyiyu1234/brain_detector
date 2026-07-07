import os
import numpy as np
import xml.etree.ElementTree as ET
import csv
import datetime
import json
import platform

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

def compute_grid_fallback_offsets(tile_paths, tile_size, overlap_pct):
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
            ax = raw_col if use_raw_offset else gj * step_px
            ay = raw_row if use_raw_offset else gi * step_px
            disp_mat_fin[gi, gj] = [ax, ay, 0]
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
    Scans local 1_tile_2d_raw/ for *_{anchor_ch_id}_result.csv files,
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

def save_run_metadata(cfg, start_time_stamp):
    save_path = os.path.join(cfg['paths']['pATHRESULT'], 'runtime_config.json')
    metadata = cfg.copy()
    
    # 动态记录所有使用的模型名称（适配新 config.json）
    models_info = {}
    if 'models' in cfg:
        for model_name, model_path in cfg['models'].items():
            models_info[model_name] = os.path.basename(model_path)
            
    metadata['run_info'] = {
        "start_time": datetime.datetime.fromtimestamp(start_time_stamp).strftime('%Y-%m-%d %H:%M:%S'),
        "platform": platform.platform(),
        "models_used": models_info # 替换原有的单一 model 字段
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)