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
    extracts tile names, and reconstructs full paths under anchor_dir."""
    suffix = f"_{anchor_ch_id}_result.csv"
    names = []
    if os.path.isdir(det_res_path):
        for fname in os.listdir(det_res_path):
            if fname.endswith(suffix):
                tile_name = fname[: -len(suffix)]
                if tile_name:
                    names.append(tile_name)
    dirnames = sorted(names)
    pATHTILE_all = [os.path.join(anchor_dir, name) for name in dirnames]
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