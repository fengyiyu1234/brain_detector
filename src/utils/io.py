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

def get_c2_file_map(tile_path_c2):
    """
    在处理 Tile 前，先建立一个 {文件名: 完整路径} 的字典
    """
    c2_map = {}
    if not os.path.exists(tile_path_c2):
        return c2_map
        
    for f in os.listdir(tile_path_c2):
        if f.startswith('.'): continue # 跳过隐藏文件
        name_no_ext = os.path.splitext(f)[0]
        c2_map[name_no_ext] = os.path.join(tile_path_c2, f)
    return c2_map

def load_cached_detections(csv_path):
    """
    读取无 Header 的 CSV 检测结果。
    假设写入格式: [filename, x1, y1, x2, y2, class_name, score, mean_val, z_real]
    索引对应: row[8] -> z_real (key)
    返回: { z_real_int: np.array([[x1, y1, x2, y2, score, class_id_placeholder], ...]) }
    """
    detection_map = {}
    if not os.path.exists(csv_path):
        return detection_map

    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 9: continue
                try:
                    # 解析行数据
                    z_val = int(float(row[8])) # 第9列是 Z
                    x1, y1, x2, y2 = float(row[1]), float(row[2]), float(row[3]), float(row[4])
                    score = float(row[6])
                    # 注意：缓存里只有 class_name，没有 class_id。为了 QC 兼容，我们这里把 class 设为 -1 或 0
                    # 如果 QC 不需要具体 class，这没问题。如果需要，得反查 labels_to_names
                    cls_placeholder = 0.0 
                    
                    bbox = [x1, y1, x2, y2, score, cls_placeholder]
                    
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
    metadata['run_info'] = {
        "start_time": datetime.datetime.fromtimestamp(start_time_stamp).strftime('%Y-%m-%d %H:%M:%S'),
        "platform": platform.platform(),
        "model": os.path.basename(cfg['model_path'])
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)