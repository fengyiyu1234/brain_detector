import json
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    base_res = cfg['paths']['pATHRESULT']
    # 派生路径 (删除了 pATH_DET_YOLO)
    cfg['derived_paths'] = {
        "pATH_VIS_TILE": os.path.join(base_res, 'visualization_tile'),
        "pATH_NORM_CHECK": os.path.join(base_res, 'normalization_check'),
        "pATH_DET_RES": os.path.join(base_res, 'detection_results'),
        "pATH_CENTROIDS": os.path.join(base_res, 'cell_centroids')
    }
    # 确保 Key 为 int 方便索引
    cfg['labels_to_names'] = {int(k): v for k, v in cfg['labels_to_names'].items()}
    cfg['colors_map'] = {int(k): v for k, v in cfg['colors_map'].items()}
    cfg['bgr_colors'] = {k: v for k, v in cfg['bgr_colors'].items()}
    return cfg
