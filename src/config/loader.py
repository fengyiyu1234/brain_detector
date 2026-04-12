import json
import os

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
        
    base_res = cfg['paths']['pATHRESULT']

    cfg['derived_paths'] = {
        "pATH_VIS_TILE": os.path.join(base_res, 'visualization_tile'),
        "pATH_NORM_CHECK": os.path.join(base_res, 'normalization_check'),
        "pATH_DET_RES": os.path.join(base_res, 'detection_results'),
        "pATH_CENTROIDS": os.path.join(base_res, 'cell_centroids')
    }

    return cfg
