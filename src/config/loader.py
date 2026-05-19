import json
import os

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    return cfg
