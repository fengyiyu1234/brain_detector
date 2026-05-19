import numpy as np

def normalize_for_detection(img_raw, p_low, p_high):
    # 🌟 [新增] 神奇的切片魔法：每隔 4 个像素抽样一次
    # 将 400 万像素浓缩成 25 万像素去估算分位数，速度提升 16 倍，但结果几乎完全一样！
    img_sample = img_raw[::4, ::4] 
    
    # 用降采样后的图去算分位数
    vmin = np.percentile(img_sample, p_low)
    vmax = np.percentile(img_sample, p_high)
    
    # ... 后续的拉伸和截断逻辑保持你原来的不变，但应用于原图 img_raw ...
    # 比如：
    img_norm = np.clip(img_raw, vmin, vmax)
    img_norm = (img_norm - vmin) / (vmax - vmin + 1e-8) * 255.0
    return img_norm.astype(np.uint8)