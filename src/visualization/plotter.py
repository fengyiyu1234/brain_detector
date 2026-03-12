def plot_histogram_comparison(fullimg_raw, fulldraw, name_no_ext, save_path):
    """
    生成 Raw (16-bit) 和 Norm (8-bit) 的双通道直方图对比
    """
    # 提取通道 (OpenCV BGR: 1是绿, 2是红)
    raw_red = fullimg_raw[:, :, 2].flatten()
    raw_green = fullimg_raw[:, :, 1].flatten()
    norm_red = fulldraw[:, :, 2].flatten()
    norm_green = fulldraw[:, :, 1].flatten()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 左图：Raw 16-bit (0-65535)
    # 过滤掉背景 0 值，否则波峰太高看不清信号
    ax1.hist(raw_red[raw_red > 0], bins=100, color='red', alpha=0.5, label='Red (C1)')
    ax1.hist(raw_green[raw_green > 0], bins=100, color='green', alpha=0.5, label='Green (C2)')
    ax1.set_title(f"{name_no_ext} - Raw (16-bit)")
    ax1.set_xlim(0, 65535)
    ax1.legend()

    # 右图：Norm 8-bit (0-255)
    ax2.hist(norm_red[norm_red > 0], bins=100, color='red', alpha=0.5, label='Red (C1)')
    ax2.hist(norm_green[norm_green > 0], bins=100, color='green', alpha=0.5, label='Green (C2)')
    ax2.set_title(f"{name_no_ext} - Norm (8-bit)")
    ax2.set_xlim(0, 255)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig) # 关键：释放内存，防止在大规模处理时崩溃

def save_visualization_samples(final_results, dir_dict, disp_mat_fin, Z, H, W, config):
    """
    针对双通道（C1=Red, C2=Green）独立归一化的全局大图拼接预览函数
    逻辑：每隔 vISUALIZATIONSAMPLESTEP 层，连续保存 vISUALIZATIONSAMPLECOUNT 张图
    """

    # 1. 参数提取
    vis_dir = os.path.join(config['paths']['pATHRESULT'], 'vis_check')
    os.makedirs(vis_dir, exist_ok=True)
    
    root_c1 = config['paths']['channel1_dir']
    root_c2 = config['paths']['channel2_dir']
    
    # 步长和连续计数
    step = config.get('vISUALIZATIONSAMPLESTEP', 70)
    count = config.get('vISUALIZATIONSAMPLECOUNT', 10)
    
    # 2. 确定采样范围 (分段连续采样逻辑)
    z_to_save = []
    for start_z in range(0, Z, step):
        for offset in range(count):
            curr_z = start_z + offset
            if curr_z < Z:
                z_to_save.append(curr_z)
    
    # 去重并排序（防止 step < count 导致重叠）
    z_to_save = sorted(list(set(z_to_save)))

    # 3. 预先建立每个 Tile 的文件列表索引 (解决 I/O 瓶颈)
    tile_files_c1 = {}
    tile_files_c2 = {}

    print(f"Indexing tile files for {len(dir_dict)} tiles...")
    for dir_name in dir_dict.keys():
        path_c1 = dir_name if os.path.isabs(dir_name) else os.path.join(root_c1, dir_name)
        path_c2 = dir_name if os.path.isabs(dir_name) else os.path.join(root_c2, dir_name)
        
        # 1. 加载 C1 (Red) 并根据参数降采样
        if os.path.exists(path_c1):
            files_c1 = sorted([os.path.join(path_c1, f) for f in os.listdir(path_c1) 
                                             if f.lower().endswith(('.tif', '.tiff'))])
            # 如果开启了降采样，每隔一张取一张 (0, 2, 4...)
            if config['detection_params'].get('dOWNSAMPLE_Z_2X', False):
                files_c1 = files_c1[::2]
                
            tile_files_c1[dir_name] = files_c1

        # 2. 加载 C2 (Green) 并根据参数降采样
        if os.path.exists(path_c2):
            files_c2 = sorted([os.path.join(path_c2, f) for f in os.listdir(path_c2) 
                                             if f.lower().endswith(('.tif', '.tiff'))])
            # 必须与 C1 保持同样的降采样逻辑，以确保双通道融合时 Z 轴对齐
            if config['detection_params'].get('dOWNSAMPLE_Z_2X', False):
                files_c2 = files_c2[::2]
                
            tile_files_c2[dir_name] = files_c2

    # 4. 进入 Z 轴循环
    for z in tqdm(z_to_save, desc="Generating RG Global Vis"):
        save_path = os.path.join(vis_dir, f"check_z{z:06d}.tif")
        if os.path.exists(save_path): continue 

        # 准备两个通道的 16-bit 全局画布
        canvas_c1_16 = np.zeros((H, W), dtype=np.uint16)
        canvas_c2_16 = np.zeros((H, W), dtype=np.uint16)
        
        # 5. 遍历并拼接各个 Tile
        for dir_name, (row, col) in dir_dict.items():
            offsets = disp_mat_fin[row, col]
            x_off, y_off = int(offsets[0]), int(offsets[1])
            
            abs_z_tile = int(offsets[2])  # Python 索引 0-based
            actual_z_idx = z + (z_start - abs_z_tile)

            # 处理 Channel 1 (Red 通道源)
            if dir_name in tile_files_c1 and actual_z_idx < len(tile_files_c1[dir_name]):
                f_path = tile_files_c1[dir_name][actual_z_idx]
                img_c1 = cv2.imread(f_path, cv2.IMREAD_ANYDEPTH)
                if img_c1 is not None:
                    if len(img_c1.shape) == 3: img_c1 = img_c1[:,:,0]
                    h_t, w_t = img_c1.shape[:2]
                    y_e, x_e = min(y_off + h_t, H), min(x_off + w_t, W)
                    canvas_c1_16[y_off:y_e, x_off:x_e] = img_c1[0:y_e-y_off, 0:x_e-x_off]

            # 处理 Channel 2 (Green 通道源)
            if dir_name in tile_files_c2 and actual_z_idx < len(tile_files_c2[dir_name]):
                f_path = tile_files_c2[dir_name][actual_z_idx]
                img_c2 = cv2.imread(f_path, cv2.IMREAD_ANYDEPTH)
                if img_c2 is not None:
                    if len(img_c2.shape) == 3: img_c2 = img_c2[:,:,0]
                    h_t, w_t = img_c2.shape[:2]
                    y_e, x_e = min(y_off + h_t, H), min(x_off + w_t, W)
                    canvas_c2_16[y_off:y_e, x_off:x_e] = img_c2[0:y_e-y_off, 0:x_e-x_off]
        
        # 6. 独立归一化 (16-bit -> 8-bit)
        p_low = config.get('dOWNSAMPLE_PERCENTILE_LOW', 1)
        p_high = config.get('dOWNSAMPLE_PERCENTILE_HIGH', 99)
        
        # 假设 normalize_for_detection 函数已在外部定义
        c1_8bit = normalize_for_detection(canvas_c1_16, p_low, p_high)
        c2_8bit = normalize_for_detection(canvas_c2_16, p_low, p_high)

        # 7. 合并为 8-bit BGR 图像 (RG 模式)
        canvas_bgr = np.zeros((H, W, 3), dtype=np.uint8)
        canvas_bgr[:, :, 2] = c1_8bit # Red 通道
        canvas_bgr[:, :, 1] = c2_8bit # Green 通道
        canvas_bgr[:, :, 0] = 0       # Blue 通道置零

        # 8. 绘制检测框 (处理 Z+1 匹配)
        target_z_value = z + 1 # CSV 中的 Z 是从 1 开始记录的
        # 使用 np.abs < 0.5 容忍浮点误差
        current_z_results = final_results[np.abs(final_results[:, 7] - target_z_value) < 0.5]
        
        c_map = config.get('colors_map', {})
        bgr_lookup = config.get('bgr_colors', {})
        l2n = config.get('labels_to_names', {})

        for res in current_z_results:
            x1, y1, x2, y2, score, _, cls_idx, _ = res
            cls_key = str(int(cls_idx))
            
            # 获取颜色
            color_name = c_map.get(cls_key, "yellow") 
            color_bgr = bgr_lookup.get(color_name, [0, 255, 255]) # 默认黄色
            cv_color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))
            
            # 绘图（OpenCV 坐标为整数）
            cv2.rectangle(canvas_bgr, (int(x1), int(y1)), (int(x2), int(y2)), cv_color, 2)
            
            # 标签文本
            label_text = f"{l2n.get(int(cls_idx), 'Cell')} {score:.2f}"
            cv2.putText(canvas_bgr, label_text, (int(x1), int(y1)-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cv_color, 1)

        # 9. 写入文件
        cv2.imwrite(save_path, canvas_bgr)
