import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import traceback

def generate_bbox_channels(
    csv_path, output_base_dir, image_width, image_height, total_z_slices, 
    scale_xy=0.5, scale_z=1.0, class_mapping=None
):
    print("==================================================")
    print("🚀 开始生成 Imaris 兼容的 16-bit Bounding Box 图层")
    print("==================================================")
    
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ 找不到输入文件: {csv_path}\n请确保您使用的是 global_bboxes.csv 而不是 centroids！")

        print("📂 正在读取 BBox 数据...")
        df = pd.read_csv(csv_path)
        
        # 确保数据包含坐标列 (x1, y1, x2, y2)
        required_cols = ['x1', 'y1', 'x2', 'y2', 'z', 'class']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"输入 CSV 缺少必要的列: '{col}'。请确认这是 global_bboxes.csv。")
                
        df = df.dropna(subset=required_cols)

        print(f"📐 应用坐标缩放: X/Y = {scale_xy}x, Z = {scale_z}x")
        df['x1_scaled'] = df['x1'] * scale_xy
        df['y1_scaled'] = df['y1'] * scale_xy
        df['x2_scaled'] = df['x2'] * scale_xy
        df['y2_scaled'] = df['y2'] * scale_xy
        df['z_scaled'] = ((df['z'] - 1) * scale_z).round().astype(int)

        if class_mapping is None:
            class_mapping = {0: "red_glia", 1: "green_glia", 2: "yellow_glia"}

        # 字体和画笔配置
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0  
        text_thickness = 2
        box_thickness = 3   # 边框粗细，在全脑大图上 3 个像素比较清晰
        WHITE_16BIT = 65535 # 16-bit 最大亮度

        unique_classes = df['class'].unique()
        
        for cls_id in unique_classes:
            c_name = class_mapping.get(int(cls_id), f"Class_{cls_id}")
            print(f"\n🟢 正在处理类别: {c_name}")
            
            class_output_dir = os.path.join(output_base_dir, f"Channel_{c_name}")
            os.makedirs(class_output_dir, exist_ok=True)
            
            class_df = df[df['class'] == cls_id]
            grouped_df = class_df.groupby('z_scaled')
            
            for z in tqdm(range(total_z_slices), desc=f"生成 {c_name} 图像"):
                save_path = os.path.join(class_output_dir, f"label_z{z:04d}.tif")
                
                # Checkpoint: 断点续传
                if os.path.exists(save_path) and os.path.getsize(save_path) > 100:
                    continue
                
                canvas = np.zeros((image_height, image_width), dtype=np.uint16)
                
                current_cells = pd.DataFrame()
                # 依然保持前后 1 层的容差，这样在 Imaris 里这个框会稍微有厚度，更容易被肉眼捕捉
                for dz in [z - 1, z, z + 1]:
                    if dz in grouped_df.groups:
                        current_cells = pd.concat([current_cells, grouped_df.get_group(dz)])

                if not current_cells.empty:
                    for _, cell in current_cells.iterrows():
                        x1, y1 = int(cell['x1_scaled']), int(cell['y1_scaled'])
                        x2, y2 = int(cell['x2_scaled']), int(cell['y2_scaled'])
                        
                        # 画空心矩形边界框 (Bounding Box)
                        cv2.rectangle(canvas, (x1, y1), (x2, y2), WHITE_16BIT, box_thickness)
                        
                        # --- 准备标签 (去掉了 score，只保留名称) ---
                        label_text = f"{c_name}"
                        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, text_thickness)
                        
                        # 默认把名字写在框的左上角外部
                        bg_x1, bg_y1 = x1, y1 - text_height - 6
                        bg_x2, bg_y2 = x1 + text_width + 4, y1
                        
                        # 防护机制：如果细胞太靠近图片顶部，字会跑到外面去，那就把字画在框的内部
                        if bg_y1 < 0:
                            bg_y1 = y1
                            bg_y2 = y1 + text_height + 6
                        
                        # 画白色实心背景框
                        cv2.rectangle(canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), WHITE_16BIT, -1)
                        # 用黑色字体 (0) 写上纯净的名字
                        cv2.putText(canvas, label_text, (bg_x1 + 2, bg_y2 - 3), font, font_scale, 0, text_thickness)

                # 保存为 16-bit TIF
                cv2.imwrite(save_path, canvas)

        print(f"\n🎉 任务圆满完成！带 BBox 的通道已分别保存在: {output_base_dir}")

    except Exception as e:
        print(f"\n💥 [全局崩溃] 发生了错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # ==========================================
    # 🔴 核心改动：必须使用 global_bboxes.csv 
    # ==========================================
    INPUT_CSV = r"y:\Fengyi\brain\batch2\sample1_120325\detection3_lowconf\global_bboxes.csv"
    OUTPUT_FOLDER = r"y:\Fengyi\brain\batch2\sample1_120325\detection3_lowconf\imaris_label_volume3"
    
    # 原始大图尺寸 (Resolution 1)
    RES1_WIDTH = 9060
    RES1_HEIGHT = 12476
    RES1_Z_SLICES = 561
    
    # 目标尺寸 (Resolution 2)
    TARGET_WIDTH = 4530
    TARGET_HEIGHT = 6238
    TARGET_Z_SLICES = 561

    # 自动计算缩放系数
    SCALE_X = TARGET_WIDTH / RES1_WIDTH
    SCALE_Y = TARGET_HEIGHT / RES1_HEIGHT
    SCALE_Z = TARGET_Z_SLICES / RES1_Z_SLICES
    SCALE_XY = (SCALE_X + SCALE_Y) / 2.0 

    # 您的专属 MADM 分类
    MY_CLASS_MAPPING = {
        0: "RG",
        1: "GG",
        2: "YG",
        3: "RN",
        4: "GN",
        5: "YN"
    }

    generate_bbox_channels(
        csv_path=INPUT_CSV,
        output_base_dir=OUTPUT_FOLDER,
        image_width=TARGET_WIDTH,
        image_height=TARGET_HEIGHT,
        total_z_slices=TARGET_Z_SLICES,
        scale_xy=SCALE_XY,
        scale_z=SCALE_Z,
        class_mapping=MY_CLASS_MAPPING
    )

