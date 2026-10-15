# 通用 pipeline QC 与独立 GT 验证

## 运行

从项目根目录运行。QC 读取结果目录中的 runtime_config.json，使用该次运行保存的通道、
最终坐标系、分辨率和 Z-link 参数；不会用后来修改的 inference config 覆盖它们。

```bash
python src/utils/pipeline_qc.py --config config/qc_config.json
python src/utils/pipeline_qc.py --config config/qc_config.json --samples T4 T70
python src/utils/global_qc.py --config config/qc_config.json --samples ETail
```

本地 config/qc_config.json 已列出 T4、T70、ET10、ETail，默认仅启用 T4。
samples[].enabled 控制默认选择；--samples 可以显式选择未启用的样本。
公共模板是 config_example/pipeline_qc.example.json。

每个样本默认输出到 <result_dir>/pipeline_qc/；可用 samples[].output_dir 指定其他目录。
连续运行会更新同名 QC 文件。建议正式发表前指定独立输出目录，保留每次冻结版本的报告。

Windows 路径映射在 windows_path_mappings 中设置；只在 Windows 生效。
通用 path_mappings 可映射其他机器上的路径前缀，两者都不修改源文件。
相对路径相对于项目根目录解析。HPC 上可直接读取原 Linux 路径。

脚本只依赖 Python 标准库，不加载模型、torch、numpy 或 PKL。
大量 ID 的关联检查使用临时 SQLite；内存不随 ID 总数线性增长，但临时磁盘和
运行时间随数据量增长。建议 HPC 设置 scratch_dir 到容量足够的节点本地临时目录。
relational_checks=false 可以关闭 ID 关联，但报告会明确标记未检查。
请不要同时对同一样本的同一 QC 输出目录运行多个作业，也不要在 inference 正在写结果时做最终 QC。

## 检查范围与输出

| 项目 | 实现内容 | 限制 |
|---|---|---|
| detection | tile/channel 输出、数值、框、score、Z 分布；输入 manifest 的处理状态、行数、大小 | manifest 缺失时不能证明原始图像完整 |
| filtering | 原始 = 保留 + 删除的计数及 ID 分区、过滤原因、保留率 | post_align 的双曝光融合另有输入语义，会明确跳过该项 |
| alignment | 活跃通道偏移存在、有限；最终坐标系通道零偏移 | 不重新评估配准精度，保留 validate_align_shifts.py 的独立入口 |
| geometry | XML 画布、tile 清单；成功接缝图连通性、各通道成功率、已有拟合残差 | 拟合残差不是 held-out 精度；不重新求解 |
| global | 框和坐标、去重原因；最终 tile 输入按共同 Z 范围裁剪后，与保留/去重 ID 的分区及 winner 关系 | 不重新测量接缝，也不搜索所有残余重复框 |
| tracks | 层数/跨度分布、单层和跨度上限比例；成员数、Z 范围、每层唯一性；保留/拒绝成员覆盖全局检测 | 不自动判断每个 track 的生物真实性 |
| coloc | 数值、阳性通道列表/数量、source track 关系、source box 关系、重复分配、临界匹配 | 需要 GT 才能评估误配/漏配；临界案例只是诊断样本 |
| exports | 每类质心文件数量、ID、类别归属和微米单位换算 | 质心按当前导出代码的 display-box 中心和代表 Z 定义 |
| provenance | 已有输出 manifests 的大小；可选 SHA256 | 不重新计算历史模型推理 |

verify_manifest_hashes=true 会校验已有 manifests 中输出内容的 SHA256，包括 raw CSV。
verify_input_files=true 会检查 manifest 中已处理 TIFF 的存在、大小和 mtime；不读取像素，
不检查 TIFF 尺寸或对所有图像计算清晰度。改动挂载方式或复制图像可能改变 mtime，需要解释。
这些检查不能发现 anchor 图像清单本身从未列入的切片。
需要独立于检测结果的 tile 完整性时，提供 samples[].xml 或完整的 samples[].tiles 清单。
tiles 表示该结果集的完整预期清单，不是从完整全脑结果中临时抽取的子集。

输出文件：

- qc_report.json / qc_report.md：汇总、设置、输入大小/mtime、QC 代码 hash。
- findings.csv：FAIL、WARN、NOT_CHECKED；PASS 仅表示已启用的完整性检查没有异常。
- tile_metrics.csv / tile_counts.svg：按 tile、通道、阶段计数；颜色每列独立按 log(count) 缩放。
- z_counts.csv：按切片检测数，用于后续曲线绘制。
- review_candidates.csv：固定 seed 的逐文件随机案例和前 100 个临界匹配 ID。
  逐文件抽样不会按细胞数加权，不能作为总体精度测试集。
- 量化分布：每文件最多 2048 条 reservoir 样本估计 p05/p50/p95；总计数和 ID 检查是精确的。

阈值全部在 thresholds 中。当前默认是排查异常的启发式参数，不是生物学合格标准。
空组织的空结果为 WARN；没有 rejected detection 属于正常情况。
平移后的边缘框越界为 WARN。缺文件、非法数值、ID 分区或关联错误为 FAIL。
退出码：完成且没有 FAIL 为 0；任一样本失败或配置/读文件错误为 1。
错误样本不阻止后续样本执行。出错时 qc_report 会明确记录失败。

旧命令仍可用，但输出改成上述通用报告，取消 T4/45 tile/Olig2 写死条件：

```bash
python src/utils/global_qc.py --result-dir /path/results --xml /path/frame.xml --sample-id sample_A
```

## GT 比较脚本

```bash
python src/utils/compare_qc_gt.py --config config_example/qc_ground_truth.example.json
```

模板中的文件位置需替换为自己的标注和预测 CSV。
此脚本不创建 GT，也不自动把全脑 CSV 当作完整标注 ROI。
先在冻结的 ROI 内独立标注，再将**同一范围内的所有预测**导出成以下格式。
一行代表一个对象；id 在每个 ROI/channel 内唯一。
rois 在配置中显式登记，并包含 specimen_id 和 exhaustive=true。
完全空白 ROI 仍需登记，即使 CSV 中没有任何行。

3D 对象（mode=points3d）：

```csv
roi_id,id,channel,class,cx,cy,cz
brainA_roi01,cell001,soma,glia,120,150,41
```

2D 框（mode=boxes2d）：

```csv
roi_id,id,channel,class,x1,y1,x2,y2,z
brainA_roi01,box001,GFP,neuron,110,140,130,160,41
```

预测与 GT 必须使用同一个 coordinate_frame，XY 单位为像素、Z 单位为切片。
3D 距离按 voxel_size_um=[x,y,z] 换算为微米；2D 框只在同一 Z 层匹配。
脚本不能自动识别输入是否真的处于声明的坐标系。
边界规则、ROI Z 范围及是否计入被截断细胞需要预先约定；不要根据预测结果选择 ROI 或排除对象。

输出 gt_report.json、per_roi.csv、matches.csv。
报告固定阈值下的一对一最大匹配数、precision/recall/F1、计数偏差、逐 ROI/逐 specimen 汇总。
匹配不是贪心最近邻；也不保证所有可行匹配中距离总和最小。
所以定位误差用于描述选定匹配，配准精度推荐采用对应点模式。
完全空的 GT/预测集合中未定义的指标为 null，不伪装成完美准确率。
max_pairs_per_roi 防止超大 ROI 的两两比较占满内存。

class_aware=true 时只允许同类对象匹配，可测类别特异的检测表现；
false 时先做几何匹配，再输出 matched-only 类别混淆，便于区分检测与分类错误。
可在两份 CSV 中增加 markers 列，值为 JSON 数组，如 ["Sox9","Olig2"]；
CSV 写入器应正确转义其中引号。[] 表示已确认阴性，空字段表示未标注。
markers 配置列出需要评估的标记，输出已匹配细胞上的每标记 precision/recall/F1/TN。
这些分类指标不包含漏检细胞，必须与检测 recall 一起报告。

mode=landmarks 用于配准或拼接：使用 3D 点格式，GT 与预测的 id/channel 必须逐一对应。
predictions 是已应用变换的点，ground_truth 是目标对应点，不通过距离阈值删掉大误差。
输出微米单位的平均、p50、p95 对应点误差。
选择同一个物理结构的对应点，不要假设不同 marker 的所有阳性细胞应一一对应。
landmarks 的 per_roi.csv TP 表示成功找到对应 ID，不能解读为检测准确率。
当前不计算 mask Dice、AP、track-ID 指标、置信区间或自动参数敏感性曲线。

## 面向论文的验证设计建议

自动 QC 检查计算过程是否自洽；独立 GT 检查结果是否正确。
没有必要重新画整个脑，也不必每个中间步骤都画像素 mask。
建议以论文的主要结论确定标注粒度：

| 结论/阶段 | 适合的人工 GT | 主要量化 |
|---|---|---|
| 2D 检测/过滤 | ROI 内所有目标的框或中心；包含阴性区域 | 每类 precision、recall、F1，过滤前后变化 |
| 通道配准/拼接 | 未参与拟合的对应点或重叠区域同一细胞对 | XY/Z 分轴误差及 TRE，微米单位；前后配对比较 |
| Z-link | 小型连续 3D subvolume 中的逐层 cell identity | split/merge、漏检、重复计数；细胞级 precision/recall |
| 共定位/marker 分类 | 从原始多通道图像独立判读的细胞和 marker 状态 | 每 marker/组合混淆矩阵、precision/recall/F1；阴性也需标注 |
| 全脑生物学计数 | 独立抽样区域的人工计数和细胞类型 | 区域计数偏差；配合对象级误差，避免 FP/FN 抵消 |

优先完整标注少量有代表性的 3D ROI，可同时支持最终细胞检测、Z-link 和共定位验证。
若不声称分割形状准确，不需要为了框/计数任务额外绘制每个像素的实例 mask。
当前 Z 间距为 8 um，标注和定位阈值必须考虑轴向采样，不能将单层核自动认作误检。

建议在看最终预测之前，按独立脑样本、实验组、脑区、深度、细胞密度分层随机选择 ROI。
拼接验证另包含接缝，留出空白/低信号区域。
专门挑出的失败案例用于诊断图，不与代表性随机测试集混合计算总体精度。
避免相邻切片或重叠 tile 泄漏：训练/调参/测试按 specimen 分开；
用于配准或 seam 拟合的数据不再作为其独立精度测试数据。
先用 pilot 数据估计变异和标注工作量，再决定样本量；没有适用于所有论文的固定 ROI 数。

在一部分测试 ROI 上，让两名标注者独立、对预测和实验组盲法标注，报告一致性，
再仲裁分歧；预先写明细胞边界、轴向截断、弱阳性和多核的判定规则。
若只能一人标注，可隔一段时间盲法重标部分 ROI，报告重复性，并说明局限。
同一个 soma 同时有 Sox9/Olig2 来源不一定表示同一个核双阳性，标注规则应与论文表述对应。

将参数固定后再评估测试集；如需调参，使用另一个 validation 集。
报告原始检测、过滤后、最终结果，以及关键步骤的消融或合理范围内参数敏感性。
统计单位是独立脑样本：逐 specimen 报告性能，区间估计采用 specimen 层面的
cluster bootstrap/层次分析；不要把百万细胞当作百万独立重复。
对非常少的 specimen，明确区间和泛化推断的限制。

保存并公开允许共享的 ROI 原图、GT、ROI 清单、划分、配置、模型与代码版本、
处理参数、matching 阈值和运行环境，保证指标可以从冻结输入重算。

这些设计建议与下列文献一致，具体标注数量和阈值需要结合本项目 pilot 结果确定：

- [Avoiding a replication crisis in deep-learning-based bioimage analysis, Nature Methods (2021)](https://www.nature.com/articles/s41592-021-01284-3)
- [Community-developed checklists for publishing images and image analyses, Nature Methods (2023)](https://www.nature.com/articles/s41592-023-01987-9)
- [Virtual alignment of pathology image series for multi-gigapixel whole slide images, Nature Communications (2023)](https://www.nature.com/articles/s41467-023-40218-9)

