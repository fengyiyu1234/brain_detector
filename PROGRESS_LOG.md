# brain_detector 工作日志

本文件记录每次工作会话的内容，方便新开的 Claude 对话快速了解当前进度，也方便同步到实验 journal。

每条记录包含：日期 / 做了什么 / 关键决定 / 遇到的问题 / 下一步。

---

## 2026-09-16 ~ 2026-09-17

背景：EGFR T4 样本的 561nm RFP 背景噪声太高，不能再像 TSC 样本那样用 561 同时做拼接参考和通道对齐参考。
新方案：488nm（Olig2，信号最密）做拼接参考，640nm GFP（稀疏细胞体，主要检测对象）做通道对齐参考。
随之而来的问题：细胞全局坐标 = XML tile 位置 + tile 内坐标（对齐参考坐标系），两个参考不一致时，
中间缺了每个 tile 的 GFP↔Olig2 位移。

### 做了什么

**1. 拼接质量比较脚本 `src/utils/compare_stitching.py`（新增）**
- A. TeraStitcher 自评：各轴被替换成机械默认位移的比例、子块位移一致性、最终摆放与相邻对位移的残差。
- B. 接缝残差：重叠区同一细胞在两个 tile 的检测做 z-link + 互为最近邻配对，中位数差即接缝误差；
  `--also CH` 先用 Stage 2.5 偏移把 CH 换到拼接通道坐标系再测（= 细胞最终落在该拼接图上的误差）。
- C. 两份 XML 交叉验证：`pos_A − pos_B` 应等于 `s_a − s_b` + 常数。
- 验证：在 XML 里人为移动一个 tile (+7,−5,+2)，相关 4 条接缝测出的变化误差 < 0.15 px；
  sample18 上构造的自洽"第二份 XML"，C 部分正确标出加噪 tile、还原常数。
- 修了 C 部分不管结果如何都打印"互相印证"的 bug，改为按 `std_residual / std_pos_diff` 给结论；
  可用 tile < 3 时跳过。

**2. T4 640 vs 488 拼接比较结果**
- 两份整体相当：64 条可比接缝，xy 误差均值 5.8 vs 5.3 µm，>10 µm 的接缝各 13 条，坏接缝基本不重合。
- 大误差几乎全部来自 TeraStitcher 被换成默认位移的接缝：算出来的接缝 488 中位误差 1.3 µm（640 为 1.9 µm），
  被替换的接缝 8.9–14.3 µm。两份拼接各有约 1/3 的相邻对被替换。

**3. Stage 2.5 并行化（`run_inference.py`、`point_cloud_aligner.py`）**
- Stage 2.5 纯 CPU，原来主进程串行、4 块 GPU 闲置。单 tile 流程抽成 `align_tile()`，CPU 进程池并行，
  进程数 `pre_align_params.n_workers`（缺省 = `$SLURM_CPUS_PER_TASK`）。
- 续跑：offsets JSON 最后写，作为完成标记；CSV/JSON 先写 `.part` 再改名；判断完成时核对对齐 CSV 行数与原始 CSV 一致；
  主通道原始 CSV 缺失的 tile 不算完成（让缺失能被报出来）。
- 验证：T4 上 3 个 tile 的偏移和 4 通道 CSV 与 HPC 旧代码完全一致；`303600_361200` 不一致，
  但该 tile 几乎无细胞、Sox9/Olig2 得分为 0（本机串行重复计算结果与并行一致，属于平坦得分面上的平台差异）。
- 新增 `scripts/inference_cpu.slurm`（不申请 GPU）。

**4. Stage 3 提速（结果与旧代码逐字节一致）**
- `combine_predictions`：每个框都 `np.concatenate` 整层 → 每 tile 每层只拼一次（平方级 → 线性）。
- `run_z_linker`：每层整脑一个匈牙利矩阵 → 只在 IoU>0 的连通块内求解（`solver='sparse'` 默认，`'dense'` 保留作对照）；
  类别字符串解析缓存、中位数改纯 Python。等价性论证见 `z_linker._match_sparse` docstring。
- 各通道"全局拼接 + Z-Link"改为每通道一个进程（新文件 `src/core/channel_stage3.py`，`stage3_n_workers`）。
- 3B `annotate_soma_with_tf_containment`：批量 KD 查询 + 数组化包含判断，只有通过包含判断的对沿用原距离逻辑。
- sample18 实测（本机）：Stage 3 **132 min → 10.5 min**；Sox9 全局拼接 19.5 min → ~2 min，Z-Link 56 min → ~3.5 min，3B 636 s → 136 s。
- 所有最终输出（coloc_result、各类别 CSV、统计报告、质心文件）与旧代码逐字节相同；
  只有中间文件 `Sox9_3d_tracked` 369 万细胞中约 140 个不同，全部是代价完全相同的并列最优解（逐层核对总代价相等）。

**5. Olig2 → GFP 通道位移诊断与修复**
- 发现 T4 上 Olig2 → GFP 位移不可信：xy 分散在 ±36 px（29% 顶到搜索边界），得分中位数 0.006；
  C 交叉验证中加上这些位移后残差（x 25.6 / y 29.3 px）反而大于两份 XML 的位置差（13.2 / 20.3 px）。
- 6 个 tile 大范围直接扫描包含得分（xy ±42 px，z −16~+8）：5 个有明显峰（峰值为背景 2.2–3 倍），
  大致在 dx −4~−7（1 个为 +8）、dy 0~3、**dz −6~−7**；细胞太少的 `303600_349900` 无峰。
  现有算法只在 2/5 个 tile 上找对，其余被 FFT 粗搜索带到错误局部峰（如 dx=33），±8 px 精搜索回不来。
- 修复：`find_shift_containment` 新增粗搜索 `coarse='displacement_hist'`（默认）——统计 soma−TF 质心位移的 3D 直方图，
  用门限半径平滑，取前 3 个峰各做 ±2 px/±1 层小范围包含打分，最好的作为精搜索中心。
  `'fft'` 保留，仅用于复现旧结果。
- 新配置项 `pre_align_params.containment_coarse`，写入 `_align_settings.json`；旧的 settings 文件视为 `'fft'`，
  所以对已用旧代码对齐过的样本重跑会报设置不一致（删 `0_channel_alignment/` 重跑，或设 `"containment_coarse": "fft"` 沿用旧结果）。
- T4 全部 45 tile 本机重算（临时目录，未动正式结果），与原结果对比：
  - A（新粗搜索，z±5）：**不够**，Olig2 仍然分散（std 14.5/12.4/2.6），真实 dz≈−7 超出了 z 搜索窗口。
  - B（新粗搜索，z±10）：Olig2 中位 (−3,0,−7)，MAD (5,1,1)，std (9.5,4.7,3.4)，得分中位 0.011（原 0.007）；
    扫描过的 5 个 tile 与扫描峰全部吻合（误差 ≤2 px / 1 层）。
  - 剩余偏差并非全是噪声：第 1 列（327300）的 tile 一致地落在 (+4~+9, 0~2, −6~−7)，其他列约 (−3~−5, 0, −7)，
    说明 Olig2↔GFP 位移随 x 位置变化；真正的离群是 5 个得分为 0 的空 tile（都给出无意义的 (−10,−10,−3)）
    和边缘行/列上细胞稀少的 tile。
  - Sox9 也走包含步骤，新方法下 22 个 tile 结果改变，std 从 (9.4,9.4,2.0) 降到约 (5,7,1.6)；RFP（FFT 步骤）只有 2 个 tile 变化。
  - C 交叉验证（640 vs 488 XML）仍不通过（B：残差 std 19.7/21.7 vs 位置差 13.8/22.7）。两份 XML 各有约 1/3 接缝被换成默认位移，
    XML 自身的误差就有 10–30 px，这个检验在 T4 上被拼接误差主导，暂时不能作为对齐是否正确的判据。
- README 里"z 搜索软上限 ±5 / 硬上限 ±10"的描述与代码不符（代码没有这个限制），已改为实际的搜索窗口说明。

**6. 溯源信息（tile_name / slice_name）**
- 确认该功能正常：sample18 的 coloc_result 20.4 万细胞全部有 tile/slice，且 slice 都属于对应 tile；质心文件同样保留。
- 修复原有漏洞：`2_global_2d_raw/*.csv` 不存 tile/slice，从它续跑 Stage 3 时溯源丢失（全是 Unknown 或匹配到别的通道的框）。
  现在全局 2D CSV 每行多存 `tile_name`/`slice_name`，续跑时据此恢复；遇到旧版文件给出警告。
- 验证（sample18）：从头跑，全局 2D CSV 原 8 列与旧代码完全相同、新增两列无空值，其余最终输出逐字节相同；
  保留全局 2D + pkl 只删 4_/5_ 续跑，coloc_result 所有列（含 tile_name/slice_name）与旧代码逐行一致，无 Unknown。
- 注意：slice_name 是**参考通道坐标系**的层；有 dz 的通道回原图要减去 `0_channel_alignment/<tile>_offsets.json` 里的 dz。

### 关键决定
- 配准用的 moving image 是 488 XML 合并出的全脑图，所以细胞最终必须落在 488 坐标系；不采用"写 nii header"的方案，
  坐标修正直接作用在细胞坐标上。
- 长期方案（待实现）：用各通道自己的接缝细胞配对求每个 tile 的位置 P_c，给每个通道写各自的 merging XML，
  TeraStitcher 只做 merge；这样通道位移 = P_GFP − P_Olig2 + 常数，不依赖 GFP 与 Olig2 标记同一批细胞。
  需注意各通道 merge 的输出范围要一致（TeraStitcher `--V0/--V1/...` 的原点语义待确认），561/730 需先 `--import`。
- T4 暂设 `stop_before_stitching: true`，Stage 2.5 结果和 tile 位置确定前不跑 Stage 3（Stage 3 现在约 10 分钟，重跑代价低）。
- 得分为 0 的偏移是任意值，下游（tile 位置求解等）应改用邻居中位数或判为无效。

### 遇到的问题
- CPU 分区提交 `--mem=240G` 报 `QOSMaxMemoryPerNode`；`inference_cpu.slurm` 改为 12 核 / 64G（上限待用 `sacctmgr` 查）。
  859055 实际是用该 CPU 脚本提交的，落在 gpu14 上但未预留 GPU。
- brain_detector 环境的 matplotlib 画图会崩进程；`compare_stitching.py` 改为所有 CSV 写完后再画图，画图用 antsreg 环境。
- 本机停止扫描进程时误带了 `taskkill //F //FI "WINDOWTITLE eq *"`（会尝试强杀所有带窗口的进程）；
  事后核对窗口程序均仍在运行，未发现被关闭的程序。之后不再使用这类大范围结束进程的命令。

### 进行中 / 下一步
- T4 正式结果（HPC 上的 `0_channel_alignment/`）仍是旧 FFT 方法算的，**Olig2/Sox9 偏移尚未更新**。
  下一步：T4 配置改 `z_search_range_slices: 10`，删 `0_channel_alignment/` 后在 HPC 上重跑 Stage 2.5。
- 离群偏移处理（待实现）：得分为 0 / 过低、或与相邻 tile 明显不一致的偏移，改用相邻 tile 的中位数（位移随位置平滑变化，
  不能用全局中位数），并在 offsets JSON 里记录原值和替换原因。
- 用 `validate_align_shifts.py` 在 held-out 数据上检验 T4 各通道偏移（RFP/Sox9 目前看起来稳定但未独立验证）。
- 写 `src/core/solve_tile_positions.py`：接缝配对 → 加权稳健最小二乘求各通道 tile 位置 → 各通道 merging XML + 残差报告。
- T10 检测：计划在 gpu14 上用 4 块 L40（gpu02 的 RTX 2080 显存不够），提交前确认节点 CPU/内存余量。
- git：1–4 以及 5 之前的改动已在 `abb1cb4 pipeline update` 提交；之后的改动（`containment_coarse` 粗搜索、
  溯源修复、`compare_stitching.py` C 部分结论修正、README、本日志）尚未提交。
  `config_EGFR_t4.json`、`inference_cpu.slurm` 被 .gitignore 忽略，不在版本库里。

---

## 2026-09-17（下午）

问题：Olig2/Sox9 这类核通道怎么做 channel alignment。想法：不用 TeraStitcher 的 merging XML 做全局坐标转换，
tile 名字已经是台面坐标、tile 间有 15% 重叠，可以直接用相邻 tile 重叠区里的检测配对求 tile 间位移。
T4 四个通道的检测都已完成，可以用来检验。

### 做了什么

**1. 在 T4 上验证了这个思路（4 通道 × 76 条接缝）**
- 用 tile 名字（台面坐标，0.1 µm/单位）当初值，复用 `compare_stitching.py` 的接缝估计
  （重叠区 z-link → 差值直方图粗对 → 互为最近邻迭代 → 取中位数），逐通道测相邻 tile 位移。
- 接缝测量本身非常干净（成功接缝 / 配对数中位 / 配对差 MAD(x,y,z)）：
  GFP 64/76, 648, (1.7,1.2,0.5)；RFP 65/76, 422, (1.8,1.0,0.5)；
  Olig2 66/76, 5218, (1.3,0.8,0.5)；Sox9 66/76, 14926, (1.5,1.2,0.5)。
  位移中位数的标准误：稠密核通道 0.02–0.03 px，soma 通道 0.07–0.15 px。
- 失败的 10–12 条/通道全部集中在同一批没有细胞的边角 tile，四通道一致，不是算法问题。
- 名义网格本身不够用：解出来的位置相对名义网格 p50 (19, 41, 3.3)、max (74, 124, 8)，
  所以接缝求解这一步必须做，但**不需要 TeraStitcher**。

**2. 关键发现：通道之间不是一个平移，而是一个倍率 / 倾斜场**
- 把 `P_Olig2(t) − P_GFP(t)` 和两份 XML 的 `pos_488(t) − pos_640(t)` 逐 tile 比（方法完全独立：
  细胞质心配对 vs 图像 NCC）：x corr +0.72、y corr +0.84、**z corr +0.98**（差的 std 0.64 层，
  两边各自离散度 ±3 层）。通道间的逐 tile 位移是真实存在的物理量。
- 它随位置线性变化：`dy` 随行号从 +19 px 走到 −21 px，`dz` 随列号从 −4 层走到 +5 层。
- 用「共用 tile 位置 + 每通道仿射通道场」联合拟合出的斜率（相对 GFP/640）：
  | 通道 | x (px/列) | y (px/行) | z (层/列) |
  |---|---|---|---|
  | Olig2 (488) | +4.43 | +5.25 | −2.09 |
  | RFP (561) | +0.09 | +0.96 | −1.02 |
  | Sox9 (730) | −0.07 | −0.03 | +0.25 |
  z 斜率按波长单调、在参考通道 640 附近过零、730 变号 —— 色差（放大率差 + 光片焦面倾斜差）该有的样子。
  y 的 5.25 px/行 = 5.25/1738 ≈ **0.30% 放大率差**，全脑累计约 40 px。
  这解释了 9-16 记录里「第 1 列 tile 的 Olig2 偏移一致偏 +4~+9」——那是场，不是噪声。
- 顺带发现**纯平移拼接模型本身的天花板**：`337500 → 348800` 那条行边界上，各通道 x 残差一致地
  随列号从 +24 px 走到 −21 px，对应两排 tile 之间约 0.4° 的相对旋转。TeraStitcher 同样表达不了。
  每通道各解各的时候，这个不自洽被各通道分摊得不一样，会冒充成 8–13 px 的"通道位移"。

**3. 新脚本 `src/core/solve_tile_positions.py`**
- 输入只有 `1_tile_2d_raw/` 和 tile 名字，不需要任何 XML。
- 模型 `joint`（默认）：`P_c(t) = P(t) + delta_c(t)`，`delta_c(t) = a_c + b_c*行 + c_c*列`，参考通道 delta ≡ 0。
  所有通道的接缝一起约束共用的 `P(t)`（旋转那类不自洽被它统一吸收），每通道只留 2 个斜率/轴。
  拟合残差 p50 x 1.9 / y 0.74 / z 0.19 px，与每通道独立解（`--model free`，1.2 / 0.53 / 0.16）相当。
  没有接缝的 tile 由「拉回名义网格」的弱先验撑住；joint 模式下只要该 tile 在**任一**通道有接缝就能定位。
- 权重 = 位移中位数的标准误倒数（1.4826·MAD/√n），IRLS + Huber 鲁棒化。
- 输出 `5_analysis_report/tile_positions/`：`seams.csv`（可复用，`--redo-seams` 重测）、
  `tile_positions.csv`、`solution.json`、`report.txt`；`--write-xml` 出各通道 merging XML
  （561/730 没有自己的 XML，借用参考通道的模板改写 `stacks_dir`/`mdata_bin`，仍需先 `--import` 生成 mdata.bin）；
  `--write-aligned` 写成 `0_channel_alignment` 格式（offsets JSON + 平移后的 CSV），默认写到
  `0_channel_alignment_solved/`，不覆盖原结果。
- 生成的 XML 用 `src/utils/io.loadTeraxml` 复核可正常解析。

### 关键决定
- 逐 tile 的几何由**同通道**接缝匹配决定（每条几百到上万个配对），不再依赖 GFP 与 Olig2 标记同一批细胞。
- 跨通道匹配只剩每通道 3 个常数 `a_c`（接缝只约束 delta 的差，Olig2 那个 dz≈−7 层就在这里面）。
  这 3 个数应当拿全脑所有 tile 汇总来估，而不是每个 tile 各求一次。
- `solve_tile_positions.py` 暂时用旧 Stage 2.5 偏移的中位数当 `a_c`（`--const-from offsets`，
  报告里会打印「旧偏移减掉通道场后的离散度」）。T4 上 RFP (2.3,3.4,1.5)、Sox9 (3.1,3.0,0.7) 尚可，
  Olig2 (23.3,25.2,6.1) 说明旧 Olig2 偏移基本是噪声，这个常数必须换更好的估计。
- 不改配置、不删已有的 `0_channel_alignment/`：新脚本独立跑，输出到新目录，要用时再顶替。
- `paths.pATHXML` **保留**，但角色反过来了：不再是 TeraStitcher 算出来的输入，而是我们解出来、
  同时喂给 `teraconverter` merge 和 Stage 3 的那一份。理由：teraconverter 只认 merging XML，
  而「细胞坐标和 merge 出配准图的 XML 必须是同一份」这条不变式（`run_inference.py` 里已有的注释）
  最好靠**同一个文件**来保证，别再多一份 CSV 各算各的。而且 pATHXML 不显式指定的话，流程会退回去
  捡 `anchor_dir/xml_merging.xml`，也就是旧的 TeraStitcher 结果——所以现在更得写死。
- **两个「参考」拆开**（脚本里是 `--ref` 和 `--frame`）：
  - `--ref`（通道对齐参考）= **GFP**。通道场 `delta_c` 和常数 `a_c` 都相对它测量。理由：Stage 3B
    判的是「GFP 细胞体里有没有 TF 核」，要让 GFP↔各核通道成为直接测量的那一对，误差才落在最该准的地方。
  - `--frame`（全局坐标系）= **Olig2**。细胞和配准用的全脑图都落在 488 的几何上，配准图最干净。
  - 两者的换算是精确的规范变换，不重测任何东西：`s_c(t) = [delta_c + a_c] − [delta_frame + a_frame]`。
    T4 上已跑出 `--ref GFP --frame Olig2` 的结果（`s_Olig2 ≡ 0`，GFP 拿到 ±20 px / ±4 层的逐 tile 偏移）。
  - 代价：GFP 细胞坐标不再是"原样不动"，会带上 Olig2 通道场和取整（≤0.5 px）；
    而且 GFP 的 `dz ≠ 0`，**GFP 细胞的 `slice_name` 也变成坐标系通道的层号**，回原图查看要减 dz。
    （T4 四个通道的层名和层数完全一致，所以只是层号语义问题，不会指到别的文件。）
- Stage 3 不读 `reference_channel`（只有 Stage 2.5 和 `validate_align_shifts.py` 读），
  所以 frame 换成 Olig2 没有阻碍；但 `validate_align_shifts.py` 会按 config 里的 GFP 解读偏移，
  用它检验新结果前要注意这一点。

### 遇到的问题
- 原型里一开始把接缝残差的符号弄反了（残差是"摆放误差"，要**减**不要加）。表现很隐蔽：
  图的自洽性（拟合残差）不受影响，只有和 XML 对比时符号相反才暴露。已在脚本里注释清楚。
- brain_detector 环境里 `np.corrcoef` / `np.polyfit` 会直接崩进程（和 matplotlib 崩溃同源，疑似 BLAS），
  无 traceback、退出码 127。脚本里避免用这些，改手写协方差；`scipy.sparse.linalg.lsqr` 正常。
- 稀疏通道（GFP）检测 CSV 的最大 z 远小于真实层数（485 vs 813），拿它当切片数会把 z 末段的细胞全剔掉。
  `count_slices` 改为优先数 tile 目录里的 TIFF，退而取**所有通道**的最大 z。

### 进行中 / 下一步
- **全局常数 `a_c`（最要紧的一项）**：在 `point_cloud_aligner` 里加"全脑汇总"模式——各 tile 先按
  `delta_c(t)` 归位，再把 soma−核的质心位移合并成一个直方图 / 包含度打分，一次求 3 个数。
  数据量比现在的逐 tile 大几十倍，而未知数只有 3 个。
- 空 tile（T4 有 5 个角落 tile 一条接缝都没有）：目前落在名义网格上。它们几乎没有细胞，
  暂不做 intensity 相位相关兜底，只在报告里标出来。
- 验证：`validate_align_shifts.py` 的 held-out 思路；新旧偏移下 GFP soma 被 TF 核包含的比例对比。
- Stage 2.5 的逐 tile 包含搜索在新方案下只用于估常数，`z_search_range_slices: 10` 那条改动可以不做了。
- T10 检测：计划在 gpu14 上用 4 块 L40。
- git：`solve_tile_positions.py` 与本日志尚未提交（连同上一条记录里未提交的改动）。

---

## 2026-09-17（晚）常数估计写进脚本；整条思路定型

### 思路总览（新会话先看这段）

细胞的全局坐标 = `tile 位置 + tile 内坐标`，而每个通道有自己的一套 tile 位置
`P_c(t) = P(t) + delta_c(t)`。整条链子拆成三段，每段用它最可靠的数据来定：

1. **tile 之间的相对位置** ← 同通道接缝配对。相邻 tile 重叠 15%，同一个细胞被两个 tile 各检测
   一次，配上就得到位移。稠密核通道每条接缝几千上万个配对，位移中位数标准误 0.02 px。
   **不需要 TeraStitcher 的位移计算。**
2. **通道之间随位置变化的部分** `delta_c(t)` ← 由两个通道各自的接缝解之差得到，用仿射场
   （行/列一次项）表达。这是真实的色差（倍率差 + 光片倾斜差），T4 上全脑累计约 40 px / 9 层。
3. **每通道一个整体平移** `a_c`（3 个数）← 只能靠跨通道匹配。接缝对它完全无约束：把某通道所有
   tile 一起平移，每条接缝依然精确自洽。所以把**全脑所有 tile 的细胞按解出来的位置摆进同一套
   坐标，一次估这 3 个数**，而不是像旧 Stage 2.5 那样每个 tile 各估一次。

两个「参考」分开：`--ref GFP`（通道对齐的测量基准，MADM 细胞是主角，Stage 3B 判的也是
「GFP 细胞体里有没有 TF 核」）、`--frame Olig2`（全局坐标系，配准图用 488 最干净）。
两者之间是精确的规范变换：`s_c(t) = [delta_c + a_c] − [delta_frame + a_frame]`。

TeraStitcher 从此只做 merge，XML 从**输入**变成**输出**（`paths.pATHXML` 指向我们生成的那份）。

### 本次代码改动

**1. `src/core/point_cloud_aligner.py` 重构（不改变 Stage 2.5 行为）**
- 从 `find_shift_containment` 里抽出三个函数：`_coarse_from_peaks`（位移直方图粗搜索）、
  `_fine_containment`（包含度精搜索）、`containment_shift_from_arrays`（数组版入口，
  给 solve_tile_positions.py 用汇总点云调用）。
- `_fine_containment` 顺带优化：候选 (soma, TF) 对**只从 KD 树取一次**，半径放大到覆盖整个精搜索窗，
  之后每个候选位移只做纯 numpy 判断。这是精确等价而非近似——包含关系本身就要求两个质心的距离
  不超过 soma 的半对角线（≤ max_radius），所以放大半径只会多出必然判不过的对。
  取候选对（每个 TF 细胞一个 Python list）原本占了绝大部分时间：单 tile 405 个候选位移
  **73 s → 12 s**；汇总点云（8 万核）上这一步是能不能跑的分界线。

**2. `src/core/solve_tile_positions.py`**
- `--const-from pooled`（默认）：并行把每个 tile 每个通道的中心 z 窗 z-link 成紧凑数组、
  加上该通道解出的 tile 位置，拼成全脑点云，然后
  soma↔TF 用包含度打分（与 Stage 3B 同一判据）、同类型通道用质心位移直方图 + 互为最近邻。
  另有 `--const`（直接给死）和 `--const-from offsets`（取旧结果中位数）两条路。
- `--ref` / `--frame` 分离（见上）。`s_*` 列是搬到 frame 的量，`field_*`/常数仍相对 ref。
- 报告里同时打印旧 Stage 2.5 的中位数与逐 tile 离散度作对照，`solution.json` 存常数来源、
  包含率/配对数、抽样参数，可追溯。
- 抽样参数：`--const-max-cells`（精搜索，默认 8 万）、`--const-coarse-cells`（粗搜索，默认 2.5 万）、
  `--const-z-window`（默认 `sample_z_center_count`）、`--const-win-xy/z`、`--const-fine-xy/z`。

**3. T4 上的汇总规模**（中心 ±25 层，45 个 tile）
GFP 48165 soma / RFP 32081 / Olig2 494716 核 / Sox9 1144307 核。
旧 Stage 2.5 每个 tile 只有约 1–3 千 GFP soma 可用——这就是 45 倍的差别。

**3b. held-out 对照（新方法 vs 旧 Stage 2.5，唯一公平的"哪个更好"）**
- 求解完成后，脚本另取每个 tile 中心窗**外**、隔开半个窗（`--holdout-gap`，默认 1.5 个窗厚）的
  同样厚度 z 段，在这段没参与任何估计的数据上，用**同一批细胞、同一判据**（Stage 3B 的包含度）
  给两套摆放各打一次分：新方法（通道场 + 3 个常数）vs 旧的逐 tile 偏移。
- 为什么必须 held-out：旧方法是逐 tile 直接最大化这个分数的（45×3 = 135 个自由参数），
  在它自己用过的 z 窗上必然占便宜。新方法每通道只有 3 个常数 + 2 个斜率/轴，
  要在没见过的数据上赢才算真赢。
- 抽样对两套摆放抽**同一批行**（行序来自同一个 parts 字典），`--holdout-max-cells` 默认 30 万。
- 记住：结论对常数的估计质量极其敏感。抽样太小时常数本身是噪声，这个对照只反映那一点。

**4. HPC 上跑的两个入口（本机太慢，正式计算搬过去）**
- `src/utils/check_containment_equivalence.py`（新增）：重构等价性检验，做两件事——
  A 穷举精搜索（逐候选调用 `_containment_score`）vs 新的 `_fine_containment`；
  B 重构前的整个 `find_shift_containment` 函数体（原样抄在脚本里作参照）vs 现在的实现，
  `displacement_hist` / `fft` 两种粗搜索都测。有不一致就非零退出。
  参数从 `runtime_config.json` 解析，与 Stage 2.5 同源；默认按 CSV 大小自动挑细胞多的 tile。
- `scripts/solve_tile_positions.slurm`（新增）：纯 CPU 作业，16 核 / 120G / 8 小时。
  `CHECK=1` 先跑等价性检验（不通过就直接退出，不出求解结果）；
  `WRITE_ALIGNED=1` 才写 `0_channel_alignment_solved/`；`REDO_SEAMS=1` 重测接缝。
  `SAMPLE/REF/FRAME/WORKERS` 都可用 `--export` 覆盖。

### 验证状态
- ✅ 新 `_fine_containment` 与穷举调用 `_containment_score`：合成数据上 argmax 与计数完全一致。
- ✅ 真实数据（T4）上也一致，本机已过的样本点：
  - `303600_327300` Olig2（soma 392 / 核 6046）：A 两者都给 (2, 10, −2)、33 个包含，新版快 26×；
    B `displacement_hist` 旧 = 新 = (2, 10, −2, 0.00546)。
  - `326200_338600` Olig2（soma 2695 / 核 20967）：B `displacement_hist` 旧 = 新 =
    (2, −7, 2, 0.00692)；A 精搜索 405 个候选位移 73 s → 12 s。
- ⏳ 完整的一轮（多 tile × Sox9/Olig2 × 含 `fft` 粗搜索）改在 HPC 上跑：`CHECK=1`。
- ⏳ T4 的三个常数改在 HPC 上跑（本机 Y: 盘 I/O 被其他任务占满，进程 CPU 时间远小于墙钟）。
- 本机用**故意调小的抽样**（每边 4000 个细胞、精搜索 ±1 px）跑通了汇总 + held-out 全流程：
  汇总 33 s、held-out 汇总 33 s。这一轮的数值**不能用来判方法**——包含率只有 0.002
  （匹配到的核只有个位数），常数本身就是噪声，held-out 对照因此显示"旧方法更好"
  （Sox9 12851 vs 2852、Olig2 3577 vs 1579）。**正式跑（默认抽样）之后才看这两行。**
  唯一一个这轮就可信的数是 RFP 的常数：(+2.6, −1.9, −2.4)，9914 个配对、MAD (1.8, 2.0, 0.7)，
  与旧 Stage 2.5 的中位数 (+2.4, −3.5, −2.4) 只差 (0.2, 1.6, 0.0)——soma↔soma 的质心配对
  这条路本来就好定，两种方法互相印证。

### 遇到的问题
- 本机同时有别的 python 任务在跑（4 个进程各 25 CPU-小时），CPU 占 ~26/56 核、Y: 盘 I/O 饱和，
  本会话的脚本因此明显变慢（进程 CPU 时间远小于墙钟时间，是 I/O 等待）。跑正式的求解建议在 HPC 上，
  或等本机空下来。

### 下一步（按顺序，都在 HPC 上）
0. **等价性检验 + 求解一次跑完**（`cd .../brain_detector/scripts`）：
   `sbatch --export=ALL,CHECK=1 solve_tile_positions.slurm`
   检验不通过会直接退出、不产出求解结果。以后重跑不用再带 `CHECK=1`。
1. **看报告**：`5_analysis_report/tile_positions/report.txt`——三个常数的包含率/配对数、
   与旧 Stage 2.5 中位数的差、接缝拟合残差、通道场斜率。
2. **写对齐结果**：`sbatch --export=ALL,WRITE_ALIGNED=1 solve_tile_positions.slurm`
   （接缝会复用已有的 `seams.csv`，只重算常数和输出），写到 `0_channel_alignment_solved/`。
3. **顶替**：`0_channel_alignment` 改名备份 → `0_channel_alignment_solved` 改名顶上 →
   删 `1_tile_2d_filtered/`（它是按旧偏移生成的；`1_tile_2d_fused`、`2_global_2d_raw`、
   `3_channel_3d`、`4_colocalization` 现在都是空的）。
4. **merge 配准用的全脑图**：用 `tile_positions/xml_merging_Olig2.xml` 跑 teraconverter merge。
5. **config**：`paths.pATHXML` 指到同一份 XML，`stop_before_stitching` 改 `false`，跑 Stage 3+。
6. **评估**（我来做，需要 1–5 的产物）：
   - 接缝拟合残差、通道场斜率是否仍符合色差的物理预期；
   - 三个常数的包含率 vs 旧偏移下的包含率（同一判据，可直接比）；
   - `compare_stitching.py` 的 B 部分用新 XML 测接缝残差，应显著优于原 640/488 XML
     （原来算出来的接缝中位误差 1.3–1.9 µm，被替换成机械默认的 8.9–14.3 µm）；
   - Stage 3 之后：coloc_result 里 GFP soma 带 TF 核的比例、各类别计数，与 sample18 对比；
   - `validate_align_shifts.py` 的 held-out 检验（注意它按 config 的 `reference_channel` 解读偏移，
     frame=Olig2 时会误判，用前要么改 config 要么给它加参数）。

---

## 2026-09-18 全局模型被证伪；加逐 tile 局部精修

### 结论先说

**「仿射通道场 + 每通道一个常数」不够用。** T4 上它在**自己的拟合数据**上就输给旧 Stage 2.5
的逐 tile 偏移 1.6–2.1×，held-out 上比值几乎不变——两边都不怎么掉分，说明逐 tile 通道位移里
有仿射场表达不了的真实成分，**不是旧方法过拟合**。9-16 那条「旧 Olig2 偏移基本是噪声」的判断
要推翻：即使一部分 tile 搜错了峰，多数 tile 各自搜对，仍然胜过「每个 tile 都错同样一截」的全局解。

所以方案改成三段不变、末尾加一步：全局解当先验，再用该 tile 自己的细胞在它附近做小范围精修。

### 做了什么

**1. 修了 held-out 对照的 bug（`solve_tile_positions.py`）**
- 原来 `off_new = off_P`，只有接缝解出的 tile 位置，**常数 `a_c` 从来没加进去**。
  Olig2 少了 (−3, −30, −6)、Sox9 少了 (+10, +6, −4)，等于拿「缺常数的新方法」比「完整的旧偏移」。
  870114 那一轮的 held-out 两行因此作废（`tile_positions.csv` 的 `s_*` 列一直是对的，只有评估环节漏了）。
- 顺带：两套摆放的 dz 不同，取 z 段时谁的 dz 离窗远谁被截掉一截。改成 TF 通道的 z 窗按各套 dz
  的最大差额往两边放宽、soma 保持原厚度，分母同样放大，对各套一视同仁。

**2. 新增 in-sample 对照**（875142 的输出里已有）
- 就在估常数用的那段 z 上也比一次。用来区分过拟合和欠拟合：新方法在这里已经是它的最优，
  旧偏移要是在**同一批细胞**上还赢，那就是模型不够。T4 的答案是后者，脚本会直接打印这句警告。

**3. 875142 的实测（修好之后，四通道 45 tile）**
| | in-sample 新 | in-sample 旧 | held-out 新 | held-out 旧 |
|---|---|---|---|---|
| Sox9  | 2490 (0.0083) | 4035 (0.0135) | 2244 (0.0075) | 3737 (0.0125) |
| Olig2 | 1293 (0.0043) | 2672 (0.0089) |  926 (0.0031) | 2089 (0.0070) |
- 最干净的证据是 **Sox9**：常数 (+9, +7, −4) 与旧中位数 (+8.3, +6.2, −4.4) 差 <1 px，
  场斜率又极小，也就是新方法给 Sox9 的摆放几乎就是「旧中位数」，照样输 1.6×。
  差的全是旧方法保留的逐 tile 残差（去场后离散度 3.1, 3.0, 0.7 px）。
  **3 px 的逐 tile 误差值 1.6× 的包含度**——这个判据要求 1–2 px 的逐 tile 精度，全局场给不了。
- **Olig2 的常数不可辨识**：870114 给 (−3, −30, −6)，875142 给 (−10, −20, −3)，同 seed、
  同 seams.csv、解算残差逐位相同，只有汇总的 GFP soma 数差了 50 个（48115 → 48165，
  应是 z-link 并列打破）。0.1% 的输入变化把 argmax 挪了 (7, 10, 3) px：粗搜索取直方图前 3 个峰，
  换个盆地就是离散跳变。得分只有 0.005 的情况下这个数靠不住。Sox9、RFP 两次一致。

**4. 新增逐 tile 局部精修 `--refine-per-tile`（默认开）**
- 以 `field_c(t) + a_c` 为中心搜 ±6 px / ±2 层（`--refine-xy/--refine-z`），判据仍是 Stage 3B 的包含度；
  同类型通道（RFP↔GFP）走质心互为最近邻。复用估常数时已经汇总好的 `parts`，不额外读盘。
- 与旧 Stage 2.5 的区别**只在搜索范围**：旧的每个 tile 从零搜 ±60 px，得分面又平，容易停在错峰；
  这里的中心是物理上说得通的量，窗口只够修掉场表达不了的那几个 px，跳不出去。
  Olig2 常数不可辨识这件事也被降级——它只决定搜索中心，不再直接决定最终偏移。
- 退回规则（就是 9-17 那条「离群偏移处理」待办）：细胞太少 `too_few_cells`、包含数 < 5 `low_score`、
  配对太少 `too_few_matches` → 退回全局解；顶到窗边记 `edge`（仍采纳，但报告里单独计数，
  超过 1/5 会警告先验偏了）。原因写进 `tile_positions.csv` 的 `refine_status_<通道>` 和 `solution.json`。
- 精修**只作用在细胞坐标**（`s_*` 列），不进各通道的 merging XML：XML 摆的是图像 tile，
  它的几何由同通道接缝决定，掺进跨通道的逐 tile 修正会破坏接缝自洽。`s_frame ≡ 0` 仍然成立。
- 对照从两套改成三套：**全局解 / 精修后 / 旧Stage2.5**，in-sample 和 held-out 各打一次
  （`score_placements`，同一批细胞、同一判据）。

**5. `solve_tile_positions.slurm` 新增开关**
- `FORCE=1` → `--force`（`0_channel_alignment_solved/` 已经写过一次，重跑会被拒）；
- `NO_REFINE=1` → `--no-refine-per-tile`，复现精修之前的行为。

### 验证状态
- ✅ `_refine_job` 合成数据自检：先验误差 0 / 3 / 6 px 都精确回到真值（包含数 98 → 280）；
  反号分支（本通道是 soma）正确；`too_few_cells` / `low_score` 退回正确；同类型通道残差正确。
- ✅ 本机用**故意调小的抽样**（每边 4000 个细胞、z 窗 ±4 层）跑通了全流程：精修的退回/edge 计数、
  三套对照、`tile_positions.csv` 的 `refine_*`/`refine_status_*` 列、`solution.json` 的 `refine` 段都正常。
  这轮的数值不能用来判方法（常数本身是噪声：Sox9 a_y 给出 +40，正式跑是 +7），
  但精修把全局解的得分翻了约一倍（held-out Sox9 0.23×→0.39×、Olig2 0.35×→0.67× 相对旧方法），
  说明机制是work的，差距来自先验太差。
- ✅ **正式抽样（878679，9 分 04 秒）过线**。held-out（验收判据）：
  | | 全局解 | 精修后 | 旧 Stage2.5 | 精修后/旧 |
  |---|---|---|---|---|
  | Sox9  | 2209 (0.0074) | 3652 (0.0122) | 3630 (0.0121) | 1.01× |
  | Olig2 | 1103 (0.0037) | 2423 (0.0081) | 2105 (0.0070) | 1.15× |
  in-sample 1.02× / 1.20×，与 held-out 几乎相同 —— 精修没有过拟合。
  逐 tile 包含数中位数 Sox9 312 → 342、Olig2 53 → 95。
- ⚠️ 两点要记住：
  - **Sox9 是打平不是赢**：3652 vs 3630 差 22 个细胞（0.6%），在 30 万/120 万的抽样噪声里。
    真正赢的是 Olig2（+15%），也正是旧方法最不可信的那个通道。
  - **Olig2 有 17/38 个采纳的 tile 顶到 ±6 px 窗边**，精修被截断了，所以 1.15× 还不是上限。
    佐证：Olig2 的 |Δ| p50 (2.7, 4.5, 1.4) 是三通道最大，常数三轮给出 (−3,−30,−6) →
    (−10,−20,−3) → (−7,−21,−4) 一直在飘。先验偏 → 精修要走更远 → 撞窗。
  - 附带收益（不体现在分数里）：7 个细胞太少的 tile 全部 `too_few_cells` 退回全局解，
    旧 Stage 2.5 在这些空 tile 上给的是 (−10,−10,−3) 那类无意义值。
- `solve_tile_positions.slurm` 再加 `REFINE_XY` / `REFINE_Z` 透传，便于放大窗口复跑。

**窗口 ±12 px / ±3 层复跑（882121，21 分 14 秒，其中精修占 12 分钟）**
放大窗口是对的，Olig2 明显继续涨，而且 in-sample 与 held-out **同步**涨——追噪声的话
held-out 不会跟着涨，所以修的是真实结构。

| held-out | 全局解 | 精修后 | 旧 Stage2.5 | 精修后/旧 | 上一轮(±6) |
|---|---|---|---|---|---|
| Sox9  | 2174 | 3670 | 3647 | 1.01× | 1.01× |
| Olig2 | 1192 | 2887 | 2099 | **1.38×** | 1.15× |

in-sample 同步：Sox9 1.02× → 1.04×，Olig2 1.20× → 1.33×。
顶到窗边：Sox9 3 → 0、RFP 4 → 0、**Olig2 17 → 11（仍未清零）**。
Olig2 逐 tile 包含数中位数 54 → 122。

**但 Olig2 还没收敛，而且卡住的 11 个 tile 有规律：**
- 第 0 列（327300）有 3 个 tile 一致地顶在 `dx ≈ +12`（314900/337500/348800/382800_327300）。
  这正是 9-16 记过的「第 1 列的 tile 一致偏 +4~+9、其他列约 −3~−5」——**通道场是行/列的一次
  函数，这一列的非线性偏离它吃不掉**，全压给了精修，精修又被窗口截住。
- 这 11 个 tile 的 `n_seams` 都是 2–4（网格边缘/角落），它们的 `P(t)` 本身也约束得最弱。
- 最直接的证据是和旧 Stage 2.5 逐 tile 偏移之差：
  全局解 vs 旧 `|差| p50 (16.5, 16.5, 7.3)` → 精修后 vs 旧 `(7.0, 11.5, 5.0)`，p90 `(30.6, 31.2, 13.0)`。
  砍掉一半，但 **y 的 p50 是 11.5，正好等于 ±12 的窗口**——不是收敛，是撞墙。
  （注意精修后并没有收敛到旧值，而是找到了**更好**的位置：held-out 上赢 1.38×。）

**要如实记住：Sox9 是打平不是赢。** 两轮都 1.01×，且这轮它的 edge 已清零，1.01× 就是它的
真实水平。新方案对 Sox9 只是追平旧方法；真正的收益在 Olig2（+38%）和 7 个空 tile 的兜底。

### 遇到的问题
- 改脚本时用 `io.open(p, 'w', newline=...)` 传了个非法的 newline 值，Python **先清空文件才校验参数**，
  把 `solve_tile_positions.py` 截成 0 字节。已从 git 恢复重打。以后写文件一律先写 `.tmp` 再 `os.replace`。
- `bjobs` 是 LSF 的命令，这个集群是 Slurm，它永远报 "No unfinished job found"。查作业用
  `squeue -u $USER` / `sacct -j <id>`。

### 下一步
0. **验收判据其实已经过线**（Olig2 1.38×、Sox9 打平、空 tile 有兜底）。下面第 1 条是
   锦上添花，不是必须——想先往下走的话，882121 这版结果可以直接用。
1. **再放大一轮看 Olig2 收不收敛**（±20 px / ±4 层，候选位移 41×41×9 = 15129，
   是 ±12 那轮的 3.5 倍，精修约 40 min，整个作业 50 min 左右，2 小时够）：
   `sbatch --export=ALL,WRITE_ALIGNED=1,FORCE=1,REFINE_XY=20,REFINE_Z=4 solve_tile_positions.slurm`
   - edge 掉到个位数、倍数稳住 → 收敛了，用这版往下走。
   - 倍数还在涨、edge 还是一堆 → **别再靠放大窗口硬掰**：±20 已接近旧方法 ±60 的无约束搜索，
     「跳不到错峰」这个保证在变弱。那说明是通道场模型不够，两条路：
     (a) 给 `delta_c(t)` 加二次项（row²/col²/row·col），先把第 0 列那种非线性吃掉；
     (b) 改成迭代——精修一轮 → 用结果重新稳健拟合通道场 → 在新先验附近用**小窗**再精修一次。
         这样总位移可以很大，但每一步搜索都是局部的，不丢「跳不到错峰」的保证。
     (b) 更可取，因为它不需要猜场的函数形式。
2. 稳定之后做顶替：`0_channel_alignment` 改名备份 → `0_channel_alignment_solved` 顶上 →
   删 `1_tile_2d_filtered/`（按旧偏移生成的）。
3. 用 `tile_positions/xml_merging_Olig2.xml` merge 配准用的全脑图；config 的 `paths.pATHXML`
   指到同一份，`stop_before_stitching` 改 `false`，跑 Stage 3+。
4. 没追上的话，候选方向：放大 `--refine-xy` 看是不是窗口太小（报告里的 `edge` 计数会先提示）；
   或者承认逐 tile 自由度必须保留，把「全局解」降格为**初值和离群判据**——用它检出旧 Stage 2.5
   里搜错峰的 tile 并替换，其余保留旧值。
5. 估常数的 z 段是按不含 `a_c` 的 dz 取的，|dz| 越大被窗口截掉越多，包含度打分**系统性偏向
   `a_z ≈ 0`**（50 层窗、Olig2 `a_z = −6` 约 12%）。干净做法是迭代一次：拿估出来的 `a_c`
   重取 z 段再估。加了精修之后这一项的重要性下降（只影响搜索中心），暂时没做。
- git：`solve_tile_positions.py`、`solve_tile_positions.slurm`、本日志的这一轮改动尚未提交
  （连同 9-17 那些未提交的）。
---

## 2026-09-21 EGFR T4: 488/Olig2 tile frame and downstream handoff

### Completed

- Confirmed the EGFR T4 channel mapping from `config_EGFR_t4.json`:
  - `GFP` = soma/reference channel for the original Stage 2.5 alignment.
  - `RFP` = soma channel.
  - `Sox9` and `Olig2` = TF/nuclear channels.
  - `Olig2` = 488 channel and final tile-stitching coordinate frame.
  - `tf_align_mode=direct`.
- First-round local tile solve completed with `MODEL=free` and `ALIGNMENT_FROM=old-offsets`:
  - Output: `EGFR_brain/T4/detection_results/5_analysis_report/tile_positions_488_oldalign_local/`.
  - 45 tiles, 76 grid seams per channel.
  - Olig2 seams: 66/76 successful, p50 match count 5218, p50 match fraction 0.76.
  - Olig2 seam MAD: `(1.33, 0.75, 0.50)`; fit residual p50: `(1.22, 0.53, 0.16)` px/slices.
  - Five tiles had no usable seams and were retained by nominal grid placement; they are edge/low-signal tiles.
- Second-round rebase completed:
  - Output: `EGFR_brain/T4/detection_results/0_channel_alignment_488frame/`.
  - 45 offset JSON files, 180 aligned CSV files (45 tiles x 4 channels), and `_align_done.flag` verified.
  - Every tile has `Olig2 = (0, 0, 0)`.
  - Other channel offsets are rebased from the original GFP reference to the Olig2/488 frame.
- Generated XMLs in the tile-position report directory. The authoritative downstream XML is:
  - `xml_merging_Olig2.xml`
- Added local GPU downstream entry points:
  - `config/config_EGFR_t4_local_gpu.json`
  - `scripts/run_inference_t4_local_gpu.cmd`
  - The local config uses `start_from_stage=2`, `stop_before_stitching=false`, local `Y:` paths, and the Olig2 XML.

### HPC submission status

- `inference_t4.slurm` was changed from `4 x L40` to `1 x A30`; memory was reduced from `240G` to `180G` to fit A30 nodes with `188000 MB` RAM.
- Jobs `906070` and `906117` remained pending with `Reason=QOSGrpGRES` under `Account=lsmsmart_gpu`, `QOS=gpu`, despite idle A30 nodes. This is an account/QOS GPU quota issue, not a node or script resource issue.
- The local GPU route is preferred until the HPC GPU quota is available.

### Next action

1. Preserve the old `0_channel_alignment` as a GFP-reference backup.
2. Rename the verified `0_channel_alignment_488frame` to active `0_channel_alignment`.
3. Rename old offset-derived caches such as `1_tile_2d_filtered` instead of deleting them; the new downstream run must regenerate them.
4. Run `scripts\run_inference_t4_local_gpu.cmd` from the repository root after confirming that the local GPU environment is active.
5. Validate Stage 3 outputs (`2_global_2d_raw`, `3_channel_3d`, `4_colocalization`) and compare GFP soma/TF containment results.
## CODEX_HANDOFF_JSON: EGFR_T4_2026-09-21

```json
{
  "record_date": "2026-09-21",
  "sample": "EGFR_T4",
  "sample_root_local": "Y:/Fengyi/EGFR_brain/T4",
  "project_root": "Y:/Fengyi/brain_detector",
  "channel_mapping": {
    "488": "Olig2",
    "561": "RFP",
    "640": "GFP",
    "730": "Sox9"
  },
  "original_stage_2_5_reference": "GFP",
  "final_tile_frame": "Olig2",
  "tf_align_mode": "direct",
  "tile_solver": {
    "model": "free",
    "alignment_source": "old-offsets",
    "n_tiles": 45,
    "grid": [9, 5],
    "first_round_output": "Y:/Fengyi/EGFR_brain/T4/detection_results/5_analysis_report/tile_positions_488_oldalign_local",
    "authoritative_xml": "Y:/Fengyi/EGFR_brain/T4/detection_results/5_analysis_report/tile_positions_488_oldalign_local/xml_merging_Olig2.xml",
    "olig2_seams_success": 66,
    "olig2_seams_total": 76,
    "olig2_match_count_p50": 5218,
    "olig2_match_fraction_p50": 0.76,
    "olig2_seam_mad_xyz": [1.33, 0.75, 0.50],
    "tiles_without_seams": [
      "303600_361200",
      "303600_372500",
      "314900_372500",
      "382800_372500",
      "394100_372500"
    ]
  },
  "rebased_alignment": {
    "directory": "Y:/Fengyi/EGFR_brain/T4/detection_results/0_channel_alignment_488frame",
    "offset_json_count": 45,
    "aligned_csv_count": 180,
    "align_done_flag": true,
    "all_olig2_offsets": [0, 0, 0],
    "meaning": "All channel coordinates were rebased from the original GFP-reference frame to the Olig2/488 frame."
  },
  "hpc": {
    "account": "lsmsmart_gpu",
    "qos": "gpu",
    "jobs": [906070, 906117],
    "reason": "QOSGrpGRES",
    "status": "blocked_by_gpu_qos_quota"
  },
  "local_downstream": {
    "config": "Y:/Fengyi/brain_detector/config/config_EGFR_t4_local_gpu.json",
    "runner": "Y:/Fengyi/brain_detector/scripts/run_inference_t4_local_gpu.cmd",
    "start_from_stage": 2,
    "stop_before_stitching": false,
    "device": "cuda",
    "required_active_alignment_directory": "Y:/Fengyi/EGFR_brain/T4/detection_results/0_channel_alignment",
    "required_xml": "Y:/Fengyi/EGFR_brain/T4/detection_results/5_analysis_report/tile_positions_488_oldalign_local/xml_merging_Olig2.xml"
  },
  "next_steps": [
    "Preserve old 0_channel_alignment as a GFP-reference backup.",
    "Rename 0_channel_alignment_488frame to active 0_channel_alignment.",
    "Rename old 1_tile_2d_filtered and other downstream caches; do not delete them.",
    "Run scripts/run_inference_t4_local_gpu.cmd from the project root.",
    "Validate 2_global_2d_raw, 3_channel_3d, and 4_colocalization outputs."
  ]
}
```
---

## 2026-09-22 Raw/Filtered contract and reusable Stage 2.75 filter

### Completed

- Added src/core/detection_filter.py as the CPU-only source of truth for Stage 2.75: ordered bbox/aspect/area/mean/IoMin filtering, per-z stable-score containment NMS, parameter validation, per-channel override resolution, statistics, legal source routing, and atomic CSV writing.
- Updated src/core/worker.py so 1_tile_2d_raw/ streams the full nine-column detector output and no longer applies configurable bbox, percentile, intensity, or containment filters. Detector-native YOLO patch stitching/model NMS and StarDist instance NMS remain.
- Updated scripts/run_inference.py: Stage 2.75 is now mandatory, has no passthrough switch, validates every required raw/aligned/fused source CSV before checkpointing, logs per-tile/channel params and counts, and leaves Stage 3 reading only filtered CSVs.
- Added CPU-only src/core/refilter_detections.py with --channels, --tiles, --output-dir, --dry-run, and explicit --overwrite. It uses existing CSV means, writes only after all filter calculations succeed, records refilter_manifest.json, and warns (without deleting) about stale downstream outputs.
- Updated 2-D visualization preview to use the shared filter before z-range cropping, read raw/post-align, aligned/pre-align, or fused/double-exposure sources, use the recorded runtime_config.json parameters when available, and distinguish source -> preview-kept from preview-rejected layers. The dual-exposure preview utility now uses the shared module as well.
- Added empty channel_filter_overrides.Olig2 placeholders to the three EGFR T4 local config files. No Olig2 QC threshold was guessed or written; Sox9 still inherits the StarDist defaults. config/ is locally ignored/skip-worktree, so these local config edits require explicit force-add or equivalent if they are intended for Git.
- Extended README with raw/filtered semantics, override/null behavior, cache invalidation, and refilter commands.

### Verification

- brain_detector environment: python -m unittest discover -s tests -p test_*.py -v passed 5 tests (filter order/stability, containment NMS, override/null validation, refilter dry-run, candidate output, and manifest).
- Compiled: shared filter, worker, main pipeline, refilter CLI, visualizer, and dual-exposure preview.
- git diff --check passed. Repository scan found no non-log stage_2_75_enabled references.

### Still requires manual QC / authorization

- Final EGFR T4 Olig2 threshold values remain pending user QC confirmation.
- No T4 data was archived, overwritten, re-filtered in place, or re-run.
- Perform the planned side-directory refilter + visual QC before using --overwrite, then regenerate Stage 3-5 outputs.

- Follow-up correction: removed the remaining obsolete stage_2_75_enabled switch and passthrough-era comments from config/config_EGFR_t4_local_gpu.json. The pipeline does not read this key; Stage 2.75 is mandatory.

---

## 2026-09-22 Visualization coordinate-frame repair

### Completed

- Added `src/utils/coordinate_context.py`. It derives visualizer geometry exclusively from `<pATHRESULT>/runtime_config.json` and its `paths.pATHXML`, validates the final frame channel, same-directory `xml_merging_<channel>.xml` files, unique tile names, complete pre-align offset JSON, and zero shift for the frame channel.
- The visualizer now uses the runtime routing/paths rather than the duplicated sample routing in `vis_config.json`. Global Stage 3/4 results use the runtime frame XML without a channel-directory or regular-grid fallback.
- Post-mode image layers and intensity-filter volumes now use the same per-channel shifts as prealign mode, including an expanded raw Z read before shift/crop so edge slices entering the requested final-frame range are retained. Post Stage-1 preview reads the filtered, final-frame CSV source.
- Startup provenance prints `P_O`, `P_c-P_O`, per-tile alignment `s`, and residual `q = P_O + s - P_c` for every channel.
- Added `tests/test_coordinate_context.py` covering global/local inversion, the two equivalent global-coordinate expressions, the T4 GFP residual, and fail-fast incomplete offsets.

### Verification

- `conda run -n brain_detector python -m unittest discover -s tests -p test_*.py -v`: 7 tests passed.
- T4 smoke test for `360100_349900`: Olig2 `P_O=(3458,8674,5)`; global `z=2` maps to local index `6` (the seventh slice); GFP `q=(-1,34,-3)`.
- Compiled `coordinate_context.py` and `visualize.py`; `git diff --check` passed.

### Notes

- No EGFR T4 detection, filtered, Stage 3, Stage 4, or colocalization result files were changed. This update changes only visualization coordinate parsing/display and its regression tests.

- Added src/utils/review_filtered_gui.py: read-only raw-image plus saved-filtered-box QC. It prints the visualize.py spatial tile grid, accepts comma-separated multi-tile selection, then opens one Napari window per selected tile with a shared Z range and selected channels.

## 2026-09-24 — score_min 后处理过滤

### 已完成
- 在共享 `src/core/detection_filter.py` 中加入 `score_min`：允许 `null` 或 `[0,1]` 有限数值，拒绝 bool/字符串/NaN/Infinity/越界值；过滤顺序在 `mean_min` 后、containment NMS 前，使用 `score >= score_min`。Filter schema 升至 `"2"`，每步统计始终包含 `removed.score_min`。
- `src/core/refilter_detections.py` 继续使用共享 filter，终端显示 score_min 独立删除数；manifest 记录参数、分步统计和 schema。Stage 2.75 日志同样报告独立删除数。
- 可视化 preview 通过共享 filter 对 CSV score 过滤，不加载像素数据；更新 README 与预览说明，区分 StarDist 推理用 `prob_thresh` 和后处理用 `score_min`。
- 所有生产模型配置加入显式 `score_min`。Olig2 redetect 的 HPC/local 配置均设为 `0.30`，其他配置为 `null`；两份 Olig2 config 阈值一致。
- 新增阈值边界、参数校验、空表、channel override、NMS 顺序、refilter manifest/overwrite 覆盖。修复过滤后 index 非连续时 containment NMS 使用行位置越界的问题。

### 验证与真实数据
- `conda run -n brain_detector python -m unittest discover -s tests -v`：12 项全部通过。
- `conda run -n brain_detector python -m compileall -q src scripts tests`：通过。
- 全部生产 config 均可通过项目 JSONC loader 解析，且 active channel 的 filter 参数校验通过。
- Olig2 本地 config dry-run：读取 21 个 raw tile，5390220 条进入 filter，5262590 条保留，累计删除 127630 条；其中 `score_min=0.30` 单独删除 0 条。dry-run 未写文件。
- 按计划生成独立候选目录 `Y:\Fengyi\EGFR_brain\T4\detection_results\redetect_norm_prob\1_tile_2d_filtered_score030`：21 个 filtered CSV 和 manifest；schema 为 `2`，每条记录 `params.score_min=0.30`，合计数据与 dry-run 一致。raw CSV 和正式 filtered 目录未修改。

### 未完成的人工验收
- Napari 视觉抽查尚未完成；需要人工检查候选目录里的信号强/暗边缘/高背景 tile。
- 不同阈值（null/0.25/0.40）的额外候选比较未生成，本次计划只指定了 score030 的输出目录。

---

## 2026-10-01 EGFR T4 共定位 QC：GFP/RFP 漏显排查

- 样本 EGFR T4、tile `371500_361200`、local z=510：RFP `glia_RFP`（中心约 652.1, 781.0；z=510–511）与 GFP `glia_GFP`（中心约 643.3, 788.5；z=510）的框满足 Stage 3 soma 3D 匹配条件。对已保存的 `3_channel_3d/*_3d_tracked.pkl` 复核，`match_soma_3d_iou` 能将两者配对。
- 已保存的 `4_colocalization/coloc_result.csv` 中存在 `glia_GFP_RFP`；global z=507 经该 tile 的 z offset=4 转换为 local z=510，tile-local 中心约 (643.3, 788.5)。可视化的 tile、z、marker 和边界筛选也保留该记录，`[coloc] GFP+RFP` 图层中包含它。
- 本例的“漏检”发生在显示环节：共定位图层创建时默认为 `visible=False`。已修改 `src/utils/visualize.py`，使加载的共定位图层默认可见；prealign 模式新增 `coloc_source`，可选 `saved`（Stage 4 CSV）或 `preview`（用当前 filtered tile CSV 与 [zlinked] 同源计算）。当前本地 `config/vis_config.json` 设为 `saved`；示例配置设为 `preview`。更新了 README。正式 T4 检测及 Stage 3/4 结果文件未改动。
- 验证：用报告的两个框确认 3D 匹配；用小型 CSV 确认 global z=507 映射到显示索引 10（local z=510）且图层可见；项目配置 loader 可解析两份可视化配置，`py_compile` 与 `git diff --check` 通过。
- 这只证实所报细胞已在正式结果中；其他可疑细胞仍需逐例对照 Stage 3/4 数据，不能由本例推断全部共定位匹配均正确。

### 同日后续：tile 348800_361200 的重叠区案例

- 用户给出的 local RFP `glia_RFP`（中心约 259.4, 360.5, z=508；z-span 506–510）和 GFP `glia_GFP`（中心约 262.3, 370.5, z=506；z-span 505–508）从该 tile 的 filtered CSV 重新 Z-link 后，`match_soma_3d_iou` 确实将两者匹配成 `glia_GFP_RFP`。
- 正式 Stage 4 也在同一物理区域有 `glia_GFP_RFP`（映射到当前 tile 后中心约 247.4, 373.9，local z=506），但其 `tile_name` 为相邻的 `348800_349900`：全局去重保留了相邻 tile 来源的检测。此前可视化优先按当前 `tile_name` 筛选，因此漏显该结果。
- 已改为按全局框与当前 tile 的空间相交关系选择 Stage 3/4 结果，且不再对全局去重后的结果重复应用 tile 左/上重叠边缘隐藏规则；同时移除保存的 Stage 3 loader 中引用未定义 `preview_rejected` 的代码。Stage 4 CSV 只在代表 Z 层画框，本例需看 local z=506，而不是 RFP 的代表层 z=508。当前 T4 QC 配置继续使用 `coloc_source: "saved"`。
- 回归测试覆盖“来源是相邻 tile、框位于当前 tile 内”的 Stage 3/4 显示；实际 T4 CSV 复核后，当前 tile 的 GFP+RFP 图层可以选出该区域的正式结果。正式检测、Z-link 和 Stage 4 CSV 未改动。
- 后续真实 CSV 验证还暴露保存的 Stage 3 可视化 loader 调用已不存在的 `_filter_df_by_size_and_intensity`；现改用共享 `filter_detection_df` 处理可视化筛选，并保留被筛掉的灰色框。实际 T4 RFP/GFP Stage 3 CSV 均可在该 tile 加载；新增测试覆盖此筛选分支。

---

## 2026-10-02 Saved-result prealign GUI

### Completed
- Reworked the prealign visualization path to load saved outputs from the current detection pipeline instead of running Z linker or channel colocalization during GUI startup.
- Added local tile and global coordinate views. The global view uses the saved runtime configuration, XML tile positions, and channel alignment offsets to place images and results consistently; missing metadata is reported instead of guessed.
- Added readers for saved tile/global 2D detections, Stage 3 cells, and Stage 4 colocalization results. Stage 4 layers show the recorded marker combinations and representative Z positions without recomputing matches.
- Kept per-Z Stage 3 track boxes available through the saved PKL files on cell selection. Added safeguards for ambiguous CSV-to-PKL matches, duplicate global rows, and invalid Z ranges.
- Updated the visualization config example and README for the new view options. The active local visualization config was also adjusted, while its sample data path remains unchanged until the new pipeline outputs exist.

### Verification and follow-up
- Synthetic tests: 12 new saved-result GUI tests passed; 5 coordinate-context tests and 1 tile-selection test passed. Visualization configs parsed, and git diff --check passed.
- A full Napari window smoke test could not run in the current environment because Qt could not create an OpenGL context. Real EGFR T4 outputs from the revised pipeline are not available yet, so image/result alignment and Stage 4 display still need validation against a complete saved dataset.


---

## 2026-10-02 Pre-align tile-position pipeline and filter configuration

### Completed

- Stage 2 can resume from existing 1_tile_2d_raw CSVs when start_from_stage is 2; it does not need a separate raw-detection path parameter.
- In pre_align, Stage 2.25 filters raw tile detections into 1_tile_2d_prefiltered before channel alignment. Stage 2.5 estimates shifts from those filtered cells. Stage 2.75 publishes aligned single-channel CSVs into 1_tile_2d_filtered without filtering them again; fused double-exposure results are filtered at publication. Post-align filters raw CSVs directly into 1_tile_2d_filtered. The unused stage_2_75_enabled setting and its passthrough comments were removed from the T70/T4 local configs and example config.
- Optional Stage 2.9 now invokes the detection-based tile-position solver after tile filtering/alignment and before Stage 3, including when stop_before_stitching is true. It reads filtered, unaligned detections and reuses the per-tile Stage 2.5 offsets. Stage 3 automatically loads the solved frame XML for global cell coordinates.
- The two meaningful references are pre_align_params.reference_channel (channel-shift measurement, GFP for T70) and stitching_reference_channel (tile/global coordinate frame, Olig2/488 nm for T70). Stage 2.5 rebases shifts into the latter. The duplicate tile_position_params.reference_channel key was removed.
- The solver writes xml_merging_<channel>.xml into the tile-position report and publishes xml_merging.xml into each original channel image directory. An existing source XML is preserved as xml_merging.original.xml. Channel image XMLs use seam-derived tile geometry plus the global channel translation; local cell-alignment residuals stay with cell coordinates.
- tile_position_params.workers controls CPU parallelism for tile seam measurement. It is separate from pre_align_params.n_workers, which controls tile-level channel alignment. The stages run sequentially, and the solver falls back to the alignment worker count when its own count is omitted.
- Solver input/output provenance is recorded in _pipeline_manifest.json. A changed solver input or XML is rejected while global Stage 3/4 checkpoints exist, preventing mixed coordinate systems.

### Verification and follow-up

- Earlier implementation verification: 32 unit tests passed, including different-reference conversion, XML output/backup, and changed-input checkpoint protection. The T70 dataset itself was not rerun during that verification.
- In this checkout, config/config_EGFR_t70.json is absent while scripts/inference_t70.slurm defaults to that path. Confirm that the T70 config exists at the submission location before running sbatch.

---

## 2026-10-08 Interactive visualization GUI updates

### Completed

- Added a light purple and white Qt launcher for `src/utils/visualize.py`. It supports sample selection, numbered Tile selection and spatial layout, editing common settings or the full configuration, larger adjustable fonts, and resizable panels. The launcher and its controls use English text; `--direct` retains the command-line flow.
- The launcher starts with no sample selected and does not scan image folders until the user selects a sample and clicks **Choose**. Changing samples clears stale Tile selections. **Refresh** becomes available after a sample is chosen.
- Scanning displays a prominent indeterminate progress dialog. Opening Tiles displays a progress dialog with the count of loaded Tiles; the dialog closes when the Napari viewer is ready. Viewer failures remain visible in the launcher.
- In prealign local mode, each selected Tile opens in its own local-coordinate viewer. In prealign global mode, selected Tiles share one viewer; all image layers are added together before result layers so they are contiguous in Napari's layer list. Images remain separate layers rather than being blended into one output file.
- Added **Start with images only**, enabled by default. TIFF image layers initially appear while detection boxes, points, colocalization, and overlap overlays start hidden in all visualization modes. Users can enable result layers inside Napari or disable the option to restore configured initial visibility.
- Reduced TIFF loading overhead for aligned image views: retain native 16-bit pixels, avoid redundant full-volume copies when applying channel shifts, and bound the image sample used to calculate display contrast. The other `load_volume` callers keep their float32 return type. These are display-only changes; saved detection CSVs and pipeline results are unchanged.
- Configured the visualization process's stdout and stderr as UTF-8 before logging. This fixes the Windows cp1252 `UnicodeEncodeError` raised while local mode printed Ctrl+click annotation instructions.
- The visualizer reads only the current result layout. Channel-relative offsets come from `3_2d_aligned`, while global Tile positions come from the configured final-frame XML, including per-channel `xml_merging.xml` files where applicable. The temporary legacy-layout read fallback was removed.
- Updated `README.md` and the local visualization configuration comments for the launcher and image-only startup behavior.

### Verification and limits

- The saved-view regression suite passed 15 tests, including multiple-Tile global layer ordering, initial visibility, uint16 TIFF alignment, and a cp1252 annotation-log reproduction. Python compilation and `git diff --check` passed.
- An offscreen launcher smoke check confirmed an empty initial sample selection, zero scan workers, and **Start with images only** enabled.
- No full multi-Tile Napari interaction or network-drive loading benchmark was run. The TIFF and CSV inputs are on a mapped network drive, so initial load time may still be limited by network throughput.

---

## 2026-10-08 YOLO 2D containment NMS: glia priority

- Confirmed that raw YOLO tile CSVs already pass the model's per-patch NMS (`yolo.nms_iou`) and a per-class, per-slice tile NMS with fixed IoU 0.4. The later `nms_containment_thresh` filter reads those raw CSVs and writes filtered tile CSVs.
- Changed the containment filter to keep glia when overlapping boxes have different classes, even if the glia box has a lower score. Boxes of the same class still use score order. The rule applies within each z slice of a tile and channel when IoMin exceeds the configured threshold.
- Corrected the active T70 config comment: `nms_containment_thresh: 0.98` means more than 98% overlap relative to the smaller box, not 60%. Earlier confidence, size, and area filters can still remove a glia box before containment NMS.
- Bumped `FILTER_SCHEMA_VERSION` to 4 so cached filtered CSVs are recomputed under the new rule. The focused detection-filter suite passed all 10 tests in the local `brain_detector` Conda environment; Python compilation and whitespace checks passed.

---

## 2026-10-08 重叠区检测丢失 bug 修复

- 原因：全局 2D 拼接时，旧逻辑无条件丢弃后处理 tile 中心落在重叠区的检测，即使相邻 tile 没有对应检测，导致检测无法进入 Stage 3。
- 修复：`combine_predictions` 只在先前 tile 的同一全局 Z 层、同一细胞类型存在 IoU > 0.4 的框时丢弃重复检测；没有匹配框则保留，并同步保留 tile/slice 溯源。
- 验证：新增重叠区独有检测、重复检测、不同类型和不同 Z 层的回归测试；在 `brain_detector` Conda 环境中，新增测试 3 项及相关流水线测试 2 项全部通过。
- 已生成的全局 2D、Stage 3 及下游 checkpoint 不会自动更新，需要重新生成才能应用修复。

---

## 2026-10-09 Pipeline-wide detection and colocalization provenance (implemented 2026-10-08)

### Completed

- Raw tile detections now carry stable `detection_id`, original TIFF path, raw slice/Z, score meaning, intensity-region meaning, and parent IDs. Each tile/channel has an input manifest recording processed, missing, or unreadable slices and validating the raw CSV. YOLO exports post-model window candidates with patch origins and tile-NMS keep/suppress decisions, winner IDs, and IoU. StarDist exports instance labels, mask areas, centroids, and available polygons.
- Both filtering passes and the standalone refilter tool retain IDs in accepted and rejected CSVs. Rejected rows record the first failing filter; containment NMS also records the winning detection ID, IoMin, and threshold. Per-tile filter manifests bind source, accepted, and rejected files by SHA256. Alignment preserves raw slice/Z and source image; its per-tile manifest validates inputs, offsets, and outputs.
- Dual-exposure fusion writes a new ID linked to both original exposure IDs, the winning box/source image, and fusion IoU. Per-tile fusion manifests record input/output hashes and counts. `fusion_summary.csv` is rebuilt from every tile manifest, including cached tiles during a partial resume.
- Global 2D stitching keeps detection ID and exact tile/slice. Cross-tile overlap rejections record the competing ID and IoU. Z-linked cells keep stable track IDs, exact member detections, representative detection/Z, observed XY envelope, and tracks rejected by the minimum-Z rule; CSV sidecars and Stage 3 manifests validate these outputs.
- Stage 4 colocalization exports stable `coloc_id`, each participating channel's original 3D track and per-Z boxes, accepted soma/TF match measurements, and exclusion decisions. `coloc_result.csv` includes a `source_3d` JSON map, exact source-trace status, and calibrated center coordinates in micrometers. The representative-centered 3D matching box and observed per-Z envelope are labeled separately. Centroid CSVs retain the cell ID and physical coordinates.
- Runtime metadata records model/code hashes, original config/run info, and run history. Tile-position solver, filtering, alignment, fusion, Stage 3, and Stage 4 checkpoints now use content signatures or completion manifests; stale downstream outputs are rebuilt where safe. The stitching XML is hashed for the global stages. README documents file joins, coordinate frames, and legacy limitations.

### Verification and limits

- `brain_detector` Conda environment: `python -B -m unittest discover -s tests -p 'test_*.py' -q` passed 76 tests. `git diff --check` found no whitespace errors.
- No T70 dataset or existing detection results were rerun for this change. New provenance fields appear when the affected stages regenerate. Historical raw detections or track summaries cannot recover original details that were never saved; raw TIFF manifests use path, size, and modification time rather than a full TIFF content hash.

---

## 2026-10-09 Stage 4 soma 3D union and nuclear-marker status

- GFP/RFP soma tracks still match by the configured 3D IoU or IoMin gate. A matched cell now uses the axis-aligned union of both source 3D boxes: min lower and max upper X/Y/Z coordinates. The merged center and per-Z display boxes are recomputed; each original channel track remains available unchanged in `source_3d`, `coloc_source_3d.csv`, and `coloc_source_boxes.csv`.
- Stage 4 applies Sox9 and Olig2 containment independently to the same merged soma geometry. The marker order does not change final cell labels. `coloc_result.csv` now exposes merged 3D center/bounds, `bounds_method`, soma/TF positive channel lists and counts, and negative/single/double/multi positivity status. Its legacy XY box is the union projection for multi-soma matches.
- The Stage 4 code hash invalidates an earlier Stage 4 manifest and rebuilds these outputs from saved Stage 3 tracks; saved raw detections or alignment shifts do not need to be recomputed for this code change. Status is defined at the soma level; two TF assignments do not by themselves prove both markers occupy one nucleus.
- Verification: the full test suite passed 78 tests after the core change; after the viewer label and source-trace test, the focused colocalization suite passed 6 tests and saved-viewer suite passed 16 tests. `git diff --check` reported no whitespace errors. T70 data were not rerun.

---

## 2026-10-12 Channel alignment quality check: offset sensitivity and per-tile plots

Recorded the completed T70 evaluation and the quality-check feature added during the preceding session.

### Implementation and decisions

- Extended `src/utils/validate_align_shifts.py` to quantify saved channel offsets by additional positive/negative X, Y and Z translations. It reuses filtered detections and saved offsets on CPU; it does not rerun inference or optimize/replace the offsets.
- Coordinates remain in the saved final offset frame (Olig2 for T70). GFP is the fixed channel for three pairwise comparisons: GFP/RFP, GFP/Sox9 and GFP/Olig2. Plot coordinate zero means the saved alignment plus zero additional translation.
- Metric: binary 3D occupancy IoU of z-linked detection boxes, including the alignment rasterizer's inferred Z extent for single-slice cells. This is detection-box IoU, not raw-image intensity or segmentation-mask IoU. Pooled IoU is sum(intersection) / sum(union); equal-tile mean and SD are also exported.
- The unaligned baseline applies GFP's saved offset to both raw channels, preserving their original relative displacement while evaluating the same ROI. All shifts and the baseline use one common valid XY/Z footprint; cells outside the ROI can move into it during a scan.
- Retained the original random-tile, random-window held-out mode. Added `sampling.tile_selection="all"` and `sampling.z_selection="center"` for all completed alignment tiles with one center window each. The center is calculated from actual common image-slice support with offset/scan/baseline margins, independently of detection density. Processed-slice manifests are preferred; legacy T70 results use original TIFF filename indices.
- Center sampling requires `held_out=false` and one region per tile. For this run, `min_cells=0` retains sparse/empty tiles rather than selecting stronger signal or replacing tiles. Both-empty union gives undefined IoU (NaN); one-empty occupancy gives IoU zero. Aggregate curves record both `n_tiles` and per-shift `n_defined_tiles`.
- All-tile evaluation uses dynamically scheduled worker jobs and writes each completed tile's CSV into `tile_curves/`, plus `progress.json`. Final tables and `sampling.json` record exact windows, offsets, settings, source support, eligibility and completion. Per-tile checkpoint CSVs provide intermediate results; automatic resume from these evaluation checkpoints is not implemented.
- Added `output.per_tile_plots=true` / `--per-tile-plots`: one 3 x 3 figure per tile, with channel rows and X/Y/Z columns, in PNG/PDF. Each plot uses that tile's own unaligned baseline, saved-alignment line, scanned maximum and exact Z bounds. Empty panels explicitly state that IoU is undefined and retain the actual scan-axis range. `tile_plots/index.csv` maps tiles to figure files.
- Added `--plot-only` to render existing scan tables without loading detections or recomputing IoU. Figure captions explain red curves/points, gray individual-tile curves in the overview, orange unaligned baselines, blue zero-shift lines and black scanned maxima. Lines connect evaluated points; there is no fitted curve. Panels use independent Y scales.

### T70 results2 run and outputs

- Config: `config/EGFR/2/align_validation_t70_all_center240.json`; versioned template: `config_example/align_validation_all_center240.example.json`. The earlier random-window template remains `config_example/align_validation.example.json`.
- Included all 45 completed tiles, one central 30-slice window per tile: 240 um at 8 um Z spacing. XY resolution is 0.65 um/px; occupancy bins are 4 px. X/Y scan +/-30 px at 2 px steps; Z scan +/-5 slices at 1-slice steps. Eight CPU workers were used.
- All 45 tiles completed with no exclusions: 9,855 per-tile scan records and 219 pooled scan points. Zero-additional-shift IoU is defined for 39 tiles per channel; six retained tiles have both-empty center occupancy.
- Current output directory: `Y:/Fengyi/EGFR_brain/T70/detection_results2/alignment_validation2_center240/`. It was renamed from the earlier `alignment_validation_all_center240`; the evaluation config now uses the current name. The earlier 10-tile/80-um output remains in `detection_results2/alignment_validation/`.

| Pair | Unaligned pooled IoU | Aligned pooled IoU | Relative improvement |
|---|---:|---:|---:|
| GFP/RFP | 0.098748 | 0.205529 | 108.14% |
| GFP/Sox9 | 0.089713 | 0.100421 | 11.94% |
| GFP/Olig2 | 0.067794 | 0.074102 | 9.31% |

- Artifacts: `T70_alignment_sensitivity.png/pdf`, `T70_curves.csv`, `T70_pooled_curves.csv`, `T70_summary.csv`, `T70_pooled_summary.csv`, `sampling.json`, `analysis_notes.md`, per-tile checkpoint CSVs, and 45 PNG plus 45 PDF figures under `tile_plots/`.
- Empty-at-zero tiles: `305500_353000`, `305500_364300`, `316800_364300`, `384700_364300`, `396000_319100`, `396000_364300`.

Reproduce the analysis on HPC (CPU only):

```bash
python src/utils/validate_align_shifts.py \
  --sample /rsstu/users/a/agrinba/DeepDesign/Fengyi/EGFR_brain/T70/detection_results2 \
  --eval-config config/EGFR/2/align_validation_t70_all_center240.json
```

Regenerate the overview and all per-tile figures from saved measurements:

```bash
python src/utils/validate_align_shifts.py \
  --sample /rsstu/users/a/agrinba/DeepDesign/Fengyi/EGFR_brain/T70/detection_results2 \
  --eval-config config/EGFR/2/align_validation_t70_all_center240.json \
  --plot-only --per-tile-plots
```

### Interpretation, verification and remaining work

- A nonzero IoU peak alone does not establish an alignment error. The plotted image-center 30-slice region can differ from the solver's detection-centered approximately 50-slice region. In sequential-joint alignment, Sox9 optimizes soma containment against the GFP/RFP collection, while Olig2 also uses fixed Sox9 spatial support; neither directly optimizes the plotted pairwise GFP IoU.
- For the 39 nonempty tiles, median relative extra IoU gains from zero to each tile's scanned X/Y peak were RFP 0.32%/0%, Sox9 0.66%/0.71%, and Olig2 1.25%/1.01%. Most nonzero peak shifts therefore reflect a shallow local IoU landscape. Larger deviations worth investigating include Sox9 X on `305500_341700` (-15.6 um, about 15% extra IoU) and Olig2 Y on `328100_364300` (+9.1 um, about 17.3% extra IoU).
- Center windows may overlap solver data, so this run is a sensitivity evaluation rather than independent held-out validation. Tile overlaps are not deduplicated into a whole-brain estimate. The active T70 job `scripts/logs/lsf_t70_360749.out` explicitly reused old offsets rather than rerunning alignment; current solver code does not by itself establish the exact historical implementation that generated those offsets.
- Verification completed during implementation: 19 focused regression tests passed (`tests.test_validate_align_shifts`, `tests.test_alignment_frames`); a separate smoke run verified serial/parallel equality and PNG/PDF rendering. All 45 real-data center windows were independently checked against source slice counts and saved offset/scan margins. Checked final table sizes, all 45 PNG/PDF pairs, the tile index, representative populated/empty plots, and `git diff --check`.
- Next diagnostic, not yet implemented or run: repeat sensitivity scans in each tile's original solver Z window using its actual optimization target (RFP IoU, Sox9 containment, Olig2 joint score), then inspect clear nonzero improvements and boundary peaks. Do not automatically replace saved offsets with peaks from the current pairwise-IoU plots.

---

## 2026-10-16 自动续跑、分阶段 checkpoint 与 T70 下游衔接

### 背景与问题

- T70 已完成 2D detection、filter、channel alignment 和人工 quality check，本次目标是保留已有结果并继续下游。
- 作业 `387352` 在主流程检查历史配置时退出：`Existing checkpoints lack runtime_config.json provenance`。当前 `config_EGFR_t70.json` 已加载成功；结果目录缺的是历史 `runtime_config.json`，但存在独立对齐流程生成的 `alignment_runtime_config.json`。
- `detection_results3` 中 raw、filtered、aligned 各有 180 份结果 CSV（45 tiles × 4 通道）；当时 `4_2d_filtered` 至下游目录尚无结果文件。
- 原 `start_from_stage>=2` 只禁止重新检测，仍会进入过滤和对齐检查。GFP/RFP 的 filter manifest 嵌入了旧 Sox9/Olig2 参数，导致本通道参数未变也可能被判失效，继而触发对齐重算。全局 2D 与 Z-Link 共用完成记录，也会让 Z-Link 中断导致重新拼接。

### 修改与关键决定

1. **统一自动续跑入口**：移除样本配置中的 `start_from_stage`。每次从入口检查各阶段，复用有效结果、补齐缺失任务。优先从图像目录发现预期 tiles；图像根目录不可用时可从已有 raw CSV 推导，但该回退不能发现从未产生 CSV 的 tile。保留 `stop_after_detection` 和 `stop_before_stitching`。
2. **取消普通通道的重复 CSV**：`pre_align` 且无双曝光时，全局拼接直接读取 `3_2d_aligned`，不再复制到 `4_2d_filtered`。双曝光通道仍经过融合及最终过滤，`post_align` 仍读取最终过滤目录；混合曝光按通道选择输入。`2_2d_filtered` 保留，供对齐和 tile position 求解使用。已有旧目录不自动删除。
3. **按阶段、通道比较配置**：raw inference 比较排除过滤、几何和报告参数；过滤 checkpoint 只依赖本通道参数及源文件内容，不再因其他通道参数变化失效。带 hash 的文件按内容比较，兼容挂载路径变化和仅时间戳变化。
4. **分离对齐与下游 Z-Link 参数**：样本配置新增 `pre_align_params.z_linker`，保存对齐内部临时 linking 原有参数；顶层 `z_linker` 继续控制下游追踪／共定位。旧配置缺少新字段时保留原回退行为。
5. **补建配置快照与历史**：缺少 `runtime_config.json` 时，以当前 `--config` 补建运行记录，并利用已有 `alignment_runtime_config.json` 提供历史对齐上下文。`resume_bootstrap_config.json` 明确标记历史 raw 来源未经验证；每次运行另存解析后的快照到 `run_configs/`，不把当前模型信息冒充为已验证的历史检测来源。
6. **保护已检查的对齐结果**：完整对齐结果的参数、输入或输出出现冲突时，明确报错，不自动覆盖；缺失的输出组可恢复。此次 checkpoint 代码变更显式兼容原 aligner 指纹，未修改其数值对齐算法。文件一致性校验不等于自动完成了人工准确性评估。
7. **同步入口**：更新保存结果的可视化、review GUI 和 pipeline QC，使其支持直接读取 aligned 数据；更新 `scripts/inference_t70.sh` 的续跑说明、10 份 EGFR 与 7 份 TSC JSON 配置及 detection 示例配置。

### 各断点的恢复行为

| 阶段 | 完成记录与恢复方式 |
|---|---|
| Detection | 按 tile／通道复用有效 CSV 和已有 input manifest；补检测缺失通道，损坏的完整结果明确报错 |
| 2D filter | 按 tile／通道 filter manifest 复用；与已有对齐冲突时保护对齐输入 |
| Channel alignment | 检查 manifest、offsets、CSV 行数和坐标系；完整冲突停止，缺失组补算 |
| Fusion／final filter | 按 tile／通道独立复用或重建 |
| Tile positions | 新增 `_solver_manifest.json`，求解完成后立即记录，XML 生成失败不丢失已求位置 |
| XML | 保留 `_pipeline_manifest.json`；求解结果有效时只补 XML |
| Global 2D | 新增 `<channel>_global_manifest.json`，在 Z-Link 开始前提交完成记录 |
| Z-Link | 新增 `<channel>_tracks_manifest.json`；中断或修改 linking 参数时复用 global 2D |
| Colocalization | 先校验上游阶段，再使用 `_provenance_manifest.json` 决定复用／重建 |
| 质心／报告导出 | 新增 `_exports_manifest.json`；从已有共定位结果补导出 |

完整旧版 Stage 3 combined manifest 可校验并拆分接入新 checkpoint，无需重算有效的 2D／3D 输出。恢复规则见 [docs/pipeline_resume.md](docs/pipeline_resume.md)。主要实现涉及 `scripts/run_inference.py`、新增 `src/core/resume.py`、`provenance.py`、`channel_stage3.py`、`point_cloud_aligner.py`、`tile_position_pipeline.py` 和 `src/utils/io.py`。

### 验证与当前状态

- **74 项相关测试通过**，覆盖自动续跑、按阶段配置比较、旧 manifest 兼容、混合曝光输入选择、XML 中断恢复、Z-Link 中断／参数变化后的恢复，以及可视化和 QC 路径。
- 新增 `tests/test_automatic_resume.py`：真实主入口使用临时小数据从已有过滤／对齐结果完成下游，随后再次运行同一命令，验证复用全部 checkpoint、对齐 CSV 和 global 2D 不被重写，且不生成 `4_2d_filtered`。
- 全部 10 份 EGFR、7 份 TSC 配置通过解析检查；修改的 Python 文件编译检查和 `git diff --check` 通过。
- T70 只读抽查：当前对齐设置与保存记录一致；tile `384700_364300` 的四通道 filter／alignment 内容校验通过，可复用。此项不是对全部 45 tiles 大型 CSV 的完整内容验证。
- **本次没有提交 HPC 作业，没有修改 T70 已有结果数据。** EGFR/TSC 配置和提交脚本受仓库现有 `.gitignore` 规则影响，修改保存在本地工作区，不能只用普通 Git diff 判断是否已更新。

### 下一步

使用原 `scripts/inference_t70.sh` 提交 T70，检查日志是否复用检测、过滤和对齐，并从 tile positions／XML、全局 2D、Z-Link、共定位及导出继续。实际 T70 全样本下游尚未运行验证；如果发现已完成结果的真实冲突，应检查具体文件及阶段记录，不直接覆盖人工检查后的结果。

---

## 2026-10-26 对齐层数改为 100、gpu30 作业卡住、续跑 checkpoint 提速

### 背景与问题

- 所有样本的对齐采样层数统一改为 100：12 份 config 的 `sample_z_center_count` 50 → 100（EGFR4、t10、t4、ET10、ETail、TSC10/11/12q/12q_local/12t/18/8）。t70、ERH 原本已是 `sample_z_count: 100`（ERH 实际取第 501–600 层）。
- LSF bme_gpu 作业在 gpu30 上全部卡在 `Starting`，从未进入 `Running`（本人 423258、423391，以及他人 423080）。Slurm 中 gpu30 为 `DOWN+NOT_RESPONDING`，且本账户在 `gpu_partners` 只有 `short_gpu` QOS，没有 bme partner QOS，暂不改用 Slurm。`gpu_l40s` 组实际有 gpu27–31，gpu31 可正常运行，临时用 `bsub -m gpu31` 提交。
- ERH 重跑时 Checkpoint 1 之后长时间无输出；而且在只改了共定位代码的情况下，Stage 2.6 融合和 Stage 3 拼接也被重新计算。

### 修改与关键决定

1. **ET10 结果归档**：改为 100 层后，ET10 因 `check_align_settings` 报 50 → 100 不一致而退出。已将 `3_2d_aligned`、`5_2d_global`、`6_3d_global`、`7_colocalization` 移到 `detection_results_20261021/old_results_align50/`（未删除），保留 `1_2d_raw`、`2_2d_filtered`。其他用 50 层跑过对齐的样本需要做同样的归档。
2. **Checkpoint 1 后的 raw 校验改为只查文件是否存在**（`scripts/run_inference.py`）：原逻辑对每个 tile×通道的 raw CSV 及 sidecar 计算 sha256，并对每张 TIFF 做 stat（ERH 约 19 万次 `/rsstu` 访问）。如今只确认 `*_result.csv` 存在。代价是不再自动发现原始 TIFF 或 raw CSV 被替换；`validate_raw_input_manifest` 保留，可在需要时手动调用。
3. **融合与拼接 checkpoint 只对实际调用的函数做 hash**：原来 Stage 2.6 的 `fusion_code_sha256` 和 Stage 3 拼接签名都对整个 `stitcher.py` 做 hash，commit `93f26c8` 只改了共定位函数 `annotate_soma_with_tf_containment`，也会触发两步重算。
   - 新增 `provenance.source_sha256(module, names)`，只对列出的函数源码和常量做 hash，支持用 `'package.module:name'` 写其他模块里的函数。
   - `stitcher.py` 末尾新增 `FUSION_CODE`（`fuse_dual_intensity_2d` 及其辅助函数、常量、`src.utils.markers:clean_markers`）和 `STITCH_CODE`（`combine_predictions` 等）。
   - `run_inference.py` 的融合签名、`channel_stage3.py` 的 `code_sha256['stitcher.py']` 改为使用上述 hash。Stage 4 共定位签名不变，仍对整个 `stitcher.py`、`run_inference.py` 做 hash。
   - 新增 `tests/test_checkpoint_code_lists.py`：用 AST 追踪列表中函数用到的项目内函数和常量（包括跨模块），有遗漏时测试失败。规则写进了新建的 `CLAUDE.md`。
4. **作业脚本**：`inference_ET10.sh` 增加带时间戳的分步日志（同时写 `.out`/`.err`），python 加 `-u`，并加 GPU 检查和退出码；`inference_ET10.slurm` 资源与 sh 对齐（4 CPU、128G、12h、`gpu:1`），也加了分步日志。

### ERH 双曝光说明

旧作业 401644（10/20）已做过 GFP 双曝光融合，Stage 3 的 GFP 用的就是融合结果（991072 框 / 524897 细胞，与这次重跑一致）。该作业在 Stage 3 共定位时以 exit 140 被终止（24h 时限），`7_colocalization` 为空。本次重跑覆盖 `5_2d_global`/`6_3d_global` 不会改变结果，共定位会首次完成。

另外，ERH 有 15 个 tile 的对齐得分为 0、偏移全为 0（主要在 x=376600 一列，可能是 501–600 层细胞太少的边缘 tile），尚未进一步核查。

### 验证与当前状态

- 本机 `brain_detector` 环境下以下测试全部通过：test_automatic_resume（9）、test_provenance_pipeline（10）、test_pipeline_steps（2）、test_coloc_sources（7）、test_stitcher_overlap（3）、test_pipeline_filter_inputs（3）、test_checkpoint_code_lists（2）。
- 首次使用新的 hash 方式时，已有的融合和拼接 checkpoint 会再失效一次（预计融合约 3 min，拼接十几分钟），之后无关改动不再触发重算。正在运行的 ERH 作业 423504 用的是旧代码。
- 代码改动尚未 commit。

### 下一步

- 用 `bsub -m gpu31` 重新提交 ET10，确认 Stage 2.5 按 100 层重算。
- 检查其他样本（ETail、TSC 等）是否已有 50 层的对齐结果，如有则归档。
- 联系 HPC：gpu30 作业卡在启动阶段；确认 bme GPU 在 Slurm 下的 QOS/分区安排。
- 核查 ERH 那 15 个对齐得分为 0 的 tile。

---

## 2026-10-27 TSC 配置统一为 pre_align，与 ETail 同步

### 背景

TSC 样本（10、8、11、12q、12t、18）准备从头重新检测。原配置混用 `post_align`（TSC8）和 `pre_align`，通道命名不一致（`GFP`/`GFP2` 与 `GFP_3`/`GFP_20`/`GFP_25`），且 12t 的 `GFP_3` 实际指向 `640nm_5`。

### 修改与关键决定

1. 6 份 `config/TSC/*.json` 由同一模板生成，对齐、z_linker、检测和过滤参数与 `config/EGFR/3/config_EGFR_ETail.json` 一致：全部改为 `pre_align`，启用 `tile_position_params`（workers 4），`n_workers` 4，`sample_z_center_count` 100，开启 alignment QC 和 direct_stitching，新增 `stage3_n_workers: null`；删除已不再读取的 `stage_2_75_enabled`、`n_permutations`、`max_soma_sample`。原配置备份在本机 scratchpad（会话临时目录）。
2. **与 ETail 不同之处（TSC 只有 RFP/GFP/Sox9）**：`sequential_joint` 硬性要求 GFP/RFP + Sox9/Olig2，因此改用 `tf_align_mode: "chain"`（GFP→RFP 用 voxel IoU，Sox9→RFP 用 containment）；`stitching_reference_channel` 和 `reference_channel` 均为 RFP（561nm）。
3. **双曝光命名统一为 `GFP` + `GFP2`**：不带下划线，避免 class 字符串中出现伪 marker（`channel_marker` 已能处理 `GFP_3`，但无下划线的命名最稳妥）。文件名匹配均为 `*_{id}_result.csv` 形式，`GFP` 与 `GFP2` 不会互相匹配。第二曝光复用主曝光的对齐偏移；融合对称（得分高者胜出，平分时主曝光胜出）。主曝光沿用各样本原设置：TSC10 为 30（长曝光），其余为短曝光（2/3/5）。
4. YOLO 模型由 `models/train18_best_0515.pt` 改为 `models/train6_1009best.pt`：前者不在 `models/` 中。
5. `pATHXML` 设为 null（tile position 求解后写到 `rfp_dir/xml_merging.xml`）；12t 的相对路径改为绝对路径；TSC8 路径保持 `sample8/aligned/*` 不变，等数据移到 research drive 后再改。

### 验证与当前状态

- 6 份配置均通过 `load_config`、`resolve_align_settings`、`expand_double_exposure_channels`、`tile_position_frame`、`resolve_stop_step` 和 QC `resolve_options` 检查。
- `test_pipeline_steps`（2）、`test_alignment_qc_runner`（5）通过；`test_qc_alignment` 进程以 exit 127 退出，没有输出汇总。该测试不读取配置文件，与本次修改无关，尚未排查。
- **注意**：XML 生成会以 `--force` 在每个通道目录写入 `xml_merging.xml`，会覆盖 12q（`561nm`、`730nm`）和 12t（`561nm`）中已有的 TeraStitcher `xml_merging.xml`。运行前应先备份，备份文件名不要以 `xml_merging_` 开头（以免被当作帧 XML 识别）。
- 尚未提交 HPC 作业。

**同日补充**：TSC 拼接参考帧改为 Sox9（`stitching_reference_channel: "Sox9"`）。Sox9 检测很密集，RFP/GFP 太稀疏，不适合求解 tile 位置。通道对齐参考仍为 RFP。`direct_stitching.enabled` 暂时全部改为 false。`tile_position_params.enabled` 保持 true，因为 XML 由它生成；生成的 XML 写到 `sox9_dir/xml_merging.xml`，会覆盖 12q、12t `730nm` 中已有的 TeraStitcher `xml_merging.xml`，运行前需要备份。

---

## 2026-10-27 模型指纹只看实际加载的 StarDist 模型；检测日志显示 tile 进度

### 背景

T10 在 CPU 节点续跑时报 `Raw detections were produced with different settings/model content (model_content)`。原因是 `model_fingerprints` 对 `stardist_basedir`（`models/`）整个目录做哈希，把 YOLO 权重等无关文件也算了进去；删掉一个不用的 `train18_best_0515.pt` 就让所有已有结果目录的检测复用失败。

### 修改

1. `src/core/provenance.py`：`stardist_basedir` 只哈希 `<basedir>/<stardist_name>/`（StarDist 实际加载的目录），文件键统一用 `/`。新增 `comparable_model_fingerprints`：比较前把 `stardist_basedir` 的记录裁剪到该子目录，兼容旧记录（旧记录哈希的是整个目录）。YOLO 仍按 `yolo_path` 单独哈希。
2. `scripts/run_inference.py`：`validate_reused_provenance` 改用 `comparable_model_fingerprints`。**此文件被 Stage 4 共定位哈希，已完成共定位的样本会重跑一次共定位。**
3. `src/core/worker.py`：Stage 2 检测的“确认通道推理路由”和“✅ Tile … 单通道 2D 特征提取完成”两行加上 `（k/N）`，k 为该 tile 在锚点通道全部 tile 中的序号。worker.py 不参与任何 checkpoint 哈希。

### 验证

- `test_provenance_pipeline`（11，含新增的指纹测试）、`test_automatic_resume`（9）、`test_checkpoint_code_lists`（2）通过。
- 用新规则比对 ET10、ETail、T10、T4、T70 的 `runtime_config.json` 中已保存的指纹，全部一致，不会触发重新检测。
- T10 因 `stitched_direct_4x_config_EGFR_t10` 是用旧 XML 拼的（Stage 2.9 重新求解后 ABS_D 变化）而失败；旧拼图已删除，改用 Slurm CPU 分区（`inference_cpu.slurm`）续跑。

---

## 2026-10-27 对齐 QC 自动挑出可疑 tile

### 背景

可视化发现部分 tile 的通道对齐不理想。原有 alignment QC（`qc.alignment.enabled`，Stage 2.5 之后自动运行）只输出灵敏度曲线，不判断好坏。

### 修改

1. 新增 `src/qc/alignment_flags.py`，在 QC 结束后逐 tile×通道给出 `ok` / `bad` / `insufficient` / `error`：
   - **Sox9 / Olig2（intensity center）**：对固定的 containment 配对，取（核亮区强度质心 − RFP 胞体亮区强度质心）残差向量，逐轴求中位数，并用 bootstrap（固定 seed）估计 SE。若某轴 |中位数| 超过 3 µm（XY）或 4 µm（Z），且不少于 3 SE，则判为 `bad`。配对少于 `sampling.min_pairs` 的判为 `insufficient`，不算 bad。
   - **GFP 等 soma 通道（voxel IoU）**：若 IoU 最优位移超过同样的阈值且比零位移高 ≥5%，或对齐后 IoU 低于未对齐的 1.1 倍（仅当保存的偏移与参考通道不同），则判为 `bad`。
   - 阈值可在 `qc.alignment.flagging` 中修改（`max_offset_xy_um`、`max_offset_z_um`、`min_z_score`、`bootstrap_samples`、`iou_min_gain`、`iou_min_vs_unaligned`），默认值无需改配置。
2. `src/qc/alignment.py`：输出 `flagged_tiles.csv`（非 ok 的 tile，最差的排在前面）和 `tile_verdicts.csv`（逐通道），`qc_report.md` 增加「Flagged tiles」表；`run_details.json` 增加 `flagged`、`insufficient`。`src/qc/run_qc.py` 在 pipeline 日志中以 WARNING 逐条列出可疑 tile。
3. QC 每个 tile 的 checkpoint 签名现在不包含 `flagging` 选项，也不再对 `alignment_flags.py`、`run_qc.py` 求 hash：之后调整阈值或判定逻辑不会触发重新计算 QC。**本次修改了 `alignment.py`，已有的 QC tile checkpoint 会失效一次。**
4. 已有 QC 结果可离线重新判定，无需重算：`python -m src.qc.alignment_flags --qc-dir <结果目录>/qc/alignment`（读取 `run_details.json`）。

### ETail（20261026）上的结果（默认阈值，只读，复制到本地运行）

45 个 tile 中 8 个可疑：293300_324900（GFP x IoU 峰在 +18.9 µm）、293300_347600（Sox9 y +4.1 µm）、315900_370200（GFP z +1 层）、327200_347600（Sox9 y +3.7 µm，206 对）、349800_336300（Sox9 x +3.3 µm）、361100_324900（GFP x −3.9 / y −3.2 µm）、361100_347600（Olig2 x −3.3 µm）、372500_370200（GFP z −1 层，对齐后 IoU 不高于未对齐）。另有 8 个边缘 tile 配对/信号不足，无法判断。

### 验证

- 新增 `tests/test_alignment_flags.py`（6 项）通过；`test_alignment_qc_runner`（5）、`test_alignment_sampling`（8）、`test_checkpoint_code_lists`（2）通过。
- `test_qc_alignment` 进程以 exit 127 退出的原因已查明：本机 matplotlib 绘图触发 Windows 原生异常 `0xc06d007f`（DLL 延迟加载失败），属于本机环境问题，修改前后表现相同。用脚本替换绘图后重跑其串行端到端流程，通过，并确认仅修改阈值时复用 tile checkpoint。

---

## 2026-10-27 ET10 对齐 QC（本机离线运行）

ET10（`detection_results_20261021`，100 层对齐）的 config 中 `qc.alignment.enabled` 为 false，因此之前没有运行过 QC。本次在本机补跑全部 45 个 tile，参数与 config 相同。本机 matplotlib 会崩溃，所以绘图函数换成了写空文件，结果只有 CSV/报告，存放在会话 scratchpad 中，没有写入结果目录。

- 核通道：**Sox9 有 10 个 tile 判为 bad**（282000_365900、304700_354600、304700_377200、316000_377200、327300_354600、327300_377200、327300_388500、327300_399800、338600_377200、361200_388500，残差 3.1–4.5 µm，方向不一致，各 tile 各自偏移）；Olig2 有 3 个（338600_354600、338600_377200、372500_365900）。判为 ok 的 tile，XY 残差中位数约 1.1–1.5 µm，Z 为 0。
- GFP：2 个 tile 的 IoU 峰值偏离（282000_365900 x +4.5 µm、338600_377200 x +7.8 µm），282000_354600 z +1 层；另有 9 个 tile 因「对齐后 IoU 不比未对齐高 1.1 倍」被标记，但 IoU 只有 0.10–0.22，差别很小，可能是 GFP 太稀疏导致规则误报。
- 8 个边缘 tile 配对不足（主要在 399800 和 388500 两列）。

---

## 2026-10-28 ET10 跨 tile 重复检测排查；按通道 2D 拼接方案（仅讨论，未改代码）

### 起因

ET10（`detection_results_20261021`）中，原图上完全重合的一个细胞（全局约 x 3712, y 6434, z 410）在 Stage 4 产生了两条共定位结果：`glia_Olig2_RFP_Sox9`（来源 tile 316000_377200）和 `glia_GFP_Olig2_RFP_Sox9`（来源 tile 316000_365900）。两者都是全局坐标，`tile_name` 只表示来源。

### 诊断

1. **同一个细胞被两个相邻 tile 各检测一次，Stage 3 没有去掉重复。** `combine_predictions` 只在同一 z 层 IoU > 0.4 时才删除重叠区里的检测：
   - Sox9/Olig2：两个 tile 中同一个核错开约 9–11 px，核宽约 12 px，IoU 约 0.1；
   - RFP：一个 tile 的框是 38×38，另一个是 62×64 且完全包住它，IoU 0.36，IoMin 1.0；
   - GFP：IoU 0.49–0.53，被正确去重，所以只有一份。
2. **Stage 4 中的连锁结果。** soma 一对一匹配时，GFP 只能配大的 RFP 框；小的 RFP 框单独成为一个细胞。同通道重复没有合并步骤，`suppress_cross_class_overlap` 只处理 neuron 与 glia 之间的重叠。两份拷贝各自配上了自己 tile 中的 Sox9/Olig2。
3. **不是个别细胞。** 在 316000_365900 | 316000_377200 这条接缝上，每个 tile 的重叠带中约有 10 万个 Olig2 框，两边都保留下来的配对约 5.5 万。
4. **tile 位置求解（Stage 2.9）没有贴合 Olig2 自己的接缝测量。**
   - 66 条 Olig2 接缝中，解出的位置与 Olig2 实测相差：x 中位数 5.0 px、p90 16.6 px、最大 31.8 px，49 条超过 3 px；y 中位数 1.5 px、p90 7.4 px。
   - 推测原因：`model=joint` 假设通道之间的差是平滑的仿射场，但 `alignment_from=old-offsets` 下，实际放细胞用的是 Stage 2.5 的逐 tile 偏移，后者不平滑。于是其他通道的接缝经仿射场混进了 Olig2 的几何。
   - 这条接缝的 Olig2 测量本身较弱（peak_contrast 17.9，其他接缝为 35–58；match_frac 0.35），四个通道测得的 dx 也互相矛盾（−2.7 / 10.6 / 19.7 / 22.0），但状态仍为 ok。
5. **刚性平移模型本身也不够。**
   - 2×2 环闭合误差：Olig2 p50 约 5 px、p90 约 12 px，Sox9 结果相同，说明测量本身是真实的。
   - 错位随 z 漂移（Olig2 重复检测，按 z 分段的直方图峰值）：

     | 接缝 | 错位随 z 的变化 |
     |---|---|
     | 316000_365900 → 316000_377200（东） | dx 恒定约 −10，dy 从 +36 变到 −4 |
     | 327300_365900 → 327300_377200（东） | 恒定约 (−20, 0)，纯求解误差 |
     | 316000_365900 → 327300_365900（南） | 从 (+22, +32) 变到 (+14, −6) |

     每个 tile 只有一个固定位置，不可能在所有深度都对齐。
   - 注意：用最近邻、半径 15 px 统计错位时，稠密通道会把随机邻居当成配对。应改用二维直方图找峰。当天曾因此误判 327300 那条接缝正常。
6. **通道对齐误差会变成其他通道的接缝错位。** 在该细胞处，Olig2 两份拷贝错开约 (−10, +2) px，Sox9 约 (−9, +6) px，多出的 y 分量与 alignment QC 判出的 316000_377200 Sox9 y +5.8 px 吻合。

### 讨论结论与计划（明天继续）

- **原则**：拼接用 2D 检测结果（更稠密，可以按 z 分段测量）。每个通道各自求拼接；GFP/RFP 检测太少时，回退到 Olig2 的结果。
- **核心设计**：把两种对齐分开。
  - 跨 tile 去重和 z-link 在各通道自己的拼接坐标里进行；
  - 共定位时，每个 3D 细胞按其主要来源 tile 换算到 Olig2 坐标系：tile 内局部坐标 + Olig2 中该 tile 的位置 + Stage 2.5 通道偏移。
- **否决硬性中线归属**：严格只保留中线一侧的检测，会丢失只在一个 tile 中检测到的细胞。改为：
  - 配上的保留离本 tile 边缘较远的那个；
  - 没配上的两边都保留。
- **分阶段**：
  - 阶段 0（只读诊断）：每通道 × 每接缝 × 每 z 段，测 2D 重复错位。确认：各通道错位是否不同；z 漂移在各通道间是否一致（决定共用漂移形状还是各自独立求解）；GFP/RFP 的配对数（决定回退阈值）；现有重复率作为基准。
  - 阶段 1：按 z 段用 2D 结果测量接缝（直方图找峰 + 互为最近邻细化；dz 在 ±几层内搜索）。
  - 阶段 2：每个通道各自求随 z 变化的 tile 位置。用稳健加权最小二乘，按接缝段把关质量；检测不足时以接缝段为单位回退到 Olig2 + 通道偏移差。
  - 阶段 3：Stage 3 去重改为配对（IoU 或 IoMin）。
  - 阶段 4：换算到 Olig2 坐标系做共定位。「通道位置 − Olig2 位置」可作为通道偏移的独立对照。
  - 阶段 5：拼接 QC（config 中默认打开）。指标包括按接缝和 z 段的残余错位、残留重复率、环闭合误差。bad 的接缝段降权或回退后重新求解，最多 N 轮，仍不合格的输出 `flagged_seams.csv` 并打 WARNING。QC 的 checkpoint 签名不含阈值。
  - 阶段 6：在 ET10 上做改进前后对比，补单元测试，维护 `STITCH_CODE` 列表。Stage 2.9、3、4 会各重跑一次。TeraStitcher XML 表达不了随 z 变化的位置，图像拼接只能用平均位置；napari 查看器的坐标换算也需相应更新。
- **待决定**：
  1. 共定位继续用 Stage 2.5 的通道偏移（建议），还是改用拼接得出的那套；
  2. 回退阈值：按阶段 0 的数据来定，还是用经验值（例如每个 z 段少于 30 对）。
- **状态**：未改任何代码。分析脚本是临时的，没有保存到仓库；结果见上文的数字。

---

## 2026-10-28 通道 tile 目录名坐标偏移的容差匹配（EGFR batch4 sample4）

### 起因

`lsf_EGFR_b4s4_428803`：488nm（Olig2）有 4 个 tile 的目录名和其他通道差 100（0.1 µm 单位，即 10 µm）：451400/474100/496700/508000 的 `_340700` 在 488nm 下叫 `_340600`，切片文件名前缀也跟着是 `_340600`（z 后缀和锚点完全一致，各 1013 张）。worker 找不到镜像目录，只打 WARNING，写出只有表头的 Olig2 CSV，并当作完成。本样本 `stitching_reference_channel` 是 Olig2，所以还会影响 tile 位置求解（`tile_position_pipeline` 会因 488 目录缺失直接报错）和 488 的 XML。

### 修改

- `src/utils/io.py`：新增 `resolve_tile_dir`（目录名完全一致优先；否则在同级目录里找各坐标差 < `TILE_NAME_TOLERANCE`=1000 的最近目录，等距多个报错，找不到时返回原路径，保持各调用方原有的缺失处理）和 `anchor_slice_key`（把 `<通道tile>_<z>` 切片名换成锚点 tile 名）。容差是代码常量，没有放进 `detection_params`，否则 `raw_detection_settings` 会让已有检测失效。
- `worker.py`：用 `resolve_tile_dir` 找镜像目录，切片 key 换成锚点名，`source_image` 仍记录真实路径；有偏移时打 WARNING。
- `tile_position_pipeline.py`、`solve_tile_positions.py`（frame 通道目录与 aligned CSV 的 slice_name）、`generate_merging_xml.py`（`DIR_NAME` 写真实目录名，切片前缀按真实目录名校验）。
- `stitch_raw_tiles.py`、`align_stitched_channels.py`：通道间按 XML 的 (ROW, COL) 比对 tile 集合，没有 ROW/COL 时退回目录名。
- 查看器/QC：`visualize.py`、`saved_gui.py`、`review_filtered_gui.py`、`alignment_intensity_qc.py`。
- 未改：`refilter_align.py` 用对齐参考通道的目录名建索引（本样本是 RFP/561nm，名字一致）。
- 测试：新增 `tests/test_tile_dir_tolerance.py`，`test_generate_merging_xml.py` 加偏移目录用例。各测试文件单独跑均通过；`test_qc_alignment.py`、`test_tile_2d_sources.py` 在本机因 Qt 崩溃（exit 127），未改动的 HEAD 上同样如此。

### Checkpoint 影响

Stage 1 只检查 CSV 是否存在，不比较代码 hash，已完成的 tile 不会重跑。未改 `stitcher.py`、`run_inference.py`，Stage 3/4 不受影响。

### 数据处理

删除了 `detection_results_20261028/1_2d_raw` 中 451400_340700、474100_340700 的全部文件（各通道 result/candidates/instances/inputs.json）和 496700_340700 的所有 `.part`，共 40 个文件；下游目录当时都是空的。重跑后需检查这 4 个 tile 的 Olig2 通道偏移与相邻 tile 的差（若 x 约多 15 px，说明是真实台面偏差，已被对齐吸收）。

---

## 2026-10-29 ET10 按通道拼接的只读诊断（阶段 0，未改代码）

### 目的

检验 10-28 的方案：channel alignment 之后，每个通道用自己的接缝单独求拼接，再借 Stage 2.5 偏移换算到 Olig2 坐标系，能否消除跨 tile 重复。

### 方法

- 全部只读，脚本和结果在本机会话 scratchpad，没有写入结果目录。
- 第 1 部分：复用 `5_2d_global/tile_positions/seams.csv`（整段 z 的 3D 接缝测量），对每个通道用 `--model free` 的同一套求解函数单独求 tile 位置。
- 第 2 部分：读 `3_2d_aligned` 各 tile 的边缘带，按「通道 × 接缝 × 50 层 z 段」统计两个 tile 之间 2D 框中心差的直方图找峰（dz 在 ±2 层内搜索）。峰附近互为唯一的配对计为重复，共 2623 个 z 段。再按 Stage 3 的判据（同一层、IoU > 0.4）计算各种摆放下重复框被去掉的比例。

### 结果

1. **现状下重复非常多。** 按 z 段计的重复配对中，没被去掉的比例：Olig2 73%，Sox9 79%，GFP 51%，RFP 41%。66 条 Olig2 接缝中，52 条超过一半的重复没被去掉。
2. **只做按通道的刚性拼接不够。**

   | 摆放方式 | Olig2 | Sox9 | GFP | RFP |
   |---|---|---|---|---|
   | 现状 | 73% | 79% | 51% | 41% |
   | 各通道各自刚性求解（free） | 49% | 57% | 30% | 25% |
   | 每条接缝取最优常数（刚性上限） | 38% | 39% | 17% | 20% |
   | 按 z 段，所有通道用 Olig2 的接缝 | 1.3% | 59% | 36% | 22% |
   | 按 z 段，各通道用自己的接缝 | 1.3% | 1.7% | 1.0% | 5.1% |

   （数值为没被去掉的重复比例）
3. **接缝错位随 z 变化。** 同一接缝内，xy 错位在 z 方向上的变化幅度 p50 约 11 px，p90 约 40 px，四个通道相同。
4. **通道之间的差基本是每条接缝一个常数。** 同一接缝、同一 z 段，Sox9 − Olig2 的峰位差：
   - 每段 p50 4.9 px；每条接缝的常数部分 p50 4.7 px、max 11.7 px；随 z 的变化（std）p50 只有 2.3 px。
   - GFP、RFP 的常数部分 p50 分别为 7.0、5.9 px。
   - 39–62% 的 z 段 dz 与 Olig2 不同（差 1 层）。

   这与 alignment QC 中核通道残差中位数约 3.4 px 的量级吻合，即逐 tile 的通道对齐误差。
5. **第 1 部分「free 几何 − Stage 2.5 偏移」约 20 px 的逐 tile 不一致是假象。** 整段 z 测出的接缝混了 z 漂移，刚性求解按通道各自分摊。该量不能直接当对齐 QC 用。
6. **即使在同一 z 段内，tile 平移模型也闭合不上。** 2×2 环闭合误差 xy p50 约 5 px、p90 约 14 px，四个通道相同，dz 有 44–58% 不为 0。说明 tile 内部有平移以外的形变（旋转/缩放）。因此无论刚性还是随 z 变化，任何「每个 tile 一个位置」的全局解，都会在部分接缝上残留 5–14 px。对 12 px 左右的核来说，IoU > 0.4 会失败。
7. **GFP/RFP 的配对数**：每个 z 段中位数约 300 对，但 13–14% 的段少于 30 对，需要回退。Olig2/Sox9 的段几乎都在 100 对以上。

### 结论（建议的方向，待确认）

- 去重不应依赖全局 tile 位置，而应在每条接缝上按「该通道 + 该 z 段」实测的局部错位（dx, dy, dz）去比对。实测上限只剩约 1–5% 的重复。
- 稀疏通道某段配对不足时，用「Olig2 该段的错位 + 本通道该接缝的常数差」回退（通道差主要是常数）。
- 最终坐标仍用现有的全局位置 + Stage 2.5 偏移，共定位不受影响。不需要求按通道、随 z 变化的全局位置，也不需要换算回 Olig2 坐标系。
- 待设计：
  - 胜出规则：选离本 tile 边缘较远的那份，并让同一个细胞的各层都由同一个 tile 胜出，避免 z-link 时两份拷贝错开几 px 被拆开；
  - 接缝错位表的存放和 checkpoint；
  - 环闭合误差带来的残余坐标误差（约 5 px）是否需要处理。

**同日补充：用接缝错位直接求全局位置的效果。** 用每个「通道 × z 段」的接缝实测错位，以稳健加权最小二乘为每个 tile 求一个位置（各通道、各 z 段各求一套）。求解后每条接缝的残余错位：xy p50 约 2 px，p90 约 7–13 px，22–31% 的接缝超过 4 px。原因是第 6 条所说的环闭合误差：一个位置不可能同时满足一个 tile 的四条接缝。

没被去掉的重复比例：

| 方案 | Olig2 | Sox9 | GFP | RFP |
|---|---|---|---|---|
| 现状 | 73% | 79% | 51% | 41% |
| 全局解（IoU > 0.4） | 14% | 19% | 3.3% | 6.9% |
| 全局解（IoU > 0.2 或 IoMin > 0.6） | 6.3% | 6.3% | 1.8% | 1.3% |
| 按接缝实测错位比对（IoU > 0.4） | 1.3% | 1.7% | 1.0% | 5.1% |
| 按接缝实测错位比对（宽松规则） | 约 0 | 约 0 | 约 0 | 约 0 |

全局解已经比现状好很多，剩下的重复集中在环闭合误差大的接缝上。两者可以结合：全局位置决定坐标，去重时再按接缝实测错位比对。（本机 numpy 的 LAPACK `lstsq` 会以 exit 127 崩溃，与 `np.corrcoef` 相同，诊断改用 scipy `lsqr`。）

---

## 2026-10-29 Stage 3 按接缝实测错位去重（Stage 2.97）

### 修改

1. 新增 `src/core/seam_offsets.py`（Stage 2.97，Stage 3 之前运行）：
   - **测量**：对每个通道 × 接缝（EAST/SOUTH）× 50 层 z 段，在 Stage 3 的输入 CSV 上，按 XML 位置摆放后测量两个 tile 拷贝之间的残余错位 (dx, dy, dz)。方法与诊断相同：中心差直方图找峰，dz 在 ±2 层内搜索。
   - **有效段**：重复配对 ≥ 20 且峰对比度 ≥ 5。
   - **回退顺序**：
     1. 坐标系通道该段 + 本通道在这条接缝上的常数差；
     2. 本通道最近的有效段；
     3. 坐标系通道该段；
     4. 坐标系通道最近的有效段。
   - **坐标系通道**：`stitching_reference_channel`，未配置时取有效重复最多的通道。
   - **输出与 checkpoint**：结果写到 `5_2d_global/seam_offsets/seam_offsets.csv`（带 manifest）。签名含输入 CSV 的大小 + mtime、几何、参数和本文件 hash。
   - **配置**：`seam_dedup.enabled`（默认 true），其他参数见 `DEFAULT_PARAMS`。
2. `stitcher.py`：`combine_predictions` 新增 seam 模式（`_combine_predictions_seam`）。
   - 一个检测落在已拼接邻居（含对角）的重叠区内时，先减去该接缝该段的实测错位，再与邻居的同类框比较。IoU > 0.4 或 IoMin > 0.6 判为重复，丢弃。
   - 对角邻居的错位取两条两步路径的平均。
   - 没配上的重叠区检测按同一错位移到邻居坐标系，使同一细胞各层不管来自哪个 tile 都对齐，便于 Z-Link。
   - 重叠区以外的检测坐标不变。
   - 不给错位表时走原来的逻辑。`STITCH_CODE` 已加入新函数和常量。
3. `channel_stage3.py`：读取错位表，传给 `combine_predictions`。启用时 Stage 3 签名加入错位表 hash。
4. `run_inference.py`：在 geom 建好之后、Stage 3 之前调用 Stage 2.97。`pipeline_steps.py` 新增 `stop_after_step: "seam_offsets"`（2.97）。
5. 新增 `tests/test_seam_offsets.py`（8 项）：
   - 合成数据上能按段恢复错位（含 dz）；
   - 回退顺序；
   - 对角路径；
   - 偏 8 px 的重复只有 seam 模式能去掉；
   - dz 匹配；
   - 未配上的检测被移入邻居坐标系；
   - 重叠区外的检测不动。
6. README 和 CLAUDE.md 已更新。

### 验证

- 本机所有测试文件逐个运行均通过（`test_qc_alignment`、`test_tile_2d_sources` 本机 Qt 崩溃，未运行）。
- ET10 2×2 块（316000/327300 × 365900/377200，含最差的几条接缝），用真实的 Stage 2.97 + `_stitch_channel`，结果只写到 scratchpad：

  | 通道 | 判为重复的框（旧 → 新） | 诊断预期 | 重叠区密度 / 单 tile 区密度（旧 → 新） |
  |---|---|---|---|
  | Olig2 | 1.5 万 → 22.1 万 | 约 16.8 万对重复 | 1.79 → 1.21 |
  | GFP | 1.3 万 → 3.1 万 | — | — |

  测量耗时 22 s。新输出中，跨 tile、同层、中心 < 4 px 的框对：
  - Olig2：930 对（共 253 万个框）；
  - GFP：1140 对，其中 1038 对是同一 soma 在两个 tile 中分别判为 glia/neuron。按类去重本来就不比较不同类，留给 Stage 4 的 `suppress_cross_class_overlap` 处理。

  GFP 有两条接缝每段只有约 10 对重复，回退到 Olig2。

### Checkpoint 影响

`stitcher.py`、`channel_stage3.py`、`run_inference.py` 均已修改。各样本 Stage 3（5_2d_global、6_3d_global）和 Stage 4 共定位会各重跑一次，预期如此。Stage 2.97 第一次运行需要读一遍所有 tile CSV 的边缘带。

### 未做

- **随 z 变化的全局 tile 位置**：XML 表达不了随 z 变化的位置，细胞坐标会与拼接图像在接缝附近错开（最多约 40 px）。2026-10-30 决定不做：全局坐标保持 XML tile 位置，本次只解决重叠区去重。
- **检测坐标与拼接图像的一致性（2026-10-30 核对，ET10）**：检测全部在 Olig2 坐标系（Olig2 XML 位置 + Stage 2.5 逐 tile 通道偏移）。Olig2 图像用同一套位置，与检测逐像素一致。其他通道图像用各自独立解出的 `P_c` + 一个全局平移，不含逐 tile 通道偏移，与检测逐 tile 相差 |dxy| p50 14–17 px、p90 21–29 px、最大约 48 px，|dz| 最多 3–4 层。其中大部分是随行列线性变化的平滑场（色差放大率，y 方向约 4–6 px/行）；去掉平滑场后 p50 3.5–5 px、p90 8–11 px，少数 tile 仍有 40 px 以上。
- **重叠区「离边缘较远者胜出」规则**：现在仍是先拼的 tile 胜出（各通道一致，按位置确定）。

---

## 2026-10-31 Stage 2.97 改为从名义重叠出发的 IoU 测量；Stage 3 跨类别去重；ET10 全片测试

### 决定

- 不做「每个通道各自求全局 tile 位置」。全局坐标仍为 Olig2 XML 位置 + Stage 2.5 逐 tile 通道偏移，通道对齐误差留在坐标里，可以接受。
- 去重只依赖按通道、按接缝、按 z 段实测的错位。
- 测量方式与 channel alignment 相同：从名义 15% 重叠、z 位移 0 出发，在重叠体积里找 IoU 最大的位移。每 50 层一块体积，不用一整块，因为错位随 z 漂移。
- soma 通道在 Stage 3 重叠区跨类别去重，glia 优先。
- 验收阈值先看 ET10 的 QC 再定。

### 修改

1. `src/core/seam_offsets.py`（测量部分重写）：
   - 名义偏移 `nominal_offset`，名义重叠区 `nominal_band`。名义步长 = tile_size × (1 − overlap_pct/100)；`overlap_pct` 默认取 `pre_align_params.tile_overlap_pct`（15）。
   - **粗搜**：两 tile 的框按 4 px 体素化，3D FFT 互相关，搜索 ±160 px、±10 层。同一接缝各段的强峰先取中位数作为共同中心，各段再在中心 ±50 px、±3 层内取峰。
   - **细搜**：在粗峰 ±5 px、±1 层内按 1 px 计算框 IoU，分子是配对框交集之和，分母是两体积框面积之和减交集，再做亚像素抛物线拟合。
   - **有效段判据**：一一配对数 ≥ 20（IoU > 0.4 或 IoMin > 0.6）；粗峰 snr ≥ 6，snr 为相关窗口中位数之上的稳健标准差倍数；与相邻 ±2 段的中位数相差 ≤ 15 px、≤ 2 层，否则标为 outlier。
   - 回退顺序不变。输出新增 `n_b, snr, iou, outlier, nom_dx/dy/dz`，`contrast` 改为 `snr`。签名 version 2，`fft_threads` 不进签名。
   - 本机 numpy 的 matmul（BLAS）会以 0xc06d007f 崩溃，细搜改为逐行相乘求和。matplotlib 的 imshow 同样会崩，本机出图改用 PIL。
2. `src/core/stitcher.py`（seam 模式）：
   - 重叠区检测同时与邻居的 glia 和 neuron 框比较。
   - B 为 glia、匹配到的都是 neuron 时，保留先拼 tile 的框，类别改为 glia，坐标不变。决策表记 `relabelled / cross_tile_glia_priority`；被删的 B 框记 `cross_tile_overlap_other_type`。
   - 新 helper `_is_glia` 已加入 `STITCH_CODE`。
   - TF 通道的 class 都是 nucleus，不受影响。
3. 测试：`tests/test_seam_offsets.py`。
   - 合成数据从名义网格出发，恢复每段 (dx, dy, dz)，稠密核和稀疏 soma 各一例。
   - 新增 outlier 和弱段不参与回退的测试。
   - 新增跨类别去重两例：glia 替 neuron；neuron 被 glia 吃掉。
   - 所有测试文件逐个运行通过（`test_qc_alignment`、`test_tile_2d_sources` 本机 Qt 崩溃，未运行）。
4. README Stage 2.97 段落已更新。

### ET10 全片测试

结果只写到 `Y:/Fengyi/EGFR_brain/ET10/seam_dedup_test_20261031/`（含 README），原结果目录未动。测试脚本在会话 scratchpad，没有进仓库。

- **耗时**：测量 8 分钟（4 通道并行，每通道 8 个 FFT 线程）。本机 Stage 3 2D 拼接：Sox9 48 分钟、Olig2 23 分钟、GFP/RFP 各约 3.5 分钟。
- **测量结果**：每通道 990 段（66 条接缝 × 15 段）。没测到的段约 300 个，几乎都是 z 两端没有组织的段（n_a < 20）；配对不足 18–31 个，snr 不足 1–32 个，outlier 3 个。
- **与旧直方图法对比**：两者都测到的段，xy 差 p50 0.1–0.5 px、p90 约 2 px。分歧段（xy > 3 px 或 dz 不同）在同一判据下比较配对数，约 85% 新位移更多，总配对多 6–16%。
- **GFP 的大错位**：接缝 327300_365900 → 338600_365900 上，GFP 每段稳定约 (−37, +48, −3)，Olig2 约 (−6, +4, 0)。差值等于两个 tile 的 Stage 2.5 GFP 偏移之差；alignment QC 也把 327300_365900 的 GFP 判为 bad。旧方法只搜 ±40 px，够不到。
- **去重决策**：被删的重复框，GFP/RFP 约为旧规则的 2 倍，Olig2/Sox9 约为 3.5–4 倍。GFP 跨类别删除 23823 个、改为 glia 8163 个；RFP 分别为 29690 和 9119。
- **2D 残留**：同一层、扣除错位后的残留重复占去重前重复数的比例：GFP 0.1%，RFP 0.2%，Olig2 2.9%，Sox9 1.5%。按峰值不分 dz 统计时为 8–14%，但其中绝大部分落在相邻层（dz = ±1），是 A 漏检、由 B 补上的同一细胞的其他层，Z-link 会把它们连成一个细胞。
- **3D 残留**：重叠区内主要来源 tile 不同、xy < 5 px（核）/ 10 px（soma）、z 范围重叠的 3D 细胞对，减去平移 30 px 的随机对照：

  | 通道 | 旧规则 | 新规则 |
  |---|---|---|
  | GFP | 28.6% | 0.11% |
  | RFP | 25.8% | 0.11% |
  | Olig2 | 49.0% | 1.5% |
  | Sox9 | 49.4% | 3.0% |

  （占重叠区 A 侧 3D 细胞数）
- **检查图**：32 张（4 通道 × 4 条接缝 × 第 175、475 层），在 `review_images/`。
  - 图像定位验证正确：B 按实测错位移动后，与 A 的核重合。
  - 每个被删的 B 框在 A 中同位置都有框。

### 未做 / 待定

- 人工检查图后再决定验收阈值。
- Sox9 剩余约 3% 的 3D 重复，可能来自段内 z 漂移和 dz 取整；可以试 25 层一段（只改配置 `seam_dedup.seg_layers`）。
- 去重 QC 尚未写进 pipeline，等确认后再做。
- 各样本下次运行时，Stage 2.97、Stage 3、Stage 4 会各重跑一次，这是预期的。

---

## 2026-10-31（晚）去重方案经人工确认；接缝去重 QC 接入 pipeline

### 确认

人工检查 ET10 的 32 张检查图后确认效果良好。新版 Stage 2.97 和 Stage 3 跨类别去重保留在 pipeline 中，与上一条记录相同。

### 修改

1. 新增 `src/qc/stitch_dedup.py`，在 Stage 3（拼接 + Z-link）之后、共定位之前运行，只读，默认打开。
   - **3D 残留**：XML 重叠区内，主要来源为 A 的 3D 细胞与主要来源为 B 的 3D 细胞，中心 < 5 px（核）/ 10 px（soma）且 z 范围重叠，一一配对。减去 B 平移 30 px 的随机对照，除以 A 侧细胞数。
   - **2D 同层残留**：按 50 层一段，同一层、< 3 / 5 px 的框对，减随机对照，除以 Stage 2.97 测得的去重前重复数。
   - **决策表计数**：rejected / relabelled 等。
   - **标记**：A 侧细胞 ≥ `min_seam_cells`（200）且 3D 残留 > `max_seam_dup_pct`（10%）的接缝写入 `flagged_seams.csv`，并打 WARNING；默认不中断（`fail_on_error`）。
   - **检查图**：每通道 `image_seams_per_channel`（2）张，分别是最差接缝和中位接缝。层取该接缝配对最多的 z 段中层，裁剪取沿接缝框最密的 512 px。三格：A + 移动后的 B、最终框、B 的删除 / 保留框。标注用英文，因为 Linux 上没有中文字体。
   - **复用**：签名包含 Stage 3 输出文件的 stamp、错位表 hash、几何、计算参数和代码 hash，不含阈值和图片数。签名不变时直接读表（ET10 5 秒），已有 PNG 跳过；签名变化时重算并删除旧 PNG。
   - **命令行**：`python -m src.qc.stitch_dedup <results> [--path-map FROM TO]`。
2. `src/qc/run_qc.py` 新增 `run_stitch_dedup_stage`；`scripts/run_inference.py` 在 `run_stage3_channel_pool` 之后调用。
3. 新增 `tests/test_stitch_dedup_qc.py`（3 项）；README 新增「Seam deduplication QC」一节。所有测试文件逐个运行通过（两个 Qt 测试本机不运行）。

### ET10 测试（测试目录 `seam_dedup_test_20261031`，先对新 2D 结果跑了 Z-link）

- **耗时**：Z-link Sox9 42 分钟、Olig2 20 分钟、GFP/RFP 各约 2.5 分钟。QC 第一次 7.5 分钟（本机，读 Y: 盘），第二次 5 秒、0 张新图。
- **结果**：

  | 通道 | 3D 残留 | 单条接缝 p50 | 单条接缝最大 |
  |---|---|---|---|
  | GFP | 0.21% | 0.02% | 12.5% |
  | RFP | 0.21% | 0.06% | 10.2% |
  | Olig2 | 1.45% | 0.43% | 13.6% |
  | Sox9 | 2.92% | 1.92% | 11.5% |

  按默认阈值标记了 1 条：Sox9 293300_388500 → 304700_388500（461 个细胞，组织边缘）。
- 第一版检查图固定取接缝中段，组织边缘的接缝会裁到空白处，已改为取框最密的一段。

### Checkpoint 影响

改了 `run_inference.py`，Stage 4 本来就会因 `stitcher.py` 的改动重跑，这次不额外增加重跑。

### 待定 / 后续

- 四个 tile 交汇的角落区：对角邻居的错位由两条接缝平均得到，QC 目前只统计相邻两 tile，角落区没有单独统计。ET10 边缘接缝的检查图里能看到第三个 tile 的框，值得以后检查。
- 327300_365900 的 GFP 通道对齐偏差约 50 px（alignment QC 判 bad），去重已吸收，但共定位坐标仍带这个误差。

**同日补充：去掉阈值标记，检查图改为随机抽样。** 按要求删除了接缝标记功能，包括 `max_seam_dup_pct`、`min_seam_cells`、`flagged_seams.csv` 和警告。残留重复统计表和日志里的数字保留。检查图改为每通道 `images_per_channel`（默认 10）张：从本通道实测的（接缝, z 段）中随机抽取，先抽不同接缝，接缝不足时再补其他段。种子由 `image_seed`（默认 0）和通道名决定，结果不变时抽到同一批，已有 PNG 跳过。ET10 测试：首次 8 分钟，生成 40 张图（每通道 10 张，各来自不同接缝）；第二次 5 秒，0 张。测试改为 5 项，全部通过。

---

## 2026-11-02 EGFR 样本复跑准备：统一到 bbox_max=100 和 Stage 2.97 拼接

### 核查结果（只读 manifest）

- 9 个 EGFR 样本的 filter manifest 都是 schema 6，代码 hash 与当前 `detection_filter.py` 一致，glia 优先的 containment NMS 都已生效。
- 与当前 config 唯一的差别：T10、T4、T70、ETail、ET1、ERH 旧结果的 `detection_params.yolo.bbox_max` 为 null，当前为 100。
- 只有 ET10 用了 Stage 2.97 新拼接，Stage 3/4 代码 hash 与当前一致，不用重跑。
- batch4 sample4/5：过滤、对齐为最新；10-30 Stage 3 Sox9 被 OOM kill（128G，4 通道同跑），Sox9（sample5 还有 Olig2）没有 3D 结果，Stage 4 未完成。

### 处理

1. batch4 sample4/5：`stage3_n_workers` null → 2；slurm `--mem` 128G → 240G。直接重新提交即可从 Stage 2.97 续跑。
2. T70 / ETail：`pATHRESULT` 的 `{date}` 改成已有目录（`detection_results3`、`detection_results_20261026`），避免新建目录重跑检测。
3. 6 个 bbox_max 样本：把 `3_2d_aligned`、`5_2d_global`、`6_3d_global`、`7_colocalization`、`qc/alignment`（ERH 另有 `3_2d_aligned_fusion`、`4_2d_filtered`）移到结果目录下的 `old_results_bboxNone/`（附 README）。否则 Stage 2.25 会报 `Filtered checkpoint conflicts with saved alignment`。
   - 模拟 checkpoint 判断：GFP/RFP（ERH 还有 GFPhi）重新过滤，Sox9/Olig2 复用；raw 检测复用（`bbox_max` 不在 raw 签名里）。
   - 重新提交后：过滤 → 对齐（参考通道是 soma，所有通道偏移会变）→ 融合（ERH）→ tile positions/XML → 2.97 → Stage 3 → 去重 QC → Stage 4。

### 未做

- 没有提交 HPC 作业。

---

## 2026-11-02 Alignment QC 兼容通道间 tile 命名微小偏差

- 问题：batch4 sample4（job 1645672）对齐 QC 有 4 个 tile 报 `Missing source image for one-based slice N`：451400/474100/496700/508000 的 `_340700`。原因是 488nm（Olig2）这 4 个 tile 的目录名和文件名是 `_340600`，其他通道是 `_340700`。主流程用 `resolve_tile_dir` + `anchor_slice_key` 能对上，QC 读图只按原始文件名匹配，所以找不到。检测/对齐/拼接/共定位结果不受影响。
- 修改：
  - `src/qc/alignment.py` `extract_pairs`：切片按 `anchor_slice_key` 换成锚点 tile 名再匹配。
  - `src/utils/validate_align_shifts.py` `tile_slice_names`：精确路径找不到时用 `resolve_tile_dir`（容差 1000）找目录，切片名同样换成锚点名。
  - 新增测试 `test_legacy_support_tolerates_drifted_channel_tile_name`。
- 真实数据验证：451400_340700 的 Olig2 1013 张切片全部对上锚点名。
- 影响：QC 代码 hash 变化，下次运行全部 81 个 tile 的对齐 QC 重跑一次；不影响 Stage 2.6/3/4 checkpoint。
- 已知：本地 Windows 上 `test_qc_alignment.py` 的 end-to-end 测试在改动前后都以 exit 127 退出（子进程环境问题），与本次改动无关。

---

## 2026-11-02 ~ 03 Sox9 假阳性：局部 SNR 与分类器离线测试（未改 pipeline）

ET10 共定位中 Sox9 假阳性多，主要来自淡的背景核状结构。StarDist score 区分不开（假阳性也有 0.73）。

- **特征**：每个框计算 mask 外 2–8 px 环形背景（排除所有检测 mask）的中位数和 MAD，SNR = (mean − ring_bg) / ring_MAD。每层计算约 1 秒；主要耗时在读原图和 CSV。
- **亮度绝对值不可用**：不同层的背景相差约 10 倍。环形背景的噪声与背景亮度成正比（MAD/bg ≈ 0.15），说明噪声来自组织纹理。人工选的每层 SNR 阈值（3–15）与背景亮度无关。
- **人工阈值**：9 个 EGFR 样本共 44 层，结果在 `Y:\Fengyi\EGFR_brain\ET10\sox9_contrast_test_20261102\`（每层有 SNR 2–16 的扫描图和标签）。
- **分类器**：输入为 SNR + 层级上下文，对 SNR 单调递增，深度 2。留一组交叉验证的平均平衡准确率：现有过滤 0.681，全局 SNR ≥ 5 0.899，分类器 0.899（最差一层 0.69，全局阈值为 0.62）。用全部特征的模型反而更差（0.883）。
- **验证**：ET10 6 层（训练没用过的 tile）+ TSC FF8/FF9/FW2/FW6 各 3 层。分类器在每层的等效阈值为 SNR 4–8，大多是 5。现有过滤在 TSC 上比分类器多保留 10–30% 的框。结果在 `Y:\Fengyi\EGFR_brain\test\sox9_classifier_20261103\report.html`。
- **待定**：人工检查验证图后，决定是用全局 SNR ≥ 5 还是用分类器。接入时在 Stage 2.25 之前新增特征阶段（读 Sox9 原图，结果缓存），再在 Stage 2.25 按概率或 SNR 过滤；Sox9 参与对齐，所以过滤和对齐以下都要重跑。

---

## 2026-11-03 Stage 4 共定位问题排查：同一细胞被拆成多个结果、TF 标签被抢（未改代码）

在 viewer 里人工检查 ET10（`ET10/detection_results_20261021`，tile 316000_377200 / 316000_388500，z 约 393–412）的 s4 结果，发现 5 个例子，归为 3 类。

### 1. 同一细胞在某个通道断成多段 track，共定位后成了重复细胞（3 例，主要问题）

- **截断原因 A**：`max_cell_z_span`（soma 5、TF 3）。track 从 `first_z` 起满 N 层就强制结束，下一层另起新 track（`z_linker.py:188`）。这批细胞在 z 上常跨 6–7 层。
  - 例：GFP 406–410 + 411。
  - 例：GFP/RFP 都是 393–397，RFP 在 398 另起；Sox9 394–396 + 397–398，正好顶到上限。
- **截断原因 B**：相邻层的框大小突变，IoU 低于 z-link `iou_thresh`，track 断开。
  - 例：RFP 在 z402 的框约 74 px，z403 缩到约 42 px，IoU ≤ 0.32。
  - 第一例 RFP 在 407/408 断开可能也是这个原因。
- **共定位是一对一贪心匹配**（`match_soma_3d_iou`，按 IoU 从高到低，每条 track 只用一次）。一个 GFP 碎片配上一个 RFP 碎片后，剩下的碎片要么互相配成第二个 GFP+RFP 细胞，要么作为单通道细胞追加进结果。
  - 例：GFP_A–RFP_A IoU 0.194 先配上；GFP_A–RFP_B IoU 0.168 因 GFP_A 已被占用而跳过；GFP_B–RFP_B 靠 IoMin = 1.0 配上。
- **匹配之后没有去重**：结果细胞 XY 几乎重合、z 区间重叠或相接，也不会合并（`run_inference.py:1086` 直接追加 unmatched）。
- **TF 标签被抢**：同一 TF 只标给中心最近的 soma（`stitcher.py:850`）。
  - 例：Sox9 `track3d:a562c7b9…` 与 GFP+RFP 细胞 `coloc:bc9cb290…` 完美重合，中心距离约 3.5 px，包含判断全部通过。但它被标给了同一位置的单 RFP 碎片 `coloc:b58df551…`（RFP z401–402，框约 74 px，中心距离 1.08 px），这个碎片成了 RFP+Sox9+Olig2。可用 `coloc_match_evidence.csv` 的 tf_containment 行查。

### 2. 合并后的框超过 bbox_max = 100（1 例）

- `bbox_max` 只作用于每个通道的 2D 框（`detection_filter.py:155`）。`merge_soma_volumes_union` 取的跨通道并集不会再检查尺寸；Stage 4 只有 TF 有 `bbox_max_w/h`。
  - 例：GFP 89×85 和 RFP 75×47 中心错开约 30 px，并集宽 112 px。

### 3. 单层、低分的 soma 拼成多阳性假阳性（2 例）

- 每个通道各一个单层、低分的框，按 IoU/IoMin 拼成多阳性细胞。在原图上看不到 GFP/RFP 细胞。
  - 例：GFP 0.20 + RFP 0.19，四阳性，mean 26078。
  - 例：GFP 被判为 neuron（0.69，2 层）+ RFP 0.155 单层，合并后类别是 glia。
- 共定位只看几何（IoU/IoMin），不看 score、`member_count`，也不看两边类别是否一致。soma 的 1 成员 track 也会保留。

### 拟修改（待定）

1. **优先**：Phase B 之后加共定位后去重。XY IoMin 高、z 区间重叠或相接的结果细胞合并为一个，soma 和 TF 标签取并集。能解决第 1 类的全部 3 例。
2. 并集之后加 soma 尺寸检查，或拒绝源框中心距离过大的配对。
3. 对单层、低分的 soma track 加更严格的跨通道匹配门槛，或拒绝 neuron/glia 跨类别配对。要放在第 1 步之后，否则会把真实细胞的 z 碎片删掉。
4. 调大 `max_cell_z_span` 前，先统计各通道 `member_count` 分布，看在上限处有没有尖峰。

以上改 `run_inference.py` 和 `stitcher.py`，Stage 4 要重跑；如果改了 `combine_predictions` 调用到的函数，还要同步更新 `STITCH_CODE`。

---

## 2026-11-03 Sox9 局部 SNR 过滤接入 pipeline（Stage 2.2），ET10 待重跑

选用全局 SNR ≥ 5（分类器平均效果相同，不值得多维护一个模型）。

- **新模块** `src/core/local_contrast.py`：
  - 计算方法与离线测试完全一致：环形背景 2–8 px，排除所有实例 mask；SNR = (mean − 环形中位数) / (1.4826 × MAD)。
  - 在 ET10 304700_365900 z150 上复算 4138 个框，与离线结果的最大差 7e-15。
  - 速度：约 230 µs/框。ET10 约 3200 万个 Sox9 框，约 2 CPU 小时。
- **配置**：顶层 `"contrast_filter": {"Sox9": {"snr_min": 5}}`，只支持单曝光 StarDist 通道（要用 instances.csv 的多边形）。不写这一节时，所有 checkpoint 签名和对齐设置都不变，其他样本不受影响。
- **Stage 2.2**（检测之后）：
  - 每个 tile×通道写 `1_2d_raw/contrast/{tile}_{ch}_contrast.csv`（detection_id, ring_bg, ring_mad, ring_n, snr）和 manifest。
  - 签名包括：raw CSV 的 hash、instances CSV 的大小、环参数、模块代码 hash。
  - 按 tile 多进程并行，进程数 = SLURM CPU 数。
- **Stage 2.25 / 2.75**：
  - 先跑原有过滤，再去掉 SNR < snr_min 的框。rejected 中的原因记为 `snr_min`，输出 CSV 多一列 `snr`。
  - 环形像素少于 10 个、无法计算 SNR 的框保留。
  - 只有配置了 contrast_filter 的通道，过滤签名里才加入 SNR 表 hash 和阈值；配置后，对齐设置里也会加入 `contrast_filter`。
- `detection_filter.py` 没改，所以其他样本已有的过滤和对齐 checkpoint 仍然有效。`run_inference.py` 改了，按惯例其他样本下次运行时会重跑 Stage 4。
- `refilter_align` / `refilter_detections` 遇到 contrast_filter 通道时直接报错：它们不算 SNR，不能静默跳过。
- **测试**：新增 `tests/test_local_contrast.py`（6 项），相关模块的测试全部通过。

### ET10
- config 已加上 `contrast_filter: {"Sox9": {"snr_min": 5}}`。
- 旧结果已移到 `detection_results_20261021/old_results_sox9_no_snr/`（附 README），包括 3_2d_aligned、5_2d_global、6_3d_global、7_colocalization、qc/stitch_dedup，以及旧的 Sox9 过滤 CSV。
- 重新提交 `scripts/egfr/inference_ET10.slurm` 后的流程：
  - Stage 2.2 计算 45 个 tile 的 SNR
  - 只重新过滤 Sox9
  - 对齐 → tile positions → 2.97 → 3 → 4
- 离线估计 Sox9 每层保留约 51%，现有过滤约 73%。

---

## 2026-11-05 可视化工具：删除 post 模式

post-alignment 流程已不再使用，可视化只保留 `prealign` 和 `2d` 两种模式。

- `src/utils/visualize.py`（2294 → 916 行）：
  - 删除 `_run_post`，以及只有它用到的 helper：tile 重叠边距遮罩、可视化端 size/intensity 预览过滤、全局 CSV → tile shapes、s4 共定位层、Shift+click 共定位 trace、s3 span 调试、pkl 读取、GUI 内 z-link 和共定位重算等。
  - 去掉相应 import（`run_z_linker`、`stitcher` 函数、`detection_filter`）。
  - `_load_tile_csv_shapes` 只负责读框：去掉 `raw_vol`/`filt`/边距参数（已无调用方传入，原 `filt` 分支实为死代码），直接返回 `(shapes, colors, meta)`；`_run_2d` 和 `review_filtered_gui.py` 已同步。
  - `--mode post` 或 config 里 `mode: "post"` 会报错退出，并提示该模式已删除。
- `visualize_gui.py`：Mode 下拉框去掉 post，删除只对 post 生效的 "Result stage" 控件。
- `config_example/visualization_config.example.json`：删除已无人读取的 `z_linker`、`stage`、`filter`、各类 opacity、`sphere_colors`、`marker_colors`；补充 `aligned_filtered` 在 layout v2 下退回 `3_2d_aligned` 的说明。本地 `config/vis_config.json` 中的这些键会被忽略，不影响运行。
- 测试：删除只覆盖 post 模式 helper 的 `tests/test_visualization_tile_selection.py`；`test_saved_pre_gui.py` 的 z-link patch 目标改为 `src.core.z_linker.run_z_linker`。`test_saved_pre_gui`（17）通过；`test_tile_2d_sources`（4）在先创建 QApplication 时全部通过。直接运行时，`_run_2d` 那一项会让进程以 127 退出，HEAD 上也一样，是已有问题。
- 不影响任何 pipeline checkpoint（`visualize.py` 不在 Stage 2.6/3/4 的哈希范围内）。

---

## 2026-11-05 Saved local 视图：s3 3D 框画回本 tile 自己的坐标系

现象：ET10 prealign local 视图里，接缝重叠区的 s3 3D 框偏离细胞，tile-local 2D 框是准的。

- **原因 1（重叠区，主要）**：Stage 3 在重叠区保留先拼接的那个 tile 的框；后拼接 tile 没配上的框会按 seam offset 移进前者的坐标系。全局 3D 框（= track 代表框的全局坐标）因此对上的是邻居 tile 的图像。GUI 只减本 tile 的 XML 位置，差一个 seam offset。316000_388500 GFP：约 40% 的 s3 框来自邻居 tile，左缝偏约 dx +16、dy −6 px，z 也错 0–2 层。RFP 统计相同。
- **原因 2（全 tile）**：s3 行的 xy 来自代表框所在层（`best_box_z`），但 `z` 列是 track 的中位层，约 34% 差 1–2 层。
- GUI 的 x/y/z 换算本身没问题（`max(ABS_D) − ABS_D` 与 Stage 3 的 z0 一致）。
- **修改**（只改 viewer，不碰任何 checkpoint）：
  - `saved_view.read_global_boxes` 加 `z_column`；s3 层按 `best_box_z` 选层、画框（老文件没有这一列时退回 `z`）。local 和 global 视图都生效。
  - local 视图新增 `s3_rows_in_tile_frame`，按 `best_detection_id` 找 Stage 3 输入 CSV 里的源框：本 tile 检测的直接用 tile-local 框和层；邻居检测的用邻居 tile-local 框 + XML 位置差 + seam offset（与 Stage 3 的 `_seam_shift` 相同，按本 tile 层所在段取值）；找不到源框时仍按 XML 放置。
  - seam 表和 seg_layers 取自 `5_2d_global/{ch}_global_manifest.json`：manifest 里的 seam 表 sha256 对不上、或 frame XML 与 manifest 几何不一致时，邻居框退回 XML 放置并打印提示。
  - 每行保留 `stage3_x1..y2`（点击查 PKL track 用）和 `source_tile`。
  - global 视图仍按 XML 放置（各 tile 图像本来就按 XML 叠加）。
  - `[global saved]` 2D 层和 Shift+click 的 track span 没改，重叠区仍按 XML 放置。
  - **s4 层**（local 视图，同一天追加）：换算逻辑抽成 `saved_view.TileFrame`，s3 和 s4 共用。
    - rep-z 行：按 `coloc_result` 的 `tile_name`/`slice_name` 找到主 track 在中位层的成员框（同一切片上宽高完全相同的唯一框；平移不改变宽高），按它换算；找不到时按主 track 代表框的位移平移。
    - union 行：每个 soma 通道的 3D 框（`source_3d` 里的 `x1_3d..`、`z_min/z_max`）按各自代表框的位移换算后重新取并集，画在 `round((z_min+z_max)/2)` 层，与 Stage 4 相同。已保存的 union 不等于这些 soma track 的并集时保持 XML 放置。TF track 不参与并集。
    - s3 和 s4 在同一次打开中共用 tile CSV 和 seam 表的读取缓存。
- **ET10 验证**（316000_388500，与本 tile 自己的检测比较，IoU>0.5 的比例）：
  - 本 tile 来源：1.00。
  - 邻居来源，同层：GFP 0.04（XML 放置、best_box_z）/ 0.10（改前）→ 0.42；±1 层内 0.64；±3 层内仍有约 22% 附近没有本 tile 的检测（本 tile 没检出）。
  - 剩余的主要是 ±1 层误差：seam dz 是每 50 层一个整数。
  - 每个邻居方向都比 XML 放置好，没有符号错误。
  - 处理一个 tile 约 4 秒。
  - s4（同一 tile，35362 行）：rep-z 本 tile 来源 0.94 → 1.00；邻居来源 0.04 → 0.39（±1 层内 0.57）。union 行 11098 个，只有 83 个本 tile 来源的框有变化（本 tile 检测在 Stage 3 被 seam offset 移过）。s4 换算单独运行约 13 秒；viewer 里与 s3 共用 CSV 读取，实际多出约 8–9 秒（读 `coloc_result.csv` 本身约 34 秒）。
- 测试：新增 `tests/test_saved_s3_tile_frame.py`（6 项，含 s4）；`test_saved_pre_gui`（17）、`test_checkpoint_code_lists`（2）通过。

---

## 2026-11-05 Stage 4：跨通道合并前，先在每个 soma 通道内做 neuron/glia 去重

现象：ET10 GFP 同一个细胞在 s3 和最终结果里都是两个（同层一个 neuron_GFP、一个 glia_GFP，IoU 0.59）。

- **原因**：
  - Stage 4 的跨类别去重原来只在 Phase A（跨通道合并）之后跑一次。
  - GFP neuron track 先和 RFP glia track 合并，合并后的细胞框变大、类别变成 glia。去重时这一对已不再是 neuron/glia，于是两个都保留。
  - 另一种情况：glia 先和别的通道合并，框变大后 IoU 掉到 0.5 以下。
  - ET10 全脑最终结果里，同通道 neuron/glia 3D IoU > 0.5、两个都保留的：GFP 5,646 对，RFP 4,876 对。
- **修改**：
  - `stitcher.suppress_cross_class_per_channel`：对每个 soma 通道各自跑一次 `suppress_cross_class_overlap`，参数相同（`cross_class_iou_thresh`、`z_pad_3d`），保留原顺序。
  - `run_inference.py` Stage 4 在 Phase A 之前调用它，Phase A 之后原有的去重不变。
  - 去掉的 track 写进 `coloc_decisions.csv`，`stage = channel_cross_class_suppression`，`channel` 为所在通道。
- **离线验证**（`X:/Fengyi/test_xclass`，ET10、batch4/sample4、TSC FF8、TSC FW2，各取一个 tile 中央 1200×1200 px × 50 层）：
  - 对“现在”的模拟与已保存的 `coloc_source_3d.csv` 完全一致。
  - 细胞数减少 0.7–1.5%；去掉的 292 对全部是两条单层 track（同层或相邻层）。
  - 抽样看图基本都是同一个细胞，多层 neuron track 不受影响。
  - 用 pipeline 里的函数重跑 ET10、FW2，结果与离线测试相同。
- **另查**：Stage 2.25 的 `nms_containment_thresh` 调低（`X:/Fengyi/test_nms`，7 个样本）解决不了这个问题。3D neuron/glia 重复约 98% 是相邻层类别不同，同层的只占约 2%；调低后 3D 重复数反而略增。所以没有改。
- **影响**：
  - Stage 4 哈希整个 `stitcher.py` 和 `run_inference.py`，下次运行会重跑共定位。
  - 新函数不在 `STITCH_CODE` / `FUSION_CODE` 里，Stage 2.6 和 Stage 3 checkpoint 不受影响。
- **待查**：batch4/sample4 的 tile `451400_340700` 2D 各步都正常，但不在 Stage 3 结果里。
- 测试：新增 `tests/test_cross_class_per_channel.py`（3 项，含 ET10 例子）；`test_checkpoint_code_lists`（2）、`test_coloc_sources`（7）、`test_stitcher_overlap`（3）通过。

---

## 2026-11-05 对齐 checkpoint 误报冲突：换行符（CRLF→LF）改变了代码 hash

现象：ET10 重跑 Stage 4（job 1719130）时，Stage 2.5 报 `Existing alignment checkpoint conflicts with inputs or outputs`（282000_354600）。

- **原因**：
  - 输入 CSV 的 hash 与 manifest 一致，只有 `aligner_sha256` 不同。
  - 保存的 `0f042f60…` = 2239e80 版 `point_cloud_aligner.py` 的 CRLF 字节；现在是同一内容的 LF 版（`0d250537…`）。
  - cd135d9 新增 `.gitattributes`（`* text=auto eol=lf`），工作区被转成 LF。代码本身没改。
- **修改**：`tile_alignment_done` 的 `compatible` 集合加入这两个 hash（LF 版也加，之后再改这个文件时，11/4 之后用 LF 跑的样本也不会报错）。
- **验证**：ET10 45 个 tile 的对齐 checkpoint 全部通过，没有报错。本机 env 没装 torch，`test_pipeline_filter_inputs` 无法导入，没跑。
- **ET10 其他 checkpoint**（同样按整个 .py 文件算 hash）：
  - Stage 2.2：`local_contrast.py` 保存的是 LF hash，仍有效。
  - Stage 2.97：`seam_offsets.py` 是 CRLF hash，会重算。
  - Stage 3：`channel_stage3.py`、`provenance.py` 是 CRLF hash，会重算。
  - 这几步只会重算、不会报错，代码没变，结果应相同。上次 2.97+3 约 35 分钟。
- 其他在 11/4 之前跑的样本，下次运行时情况相同。
- **续（同日）**：再次提交（job 1719932）后，对齐已通过，但 Stage 2.9 报 `Tile positions or their inputs changed while global checkpoints exist`。原因相同：`solve_tile_positions.py`、`generate_merging_xml.py` 的 hash 是 CRLF 版。
  - 新增 `local_scripts/migrate_crlf_code_hashes.py`：对每个当前 .py 文件，算出同一内容 CRLF 版的 hash；结果目录里 checkpoint manifest 中出现这个 hash 的，替换成现在的 LF hash。默认只预览，`--apply` 才写入，并在原处保留 `.bak_crlf` 备份。跳过 `run_configs/`、`runtime_config.json`、`old_results*`。
  - ET10 预览：8 个 manifest（tile_positions 2 个、seam_offsets 1 个、`6_3d_global/*_stage3_manifest.json` 4 个、stitch_dedup QC 1 个）。应用后 2.9 / 2.97 / 3 都不用重算，只重跑 Stage 4。
  - 后续阶段只有 Stage 4 的签名包含 stage3 manifest 的 hash，Stage 4 本来就要重跑。

---

## 2026-11-06 Stage 2.9：求解器只哈希影响几何的代码

现象：ET10（job 1720756）手动迁移 checkpoint 后，Stage 2.9 仍报 `Tile positions or their inputs changed while global checkpoints exist`。

- **原因**：`migrate_crlf_code_hashes.py` 只替换了 hash 文本。`_solver_manifest.json` 里 `solve_tile_positions.py` 的 `size` 仍是 CRLF 版（80592，LF 版 79163，正好差 1429 行）。manifest 按 sha256 + size 比较，所以判为输入变了。其余 225 个输入（filtered CSV、offsets JSON）都一致。
  - 已手动把 ET10 的 size 改为 79163（备份 `_solver_manifest.json.bak_size`）。其余 7 个迁移过的 manifest 没有 size 字段问题。
  - `migrate_crlf_code_hashes.py` 现在同时更新 `size`。
- **修改**：
  - `tile_position_pipeline.SOLVE_CODE`：`solve_tile_positions.main()` 用到的函数/常数（含 compare_stitching、point_cloud_aligner、z_linker、io、markers、config.loader 里的 helper）。`SOLVE_CODE_SKIP`：流程不会跑的 XML / `--write-aligned` 写出函数。
  - `provenance.code_sha256`：按源码行算 hash，忽略注释、空行、docstring、换行符，以及 `say`/`print`/`say_scores` 这类日志语句。不用 `ast.dump`，因为它在不同 Python 版本输出不同。
  - `run_tile_position_stage(..., solver_code=)`：给了就用它代替整个 `solve_tile_positions.py` 的文件 hash。
  - 旧 manifest（整文件 hash）：如果当前文件与记录的 hash 一致（LF 或 CRLF 都算），自动升级，不重算。文件真改了的仍会报错。
  - 列表放在 `tile_position_pipeline.py` 而不是求解器里：这样 `solve_tile_positions.py` 一个字节都没变，旧样本才能自动升级。旧样本第一次用新代码跑之前，不要改 `solve_tile_positions.py`，否则还是会报错。
  - 影响：现在 point_cloud_aligner / z_linker / compare_stitching 里求解器用到的函数改了，也会让 Stage 2.9 失效（以前没覆盖）。
- 测试：`test_checkpoint_code_lists`（4，新增 SOLVE_CODE 闭包检查、hash 归一化）、`test_tile_position_pipeline`（6，新增 3 项）通过。

---

## 2026-11-06 可视化：s3 层显示同通道细胞类别去重之后的结果

现象：`6_3d_global/{cid}_3d_tracked.csv` 是 Stage 3 写出的，同通道 neuron/glia 去重（`suppress_cross_class_per_channel`）在 Stage 4 才做，所以 GUI 的 s3 层是去重前的结果，无法检查单通道 3D 去重效果。

- **修改**（`src/utils/saved_gui.py`，只影响 GUI，不影响 checkpoint）：读 `7_colocalization/coloc_decisions.csv` 中 `stage=channel_cross_class_suppression` 的 track_id，从 s3 中去掉这些 track，层名改为 `[s3 after class dedup] {cid}`（球体层也用去重后的行）。被删的 track 不单独建层，免得加载太多。
  - 没有 Stage 4 结果时照旧显示 `[s3 saved summary]`；`3d_tracked.csv` 比 decisions 新时打印提示（需重跑 Stage 4）。
- **验证**：ET10 GFP 10716 / RFP 10439 条删除记录全部能在 s3 中找到，且都是 neuron。`test_saved_pre_gui`（17）通过。

---

## 2026-11-07 Stage 2.25：去掉被 YOLO patch 边切断的重复框（patch_fragment_iomin）

现象：ET10 RFP 同一个 glia 细胞在 s3 里是两条 track（`track3d:2b1d98…` z416–419、`track3d:d878ee…` z414–417），两条都是 glia_RFP，与类别去重无关。

- **原因**（tile 316000_365900，全部 8 个成员框都来自这个 tile）：
  - 细胞位于 4 个 YOLO patch 的交界处（512 px patch，步长 384，交界在 x≈1920、y≈1280）。
  - raw z=423：patch (1536,1152) 看到完整细胞（a445，0.325）；patch (1920,1152) 的框被 patch 左边 x=1920 切断（bec6，0.338）；patch (1536,768) 的框被 patch 下边 y=1280 切断（e7d5，0.293）。
  - tile 内 NMS（`stitchDetection`，只看 IoU>0.4）保留分数最高的 bec6，压掉了完整框 a445；e7d5 与 bec6 IoU 0.37（IoMin 0.81），也留了下来。z=424 情况相同。
  - 同一层有两个框，z-linker 一对一匹配，只能分成两条 track。
  - ET10 全样本：保留框中被 patch 内边切断的 GFP 12.4%、RFP 15.0%；同层同类 IoMin>0.6 的保留框对里，含切断框的 GFP 48%（71k/147k）、RFP 77%（56k/74k），所以 RFP 更明显。
- **修改**（不重跑 YOLO）：
  - 新模块 `src/core/patch_fragments.py`：每层（不分类别）按面积从小到大，框的某条边落在自己 patch 的内边上（±2 px，tile 外边不算），且同层另一个不小于它的保留框盖住它面积的 > `patch_fragment_iomin`，就删掉。patch 位置取自 `1_2d_raw/{tile}_{ch}_candidates.csv`（fused 行用 `winner_detection_id`）。删掉的行写进 rejected CSV，`rejection_reason = patch_fragment`，带 `suppressed_by_detection_id`、`suppression_iomin`。
  - 开关：`detection_params.yolo.patch_fragment_iomin`（null/缺省 = 关）。只对 YOLO；设在 stardist 上报错。
  - Stage 2.25：在共享 filter 之后、SNR 之前执行。Stage 2.75：只在 post_align 时执行（pre_align 的 fused 框已对齐、坐标不在 patch 边上，每个曝光已在 2.25 处理过）。
  - 开启时 filter manifest 的 settings 多一项 `patch_fragment`（参数、本模块 hash、candidates CSV 的 sha256）；关闭时签名不变。**没有改 `detection_filter.py`**，所以其他样本的 2.25 checkpoint 不受影响。
  - `resume.raw_detection_settings` 忽略这个键（不算 raw 检测设置的变化）。
  - `refilter_align` / `refilter_detections` 遇到开启的通道直接报错（与 contrast_filter 相同）。
  - `run_inference.py` 改了，按惯例其他样本下次运行会重跑 Stage 4。
- **ET10 验证**（tile 316000_365900，阈值 0.5，用 runtime_config 的 filter 参数）：
  - 例子里两个切断框（e7d5 z423、1063 z424，覆盖率 0.81 / 0.59）都被删。
  - 删框 GFP 671（0.62%）、RFP 1021（1.08%）；按 Stage 2.5 的 z_linker 参数单 tile 连 3D，track 数 GFP −0.8%、RFP −1.4%。
  - 每通道随机抽 16 个看图：绝大多数是同一细胞被 patch 边切下的一块；覆盖率 0.55–0.57 的个别例子像是相邻两个细胞。阈值 0.6 会漏掉本例的 z=424。
- **对已有样本生效**：在 config 的 `detection_params.yolo` 加 `"patch_fragment_iomin": 0.5`，把 `3_2d_aligned` 及之后的结果（5_2d_global、6_3d_global、7_colocalization）挪走，从 Stage 2.25 重跑（2.25 已有对齐结果时会报错，这是原有的保护）。YOLO 不重跑。
- 测试：新增 `tests/test_patch_fragments.py`（8 项）；`test_checkpoint_code_lists`（4）通过。`test_pipeline_filter_inputs`、`test_automatic_resume` 本机缺 torch 无法导入（已有问题）。

## 2026-11-07 同通道 neuron/glia 去重：加包含判据（cross_class_contain_thresh）

现象：s3 after class dedup 仍有 GFP neuron（z396，31×31）完全落在 GFP glia（z397，64×62）框内（seam tile 316000_377200，`track3d:73937e…` / `track3d:63a612…`）。

- **原因**：`suppress_cross_class_overlap` 只看 3D IoU > 0.5。小框套在大框里，IoU 只有 ≈0.24（IoMin = 1.0）。两框在相邻层，2D containment NMS 也看不到。
- **修改**：`suppress_cross_class_overlap` / `suppress_cross_class_per_channel` 新参数 `contain_thresh`（None = 关）：neuron 体积 ≤ glia 且 IoMin ≥ 阈值时也删 neuron；反过来（大 neuron 包小 glia）不删。decision 行 `reason = inside_glia`，metric 为 IoMin。`run_inference.py` 读 `z_linker.soma.cross_class_contain_thresh`（默认 0.8），Phase A 前后两次去重都用。
- `run_inference.py`、`stitcher.py` 改了：下次运行所有样本重跑 Stage 4（colocalization）。
- 测试：`test_cross_class_per_channel` 新增 2 项（本例被删；大 neuron 包小 glia 保留），共 5 项通过；`test_checkpoint_code_lists`、`test_detection_filter` 通过。

## 2026-11-08 TF 共定位改为逐 soma 通道判定（soma_channel_rule = all）

现象：GFP+RFP 细胞（seam 338600_388500+338600_399800，z419–423，`coloc:be92311d…`）被标成 Sox9+，但细胞本身不表达 Sox9。

- **原因**：Phase A 合并后 soma 框取 GFP/RFP 的并集。这里 RFP 框（score 0.32，74×70 px）是 GFP 框（38×38 px）的两倍，并集就等于 RFP 框。旁边的 Sox9 核（多半是贴着神经元的星形胶质细胞）在 RFP 框内，中心距 17.3 px，上限 25.5 px，能通过。用 GFP 框判的话两条都不过：Sox9 下沿超出 GFP 框 4 px，中心距 20.5 px，上限 13.5 px。
- **修改**：`annotate_soma_with_tf_containment` 新参数 `soma_channels`。给出时，合并细胞里每个 soma 通道在 `source_tracks` 中的原始框各算一个 gate（包含 + 中心距），TF 要通过全部 gate 才归给这个细胞。多个细胞都接受同一个 TF 时，最远 gate 距离最小的胜出。没有 soma 通道 track 的细胞回退用自身框。单通道细胞的结果和以前一样。
- 配置：`z_linker.tf.soma_channel_rule`，可选 `all`（默认）或 `union`（旧行为），写其他值报错。
- `match_evidence.csv` 新增 `tf_gates` 列（JSON，列出每个 soma 通道的中心距和比值）。`center_distance*` 两列改记比值最大的那个 gate。
- `run_inference.py`、`stitcher.py`、`coloc_sources.py` 改了：所有样本下次运行会重跑 Stage 4。Sox9+/Olig2+ 的计数预计会下降（只影响多通道 soma）。
- 测试：`test_coloc_sources` 新增 2 项（本例在 all 下不再是 Sox9+、union 下仍是；Olig2 在中心仍匹配；无 soma 通道 source 时回退），共 9 项通过；`test_checkpoint_code_lists` 通过。其余测试文件逐个跑通过；`test_qc_alignment`、`test_tile_2d_sources` 在本机原生崩溃，原代码上也一样（与本次无关）。

## 2026-11-08 TF 共定位 all 规则：z 改回按合并 soma 判定

现象：ET10 tile 327300_388500 的 GFP+RFP glia（`coloc:3844a55f…`），Sox9 核（`track3d:e4981edb…`，z425–427）在两个框的中心，却是 `unmatched_tf / no_accepted_soma`。

- **原因**：上一条改动把 z 包含也改成了逐通道判断。GFP track 只有一层（z427，member_count=1），ET10 的 `containment_z_pad = 1`，要求 Sox9 zmin ≥ 426，实际 425，不过；RFP（z425–428）能过。旧 union 规则下能匹配，这是 all 规则带来的假阴性。
- **修改**（`annotate_soma_with_tf_containment`）：xy 包含和中心距仍逐个 soma 通道判；z 包含改用该 soma 自身（合并后）的 z_min/z_max。上一条的 Sox9 误配例子是被 GFP 框的 xy 条件拒掉的，不受影响。
- `stitcher.py` 改了：所有样本下次运行重跑 Stage 4。
- 测试：`test_coloc_sources` 新增本例回归测试，共 10 项通过；`test_checkpoint_code_lists`（4）、`test_cross_class_per_channel`（5）通过。

## 2026-11-09 tsc11 在 gpu18 上 StarDist 加载失败（节点问题，未改代码）

现象：tsc11（job 1773560，gpu18）阶段 2 一开始就退出，StarDist 报 `CUDA_ERROR_DEVICE_UNAVAILABLE`（device ordinal 1）。TSC10（gpu43）同样的配置正常。

- 最初怀疑 GPU 是 Exclusive_Process 模式（双卡时每个 worker 的 TF 放在另一块卡上，每块卡有两个进程）。在 gpu18 上用 `srun` 查到 compute_mode 为 `Default`，排除了这个原因，为它加的 worker 改动已撤回。
- 目前判断是 gpu18 的某块卡本身不可用（被占用或状态异常）。处理方式：提交时加 `--exclude=gpu18`。
- `inference_tsc11.slurm` 的 GPU 检查改为同时输出 compute_mode，以后能直接从 .out 里看到。

## 2026-11-09 TF 共定位 z 条件改为“TF 至少一半的层落在 soma z 范围内”（未提交，规则待定，见文末“交接”）

现象：tile 327300_388500 的 GFP neuron（`track3d:9177120871d0…`，只在 z409 检出，score 0.247）和 Olig2 核（`track3d:ec7612ce5c0b…`，z408–410，xy 在 GFP 框中心）没有共定位。同一位置的 Sox9（`track3d:245491e904…`，z409）按 GFP 框算三项都能过，但也没挂上，原因未查（可能 GFP 在 Stage 4 被跨类 dedup 删了、和 RFP/GFP2 合并后过不了 all 规则，或者 Sox9 被别的 soma 抢走）。要查需要该样本（含 Olig2 通道）的 `7_colocalization/coloc_decisions.csv`、`coloc_source_3d.csv`，按这三个 track_id grep。

- **原因**：旧规则要求 TF 的整个 z 范围都在 soma z 范围内，`containment_z_pad` 只能加在一侧。单层 soma 包不住上下各多一层的核。
- **修改**（`stitcher.py`）：新增 `_tf_z_contained(t_zmin, t_zmax, s_zmin, s_zmax)`，规则是 `soma.z_min − 0.5 ≤ (tf.z_min + tf.z_max)/2 ≤ soma.z_max + 0.5`，即 TF 至少一半的层与 soma 重叠，不加 padding。`annotate_soma_with_tf_containment` 调用它，去掉了 `z_pad` 参数；xy 包含、中心距、all 规则都不变。
  - 中间版本用过 `cz`（`int(median)`），偶数层 TF 会被向下取整（[408,409] → 408），导致本来能过的被拒，所以改用真实中点。
  - TF 的 `max_cell_z_span=3`，所以新规则下 TF 最多在 soma 上下各多出 1 层。
- `run_inference.py`：Stage 4 不再读取 `z_linker.tf.containment_z_pad`；`coloc_match_evidence.csv` 中 tf_containment 行的 `z_pad` 列为空。
- **没改** `point_cloud_aligner._containment_score`（Stage 2.9 / 通道对齐用，在 `SOLVE_CODE` 里），它仍用 `containment_z_pad`。改了会让已有 tile 位置失效、流程停下。所以配置里的 `containment_z_pad` 不能删。
- `stitcher.py`、`run_inference.py` 改了：所有样本下次运行会重跑 Stage 4。
- 测试：`test_coloc_sources` 新增本例回归测试（[408,410]、[408,409]、[409,410] 匹配；[409,411]、[407,409]、[407,408] 不匹配），共 11 项通过；`test_checkpoint_code_lists`、`test_cross_class_per_channel` 通过。

**新旧规则对比**：脚本 `src/utils/compare_tf_z_rule.py`（读已有的 `6_3d_global/*_3d_tracked.pkl`，按 `run_inference.py` 复现 Phase A，然后对每个 TF 通道分别用新、旧 z 规则跑 Phase B；旧规则通过替换 `stitcher._tf_z_contained` 实现）。在其他电脑上重跑：

```
cd Y:\Fengyi\brain_detector
set PYTHONPATH=.
<brain_detector env python> src/utils/compare_tf_z_rule.py --out W:/Fengyi/tf_z_rule_compare ^
    Y:/Fengyi/TSC_brain/sample10/detection_results_20261107 ^
    Y:/Fengyi/TSC_brain/sample8/detection_results_20261107
```

输出在 `W:\Fengyi\tf_z_rule_compare\`：`summary.csv`、`changed_tfs.csv`（每个变化的 TF 的 track_id、坐标、新旧 soma，可在 GUI 里定位）、`1_summary.png`、`2_z_patterns.png`、`3/4/5_examples_*.png`（XY + XZ 示意图）。需要 Sox9 pkl（各约 2 GB），本机内存够用。

**对比结果（sample10 / sample8，Sox9，旧规则 `containment_z_pad=1`）**——新规则整体**更严**，不是更宽：

| | sample10 | sample8 |
|---|---|---|
| 旧规则匹配的 TF | 94,958 | 98,746 |
| 新规则匹配的 TF | 77,901 | 81,538 |
| 只有新规则匹配（gained） | 2,323 | 1,982 |
| 只有旧规则匹配（lost） | 19,380 | 19,190 |
| 匹配到不同 soma | 911 | 841 |
| Sox9 阳性 soma：旧 → 新 | 74,340 → 64,423（−13%） | 81,739 → 70,821（−13%） |

- gained 共 4,305 个，全部是“TF 有 3 层，soma 只在中间那 1 层”（即 Olig2 这种情况）。
- lost 共 38,570 个，全部是“TF 只有 1 层，与 soma 的 z 范围**完全不重叠**，紧挨在 soma 上一层（20,919）或下一层（17,651）”。旧规则的单侧 z_pad=1 放过了它们。这些 soma 的层数：1 层 21,831，2 层 9,935，3 层 4,076，4 层及以上 2,728；neuron 24,501，glia 14,069。
- switched 中 1,545 个是“1 层 TF 与 soma 重叠，但被分给了别的 soma”（因为 lost 的那些不再参与竞争）。

**交接（换电脑后从这里继续）**

1. **待定（需要人决定）**：只有 1 层、在 soma 相邻层（相距 8 µm）、xy 和中心距都满足的核，算不算共定位？
   - 算 → 规则改成“新规则 OR 旧规则”：`_tf_z_contained` 里返回新条件 `|` 旧的单侧 pad 条件（需要重新传入 `containment_z_pad`，即把 `run_inference.py` 和函数签名里的 z_pad 加回去）。效果：旧的全部保留，再加 4,305 个。
   - 不算 → 保持现在的代码。
   - 先看示例图再决定（第 2 步）。
2. **出图还没完成**。`W:\Fengyi\tf_z_rule_compare\` 里只有第一次运行的 `summary.csv`、`changed_tfs.csv`，没有 `soma_gates` 列，所以示例图要完整重跑一次（约 8 分钟）：
   ```
   set MKL_THREADING_LAYER=SEQUENTIAL
   set PYTHONPATH=.
   <env python> src/utils/compare_tf_z_rule.py --out W:/Fengyi/tf_z_rule_compare ^
       Y:/Fengyi/TSC_brain/sample10/detection_results_20261107 ^
       Y:/Fengyi/TSC_brain/sample8/detection_results_20261107
   ```
   之后只改图的话用 `--out W:/Fengyi/tf_z_rule_compare --replot`，不用重新读 pkl。脚本在 F: 盘那台电脑上的首次运行，画图时崩了（`0xc06d007f`，numpy 默认 MKL 线程层下矩阵乘法崩溃）；设 `MKL_THREADING_LAYER=SEQUENTIAL` 就好了。别的电脑不一定需要。`compare_tf_z_rule.py` 出图后还没人看过效果，需要检查标签是否重叠。
3. 决定后：更新 `test_coloc_sources` 中 `test_tf_z_uses_representative_layer_inside_soma_range` 的期望值，跑 `test_coloc_sources`、`test_checkpoint_code_lists`，再提交。目前的改动都**还没提交**（`stitcher.py`、`run_inference.py`、`tests/test_coloc_sources.py`、`src/utils/compare_tf_z_rule.py`、本日志）。工作区里原本就有的 `saved_gui.py` / `test_saved_pre_gui.py` 改动和本次无关。
4. 原始问题里的 Sox9（`track3d:245491e904…`）为什么没挂上 GFP，还没查（见本条开头）。

## 2026-11-09（续）TF z 规则对比：ET10 4 个 tile；GUI 可显示变化的 TF（未提交，规则仍待定）

换到新电脑（conda env 在 `C:\ProgramData\miniconda3\envs\brain_detector`；旧电脑的 `W:\Fengyi` = 这台的 `X:\Fengyi`）。

- **TSC 重跑出图**：sample10 / sample8 数字与上一条完全一致，图在 `X:\Fengyi\tf_z_rule_compare\`。
- **ET10**（只有 ET10 用的是当前 pipeline）：tile 327300_388500、327300_377200、316000_388500、316000_377200，结果在 `X:\Fengyi\tf_z_rule_compare_ET10\`。

  | | Sox9 | Olig2 |
  |---|---|---|
  | 旧 / 新规则匹配的 TF | 58,040 / 54,275 | 36,280 / 34,152 |
  | gained / lost / switched | 2,070 / 5,835 / 494 | 1,802 / 3,930 / 286 |
  | TF 阳性 soma 旧 → 新 | 47,406 → 46,183（−2.6%） | 31,465 → 30,635（−2.6%） |
  | matched soma 为 neuron 的比例：lost / 新旧相同 / gained | 59% / 51% / 52% | 63% / 55% / 72% |

  - 变化模式与 TSC 相同（gained = 3 层 TF 包住 1 层 soma；lost = 1 层 TF 紧贴 soma 上/下一层）。影响比 TSC（−13%）小得多。
  - 原始 Olig2 例子（`track3d:ec7612ce5c0b…`）在新规则下匹配到 GFP+RFP soma（gained）。
  - neuron 比例：新旧都匹配的 Sox9 也有一半是 neuron，所以这个比例不能用来判断 lost 是否错配。
  - Sox9 `track3d:245491e904…`（交接第 4 条）：按现在的代码，新旧规则下都挂在 GFP `9177120871d0…` + RFP `e61e163353ab…` 这个细胞上（旧 GFP+RFP+Sox9 → 新 GFP+Olig2+RFP+Sox9）。之前看到没挂上，应是 ET10 已保存的 s4 用较早的代码跑的，不需要再查。
- **`compare_tf_z_rule.py`**：
  - 新增 `--tiles a,b,c`（逗号分隔；保留任一成员框在这些 tile 的 track）；
  - TF 通道逐个载入；
  - `summary.csv` 加 `{kind}_neuron`；
  - `changed_tfs.csv` 加 `tf_best_detection_id`、`tf_best_box_z`，`soma_gates` 每项末尾加 best_detection_id、best_box_z；
  - 修了 `1_summary.png` 标签重叠；
  - 新增 `somas.csv`：每个 Phase A soma 一行，新旧规则下的 marker 组合（`old_combo`/`new_combo`）、挂上的 TF track（`old_tfs`/`new_tfs`）、各 soma 通道 track 的框和代表框（`gates`，GUI 定位用）。
- **GUI 新模式 `zrule_compare`**（`saved_gui.run_saved_zrule_compare`；launcher Mode 下拉框，“Z-rule comparison” 填对比输出文件夹 → config `tf_rule_compare`；命令行 `--mode zrule_compare`）：
  - 只显示原始图像和对比结果，tile-local。每个 marker 组合一层（与 s4 相同，neuron/glia 不分），层名带删除/保留/新增个数。
  - 每个 soma 画成各 soma 通道 track 的 3D 并集框，画在其 z 范围的每一层：白 = 新旧组合相同（保留），黄 = 只有旧规则是这个组合（删除），蓝 = 只有新规则是（新增）。组合变化的 soma 同时出现在两层。
  - 各 soma 通道 track 按 s3 层的方式（代表检测框 + seam offset）放回本 tile 坐标系。
  - 不含 TF 的组合（GFP、RFP、GFP+RFP）默认隐藏，因为它们的变化只是 TF 层的镜像。
  - 点框时状态栏显示状态、新旧组合、soma z 范围，终端打印 TF track id。
  - ET10 tile 327300_388500（GUI 编号 [023]）离线检查：50,238 个 soma，GFP/RFP track 全部找到源检测，没有退回 XML 放置。
  - 这次加在 prealign 模式里的 6 层 overlay 已删除，被这个模式取代。
  - `test_saved_pre_gui`（17）、`test_saved_s3_tile_frame`（6）通过。不影响任何 checkpoint。
- **下一步**：在 GUI 里看 lost 的例子，决定“新规则”还是“新 OR 旧”（见上一条交接第 1、3 条）。

## 2026-11-11 TF z 规则对比：ET10 old vs nucleus（未提交，规则仍待定）

- **补记（2026-11-09 17:15 的改动，当时没写日志）**：`compare_tf_z_rule.py` 新增 `--base`/`--cand`（`old` / `new` / `nucleus`）和 `--nucleus-iomin`（默认 0.7）。`nucleus` = new 规则 + 同层 2D IoMin ≥ 0.7、一一对应的不同通道 TF track 合并为同一个核；合并后任一 track 按 new 规则过 z 即整个核通过（例：Olig2 只有 1 层在 soma 旁，但配对的 Sox9 伸进 soma）。未合并的 TF 与 new 完全相同；xy 包含、中心距仍各用自己的框。只在对比脚本里，pipeline（`stitcher._tf_z_contained`）没改。
  - 当时已跑 ET10 new vs nucleus：`X:\Fengyi\tf_z_rule_compare_ET10_nucleus\`。
- **本次**：ET10 同 4 个 tile，`--base old --cand nucleus`，结果在 `X:\Fengyi\tf_z_rule_compare_ET10_old_vs_nucleus\`（约 8 分钟）。"old" 是在同一份 Stage 3 上用当前 Stage 4 代码按旧 z 规则重算，不是读已保存的 s4（那是更早的代码跑的）。

  | | Sox9 | Olig2 |
  |---|---|---|
  | old / nucleus 匹配的 TF | 58,040 / 55,395 | 36,280 / 35,535 |
  | gained / lost / switched | 2,840 / 5,485 / 552 | 2,821 / 3,566 / 301 |
  | TF 阳性 soma old → nucleus | 47,406 → 46,991（−0.9%） | 31,465 → 31,624（+0.5%） |

  - Sox9-Olig2 配对的核 368,180 个。
  - 与 old vs new（−2.6% / −2.6%）相比，nucleus 补回了一部分：Sox9 lost 5,835 → 5,485，Olig2 lost 3,930 → 3,566，gained 都增加约 1,000。
- **下一步**：GUI `zrule_compare` 模式填这个文件夹，看 lost（黄）的例子，决定用 new / nucleus / new OR old。

## 2026-11-11（续）残片 soma 的来源；测试脚本 relink_soma_test.py（未提交，pipeline 未改）

- **现象**（GUI 例子，ET10 全局约 x 5176, y 6970, z 426–431）：一个 GFP+RFP 细胞在 Phase A 里被拆成 3 个 soma：本体（GFP d36db2b1 + RFP ef1aa108）、RFP 261e19e9（z428 单层 glia，score 0.16）、RFP a4465ac5（z431 单层 neuron，score 0.20）。Sox9 在不同规则下在这些碎片之间挪动（nucleus 下挪到 261e，old 下 a4465 多算一个 RFP+Sox9）。
- **原因链**：
  1. Stage 3 z 连接只连同类（`z_linker._match_sparse` 的 base 检查），z428 的 glia 框连不上 z429–431 的 neuron track；
  2. z431 两个来自不同 tile（316000_388500 / 327300_377200）的同一细胞框 IoU 0.33 / IoMin 0.49，没过跨 tile 去重阈值（0.4 / 0.6）；
  3. Phase A `match_soma_3d_iou` 一对一，GFP 只配 ef1a，另两条 RFP 成了独立 soma（与 GFP 的 3D IoMin 0.49 / 0.94）；
  4. `suppress_cross_class_overlap` 只删 neuron 且要求 neuron 不大于 glia；同类之间无去重。
- **规模**（old vs nucleus 的 somas.csv）：残片 soma（与更大 soma 的 3D IoMin ≥ 0.5 / 0.8）12,897 / 6,914 个（8.9% / 4.8%），多为单通道单层。switched TF 约 60% 涉及残片，lost 只约 7%。
- **测试脚本 `src/utils/relink_soma_test.py`**：复用 compare_tf_z_rule.py（不改它），只对 soma 通道从 `5_2d_global` 重做 Stage 3 后半段：
  - 修复 B：同层不同 tile 的框 IoU > 0.25 或 IoMin > 0.45 视为同一细胞，留高分框（glia 优先改标）；
  - 修复 A：z 连接不分 neuron/glia；含 glia 成员的 track 记为 glia（用户决定）。
  - 额外输出 `relink_changes.csv`、`relink_dropped.csv`、`relink_settings.json`，somas.csv 多 `relinked` 列；GUI `zrule_layer_rows` 有此列时点框打印出来。
- **ET10 4 tile，old vs nucleus**（`X:\Fengyi\tf_z_rule_compare_ET10_relink_old_vs_nucleus\`，约 8 分钟）：
  - 删框 GFP 4,735 / RFP 4,509；混合类别 track GFP 7,308 / RFP 8,503；与 Stage 3 不同的 soma track 13,871。
  - Phase A soma 144,972 → 139,679；neuron 105,573 → 100,063，glia 39,399 → 39,616。
  - 残片（IoMin ≥ 0.5 / 0.8）12,897 / 6,914 → 9,994 / 5,188。
  - Sox9：lost / gained / switched 5,485 / 2,840 / 552 → 5,316 / 2,480 / 404；Olig2：3,566 / 2,821 / 301 → 3,412 / 2,664 / 217。
  - 例子细胞合成一个 soma（RFP track 493fda0a，z428–431），两种规则下都是 GFP+Olig2+RFP+Sox9，类别 glia（因含 261e 的 glia 层）。
- **下一步**：GUI 里对照两个文件夹看修复效果，抽查 `relink_dropped.csv` 有无误删相邻细胞；剩余残片是否还需要 Phase A 吸收。

## 2026-11-12 四通道细胞被 nucleus 规则删掉 Sox9：z 跨度上限造成的残片抢走 TF（未改代码）

- **现象**（GUI zrule_compare，relink old vs nucleus，tile 坐标 z431, x1759, y923；全局约 x 6970, y 7867）：GFP eca5a303 + RFP 61c848ee（z424–429）旧规则 GFP+Olig2+RFP+Sox9，nucleus 规则变 GFP+Olig2+RFP。Sox9 6777d010（z428–430）和 Olig2 11e0c4ac（z428–430）都 switched 到 GFP 07825977。
- **原因**：
  1. GFP 07825977 只有 z429 一层，xy 框与 eca5 几乎相同（IoU 约 0.95），是同一个细胞。Stage 3 `z_linker` 在 track 达到 `max_cell_z_span=5`（eca5 = z424–428）后关闭，z429 只能另起一条 track（span-cap 残片，与 11-11 那条的类别/跨 tile 残片来源不同，`relink_soma_test.py` 不处理）。
  2. Phase A 一对一配对：RFP 61c8 已配 eca5，07825977 成了单独的 soma。
  3. 旧规则下 3 层 TF 包不进 1 层的残片，只有本体能接；new/nucleus 规则下两者都能接，`annotate_soma_with_tf_containment` 选“最远 gate 距离最小”的：残片 1 个 gate 4.45 px，本体 GFP gate 5.16 px / RFP 3.96 px，残片以 0.7 px 之差胜出。
- **规模**（ET10 4 tile，`somas.csv`；残片判据 = 同通道 track 紧接在一个满 5 层 track 之后/之前，xy IoU ≥ 0.5）：relink 文件夹 5,337 个 span-cap 残片 soma（3.8%；1 层 2,480，2 层 1,179，≥3 层 1,678），其中持有 TF 的 old 1,191 / nucleus 1,135。old→nucleus 中 TF 从本体换到残片 86 个（Sox9 54、Olig2 32），反方向 87 个；对 z 规则对比的净影响很小，但两种规则下残片都在多计细胞、抢本体的 TF。
- 统计脚本只在临时目录，未加入仓库。
- **待定**：是否在 Phase A 把残片并回本体（同通道、z 相邻或落在本体 z 范围内、xy IoU 高），或者 TF 竞争时优先 z 重叠层数多的 soma，或者放宽 Stage 3 的 span 上限（会重跑 Stage 3）。
- **soma `max_cell_z_span` 5 是否太小**（ET10 4 tile，old_vs_nucleus 的 somas.csv，未 relink）：
  - GFP track 层数 1/2/3/4/5 = 29,913 / 26,660 / 19,934 / 10,911 / 10,091；RFP 30,738 / 27,140 / 16,367 / 8,988 / 8,319。5 层明显堆积（按前面的下降趋势应约 6k），说明更长的细胞被截断了。
  - 5 层 track 的下一层有同通道、xy IoU ≥ 0.5 的框：GFP 30%、RFP 27%（IoU 中位数 0.63 / 0.61）。
  - 这些后续段的层数：GFP 1 层 1,901、2 层 670、3 层 219、4 层 91、5 层 110；RFP 1,352 / 494 / 208 / 96 / 117。后续段偏短（1 层占约 64%），而独立细胞的层数分布是 31% / 27% / 20% / …，所以大多是同一细胞的尾巴。
  - cap 6 可接回约 64%，cap 7 约 86%。
  - 注意：Stage 2.5 对齐（`point_cloud_aligner.resolve_align_settings` 的 `z_link`）也读 `z_linker.soma`，并保存在 `_align_settings.json`。直接改配置会让 `check_align_settings` 报错，需要删对齐目录从 2.5 重跑。要只改 Stage 3，需要给 Stage 3 单独的配置键。
- **6 还是 7**（ET10 全脑 GFP/RFP 的 `6_3d_global/*_3d_tracked.pkl`，逐层 member 框；脚本在临时目录）：把每个 5 层 track 和它下一层 IoU ≥ 0.35、同类的 track（linker 没有 cap 时会连上的那个）接起来，看接缝之后面积或平均亮度是否再升高 ≥ 1/0.85（第二个细胞的迹象）。对照 = 普通 5 层 track 内部在第 3 层后切开、后面 2 层的同一检验。

  | GFP（RFP） | 后续 1 层 | 2 层 | 3 层 | 4 层 | 5 层 |
  |---|---|---|---|---|---|
  | 个数 | 9,292（6,854） | 2,814（2,480） | 1,115（1,162） | 579（654） | 721（720） |
  | 接缝后面积再升高 | — | 8.0%（20%） | 10.6%（28%） | 13.6%（37%） | 21%（43%） |
  | 接缝后亮度再升高 | — | 3.4%（2.5%） | 4.8%（9.9%） | 13%（28%） | 29%（47%） |
  | 后续段最亮 / 本体最亮 | 0.81（0.71） | 0.87（0.84） | 0.95（0.95） | 0.99（0.99） | 1.01（1.01） |
  | 后续段最高 score（本体约 0.74 / 0.43） | 0.41（0.28） | 0.65（0.36） | 0.69（0.39） | 0.70（0.40） | 0.73（0.41） |

  - 对照（同一细胞内部）：面积再升高 6.8%（17%），亮度再升高 2.4%（4.0%）。后续 2 层与对照几乎一样，而且更暗，是同一细胞的尾巴；4–5 层的亮度与本体相同、再升高比例高很多，像是叠在下面的另一个细胞；3 层介于两者之间。
  - cap 6 / 7 / 8 后仍被切断的链：GFP 36% / 17% / 9%，RFP 42% / 21% / 12%。
- **8 是否更好**：8 比 7 多接回的只有“后续 3 层”的链（GFP 1,115 / RFP 1,162）。这组后续段最亮层是本体最亮层的 0.95，几乎一样亮；按“比本体暗 10% 以上、且之后不再变亮”判为尾巴的只有 34%（GFP）/ 34%（RFP），2 层后续段是 57% / 64%，1 层是 82% / 88%。所以 cap 8 多连上的大多是另一个细胞；同时后续 4–5 层（多半是另一个细胞，每通道约 1,300）会被多并 1 层。8 层按 8 µm 层距跨约 56–64 µm，也远大于胞体。结论：7 合适，不用 8。

## 2026-11-12（续）relink 测试：soma z 跨度上限 6 / 7（未提交，pipeline 未改）

- **`relink_soma_test.py`**：
  - 新增 `--soma-span N`，覆盖配置里的 `z_linker.soma.max_cell_z_span`，只用于 relink 的 soma track；值写进 `relink_settings.json`。
  - 新输出 `soma_tracks_{GFP,RFP}.csv`：每个 soma 通道的 relink track 经过同通道 neuron/glia 去重之后的结果（Phase A 跨通道合并的输入），一行一个 track。列包括 3D 框、代表框 `rep_*`、`layers`（每层的框）、`relinked`、`stage3_track_ids`（与 Stage 3 不同的 track 记原来的 track id）。在 `suppress_cross_class_per_channel` 返回时立刻取行，因为之后 Phase A 合并会原地改这些 dict。
- **GUI `zrule_compare` 模式**（`saved_gui.zrule_soma_tracks`、`soma_tracks_in_tile`）：文件夹里有 `soma_tracks_{ch}.csv` 时，每个 soma 通道加一层 `[relink s3 after class dedup] {ch}`，默认隐藏。每个 track 在它连上的每一层画该层自己的框（neuron 蓝、glia 橙），用 `_place_gates` 按代表检测放回本 tile 坐标系，和 z-rule 层一样。点框会打印 track_id、`track_z`、`n_layers`、`relinked`、`stage3_track_ids`。
- **ET10 4 tile，old vs nucleus**，结果在 `X:\Fengyi\tf_z_rule_compare_ET10_relink_span{6,7}_old_vs_nucleus\`（两个并行跑，约 8 分钟）。对照组是 span 5 relink（`..._relink_old_vs_nucleus`）：

  | | span 5 | span 6 | span 7 |
  |---|---|---|---|
  | Phase A soma | 139,679 | 136,300 | 135,096 |
  | span-cap 残片 soma（持有 TF：old / nucleus） | 5,337（1,191 / 1,135） | 1,882（475 / 444） | 837（232 / 211） |
  | 残片 IoMin ≥ 0.5 / ≥ 0.8 | 9,994 / 5,188 | 8,300 / 3,670 | 7,777 / 3,233 |
  | Sox9 lost / gained / switched | 5,316 / 2,480 / 404 | 5,222 / 2,437 / 316 | 5,208 / 2,410 / 303 |
  | Olig2 lost / gained / switched | 3,412 / 2,664 / 217 | 3,384 / 2,630 / 171 | 3,359 / 2,611 / 161 |
  | GFP track 6 / 7 层（去重后） | — | 4,790 / — | 2,859 / 1,943 |
  | RFP track 6 / 7 层（去重后） | — | 3,752 / — | 2,203 / 1,547 |

  - 本次的例子细胞：span 6、7 都合成一个 soma（GFP 85de1ea3 + RFP e4610c8d，z424–430，neuron），old 和 nucleus 下都是 GFP+Olig2+RFP+Sox9。
  - span-cap 残片判据和上一条一样：同通道 track 紧接在一个满 cap 层的 track 之后或之前，xy IoU ≥ 0.5。
- 测试：`test_saved_pre_gui` 新增 1 项（共 18 项通过），`test_saved_s3_tile_frame`（6）通过。
- **下一步**：在 GUI 里看 span 7 的 6–7 层 track 有没有把上下两个细胞连成一个（`[relink s3 after class dedup]` 层，点框看 `n_layers`）；没问题就给 Stage 3 加单独的 span 配置键（对齐继续用 5）。

## 2026-11-13 Olig2 假阳性：没有 SNR 过滤；挑阈值用的图（未改 pipeline）

- **现象**（GUI zrule_compare，relink 文件夹，ET10 tile 327300_388500）：两个 GFP+RFP 细胞被判成 GFP+Olig2+RFP+Sox9，实际没有 Olig2。old 和 nucleus 规则下都分到了这个 Olig2，所以与 z 规则无关。
  - cell #2（全局约 x 6170, y 7008, z479）：Olig2 track cc8ae5de 只有 1 层，只在相邻 tile 327300_388500 的重叠区检测到，score 0.40，mean 800，SNR 2.5（该层最低的 5%）。检测到 Sox9 核的 tile 316000_388500 在这里没有 Olig2 框；它旁边的真 Olig2 核 mean 约 2200，SNR 5.7–7.5。
  - cell #3（全局约 x 5866, y 7660, z484–485）：Olig2 track 5a982b03 与 Sox9 track 94b2e50d 的框几乎重合（同一 tile），SNR 3.9 / 5.6，可能是 Sox9 核在 Olig2 通道里的弱信号；Sox9 的 Stage 3 detection id 与现在的 Sox9 raw CSV 对不上，没能核实。
- **原因**：`contrast_filter` 只配了 Sox9，Olig2 没有 SNR 过滤。
- **SNR 阈值影响**（6 层，Stage 3 的 Olig2 框 4,015 个）：SNR < 4 / 5 / 6 的占 12% / 22% / 34%。snr_min 5 能删掉 cell #2 的 Olig2，删不掉 cell #3 的（z485 那层 5.6，track 只要 1 层就保留）。
- **挑阈值用的图**：`Y:\Fengyi\EGFR_brain\ET10\olig2_contrast_test_20261113\`，ET10 5 个 tile × 3 层（含两个病例层），每层 SNR ≥ 3…15 的图和标签。SNR 直接调用 `local_contrast.slice_ring_stats`；现有过滤已删的框画灰色、不计数。脚本 `scripts/build_olig2_slices.py`，约 2.5 分钟。
- **下一步**：用户看图填 `threshold_choices.csv`，再定 Olig2 的 `snr_min`（Olig2 参与对齐的话，要从 Stage 2.25 重跑）。

## 2026-11-13（续）Olig2 SNR ≥ 4 加入 relink 测试（未改 pipeline）

- 用户按挑阈值的图选定 Olig2 `snr_min 4`（15 层里大多选 4，范围 3–5.5）。核对过：加上绝对亮度（mean − 背景、mean / 背景）没有提升，见上一条。
- **SNR 表**：ET10 区域内的 15 个 tile（4 个测试 tile + 邻 tile）用 Stage 2.2 的 `tile_contrast` 计算，结果在 `X:\Fengyi\olig2_contrast_ET10\`（脚本 `compute_olig2_snr.py` 把 /rsstu 路径映射到盘符；15 个并行约 15 分钟）。没写 manifest，以后正式跑 Stage 2.2 时会重算。
- **`relink_soma_test.py` 新增 `--tf-snr CH=V`、`--snr-dir`**：从区域内的 5_2d_global 删掉 SNR < V 的 TF 框（无 SNR 的保留，与 Stage 2.25 一致），再用配置里的 `z_linker.tf` 参数重新 z 连接这个 TF 通道（Stage 3 对 TF 的做法）。没有模拟的：pipeline 里过滤在对齐和拼接去重之前，所以拼接时输给被删框的重复框会回来（本次 9,717 个，只计数）。`relink_settings.json` 记 `tf_snr_min`、`tf_snr_stats`。
- **ET10 4 tile，old vs nucleus**：`X:\Fengyi\tf_z_rule_compare_ET10_relink_span{6,7}_olig2snr4_old_vs_nucleus\`
  - 区域 Olig2 框删掉 193,808 / 2,356,673（8.2%）；Olig2 track 850,768 → 775,732。自检：没丢框的 Stage 3 track 有 701,038 / 715,524 原样回来，其余约 2% 因为相邻 track 丢层后 3 层上限的位置变了，重新连接。

  | span 7（span 6 相近） | 无 Olig2 SNR | Olig2 SNR ≥ 4 |
  |---|---|---|
  | Olig2 阳性 soma old / nucleus | 31,241 / 31,423 | 29,912 / 30,021 |
  | GFP+Olig2+RFP+Sox9（nucleus） | 7,928 | 7,604 |
  | GFP+RFP+Sox9（nucleus） | 7,139 | 7,458 |
  | Olig2 lost / gained / switched | 3,359 / 2,611 / 161 | 2,895 / 2,212 / 153 |
  | Sox9 匹配 old / nucleus | 58,358 / 55,560 | 58,358 / 55,497 |

  - 病例：cell #2 两种规则下都变成 GFP+RFP+Sox9（Olig2 只有 SNR 2.5 那一层，被删）。cell #3 的 Olig2 只剩 z485 那层（SNR 5.6）：old 规则下仍是 GFP+Olig2+RFP，nucleus 规则下变成 GFP+RFP。中点 485 超出 soma z481–484 半层，它的 Sox9 搭档（z484–486）原来也是靠这个 Olig2 才通过的，现在一起被拒。

## 2026-11-13（续）soma max_z_gap = 1 测试（未改 pipeline）

- **起因**：GUI 里两个四通道细胞在 nucleus 规则下被拆开，都是因为 GFP 或 RFP 漏检一层、`max_z_gap = 0` 让 track 断开：
  - 6c37/c164（全局约 x 6370, y 7720）：RFP 在 z435 漏检，拆成 GFP+RFP（z434–435）和 RFP（z436–437）。z436 的 Sox9 和 Olig2 在 nucleus 规则下都去了后者。
  - cdb2/26e5（约 x 7030, y 7790）：RFP 缺 z420、GFP 缺 z421，拆成 z418–420 和 z421–423 两个 soma。跨在断点上的 Sox9 和 Olig2 按 xy 距离分到两边，变成 GFP+RFP+Sox9 和 GFP+Olig2+RFP。
- **`relink_soma_test.py` 新增 `--soma-gap N`**，覆盖配置里的 `z_linker.soma.max_z_gap`，只用于 relink 的 soma track；值记进 `relink_settings.json`。
- **ET10 4 tile，Olig2 SNR ≥ 4，old vs nucleus**：`X:\Fengyi\tf_z_rule_compare_ET10_relink_span{6,7}_gap1_olig2snr4_old_vs_nucleus\`，对照组是同 span 的 `..._span{6,7}_olig2snr4_...`。soma track 用 `soma_tracks_{ch}.csv`（同通道 neuron/glia 去重之后）按 track_id 比较：

  | | span 6 GFP | span 6 RFP | span 7 GFP | span 7 RFP |
  |---|---|---|---|---|
  | gap 0 → gap 1 track 数 | 92,859 → 90,699 | 86,567 → 84,330 | 92,007 → 89,590 | 85,935 → 83,442 |
  | 净变化 | −2,160 | −2,237 | −2,417 | −2,493 |
  | 消失（并入别的 track） | 5,924 | 6,931 | 6,125 | 7,137 |
  | 新增 | 3,764 | 4,694 | 3,708 | 4,644 |
  | 新增中跨过一层空缺的 | 3,137 | 3,761 | 3,251 | 3,866 |

  - 1 层 track 减少约 2,400 个/通道。Phase A soma：span 6 136,300 → 132,541，span 7 135,096 → 130,934。
  - span 7 nucleus：Sox9 lost / gained / switched 5,220 / 2,359 / 297 → 5,044 / 2,186 / 327；Olig2 2,895 / 2,212 / 153 → 2,823 / 2,068 / 143。
  - 两个病例都合成一个 soma（z434–437、z418–423），两种规则下都是 GFP+Olig2+RFP+Sox9。
- **GUI（11-13 续）**：`zrule_compare` 模式的 `[relink s3 after class dedup] {ch}` 改成一个 track 一个框（代表框画在代表层，同 s3 层），统一白色，不再按 neuron/glia 分色；点框多打印 `bridged_layers`（z 范围内跨过的空缺层数）。
  - `relink_soma_test.py` 带 `--before` 时写 `soma_tracks_removed_{ch}.csv`：before 文件夹里本次不再存在的 track，加 `absorbed_by`（本次哪些 track 包含它的逐层框）。两个 gap1 文件夹已补写：span 7 GFP 6,125（6,123 被吸收）、RFP 7,137（7,107）；span 6 GFP 5,924（5,922）、RFP 6,931（6,907）。
  - GUI 读到这个文件时加两层：`[gap removed] {ch}`（黄色，消失的 track）和 `[gap merged] {ch}`（蓝色，吸收它们的 track），默认都显示。
  - 测试：`test_saved_pre_gui` 19 项通过（改 1 项、新增 1 项），`test_saved_s3_tile_frame` 6 项通过。

## 2026-11-13（续）soma z 连接加 IoMin（大框套小框也连）；GUI 打印逐层框（未改 pipeline）

- **起因**（GUI，span7 gap1，全局约 x 6765, y 8420, z445–452）：同一细胞的 RFP 框在大框（约 85 px）和小框（约 27×38）之间来回变，相邻层 IoU 约 0.14，过不了 0.35。gap 1 让 41ce 跳过 z449、z451 连上 z450、z452，但 z449、z451 的框各自成了单层 track（da6b、90c7）。z445 的大框 f9c7 成了单独的 RFP soma，还拿到 Sox9/Olig2。
- `z_linker` 本身只按 IoU 连接。用户记得的包含判断在别处：Stage 2.25 containment NMS（同层 2D）、Phase A `cross_class_contain_thresh`、`patch_fragment_iomin`，都不管 z 连接。
- **`relink_soma_test.py` 新增 `--link-iomin T`**：soma z 连接时 IoU ≥ `iou_thresh` 或 IoMin ≥ T 都可连（同类 base；修复 A 下不分 neuron/glia）。每个连通块一对一，先使连接数最多，再使 IoU 和最大。不加此项时仍用 Stage 3 原来的匹配函数。值记进 `relink_settings.json`（`soma_link_iomin`）。
- **ET10 4 tile，span {6,7} + gap 1 + IoMin 0.8 + Olig2 SNR ≥ 4**：`X:\Fengyi\tf_z_rule_compare_ET10_relink_span{6,7}_gap1_iomin08_olig2snr4_old_vs_nucleus\`（`--before` = 同 span 的 gap1 文件夹）

  | span 7（span 6 相近） | gap1 | gap1 + IoMin 0.8 |
  |---|---|---|
  | GFP / RFP track | 89,590 / 83,442 | 88,300 / 79,017 |
  | RFP 单层 track | 22,620 | 18,263 |
  | RFP 7 层 track | 1,481 | 2,013 |
  | Phase A soma | 130,934 | 125,560 |
  | 残片 soma IoMin ≥ 0.5 / 0.8 | 7,602 / 3,025 | 6,414 / 1,867 |
  | Sox9 lost / gained / switched | 5,044 / 2,186 / 327 | 4,846 / 1,944 / 117 |
  | Olig2 lost / gained / switched | 2,823 / 2,068 / 143 | 2,660 / 1,932 / 54 |
  | Sox9 阳性 soma old / nucleus | 46,293 / 45,666 | 45,303 / 44,647 |

  - 相对 gap1 消失的 track：GFP 2,686、RFP 10,715（`soma_tracks_removed_*`）。
  - 例子细胞：RFP 合成 e42d（z445–451，7 层大小框连续）；f9c7、da6b、90c7 三个 RFP 单独 soma 消失。z452 的大框 cb3c 因为 7 层上限又成了单层 track（无 TF）。附近另有 RFP a83d（z449–452）配上 GFP 96eb，成了新的 GFP+RFP soma，待 GUI 确认是不是另一个细胞。
- **GUI**：点 relink 层（含 gap removed / merged）的框，会逐层打印该 track 的框（全局坐标、宽×高），跨过的空缺层标 `bridged`；状态栏显示 bridged 层数。
- 测试：`test_saved_pre_gui` 21 项通过（新增 IoMin 连接、逐层打印各 1 项），`test_saved_s3_tile_frame` 6 项通过。
- **待查**：IoMin 连接可能把上下相邻的两个细胞连成一个（大 glia 框盖住下一层另一个细胞的小框）。RFP 7 层 track 增加约 530 个，需要在 GUI 抽查。

## 2026-11-13（续）z 规则对比结果文件夹改为数字编号

- 用户要求测试输出文件夹名改短。`X:\Fengyi\tf_z_rule_compare*` 共 13 个按完成时间移到 `X:\Fengyi\zrule_tests\01` … `13`。`index.csv` 记每个编号的旧名、完成时间、样本、tile、base/cand 规则、relink、soma span/gap、link_iomin、TF SNR 和 `before`（对照组编号）。
- 本日志之前写的旧路径对应：ET10_relink_old_vs_nucleus = 05；relink_span{6,7} = 06/07；relink_span{6,7}_olig2snr4 = 08/09；…_gap1_… = 10/11；…_gap1_iomin08_… = 12/13（偶数 span 6，奇数 span 7）。
- `relink_settings.json` 以后也记 `before`（之前没存，08–13 的对照组是按本次会话补记的）。以后新的测试接着编号（14 起），并更新 `index.csv`。
- **决定（11-13）**：soma z 连接上限先定 7（GUI 里看不出 6 和 7 的差别，7 的残片更少：12 vs 13 残片 soma IoMin ≥ 0.8 为 2,292 vs 1,867，单层 1,860 vs 1,367）。以后的测试以 13（span 7 + gap 1 + IoMin 0.8 + Olig2 SNR ≥ 4）为基线，不再同时跑 span 6。

## 2026-11-13（续）测试 13 的改动正式写进 pipeline；测试脚本和 GUI zrule_compare 模式删除

用户决定：Stage 4 用 nucleus 规则，soma z linker 用 gap 1，以及测试 13 的其余改动都进 pipeline，作为所有样本的默认。Olig2/Sox9 的 SNR 阈值先不改配置，等每个样本的抽样报告。

- **Stage 3 soma 连接**（`src/core/z_linker.py`）：
  - 新增 `STAGE3_LINK_DEFAULTS` / `stage3_link_params(config z_linker)`（run_inference 和 pipeline_qc 共用）、`dedup_cross_tile`、`link_tracks`。`run_z_linker` 及其 helper 一行没改：它们在 Stage 2.9 的 `SOLVE_CODE` 里，也被 Stage 2.5 对齐调用。核对过：ET10 的 solver hash 不变，下次运行不会被 2.9 的保护拦下。
  - `link_tracks` = 连接前跨 tile 去重（同层不同 tile，IoU > 0.25 或 IoMin > 0.45，留高分框，glia 优先改标）+ IoU ≥ 0.35 或 IoMin ≥ 0.8 都可连（一对一，先使连接数最多，再使 IoU 和最大）+ neuron/glia 可互连（含 glia 成员的 track 为 glia）+ span 7 / gap 1。每项都可在 config 里设 null 关掉；全关时与 `run_z_linker` 相同（有测试）。匹配函数通过 `_matchers` 临时替换，`run_z_linker` 调用时才去取。
  - TF 通道参数不变（span 3、gap 0、只看 IoU）。
- **Stage 3 checkpoint**（`channel_stage3.py`）：去重删掉的框写 `6_3d_global/{ch}_link_dedup.csv`，列入 track manifest 的输出。track 签名含新参数和整个 `z_linker.py` 的 hash，所以所有样本下次运行都会重新连接（不重新拼接，5_2d_global 保留）。
- **Stage 4 nucleus 规则**（`stitcher.py`、`run_inference.py`）：`tf_nucleus_groups`、`tf_nucleus_z_ranges` 从测试脚本移过来。`annotate_soma_with_tf_containment` 新参数 `tf_z_ranges`；`_tf_z_contained` 改为对中点区间 [lo, hi] 判断（单个 TF 时 lo = hi，与原 new 规则相同）。Phase B 先对所有 TF 通道做尺寸过滤，再配对，再逐通道标注。配对记进 `coloc_decisions.csv`（stage `tf_nucleus`，decision `same_nucleus`，metric = IoMin），已在 Stage 4 manifest 里。config `z_linker.tf.nucleus_iomin`（默认 0.7，null = 关）。
  - 测试脚本的 docstring 说"同一核的成员中点最多差一层"，不成立（z400–402 与 z402–404 共享 z402，中点差 2）。实际用的是区间判断，与测试 13 一致，docstring 已改。
- **配置**：25 个 config（EGFR、TSC、TSC2）的顶层 `z_linker.soma` 改为 span 7、gap 1，加 `link_iomin_thresh`、`link_across_classes`、`cross_tile_dedup_iou/_iomin`，`tf` 加 `nucleus_iomin`。`pre_align_params.z_linker`（对齐用）不动；脚本核对过每个 config 除 `z_linker` 外其余设置完全不变。原文件备份在会话临时目录。
- **QC**（`pipeline_qc.py`）：span/gap/min_z_layers 用 `stage3_link_params`（原来缺省 5/0 会误报）；`{ch}_link_dedup.csv` 里的框计入 detection_partition。
- **删除**：`src/utils/compare_tf_z_rule.py`、`src/utils/relink_soma_test.py`；GUI `zrule_compare` 模式（saved_gui 的 zrule/relink 层、visualize/visualize_gui 的模式选项）和对应测试。同期的 contrast 面板、点击编号保留。`X:\Fengyi\zrule_tests\01–13` 留在磁盘上，GUI 不能再打开。
- **核对**：用新的 pipeline 代码（ET10 config）对 4 个 tile 区域的 5_2d_global 重连 GFP/RFP，再做同通道类别去重，与测试 13 完全一致：GFP 88,300、RFP 79,017 条 track 的 id 和类别全部相同，去重删掉的框 4,735 / 4,509 个也相同。
- **测试**：新增 `tests/test_stage3_link_tracks.py`（6 项）；`test_coloc_sources` 新增 nucleus 1 项；`test_pipeline_qc` 新增 link_dedup 1 项；`test_automatic_resume` 的 patch 目标改为 `link_tracks`。全部测试文件逐个运行通过（238 项），`test_tile_2d_sources.py` 进程崩溃，改动前就如此。
- **注意**：顶层 `z_linker.tf.containment_z_pad` 已无人读取（Stage 4 从 11-09 起不用，对齐读 `pre_align_params` 里的），config 注释还是旧的。
- **下一步**：HPC 上重跑样本时，Stage 3 只重新连接，Stage 4 重算；Olig2/Sox9 每个样本的 SNR 抽样报告。

## 2026-11-13（续）各样本 Olig2/Sox9 SNR 抽样报告（未改配置）

- **新工具 `src/utils/snr_survey.py`**（只读）：对每个 config 的每个单曝光 StarDist TF 通道，从组织多的 tile 里随机抽 4 个（seed 20261113），每个 tile 按深度三等分抽 3 层（常规过滤后 ≥ 100 个框）。SNR 优先用 Stage 2.2 已有的 contrast 表，没有就用 `slice_ring_stats` 现算（公式相同）。只统计常规过滤保留的框（2_2d_filtered 的结果 + 原因为 snr_min 的 rejected），报告各阈值去掉的比例；以 ET10 为参照（Olig2 4、Sox9 5），并给出"匹配阈值"（去掉与 ET10 相同比例时的 SNR）。每个样本写 report.html（表、SNR 分布图、每层一张按 SNR 分色的图），总表 summary.csv / index.html。每个样本×通道完成后缓存，断了可续跑。Windows 盘符从 `net use` 读取（这台机器 Chlamydia = X:）。测试 `tests/test_snr_survey.py`（3 项）；连同上一条的 238 项，全部 241 项通过。
  - 修正：config 路径先 normpath 再去重，否则参照样本（ET10）因路径分隔符不同会被排两次。
- **结果** `X:\Fengyi\snr_survey\01\`（25 个样本，44 个样本×通道，8 进程约 1.5 小时；`X:\Fengyi\snr_survey\index.csv` 记设置）：
  - Olig2：ET10 在 SNR 4 去掉 11.5%。多数样本在 7–19%，匹配阈值 3.1–5.8。偏离较大的有 TSC_FW2（18.9%，匹配 3.1）、batch4_sample5（16.2%，3.5）、TSC_FW8（3.7%，5.8）。
  - Sox9：ET10 在 SNR 5 去掉 35.1%。多数样本在 26–43%，匹配阈值 4.2–6.1。batch4_sample5（69.5%，环形背景中位数仅 176，SNR 中位数 3.2，匹配 2.4）和 batch4_sample4（51.6%，3.7）明显偏离。
  - 层与层之间差别很大（同一样本的单层去除比例可差 3–10 倍），所以每个样本的数字只是 12 层的合计。
  - 注意：匹配阈值假设各样本假阳性比例与 ET10 相同，不一定成立，需要看图判断。
- **下一步**：用户看报告后决定每个样本的 `contrast_filter.{Olig2,Sox9}.snr_min`；改了之后该样本要从 Stage 2.2/2.25 重跑（对齐也会重跑）。
