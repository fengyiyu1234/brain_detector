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

**1. 拼接质量比较脚本 `scripts/compare_stitching.py`（新增）**
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
- 写 `scripts/solve_tile_positions.py`：接缝配对 → 加权稳健最小二乘求各通道 tile 位置 → 各通道 merging XML + 残差报告。
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

**3. 新脚本 `scripts/solve_tile_positions.py`**
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

**2. `scripts/solve_tile_positions.py`**
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
- `scripts/check_containment_equivalence.py`（新增）：重构等价性检验，做两件事——
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
