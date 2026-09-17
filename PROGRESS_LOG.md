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
