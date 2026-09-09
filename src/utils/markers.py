# -*- coding: utf-8 -*-
"""Channel-id ↔ class-string marker helpers.

Class 字符串是下划线分隔的 "<base>_<marker>_<marker>..."，例如 "neuron_GFP_RFP"。
因此通道 id 本身如果带下划线，就会在每一次 split('_') 时多出一个伪 marker：
曝光时长后缀 "GFP_3" 会产生 "neuron_3_GFP_RFP" 这样的 class，以及共定位视图和
统计报告里那个凭空多出来的 "3"。

  channel_marker()  写入端：把通道 id 规范成单个 token 再拼进 class。
  split_class()     读取端：解析 class，顺带丢掉伪 marker，
                    这样按旧命名已经生成的结果也能正确解析。

注意：文件名（*_{ch_id}_result.csv、{ch_id}_3d_tracked.pkl 等）继续使用原始
通道 id，本模块只规范 class 字符串。
"""

import re

_EXPOSURE_SUFFIX = re.compile(r'_\d+$')


def channel_marker(ch_id):
    """通道 id → class 字符串里的单个 marker token。

    "GFP_3" → "GFP"（去掉曝光时长后缀）；"GFP_low" → "GFPlow"（去掉残余下划线）。
    "Sox9"、"Olig2" 这类以数字结尾但不含下划线的 id 保持原样。
    """
    s = _EXPOSURE_SUFFIX.sub('', str(ch_id))
    s = s.replace('_', '')
    return s or str(ch_id)


def clean_markers(markers):
    """丢掉纯数字的伪 marker（旧结果里由 "GFP_3" 拆出来的 "3"）。

    如果全部 marker 都是纯数字则原样返回，避免误删真的以数字命名的通道。
    """
    markers = [str(m) for m in markers]
    kept = [m for m in markers if not m.isdigit()]
    return kept if kept else markers


def split_class(class_str):
    """class 字符串 → (base_type, [markers])，marker 已过滤伪 marker。"""
    parts = str(class_str).split('_')
    return parts[0], clean_markers(parts[1:])


def class_markers(class_str):
    """class 字符串 → [markers]，已过滤伪 marker。"""
    return split_class(class_str)[1]
