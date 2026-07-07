import json
import os


def _strip_comments(s):
    """Strip // line-comments from JSON text, skipping content inside strings."""
    result = []
    i = 0
    while i < len(s):
        if s[i] == '"':
            result.append(s[i])
            i += 1
            while i < len(s):
                if s[i] == '\\':        # escaped character
                    result.append(s[i])
                    result.append(s[i + 1])
                    i += 2
                elif s[i] == '"':       # end of string
                    result.append(s[i])
                    i += 1
                    break
                else:
                    result.append(s[i])
                    i += 1
        elif s[i:i + 2] == '//':        # line comment — skip to newline
            while i < len(s) and s[i] != '\n':
                i += 1
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = f.read()
    return json.loads(_strip_comments(raw))


def expand_double_exposure_channels(routing_config):
    """For each active channel with double_exposure=true, append a synthetic
    detection-only entry for its second_intensity_id/dir_key. Used ONLY for Stage 2
    raw per-tile detection; every other stage keeps using the unexpanded
    routing_config, since the two exposures become one logical channel after the
    dual-intensity fusion stage."""
    existing_ids = {ch['id'] for ch in routing_config}
    expanded = list(routing_config)
    for ch in routing_config:
        if not ch.get('double_exposure'):
            continue
        second_id = ch.get('second_intensity_id')
        second_dir_key = ch.get('second_intensity_dir_key')
        if not second_id or not second_dir_key:
            raise ValueError(
                f"❌ 通道 {ch['id']} 设置了 double_exposure=true，但缺少 "
                f"second_intensity_id/second_intensity_dir_key！"
            )
        if second_id in existing_ids:
            raise ValueError(
                f"❌ 通道 {ch['id']} 的 second_intensity_id='{second_id}' 与已有通道 id 冲突，"
                f"请确认 channels_routing 中没有残留的独立第二曝光通道条目。"
            )
        expanded.append({
            'id': second_id,
            'type': ch.get('type', 'soma'),
            'model': ch['model'],
            'dir_key': second_dir_key,
            'active': True,
        })
    return expanded
