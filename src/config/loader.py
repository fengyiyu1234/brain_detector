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
