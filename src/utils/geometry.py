def mapTile(dirname_list, left2right=True, top2bottom=True):
    y_pos = []; x_pos = []
    for dirname in dirname_list:
        if dirname[:6] not in y_pos: y_pos.append(dirname[:6])
        if dirname[-6:] not in x_pos: x_pos.append(dirname[-6:])
    if not left2right: x_pos.sort(reverse=True)
    if not top2bottom: y_pos.sort(reverse=True)
    dir_map = {}
    for dirname in dirname_list:
        dir_map[dirname] = (y_pos.index(dirname[:6]), x_pos.index(dirname[-6:]))
    return dir_map
