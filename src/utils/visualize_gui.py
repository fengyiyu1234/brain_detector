# -*- coding: utf-8 -*-
"""Interactive launcher for the existing napari visualization modes."""
import json
import os
import sys
import tempfile

from qtpy.QtCore import QLocale, QProcess, QThread, QTimer, Signal, Qt
from qtpy.QtGui import QBrush, QColor, QIntValidator
from qtpy.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressDialog, QPushButton, QScrollArea, QSpinBox, QSplitter, QTabWidget,
    QStyledItemDelegate, QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from src.config.loader import load_config


STYLE = """
QWidget { background: #f7f5fc; color: #302843; font-family: "Segoe UI", "Microsoft YaHei"; font-size: __BODY_SIZE__pt; }
QMainWindow { background: #f7f5fc; }
QFrame#card { background: white; border: 1px solid #e5def2; border-radius: 14px; }
QFrame#card QLabel, QFrame#card QCheckBox { background: transparent; }
QLabel#title { font-size: __TITLE_SIZE__pt; font-weight: 700; color: #49306d; }
QLabel#section { font-size: __SECTION_SIZE__pt; font-weight: 700; color: #67459a; }
QLabel#muted { color: #746b83; }
QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QListWidget {
    background: white; border: 1px solid #dcd3e9; border-radius: 8px;
    padding: 8px; selection-background-color: #dfccfa; selection-color: #302843;
}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QListWidget:focus { border: 2px solid #9b6ed3; }
QPushButton { background: #eee8f7; border: 1px solid #ded3ed; border-radius: 8px; padding: 10px 16px; font-weight: 600; }
QPushButton:hover { background: #e5d8f5; }
QPushButton#primary { background: #8054bc; color: white; border: 1px solid #8054bc; }
QPushButton#primary:hover { background: #6c40a8; }
QTabWidget::pane { border: 1px solid #e5def2; border-radius: 8px; background: white; }
QTabBar::tab { background: #eee8f7; padding: 11px 18px; border-radius: 7px; margin-right: 4px; }
QTabBar::tab:selected { background: #dfccfa; color: #49306d; }
QSplitter#mainSplitter::handle:horizontal { background: #d9c8ee; border-radius: 4px; }
QSplitter#mainSplitter::handle:horizontal:hover { background: #aa80d8; }
QProgressBar { background: #eee8f7; border: 1px solid #d9c8ee; border-radius: 6px; min-height: 18px; text-align: center; }
QProgressBar::chunk { background: #8054bc; border-radius: 5px; }
"""


def _card():
    frame = QFrame()
    frame.setObjectName('card')
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(11)
    return frame, layout


def _heading(text):
    label = QLabel(text)
    label.setObjectName('section')
    return label


def _line(value=None):
    edit = QLineEdit()
    edit.setText('' if value is None else str(value))
    return edit


def _nullable_int(edit, field):
    value = edit.text().strip()
    if not value:
        return None
    number = int(value)
    if number < 0:
        raise ValueError(f'{field} must be zero or greater.')
    return number


def _tile_grid_positions(tiles):
    """Match the CLI grid: rank the first two stage coordinates as row and column."""
    coordinates = {}
    for tile in tiles:
        parts = tile.split('_')
        if len(parts) < 2:
            return None
        try:
            row_stage, col_stage = int(parts[0]), int(parts[1])
        except ValueError:
            return None
        coordinates[tile] = (row_stage, col_stage)
    rows = {value for value, _ in coordinates.values()}
    cols = {value for _, value in coordinates.values()}
    row_index = {value: index for index, value in enumerate(sorted(rows))}
    col_index = {value: index for index, value in enumerate(sorted(cols))}
    positions = {
        tile: (row_index[row], col_index[col])
        for tile, (row, col) in coordinates.items()
    }
    if len(set(positions.values())) != len(tiles):
        return None
    return positions


class TileScanner(QThread):
    scanned = Signal(str, list, str)

    def __init__(self, sample, anchor_dir, parent=None):
        super().__init__(parent)
        self.sample = sample
        self.anchor_dir = anchor_dir

    def run(self):
        if not os.path.isdir(self.anchor_dir):
            self.scanned.emit(self.sample, [], f'Image directory does not exist: {self.anchor_dir}')
            return
        tiles = []
        for root, dirs, _ in os.walk(self.anchor_dir):
            if not dirs:
                tiles.append(root)
        ordered_names = list(dict.fromkeys(os.path.basename(path) for path in sorted(tiles)))
        self.scanned.emit(self.sample, ordered_names, '')


class ValueDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        if index.column() != 1:
            return None
        return super().createEditor(parent, option, index)


class VisualizeLauncher(QMainWindow):
    def __init__(self, config_path):
        super().__init__()
        self.config_path = os.path.abspath(config_path)
        self.config = {}
        self.workers = []
        self.processes = []
        self.process_output = {}
        self.process_paths = {}
        self.process_dialogs = {}
        self.process_buffers = {}
        self.scan_dialog = None
        self._loading = False
        self._advanced_dirty = False
        self._scan_number = 0
        self._last_sample_name = ''
        self._chosen_sample_name = ''
        self._tiles_sample = ''
        self._pending_tiles = set()
        self._grid_items = {}
        self._tile_list_items = {}
        self.font_size = 13
        self._build_ui()
        self._load_file(self.config_path)

    def _build_ui(self):
        self.setWindowTitle('Brain Detector · Visualize')
        self.resize(1280, 860)
        self.setMinimumSize(960, 680)
        self._apply_font_size()
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 18, 22, 18)
        outer.setSpacing(14)

        top = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel('Brain Detector')
        title.setObjectName('title')
        title_box.addWidget(title)
        subtitle = QLabel('Select a sample and tiles, adjust settings, then open Napari')
        subtitle.setObjectName('muted')
        title_box.addWidget(subtitle)
        top.addLayout(title_box)
        top.addStretch()
        font_label = QLabel('Font')
        font_label.setObjectName('muted')
        top.addWidget(font_label)
        smaller = QPushButton('A-')
        smaller.setToolTip('Decrease font size')
        smaller.clicked.connect(lambda: self._change_font_size(-1))
        top.addWidget(smaller)
        self.font_indicator = QLabel('')
        self.font_indicator.setObjectName('muted')
        top.addWidget(self.font_indicator)
        larger = QPushButton('A+')
        larger.setToolTip('Increase font size')
        larger.clicked.connect(lambda: self._change_font_size(1))
        top.addWidget(larger)
        self._update_font_indicator()
        load_button = QPushButton('Load config...')
        load_button.clicked.connect(self._choose_config)
        top.addWidget(load_button)
        save_button = QPushButton('Save config as...')
        save_button.clicked.connect(self._save_config)
        top.addWidget(save_button)
        outer.addLayout(top)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setObjectName('mainSplitter')
        self.main_splitter.setHandleWidth(10)
        self.main_splitter.setChildrenCollapsible(False)
        outer.addWidget(self.main_splitter, 1)

        left_panel = QWidget()
        left_panel.setMinimumWidth(310)
        left = QVBoxLayout(left_panel)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(14)
        self.main_splitter.addWidget(left_panel)

        sample_card, sample_layout = _card()
        sample_layout.addWidget(_heading('01  Sample'))
        self.sample_box = QComboBox()
        self.sample_box.currentTextChanged.connect(self._sample_changed)
        sample_row = QHBoxLayout()
        sample_row.addWidget(self.sample_box, 1)
        self.choose_sample_button = QPushButton('Choose')
        self.choose_sample_button.setToolTip('Choose this sample and scan its tiles')
        self.choose_sample_button.clicked.connect(self._choose_sample)
        sample_row.addWidget(self.choose_sample_button)
        sample_layout.addLayout(sample_row)
        self.sample_path = QLabel('')
        self.sample_path.setObjectName('muted')
        self.sample_path.setWordWrap(True)
        sample_layout.addWidget(self.sample_path)
        left.addWidget(sample_card)

        tile_card, tile_layout = _card()
        tile_layout.addWidget(_heading('02  Tile'))
        search_row = QHBoxLayout()
        self.tile_search = QLineEdit()
        self.tile_search.setPlaceholderText('Search tile name or number')
        self.tile_search.textChanged.connect(self._filter_tiles)
        search_row.addWidget(self.tile_search)
        self.refresh_button = QPushButton('Refresh')
        self.refresh_button.setEnabled(False)
        self.refresh_button.clicked.connect(self._refresh_tiles)
        search_row.addWidget(self.refresh_button)
        tile_layout.addLayout(search_row)
        self.tile_splitter = QSplitter(Qt.Vertical)
        self.grid_hint = QLabel('Spatial layout | row / column | click a number to select')
        self.grid_hint.setObjectName('muted')
        tile_layout.addWidget(self.grid_hint)
        self.tile_grid = QTableWidget()
        self.tile_grid.setSelectionMode(QAbstractItemView.NoSelection)
        self.tile_grid.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header_style = (
            'QHeaderView::section { background: #eee5fa; color: #7542ad; '
            'border: 1px solid #d9c8ee; font-weight: 700; padding: 4px; }'
        )
        self.tile_grid.verticalHeader().setStyleSheet(header_style)
        self.tile_grid.horizontalHeader().setStyleSheet(header_style)
        self.tile_grid.verticalHeader().setDefaultSectionSize(38)
        self.tile_grid.horizontalHeader().setDefaultSectionSize(64)
        self.tile_grid.itemClicked.connect(self._grid_clicked)
        self.tile_grid.setMinimumHeight(180)
        self.tile_splitter.addWidget(self.tile_grid)
        self.tile_list = QListWidget()
        self.tile_list.setSelectionMode(QListWidget.MultiSelection)
        self.tile_list.itemSelectionChanged.connect(self._sync_grid_selection)
        self.tile_list.setMinimumHeight(140)
        self.tile_splitter.addWidget(self.tile_list)
        self.tile_splitter.setSizes([280, 180])
        tile_layout.addWidget(self.tile_splitter, 1)
        self.tile_status = QLabel('No tiles loaded')
        self.tile_status.setObjectName('muted')
        tile_layout.addWidget(self.tile_status)
        left.addWidget(tile_card, 1)

        settings_card, settings_layout = _card()
        settings_card.setMinimumWidth(460)
        self.main_splitter.addWidget(settings_card)
        self.main_splitter.setStretchFactor(0, 4)
        self.main_splitter.setStretchFactor(1, 6)
        self.main_splitter.setSizes([430, 800])
        settings_layout.addWidget(_heading('03  Visualization settings'))
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self._tab_changed)
        settings_layout.addWidget(self.tabs, 1)

        basic_scroll = QScrollArea()
        basic_scroll.setWidgetResizable(True)
        basic_scroll.setFrameShape(QFrame.NoFrame)
        basic = QWidget()
        basic_layout = QVBoxLayout(basic)
        basic_layout.setSpacing(14)
        form = QFormLayout()
        self.settings_form = form
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(11)
        basic_layout.addLayout(form)
        self.mode = QComboBox()
        self.mode.addItems(['prealign', '2d', 'post'])
        form.addRow('Mode', self.mode)
        self.view_space = QComboBox()
        self.view_space.addItems(['local', 'global'])
        form.addRow('View coordinates', self.view_space)
        self.source_2d = QComboBox()
        self.source_2d.addItems(['filtered', 'raw'])
        form.addRow('2D source', self.source_2d)
        self.stage = QComboBox()
        self.stage.addItems(['all', 's1', 's3', 's4'])
        form.addRow('Result stage', self.stage)
        self.z_start = _line()
        self.z_start.setPlaceholderText('Blank = 0')
        self.z_start.setValidator(QIntValidator(0, 2147483647))
        form.addRow('z_start', self.z_start)
        self.z_count = _line()
        self.z_count.setPlaceholderText('Blank = to the end')
        self.z_count.setValidator(QIntValidator(0, 2147483647))
        form.addRow('z_count', self.z_count)
        self.global_z_start = _line()
        self.global_z_start.setPlaceholderText('Blank = automatic')
        self.global_z_start.setValidator(QIntValidator(0, 2147483647))
        form.addRow('global_z_start', self.global_z_start)
        self.fn_ann_path = _line()
        form.addRow('Annotation CSV', self._path_row(self.fn_ann_path, False))
        self.fn_crop_dir = _line()
        form.addRow('Crop folder', self._path_row(self.fn_crop_dir, True))
        self.fn_crop_size = QSpinBox()
        self.fn_crop_size.setRange(1, 10000)
        form.addRow('Crop size', self.fn_crop_size)

        basic_layout.addWidget(_heading('Display layers'))
        switches = QHBoxLayout()
        left_switches = QVBoxLayout()
        right_switches = QVBoxLayout()
        switches.addLayout(left_switches)
        switches.addLayout(right_switches)
        self.checks = {}
        for index, (key, label) in enumerate([
            ('no_images', 'Do not load raw images'),
            ('images_only_initially', 'Start with images only'),
            ('show_before', 'Raw 2D overlay'),
            ('show_zlinked', 'Z-linked results'),
            ('show_coloc', 'Colocalization results'),
            ('show_filtered_2d', 'Filtered 2D overlay'),
            ('spheres', 'Show 3D points'),
        ]):
            check = QCheckBox(label)
            self.checks[key] = check
            (left_switches if index < 3 else right_switches).addWidget(check)
        basic_layout.addLayout(switches)
        self.mode.currentTextChanged.connect(self._update_mode_controls)
        self.view_space.currentTextChanged.connect(self._update_mode_controls)
        self._update_mode_controls()
        hint = QLabel('Blank paths are stored as null. Edit sample paths, channels, colors, filters, and other settings in Full config.')
        hint.setWordWrap(True)
        hint.setObjectName('muted')
        basic_layout.addWidget(hint)
        basic_layout.addStretch()
        basic_scroll.setWidget(basic)
        self.tabs.addTab(basic_scroll, 'Common settings')

        advanced = QWidget()
        advanced_layout = QVBoxLayout(advanced)
        advanced_help = QLabel('Edit every visualization setting in the parameter tree or JSON. Click Apply JSON to update the sample and common settings.')
        advanced_help.setWordWrap(True)
        advanced_help.setObjectName('muted')
        advanced_layout.addWidget(advanced_help)
        self.advanced_tabs = QTabWidget()
        self.parameter_tree = QTreeWidget()
        self.parameter_tree.setItemDelegate(ValueDelegate(self.parameter_tree))
        self.parameter_tree.setHeaderLabels(['Parameter', 'Value'])
        self.parameter_tree.setColumnWidth(0, 240)
        self.parameter_tree.itemChanged.connect(self._tree_changed)
        self.advanced_tabs.addTab(self.parameter_tree, 'Parameters')
        self.json_editor = QPlainTextEdit()
        self.json_editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.json_editor.textChanged.connect(self._json_changed)
        self.advanced_tabs.addTab(self.json_editor, 'JSON')
        advanced_layout.addWidget(self.advanced_tabs, 1)
        apply_button = QPushButton('Apply JSON')
        apply_button.clicked.connect(self._apply_json)
        advanced_layout.addWidget(apply_button, 0, Qt.AlignRight)
        self.tabs.addTab(advanced, 'Full config')

        bottom = QHBoxLayout()
        self.run_status = QLabel('Adjust settings, then select tiles')
        self.run_status.setObjectName('muted')
        bottom.addWidget(self.run_status, 1)
        self.launch_button = QPushButton('Open selected tiles')
        self.launch_button.setObjectName('primary')
        self.launch_button.clicked.connect(self._launch)
        bottom.addWidget(self.launch_button)
        outer.addLayout(bottom)

    def _update_mode_controls(self, *_args):
        mode = self.mode.currentText()
        for widget, visible in (
            (self.source_2d, mode == '2d'),
            (self.view_space, mode == 'prealign'),
            (self.global_z_start,
             mode == 'prealign' and self.view_space.currentText() == 'global'),
            (self.stage, mode == 'post'),
        ):
            label = self.settings_form.labelForField(widget)
            if label is not None:
                label.setVisible(visible)
            widget.setVisible(visible)
        for key, check in self.checks.items():
            check.setVisible(key in ('no_images', 'images_only_initially') or mode == 'prealign')

    def _apply_font_size(self):
        style = (STYLE
                 .replace('__BODY_SIZE__', str(self.font_size))
                 .replace('__TITLE_SIZE__', str(self.font_size + 9))
                 .replace('__SECTION_SIZE__', str(self.font_size + 2)))
        self.setStyleSheet(style)

    def _update_font_indicator(self):
        self.font_indicator.setText(f'{self.font_size} pt')

    def _change_font_size(self, delta):
        self.font_size = min(20, max(10, self.font_size + delta))
        self._apply_font_size()
        self._update_font_indicator()

    def _path_row(self, edit, directory):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        button = QPushButton('Browse')
        button.clicked.connect(lambda: self._browse_path(edit, directory))
        layout.addWidget(button)
        return row

    def _browse_path(self, edit, directory):
        start = edit.text().strip() or os.path.dirname(self.config_path)
        if directory:
            chosen = QFileDialog.getExistingDirectory(
                self, 'Choose folder', start, QFileDialog.DontUseNativeDialog)
        else:
            chosen, _ = QFileDialog.getSaveFileName(
                self, 'Choose annotation CSV', start, 'CSV (*.csv)',
                options=QFileDialog.DontUseNativeDialog)
        if chosen:
            edit.setText(chosen)

    def _choose_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load visualization config', os.path.dirname(self.config_path),
            'JSON (*.json)', options=QFileDialog.DontUseNativeDialog)
        if path:
            self._load_file(path)

    def _load_file(self, path):
        try:
            config = load_config(path)
            if not isinstance(config, dict):
                raise ValueError('The top-level JSON value must be an object')
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, 'Could not load config', str(exc))
            return
        self.config_path = os.path.abspath(path)
        self.config = config
        self._chosen_sample_name = ''
        self._fill_controls()
        self.run_status.setText(self.config_path)

    def _fill_controls(self, refresh_tiles=True):
        self._loading = True
        try:
            self.sample_box.clear()
            samples = self.config.get('samples') or {}
            self.sample_box.addItems(list(samples) if samples else ['(default)'])
            if self._chosen_sample_name and (
                    self._chosen_sample_name in samples or
                    (not samples and self._chosen_sample_name == '(default)')):
                self.sample_box.setCurrentText(self._chosen_sample_name)
            else:
                self._chosen_sample_name = ''
                self.sample_box.setCurrentIndex(-1)
            for widget, key, default in [
                (self.mode, 'mode', 'prealign'),
                (self.view_space, 'view_space', 'local'),
                (self.source_2d, '2d_source', 'filtered'),
                (self.stage, 'stage', 'all'),
            ]:
                value = str(self.config.get(key, default))
                if widget.findText(value) >= 0:
                    widget.setCurrentText(value)
            for widget, key in [
                (self.z_start, 'z_start'), (self.z_count, 'z_count'),
                (self.global_z_start, 'global_z_start'),
                (self.fn_crop_dir, 'fn_crop_dir'),
            ]:
                widget.setText('' if self.config.get(key) is None else str(self.config[key]))
            sample = self._sample_data()
            ann = sample.get('fn_ann_path', self.config.get('fn_ann_path'))
            self.fn_ann_path.setText('' if ann is None else str(ann))
            self.fn_crop_size.setValue(int(self.config.get('fn_crop_size') or 256))
            for key, check in self.checks.items():
                check.setChecked(bool(self.config.get(key, key == 'images_only_initially')))
            self._last_sample_name = self.sample_box.currentText()
            self._set_json_text()
        finally:
            self._loading = False
        self._update_mode_controls()
        if refresh_tiles:
            self._clear_tiles('Select a sample and click Choose to scan tiles')
        self.refresh_button.setEnabled(bool(self._chosen_sample_name))

    def _sample_data(self):
        return (self.config.get('samples') or {}).get(self.sample_box.currentText(), {})

    def _sample_changed(self):
        if self._loading:
            return
        name = self.sample_box.currentText()
        previous = (self.config.get('samples') or {}).get(self._last_sample_name, {})
        if self._last_sample_name:
            target = previous if 'fn_ann_path' in previous else self.config
            target['fn_ann_path'] = self.fn_ann_path.text().strip() or None
        self._last_sample_name = name
        self._chosen_sample_name = ''
        self.refresh_button.setEnabled(False)
        self._clear_tiles('Click Choose to scan the selected sample' if name else
                          'Select a sample and click Choose to scan tiles')
        sample = self._sample_data()
        ann = sample.get('fn_ann_path', self.config.get('fn_ann_path'))
        self.fn_ann_path.setText('' if ann is None else str(ann))

    def _choose_sample(self):
        name = self.sample_box.currentText()
        if not name:
            QMessageBox.information(self, 'Choose a sample', 'Select a sample first.')
            return
        self._chosen_sample_name = name
        if self.config.get('samples'):
            self.config['active_sample'] = name
        self.refresh_button.setEnabled(True)
        self._refresh_tiles()

    def _clear_tiles(self, status):
        self._scan_number += 1
        self._tiles_sample = ''
        self._pending_tiles.clear()
        if self.scan_dialog is not None:
            self.scan_dialog.close()
            self.scan_dialog.deleteLater()
            self.scan_dialog = None
        self.sample_path.setText('')
        self.tile_list.blockSignals(True)
        self._grid_items.clear()
        self._tile_list_items.clear()
        self.tile_list.clear()
        self.tile_list.blockSignals(False)
        self.tile_grid.clear()
        self.tile_grid.setRowCount(0)
        self.tile_grid.setColumnCount(0)
        self.tile_status.setText(status)

    def _refresh_tiles(self):
        if not self._chosen_sample_name or self.sample_box.currentText() != self._chosen_sample_name:
            return
        name = self.sample_box.currentText()
        self._pending_tiles = (
            {item.data(Qt.UserRole) for item in self.tile_list.selectedItems()}
            if self._tiles_sample == name else set()
        )
        sample = self._sample_data()
        paths = sample.get('paths', self.config.get('paths', {}))
        routing = sample.get('channels_routing', self.config.get('channels_routing', []))
        active = [ch for ch in routing if ch.get('active', True)]
        anchor = paths.get(active[0].get('dir_key'), '') if active else ''
        self.sample_path.setText(anchor or 'Image path or active channel is missing')
        self.tile_list.blockSignals(True)
        self._grid_items.clear()
        self._tile_list_items.clear()
        self.tile_list.clear()
        self.tile_list.blockSignals(False)
        self.tile_grid.clear()
        self.tile_grid.setRowCount(0)
        self.tile_grid.setColumnCount(0)
        self.tile_status.setText('Scanning tiles...' if anchor else 'Cannot list tiles: image path is missing')
        self._scan_number += 1
        scan_number = self._scan_number
        if self.scan_dialog is not None:
            self.scan_dialog.close()
            self.scan_dialog.deleteLater()
            self.scan_dialog = None
        if not anchor:
            return
        self.scan_dialog = self._progress_dialog(
            'Scanning tiles', 'Scanning image folders for tiles...', 0, 0)
        worker = TileScanner(name, anchor, self)
        self.workers.append(worker)
        worker.scanned.connect(
            lambda sample_name, tiles, error, n=scan_number:
            self._tiles_ready(n, sample_name, tiles, error))
        worker.finished.connect(lambda w=worker: self._worker_done(w))
        worker.start()

    def _progress_dialog(self, title, label, minimum, maximum):
        dialog = QProgressDialog(label, '', minimum, maximum, self)
        dialog.setWindowTitle(title)
        dialog.setCancelButton(None)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setMinimumDuration(0)
        dialog.setMinimumWidth(450)
        dialog.setWindowModality(Qt.WindowModal)
        dialog.show()
        return dialog

    def _worker_done(self, worker):
        if worker in self.workers:
            self.workers.remove(worker)
        worker.deleteLater()
        if self.isHidden() and not self.workers and not self.processes:
            QApplication.instance().quit()

    def _tiles_ready(self, scan_number, sample_name, tiles, error):
        if scan_number != self._scan_number or sample_name != self.sample_box.currentText():
            return
        if self.scan_dialog is not None:
            self.scan_dialog.close()
            self.scan_dialog.deleteLater()
            self.scan_dialog = None
        if error:
            self.tile_status.setText(error)
            return
        selected = self.config.get('tile')
        selected = self._pending_tiles or set(
            selected if isinstance(selected, list) else [selected])
        self._tiles_sample = sample_name
        self.tile_list.blockSignals(True)
        self.tile_list.clear()
        self._tile_list_items.clear()
        for index, tile in enumerate(tiles):
            item = QListWidgetItem(f'[{index:03d}]  {tile}')
            item.setData(Qt.UserRole, tile)
            self.tile_list.addItem(item)
            self._tile_list_items[tile] = item
            if tile in selected:
                item.setSelected(True)
        self.tile_list.blockSignals(False)
        self._build_tile_grid(tiles)
        self.tile_status.setText(f'{len(tiles)} tiles | numbers match the list | multiple selection')
        self._filter_tiles(self.tile_search.text())

    def _build_tile_grid(self, tiles):
        self.tile_grid.clear()
        self._grid_items.clear()
        positions = _tile_grid_positions(tiles)
        if not positions:
            self.tile_grid.setRowCount(0)
            self.tile_grid.setColumnCount(0)
            self.grid_hint.setText('Tile names have no recognizable row and column coordinates; spatial layout is unavailable')
            return
        self.grid_hint.setText('Spatial layout | row / column | click a number to select')
        row_count = max(row for row, _ in positions.values()) + 1
        col_count = max(col for _, col in positions.values()) + 1
        self.tile_grid.setRowCount(row_count)
        self.tile_grid.setColumnCount(col_count)
        self.tile_grid.setVerticalHeaderLabels([str(row) for row in range(row_count)])
        self.tile_grid.setHorizontalHeaderLabels([str(col) for col in range(col_count)])
        for index, tile in enumerate(tiles):
            row, col = positions[tile]
            item = QTableWidgetItem(str(index))
            item.setData(Qt.UserRole, tile)
            item.setToolTip(f'[{index}] {tile}')
            item.setTextAlignment(Qt.AlignCenter)
            self.tile_grid.setItem(row, col, item)
            self._grid_items[tile] = item
        self._sync_grid_selection()

    def _grid_clicked(self, grid_item):
        tile = grid_item.data(Qt.UserRole)
        list_item = self._tile_list_items.get(tile)
        if list_item is not None:
            list_item.setSelected(not list_item.isSelected())

    def _sync_grid_selection(self):
        query = self.tile_search.text().strip().lower()
        for tile, grid_item in self._grid_items.items():
            list_item = self._tile_list_items.get(tile)
            selected = list_item is not None and list_item.isSelected()
            matched = list_item is not None and (
                not query or query in list_item.text().lower())
            color = '#d9c3f5' if selected else ('#ffffff' if matched else '#f1eef5')
            grid_item.setBackground(QBrush(QColor(color)))
            grid_item.setForeground(QBrush(QColor('#302843' if matched or selected else '#aaa2b7')))

    def _filter_tiles(self, query):
        term = query.strip().lower()
        for index in range(self.tile_list.count()):
            item = self.tile_list.item(index)
            item.setHidden(term not in item.text().lower())
        self._sync_grid_selection()

    def _read_controls(self):
        self.config['mode'] = self.mode.currentText()
        self.config['view_space'] = self.view_space.currentText()
        self.config['2d_source'] = self.source_2d.currentText()
        self.config['stage'] = self.stage.currentText()
        self.config['z_start'] = _nullable_int(self.z_start, 'z_start')
        self.config['z_count'] = _nullable_int(self.z_count, 'z_count')
        self.config['global_z_start'] = _nullable_int(self.global_z_start, 'global_z_start')
        self.config['fn_crop_dir'] = self.fn_crop_dir.text().strip() or None
        self.config['fn_crop_size'] = self.fn_crop_size.value()
        sample = self._sample_data()
        target = sample if 'fn_ann_path' in sample else self.config
        target['fn_ann_path'] = self.fn_ann_path.text().strip() or None
        for key, check in self.checks.items():
            self.config[key] = check.isChecked()

    def _set_json_text(self):
        self.json_editor.blockSignals(True)
        self.json_editor.setPlainText(json.dumps(self.config, ensure_ascii=False, indent=2))
        self.json_editor.blockSignals(False)
        self._advanced_dirty = False
        self.parameter_tree.setEnabled(True)
        self._build_tree()

    def _build_tree(self):
        expanded = set()

        def remember(item):
            if item.isExpanded():
                expanded.add(tuple(item.data(0, Qt.UserRole)))
            for index in range(item.childCount()):
                remember(item.child(index))

        for index in range(self.parameter_tree.topLevelItemCount()):
            remember(self.parameter_tree.topLevelItem(index))
        self.parameter_tree.blockSignals(True)
        self.parameter_tree.clear()

        def add(parent, key, value, path):
            item = QTreeWidgetItem(parent, [str(key), ''])
            item.setData(0, Qt.UserRole, path)
            if isinstance(value, dict):
                item.setText(1, f'{{{len(value)} fields}}')
                for child_key, child_value in value.items():
                    add(item, child_key, child_value, path + (child_key,))
            elif isinstance(value, list):
                item.setText(1, f'[{len(value)} items]')
                for index, child_value in enumerate(value):
                    add(item, index, child_value, path + (index,))
            else:
                item.setText(1, json.dumps(value, ensure_ascii=False))
                item.setFlags(item.flags() | Qt.ItemIsEditable)
            item.setExpanded(path in expanded)
            return item

        for key, value in self.config.items():
            add(self.parameter_tree, key, value, (key,))
        self.parameter_tree.blockSignals(False)

    def _tree_changed(self, item, column):
        if column != 1 or self._loading:
            return
        path = item.data(0, Qt.UserRole)
        if not path:
            return
        parent = self.config
        for key in path[:-1]:
            parent = parent[key]
        old_value = parent[path[-1]]
        if isinstance(old_value, (dict, list)):
            return
        raw = item.text(1).strip()
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        parent[path[-1]] = value
        self.json_editor.blockSignals(True)
        self.json_editor.setPlainText(json.dumps(self.config, ensure_ascii=False, indent=2))
        self.json_editor.blockSignals(False)
        self._advanced_dirty = False
        needs_scan = path[0] in ('samples', 'paths', 'channels_routing', 'active_sample', 'tile')
        if needs_scan:
            self._chosen_sample_name = ''
        QTimer.singleShot(0, lambda refresh=needs_scan: self._fill_controls(refresh))

    def _json_changed(self):
        if not self._loading:
            self._advanced_dirty = True
            self.parameter_tree.setEnabled(False)

    def _tab_changed(self, index):
        if index == 1 and not self._advanced_dirty:
            try:
                self._read_controls()
                self._set_json_text()
            except ValueError as exc:
                QMessageBox.warning(self, 'Invalid setting', str(exc))

    def _apply_json(self):
        try:
            config = json.loads(self.json_editor.toPlainText())
            if not isinstance(config, dict):
                raise ValueError('The top-level JSON value must be an object')
            self.config = config
            self._advanced_dirty = False
            self._chosen_sample_name = ''
            self._fill_controls()
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, 'Invalid JSON', str(exc))
            return False
        return True

    def _current_config(self):
        if self._advanced_dirty and not self._apply_json():
            return None
        if self.tabs.currentIndex() == 0:
            try:
                self._read_controls()
            except ValueError as exc:
                QMessageBox.warning(self, 'Invalid setting', str(exc))
                return None
        return self.config

    def _save_config(self):
        config = self._current_config()
        if config is None:
            return
        suggested = os.path.join(
            os.path.dirname(self.config_path), 'vis_config.gui.json')
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save visualization config as', suggested, 'JSON (*.json)',
            options=QFileDialog.DontUseNativeDialog)
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as file:
                json.dump(config, file, ensure_ascii=False, indent=2)
                file.write('\n')
        except OSError as exc:
            QMessageBox.critical(self, 'Save failed', str(exc))
            return
        self.config_path = os.path.abspath(path)
        self.run_status.setText(f'Saved: {path}')

    def _launch(self):
        if self._advanced_dirty:
            if self._apply_json():
                QMessageBox.information(
                    self, 'JSON applied',
                    'The configuration has been applied. Select tiles for the updated sample, then open the viewer.')
            return
        if not self._chosen_sample_name or self.sample_box.currentText() != self._chosen_sample_name:
            QMessageBox.information(self, 'Choose a sample', 'Select a sample and click Choose first.')
            return
        tiles = [self.tile_list.item(i).data(Qt.UserRole) for i in range(self.tile_list.count())
                 if self.tile_list.item(i).isSelected()]
        config = self._current_config()
        if config is None:
            return
        if not tiles:
            QMessageBox.information(self, 'Select tiles', 'Select at least one tile on the left.')
            return
        snapshot = dict(config)
        snapshot['tile'] = tiles[0] if len(tiles) == 1 else tiles
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', prefix='.vis_run_', delete=False,
                dir=os.path.dirname(self.config_path), encoding='utf-8'
            ) as file:
                json.dump(snapshot, file, ensure_ascii=False, indent=2)
                temp_path = file.name
        except OSError as exc:
            QMessageBox.critical(self, 'Start failed', f'Could not write temporary config: {exc}')
            return
        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments([
            '-u', os.path.join(os.path.dirname(__file__), 'visualize.py'),
            '--config', temp_path, '--direct',
        ])
        process.setWorkingDirectory(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(lambda p=process: self._read_output(p))
        process.errorOccurred.connect(
            lambda error, p=process: self._process_error(p, error))
        process.finished.connect(
            lambda code, status, p=process, path=temp_path:
            self._process_finished(p, path, code))
        self.processes.append(process)
        self.process_output[process] = ''
        self.process_paths[process] = temp_path
        self.process_buffers[process] = ''
        dialog = self._progress_dialog(
            'Opening tiles', f'Preparing viewer for {len(tiles)} tile(s)...',
            0, len(tiles))
        self.process_dialogs[process] = dialog
        process.start()
        self.run_status.setText(f'Opening {len(tiles)} tiles...')

    def _read_output(self, process):
        output = bytes(process.readAllStandardOutput()).decode('utf-8', errors='replace')
        if not output:
            return
        self.process_output[process] = (self.process_output.get(process, '') + output)[-4000:]
        pending = self.process_buffers.get(process, '') + output
        lines = pending.split('\n')
        self.process_buffers[process] = lines.pop()
        for line in lines:
            line = line.strip()
            if line.startswith('VIS_PROGRESS '):
                parts = line.split(' ', 3)
                if len(parts) == 4:
                    try:
                        current, total = int(parts[1]), int(parts[2])
                    except ValueError:
                        pass
                    else:
                        dialog = self.process_dialogs.get(process)
                        if dialog is not None:
                            dialog.setMaximum(total)
                            dialog.setValue(current)
                            dialog.setLabelText(f'Opening tile {current}/{total}: {parts[3]}')
                        self.run_status.setText(f'Loaded {current}/{total} tiles')
            elif line == 'VIS_READY':
                self._close_process_dialog(process)
                self.run_status.setText('Viewer ready')
            else:
                print(line, flush=True)

    def _close_process_dialog(self, process):
        dialog = self.process_dialogs.pop(process, None)
        if dialog is not None:
            dialog.close()
            dialog.deleteLater()

    def _process_error(self, process, error):
        if error != QProcess.FailedToStart:
            return
        path = self.process_paths.pop(process, None)
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass
        if process in self.processes:
            self.processes.remove(process)
        self.process_output.pop(process, None)
        self.process_buffers.pop(process, None)
        self._close_process_dialog(process)
        QMessageBox.critical(self, 'Viewer start failed', process.errorString())
        process.deleteLater()
        if self.isHidden() and not self.processes and not self.workers:
            QApplication.instance().quit()

    def _process_finished(self, process, temp_path, code):
        if process not in self.processes:
            return
        self._read_output(process)
        self._close_process_dialog(process)
        self.process_buffers.pop(process, None)
        self.process_paths.pop(process, None)
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        self.processes.remove(process)
        output = self.process_output.pop(process, '')
        process.deleteLater()
        if code == 0:
            self.run_status.setText('Viewer closed')
        else:
            self.run_status.setText(f'Viewer exited with code {code}')
            QMessageBox.warning(
                self, 'Viewer error', output[-1800:] or f'Exit code: {code}')
        if self.isHidden() and not self.processes and not self.workers:
            QApplication.instance().quit()

    def closeEvent(self, event):
        if self.processes or self.workers:
            self.hide()
            event.ignore()
        else:
            event.accept()
            QApplication.instance().quit()


def launch_gui(config_path):
    QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates))
    if QApplication.instance() is None:
        for name in ('AA_EnableHighDpiScaling', 'AA_UseHighDpiPixmaps'):
            attribute = getattr(Qt, name, None)
            if attribute is not None:
                QApplication.setAttribute(attribute, True)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    window = VisualizeLauncher(config_path)
    window.show()
    app.exec_()

