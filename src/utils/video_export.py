"""Frame-by-frame Z-sweep export of the current napari canvas."""
from pathlib import Path
import time
import uuid


def sweep_steps(start, end, stride=1):
    """Include both endpoints, also for reverse and non-divisible sweeps."""
    if stride < 1:
        raise ValueError('Slice step must be positive')
    direction = 1 if end >= start else -1
    steps = list(range(start, end + direction, direction * stride))
    if steps[-1] != end:
        steps.append(end)
    return steps


def add_video_export_panel(viewer, vis_cfg=None):
    """Attach a cancellable MP4/AVI exporter; no additional dependencies."""
    import cv2
    from qtpy.QtCore import QEvent, QTimer
    from qtpy.QtWidgets import (
        QCheckBox, QFileDialog, QGridLayout, QLabel, QMessageBox,
        QProgressBar, QPushButton, QSpinBox, QWidget,
    )

    class VideoExportPanel(QWidget):
        def __init__(self):
            super().__init__()
            self.writer = None
            self.active = False
            self.temporary = None
            self.timer = QTimer(self)
            self.timer.setSingleShot(True)
            self.timer.timeout.connect(self.capture)
            layout = QGridLayout(self)
            hint = QLabel('Export visible layers at the current zoom.\n'
                          'Use 2D XY view; indices match the Z slider (zero based).\n'
                          'Keep the view and window size unchanged during export.')
            hint.setWordWrap(True)
            layout.addWidget(hint, 0, 0, 1, 2)
            self.start = QSpinBox()
            self.end = QSpinBox()
            self.stride = QSpinBox()
            self.stride.setRange(1, 1000000)
            self.fps = QSpinBox()
            self.fps.setRange(1, 120)
            self.fps.setValue(int((vis_cfg or {}).get('video_export', {}).get('fps', 10)))
            for row, (label, control) in enumerate([
                ('Start slice', self.start), ('End slice (inclusive)', self.end),
                ('Slice step', self.stride), ('Frames / second', self.fps),
            ], 1):
                layout.addWidget(QLabel(label), row, 0)
                layout.addWidget(control, row, 1)
            self.stamp = QCheckBox('Add Z coordinate to video')
            self.stamp.setChecked(True)
            layout.addWidget(self.stamp, 5, 0, 1, 2)
            self.export = QPushButton('Export video...')
            self.cancel = QPushButton('Cancel')
            self.cancel.setEnabled(False)
            layout.addWidget(self.export, 6, 0)
            layout.addWidget(self.cancel, 6, 1)
            self.progress = QProgressBar()
            layout.addWidget(self.progress, 7, 0, 1, 2)
            self.status = QLabel('Ready')
            self.status.setWordWrap(True)
            layout.addWidget(self.status, 8, 0, 1, 2)
            self.export.clicked.connect(self.begin)
            self.cancel.clicked.connect(lambda: self.finish(cancelled=True))
            self.refresh_bounds()
            viewer.dims.events.range.connect(self.refresh_bounds)
            viewer.window._qt_window.installEventFilter(self)

        def refresh_bounds(self, event=None):
            if self.active:
                return
            maximum = max(0, int(viewer.dims.nsteps[0]) - 1)
            self.start.setRange(0, maximum)
            self.end.setRange(0, maximum)
            self.end.setValue(maximum)

        def begin(self):
            if viewer.dims.ndim != 3 or viewer.dims.ndisplay != 2 or tuple(viewer.dims.displayed) != (1, 2):
                QMessageBox.warning(self, 'Video export', 'Switch to 2D XY view with the Z slider first.')
                return
            default_dir = Path((vis_cfg or {}).get('video_export', {}).get('output_dir', 'exports/videos'))
            path, _ = QFileDialog.getSaveFileName(
                self, 'Save Z-sweep video', str(default_dir / 'z_sweep.mp4'),
                'MP4 video (*.mp4);;AVI video (*.avi)')
            if not path:
                return
            self.target = Path(path)
            if not self.target.suffix:
                self.target = self.target.with_suffix('.mp4')
                if self.target.exists() and QMessageBox.question(
                    self, 'Replace video?', f'Replace {self.target}?',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                ) != QMessageBox.Yes:
                    return
            if self.target.suffix.lower() not in ('.mp4', '.avi'):
                QMessageBox.warning(self, 'Video export', 'Choose an .mp4 or .avi filename.')
                return
            self.original_step = tuple(viewer.dims.current_step)
            self.steps = sweep_steps(self.start.value(), self.end.value(), self.stride.value())
            self.index = 0
            self.frame_size = None
            self.active = True
            self.export.setEnabled(False)
            self.cancel.setEnabled(True)
            for control in (self.start, self.end, self.stride, self.fps, self.stamp):
                control.setEnabled(False)
            self.progress.setRange(0, len(self.steps))
            self.progress.setValue(0)
            try:
                self.target.parent.mkdir(parents=True, exist_ok=True)
                # Preserve existing exports on failure or cancellation.
                self.temporary = self.target.with_name(
                    f'.{self.target.stem}.{uuid.uuid4().hex}.partial{self.target.suffix}')
                self.advance()
            except Exception as exc:
                self.fail(exc)

        def advance(self):
            viewer.dims.set_current_step(0, self.steps[self.index])
            self.deadline = time.monotonic() + 30
            # Yield to Qt for slicing and painting. FPS is independent of
            # render speed, so slow frames are never skipped.
            self.timer.start(100)

        def capture(self):
            if not self.active:
                return
            try:
                if viewer.dims.ndisplay != 2 or tuple(viewer.dims.displayed) != (1, 2):
                    raise RuntimeError('The display axes changed during export. Keep the view unchanged.')
                if any(layer.visible and not layer.loaded for layer in viewer.layers):
                    if time.monotonic() >= self.deadline:
                        raise RuntimeError('Timed out waiting for a slice to load.')
                    self.timer.start(50)
                    return
                if viewer.dims.current_step[0] != self.steps[self.index]:
                    raise RuntimeError('The Z slider changed during export. Keep the view unchanged.')
                rgba = viewer.screenshot(canvas_only=True, flash=False)
                frame = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                height, width = frame.shape[:2]
                # Pad odd sizes to meet MP4 encoder requirements.
                frame = cv2.copyMakeBorder(frame, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT)
                size = (frame.shape[1], frame.shape[0])
                if self.frame_size is None:
                    self.frame_size = size
                    codec = 'mp4v' if self.target.suffix.lower() == '.mp4' else 'MJPG'
                    self.writer = cv2.VideoWriter(
                        str(self.temporary), cv2.VideoWriter_fourcc(*codec), self.fps.value(), size)
                    if not self.writer.isOpened():
                        raise RuntimeError('Video encoder unavailable. Try AVI instead of MP4.')
                elif size != self.frame_size:
                    raise RuntimeError('Canvas size changed. Keep the window size unchanged during export.')
                if self.stamp.isChecked():
                    text = f'Z = {viewer.dims.point[0]:g}'
                    cv2.putText(frame, text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 1, cv2.LINE_AA)
                self.writer.write(frame)
                self.index += 1
                self.progress.setValue(self.index)
                self.status.setText(f'Exporting {self.index}/{len(self.steps)} frames')
                if self.index == len(self.steps):
                    self.finish()
                else:
                    self.advance()
            except Exception as exc:
                self.fail(exc)

        def finish(self, cancelled=False, error=None, restore=True):
            if not self.active:
                return
            self.active = False
            self.timer.stop()
            try:
                if self.writer is not None:
                    self.writer.release()
                    self.writer = None
                if self.temporary is not None and self.temporary.exists():
                    if cancelled or error:
                        self.temporary.unlink()
                    elif self.temporary.stat().st_size > 0:
                        self.temporary.replace(self.target)
                    else:
                        self.temporary.unlink()
                        error = 'The encoder produced an empty file.'
                elif not cancelled and not error:
                    error = 'The encoder did not create a video.'
            except OSError as exc:
                error = str(exc)
            finally:
                self.temporary = None
                if restore:
                    viewer.dims.current_step = self.original_step
                self.export.setEnabled(True)
                self.cancel.setEnabled(False)
                for control in (self.start, self.end, self.stride, self.fps, self.stamp):
                    control.setEnabled(True)
            if error:
                self.status.setText(f'Export failed: {error}')
                QMessageBox.critical(self, 'Video export failed', str(error))
            elif cancelled:
                self.status.setText('Cancelled; no video saved.')
            else:
                self.status.setText(f'Saved {self.target} ({len(self.steps) / self.fps.value():.1f} s)')

        def fail(self, error):
            self.finish(error=str(error))

        def eventFilter(self, watched, event):
            if event.type() == QEvent.Close:
                self.finish(cancelled=True, restore=False)
            return super().eventFilter(watched, event)

        def closeEvent(self, event):
            self.finish(cancelled=True)
            super().closeEvent(event)

    panel = VideoExportPanel()
    viewer.window.add_dock_widget(panel, name='Video export', area='right')
    return panel
