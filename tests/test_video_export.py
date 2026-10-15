"""Exercise real Qt controls and OpenCV encoding with a deterministic canvas."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from qtpy.QtWidgets import QApplication, QWidget
from src.utils.video_export import add_video_export_panel, sweep_steps


class Signal:
    def connect(self, callback):
        pass


class Dims:
    ndim = 3
    ndisplay = 2
    displayed = (1, 2)
    nsteps = (5, 60, 80)
    current_step = (2, 0, 0)
    events = SimpleNamespace(range=Signal())

    @property
    def point(self):
        return (500 + self.current_step[0], 0, 0)

    def set_current_step(self, axis, step):
        self.current_step = (step, 0, 0)


class VideoExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.target = Path(self.tmp.name) / 'movie.mp4'
        self.viewer = SimpleNamespace(
            dims=Dims(), layers=[], events=SimpleNamespace(close=Signal()),
            window=SimpleNamespace(_qt_window=QWidget(), add_dock_widget=lambda *a, **k: None),
            screenshot=self.screenshot,
        )
        self.captured = []
        self.panel = add_video_export_panel(self.viewer)
        self.panel.stamp.setChecked(False)

    def tearDown(self):
        if self.panel.active:
            self.panel.finish(cancelled=True)
        self.panel.deleteLater()
        self.tmp.cleanup()

    def screenshot(self, **kwargs):
        z = self.viewer.dims.current_step[0]
        self.captured.append(z)
        rgba = np.zeros((61, 81, 4), dtype=np.uint8)
        rgba[:, :, 0] = z * 45
        rgba[:, :, 3] = 255
        return rgba

    def begin(self):
        with patch('qtpy.QtWidgets.QFileDialog.getSaveFileName', return_value=(str(self.target), '')):
            self.panel.begin()
        self.panel.timer.stop()

    def test_sweep_endpoints(self):
        self.assertEqual(sweep_steps(0, 4, 3), [0, 3, 4])
        self.assertEqual(sweep_steps(4, 0, 3), [4, 1, 0])
        self.assertEqual(sweep_steps(2, 2), [2])
        with self.assertRaises(ValueError):
            sweep_steps(0, 4, 0)

    def test_encoded_reverse_sweep_matches_slices_and_restores_position(self):
        self.panel.start.setValue(4)
        self.panel.end.setValue(0)
        self.panel.stride.setValue(2)
        self.begin()
        while self.panel.active:
            self.panel.capture()
            self.panel.timer.stop()
        self.assertEqual(self.captured, [4, 2, 0])
        self.assertEqual(self.viewer.dims.current_step, (2, 0, 0))
        video = cv2.VideoCapture(str(self.target))
        try:
            self.assertTrue(video.isOpened())
            self.assertEqual(video.get(cv2.CAP_PROP_FPS), 10)
            frames = []
            while True:
                ok, frame = video.read()
                if not ok:
                    break
                self.assertEqual(frame.shape[:2], (62, 82))
                frames.append(float(frame[10:40, 10:40, 2].mean()))
            self.assertEqual(len(frames), 3)
            np.testing.assert_allclose(frames, [180, 90, 0], atol=8)
        finally:
            video.release()

    def test_cancel_preserves_existing_file_and_removes_partial(self):
        self.target.write_bytes(b'existing movie')
        self.begin()
        self.panel.capture()
        self.panel.timer.stop()
        self.panel.finish(cancelled=True)
        self.assertEqual(self.target.read_bytes(), b'existing movie')
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [self.target])
        self.assertEqual(self.viewer.dims.current_step, (2, 0, 0))

    def test_unloaded_layer_waits_before_capture(self):
        self.viewer.layers.append(SimpleNamespace(visible=True, loaded=False))
        self.begin()
        self.panel.capture()
        self.panel.timer.stop()
        self.assertEqual(self.captured, [])
        self.assertEqual(self.panel.index, 0)
        self.viewer.layers[0].loaded = True
        self.panel.capture()
        self.panel.timer.stop()
        self.assertEqual(self.captured, [0])

    def test_capture_failure_preserves_existing_file(self):
        self.target.write_bytes(b'existing movie')
        self.begin()
        self.viewer.screenshot = lambda **kwargs: (_ for _ in ()).throw(RuntimeError('render failed'))
        with patch('qtpy.QtWidgets.QMessageBox.critical'):
            self.panel.capture()
        self.assertFalse(self.panel.active)
        self.assertIn('render failed', self.panel.status.text())
        self.assertEqual(self.target.read_bytes(), b'existing movie')
        self.assertEqual(self.viewer.dims.current_step, (2, 0, 0))


if __name__ == '__main__':
    unittest.main()
