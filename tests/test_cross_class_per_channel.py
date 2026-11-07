import unittest

from src.core.stitcher import (match_soma_3d_iou, merge_soma_volumes_union, _merge_class,
                               suppress_cross_class_overlap, suppress_cross_class_per_channel)


def cell(track_id, cls, box, z):
    x1, y1, x2, y2 = box
    return {'track_id': track_id, 'class': cls, 'x1_3d': x1, 'y1_3d': y1, 'x2_3d': x2,
            'y2_3d': y2, 'z_min': z[0], 'z_max': z[1], 'per_z_boxes': {z[0]: list(box)},
            'observed_x1': x1, 'observed_y1': y1, 'observed_x2': x2, 'observed_y2': y2}


def phase_a(vols, channels):
    """Stage 4 Phase A as in run_inference.py, followed by the cross-class dedup."""
    merged = list(vols.get(channels[0], []))
    for cid in channels[1:]:
        pairs, _, unmatched = match_soma_3d_iou(merged, vols.get(cid, []), iou_thresh=0.15,
                                                iomin_thresh=0.5, z_pad=1)
        for a, b in pairs:
            merge_soma_volumes_union(a, b)
            a['class'] = _merge_class(a['class'], b['class'])
        merged = merged + unmatched
    return suppress_cross_class_overlap(merged, iou_thresh=0.5, z_pad=1)


class CrossClassPerChannelTest(unittest.TestCase):
    def test_removes_same_channel_neuron_and_logs_channel(self):
        vols = {'GFP': [cell('n', 'neuron_GFP', (0, 0, 30, 30), (5, 5)),
                        cell('far', 'neuron_GFP', (200, 200, 230, 230), (5, 5)),
                        cell('g', 'glia_GFP', (2, 2, 30, 30), (6, 6))]}
        rows = []
        suppress_cross_class_per_channel(vols, ['GFP', 'RFP'], iou_thresh=0.5, z_pad=1,
                                         decision_rows=rows)
        self.assertEqual([c['track_id'] for c in vols['GFP']], ['far', 'g'])
        self.assertNotIn('RFP', vols)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['stage'], 'channel_cross_class_suppression')
        self.assertEqual(rows[0]['channel'], 'GFP')
        self.assertEqual((rows[0]['track_id'], rows[0]['other_track_id']), ('n', 'g'))

    def test_does_not_compare_across_channels(self):
        vols = {'GFP': [cell('n', 'neuron_GFP', (0, 0, 30, 30), (5, 5))],
                'RFP': [cell('g', 'glia_RFP', (0, 0, 30, 30), (5, 5))]}
        suppress_cross_class_per_channel(vols, ['GFP', 'RFP'], iou_thresh=0.5, z_pad=1)
        self.assertEqual(len(vols['GFP']), 1)
        self.assertEqual(len(vols['RFP']), 1)

    def test_fixes_duplicate_hidden_by_cross_channel_merge(self):
        # ET10 GFP z=423: a GFP neuron and a GFP glia on one soma; a larger RFP glia
        # track matches the neuron first, so the merged cell is glia and the dedup after
        # Phase A sees two glia cells.
        def make():
            return {'GFP': [cell('n', 'neuron_GFP', (4645.8, 6382.0, 4681.2, 6410.8), (423, 423)),
                            cell('g', 'glia_GFP', (4649.9, 6378.7, 4677.3, 6405.9), (423, 423))],
                    'RFP': [cell('r', 'glia_RFP', (4620.9, 6339.9, 4702.7, 6424.8), (422, 424))]}
        self.assertEqual(len(phase_a(make(), ['GFP', 'RFP'])), 2)
        vols = make()
        suppress_cross_class_per_channel(vols, ['GFP', 'RFP'], iou_thresh=0.5, z_pad=1)
        final = phase_a(vols, ['GFP', 'RFP'])
        self.assertEqual(len(final), 1)
        self.assertTrue(final[0]['class'].startswith('glia'))

    def test_containment_removes_small_neuron_inside_glia(self):
        # Seam tile 316000_377200: GFP neuron z=396 inside a GFP glia z=397, IoU ~0.24.
        def make():
            return {'GFP': [cell('n', 'neuron_GFP', (5389.5, 5982.9, 5420.1, 6013.9), (396, 396)),
                            cell('g', 'glia_GFP', (5377.5, 5969.4, 5441.3, 6031.8), (397, 397))]}
        vols = make()
        suppress_cross_class_per_channel(vols, ['GFP'], iou_thresh=0.5, z_pad=1)
        self.assertEqual(len(vols['GFP']), 2)
        vols, rows = make(), []
        suppress_cross_class_per_channel(vols, ['GFP'], iou_thresh=0.5, z_pad=1,
                                         decision_rows=rows, contain_thresh=0.8)
        self.assertEqual([c['track_id'] for c in vols['GFP']], ['g'])
        self.assertEqual(rows[0]['reason'], 'inside_glia')

    def test_containment_keeps_large_neuron_around_small_glia(self):
        vols = {'GFP': [cell('n', 'neuron_GFP', (0, 0, 64, 64), (5, 5)),
                        cell('g', 'glia_GFP', (16, 16, 46, 46), (5, 5))]}
        suppress_cross_class_per_channel(vols, ['GFP'], iou_thresh=0.5, z_pad=1,
                                         contain_thresh=0.8)
        self.assertEqual(len(vols['GFP']), 2)


if __name__ == '__main__':
    unittest.main()
