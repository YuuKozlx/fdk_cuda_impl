import unittest
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from report_material_roi import measure


class RoiTests(unittest.TestCase):
    def test_excludes_boundary_and_empty_regions(self):
        labels = np.zeros((9, 9, 9), dtype=np.uint8)
        labels[2:7, 2:7, 2:7] = 1
        volume = np.ones(labels.shape)
        volume[3:6, 3:6, 3:6] = 2
        result = measure(volume, labels, 1, 1)
        self.assertEqual(result['1']['count'], 27)
        self.assertEqual(result['1']['mean_mm_inv'], 2)

    def test_rejects_nonfinite(self):
        volume = np.zeros((9, 9, 9))
        volume[4, 4, 4] = np.nan
        with self.assertRaises(ValueError):
            measure(volume, np.zeros(volume.shape), 1, 1)

    def test_cli_reference_and_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.ones((9,9,9), dtype=np.uint8).tofile(root/'labels.raw')
            np.full((9,9,9), 0.02, dtype='<f4').tofile(root/'volume.raw')
            text = """schema_version=2
[workflow]
mode='project_and_reconstruct'
[projection]
label_volume='labels.raw'
[[projection.materials]]
label=1
name='water'
formula='H2O'
[geometry]
kind='flat_cbct'
volume_x=9
volume_y=9
volume_z=9
voxel_x_mm=1
voxel_y_mm=1
voxel_z_mm=1
[reconstruction]
type='analytic'
output_volume_file='volume.raw'
[reconstruction.analytic]
pipeline='fdk'
"""
            config = root/'test.toml'
            config.write_text(text)
            reference = root/'reference.json'
            command = [sys.executable, str(Path(__file__).with_name('report_material_roi.py')),
                       str(config), '--reference', str(reference), '--fixed-radius-mm', '2']
            for expected, status in [(0.02, 0), (0.03, 1)]:
                reference.write_text(json.dumps({'labels':{'1':{
                    'mean_mm_inv':expected, 'tolerance_mm_inv':0.0001}}}))
                run = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(run.returncode, status, run.stderr)
            report = json.loads((root/'volume.roi.json').read_text())
            self.assertEqual(report['fixed_roi']['z_range_mm_inv'], 0)
            config.write_text(text.replace('[geometry]', '[geometry]\nphantom_offset_z_mm=1'))
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
