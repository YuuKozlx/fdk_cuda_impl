"""Continue a completed full-physics helical reconstruction with bone BHC.

Production executables perform all projection, reconstruction and segmentation.
NumPy only assembles the projection residual in chunks and evaluates volumes;
Matplot++ renders figures. Original labels are used exclusively for evaluation
and the independent monochromatic reference, never for correction segmentation.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time
import tomllib

import numpy as np
from brainweb_bone_bhc_experiment import dump_toml, erode


def mapped(path, shape, dtype='<f4'):
    if path.stat().st_size != int(np.prod(shape)) * np.dtype(dtype).itemsize:
        raise ValueError(f'Unexpected RAW size: {path}')
    return np.memmap(path, dtype=dtype, mode='r', shape=shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('exe', 'prepare', 'plotter', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--evaluate-only', action='store_true', help='Re-evaluate completed RAW results without recomputing projection/reconstruction')
    parser.add_argument('--singlepass', type=Path, help='Ideal shared-path bone correction executable; replaces two model projections')
    parser.add_argument('--native-images', action='store_true', help='Native bone/artefact BMP exports instead of slow Gnuplot image montages')
    args = parser.parse_args()
    output = args.output.resolve()
    base_path = output / 'water-bhc/run.toml'
    base = tomllib.loads(base_path.read_text(encoding='utf-8'))
    g = base['geometry']
    dims = tuple(g['reconstruction_volume_' + a] for a in 'xyz')
    spacing = np.array([g['reconstruction_voxel_' + a + '_mm'] for a in 'xyz'])
    center = np.array([g['reconstruction_offset_' + a + '_mm'] for a in 'xyz'])
    origin = center - (np.array(dims) - 1) * spacing / 2
    shape = dims[::-1]
    projection_shape = (g['views'], g['detector_v'], g['detector_u'])
    # Validate completed inputs before starting any dependent process.
    mapped(output / 'water-bhc/reconstruction.raw', shape)
    mapped(output / 'water-bhc/projection.raw.corrected.raw', projection_shape)
    if any(g['phantom_rotation_' + a + '_rad'] != 0 for a in 'xyz'):
        raise ValueError('This experiment evaluator requires an unrotated source phantom')
    if g['kind'] != 'flat_helical':
        raise ValueError('Expected flat_helical geometry')
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[3]
    exe = args.exe.resolve()
    env['PATH'] = os.pathsep.join([str(exe.parents[2]), str(root / 'out/vtk-env/Lib/site-packages/vtk.libs'),
                                 str(root / 'out/plot-runtime/gnuplot/bin'), str(root / 'out/plot-deps/x64-windows/bin'), env['PATH']])
    times = {}

    def run(name, command):
        print(f'Running {name}', flush=True)
        start = time.perf_counter()
        with (output / f'{name}.log').open('wb') as log:
            subprocess.run([str(x) for x in command], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        times[name] = time.perf_counter() - start
        (output / 'continuation-timings.json').write_text(json.dumps(times, indent=2))

    mono_file = output / 'mono70.spec'
    mono_file.write_text('# Format: YK-SPECTRUM 1.0\n# Energy[keV] N[keV cm^2 mAs]^-1 @ 1 meter bin_width[keV]\n70 1e8 1\n')

    def config(name, *, mono=False, segmented=False, mode='project', input_file=None):
        c = copy.deepcopy(base)
        d = output / name
        d.mkdir(exist_ok=True)
        c['workflow']['mode'] = mode
        p, r = c['projection'], c['reconstruction']
        p.update(output_file=(d / 'projection.raw').as_posix(), energy_output_file=(d / 'energy.raw').as_posix(),
                 air_output_file=(d / 'air.raw').as_posix())
        p['spectrum']['export_file'] = (d / 'spectrum.spec').as_posix()
        r.update(input_projection_file=(input_file or d / 'projection.raw').as_posix(),
                 output_volume_file=(d / 'reconstruction.raw').as_posix(), save_slices=False,
                 slice_prefix=(d / 'slice').as_posix())
        c['visualization'].update(enabled=False, interactive=False, snapshot_directory='')
        r['preprocessing']['enabled'] = segmented and not mono
        r['preprocessing']['scatter']['enabled'] = False
        p['scatter']['enabled'] = False
        p['quantum_noise']['enabled'] = False
        if segmented:
            p['detector_postprocess']['das']['electronic_noise_std_electrons'] = 0.
            p['label_volume'] = (output / 'segmentation/segmented-u8.raw').as_posix()
            p['materials'] = [
                dict(label=0, name='air', preset='mixture.air', density_g_cm3=.001205),
                dict(label=1, name='soft-water', preset='compound.water', density_g_cm3=1.03),
                dict(label=2, name='bone', preset='tissue.ncat_skull', density_g_cm3=1.61)]
            # Segmentation lives on the reconstruction grid, not the input label grid.
            for i, a in enumerate('xyz'):
                c['geometry']['volume_' + a] = dims[i]
                c['geometry']['voxel_' + a + '_mm'] = float(spacing[i])
                c['geometry']['phantom_offset_' + a + '_mm'] = float(center[i])
                c['geometry']['phantom_rotation_' + a + '_rad'] = 0.
        else:
            p['focal_spot'].update(enabled=False, defocus_enabled=False)
            p['detector_response']['enabled'] = False
            for effect in p['detector_postprocess'].values():
                effect['enabled'] = False
        if mono:
            p['spectrum'] = {'source': 'file'}
            p['spectrum_file'] = mono_file.as_posix()
        # Reconstruction and TV stay byte-for-value identical to the baseline.
        assert r.get('iterative') == base['reconstruction'].get('iterative')
        assert r.get('analytic') == base['reconstruction'].get('analytic')
        path = d / 'run.toml'
        path.write_text(dump_toml(c), encoding='utf-8')
        return path

    if not args.evaluate_only:
        run('segment', [args.prepare.resolve(), base_path, output / 'water-bhc/reconstruction.raw', output / 'segmentation'])
        if args.singlepass:
            run('model-singlepass', [args.singlepass.resolve(), base_path,
                                    output / 'segmentation/segmented-u8.raw',
                                    output / 'water-bhc/projection.raw.corrected.raw', output])
        else:
            poly_config = config('model-poly', segmented=True)
            run('model-poly', [exe, poly_config])
            run('model-water-bhc', [exe, '--profile-preprocess', poly_config])
            run('model-mono', [exe, config('model-mono', mono=True, segmented=True)])
            water = mapped(output / 'water-bhc/projection.raw.corrected.raw', projection_shape)
            model_water = mapped(output / 'model-poly/projection.raw.profile-preprocessed.raw', projection_shape)
            model_mono = mapped(output / 'model-mono/projection.raw', projection_shape)
            with (output / 'bone-corrected.raw').open('wb') as corrected_file, (output / 'model-residual.raw').open('wb') as residual_file:
                for begin in range(0, projection_shape[0], 16):
                    sl = slice(begin, begin + 16)
                    residual = model_mono[sl] - model_water[sl]
                    corrected = water[sl] + residual
                    if not np.isfinite(corrected).all():
                        raise ValueError('Nonfinite bone-corrected projection')
                    residual.astype('<f4').tofile(residual_file)
                    corrected.astype('<f4').tofile(corrected_file)
        run('bone-bhc', [exe, config('bone-bhc', mode='reconstruct', input_file=output / 'bone-corrected.raw')])
        run('mono-reference', [exe, config('mono-reference', mono=True, mode='project_and_reconstruct')])
    else:
        times.update(json.loads((output / 'continuation-timings.json').read_text()))

    # Nearest label sampling in physical coordinates. Evaluation only.
    source_dims = tuple(g['volume_' + a] for a in 'xyz')
    source_spacing = np.array([g['voxel_' + a + '_mm'] for a in 'xyz'])
    source_center = np.array([g['phantom_offset_' + a + '_mm'] for a in 'xyz'])
    source_origin = source_center - (np.array(source_dims) - 1) * source_spacing / 2
    source = mapped(Path(base['projection']['label_volume']), source_dims[::-1], 'u1')
    indices = [np.floor((origin[i] + np.arange(dims[i])*spacing[i] - source_origin[i])/source_spacing[i] + .5).astype(int) for i in range(3)]
    valid = [(indices[i] >= 0) & (indices[i] < source_dims[i]) for i in range(3)]
    labels = np.zeros(shape, dtype=np.uint8)
    xy = valid[1][:, None] & valid[0][None, :]
    for z in range(shape[0]):
        if valid[2][z]:
            plane = source[indices[2][z]][np.ix_(indices[1].clip(0, source_dims[1]-1), indices[0].clip(0, source_dims[0]-1))]
            labels[z] = np.where(xy, plane, 0)
    labels_path = output / 'evaluation-labels-reconstruction-grid-u8.raw'
    labels.tofile(labels_path)
    info = json.loads((output / 'segmentation/segmentation.json').read_text())
    mu = info['mu_water_mm_inv']
    reference = mapped(output / 'mono-reference/reconstruction.raw', shape)
    masks = {'all_tissue_including_interfaces': labels > 0, 'skull_including_interfaces': labels == 7,
             'soft_tissue': erode(erode(np.isin(labels, [1, 2, 3, 8]))), 'skull': erode(erode(labels == 7))}
    artifact_mask = (labels > 0) & (labels != 7)
    for _ in range(4):
        artifact_mask = erode(artifact_mask)
    sigma = 2 / spacing[0]
    radius = int(np.ceil(sigma * 3))
    t = np.arange(-radius, radius+1)
    kernel = np.exp(-.5*(t/sigma)**2)
    kernel /= kernel.sum()
    algorithm = base['reconstruction']['analytic']['pipeline'] if base['reconstruction']['type'] == 'analytic' else base['reconstruction']['iterative']['algorithm']
    metrics = dict(reference=f'70 keV mono, identical helical {algorithm} reconstruction settings (evaluation ONLY)',
                   simulation_profile='full-physics', source_labels=labels_path.as_posix(),
                   correction_method='ideal-shared-path' if args.singlepass else 'dual-reprojection-with-detector-effects',
                   reconstruction_algorithm=algorithm, mAs_per_view=base['projection']['mAs_per_view'],
                   source_volume=base['projection']['label_volume'], dimensions_xyz=dims,
                   spacing_mm=spacing.tolist(), origin_mm=origin.tolist(),
                   display_z=int(round((-64 - origin[2])/spacing[2])), mu_water_mm_inv=mu,
                   slice_artifact_policy='2 mm erosion; XY Gaussian sigma 2 mm on ERROR for analysis only',
                   mask_policy='All slices evaluated; soft/skull masks eroded 1 mm; interface masks un-eroded',
                   comparison_cases=[dict(name='water-bhc', title='Scatter + water BHC'),
                                     dict(name='bone-bhc', title='Scatter + water + bone BHC'),
                                     dict(name='mono-reference', title='Mono 70 keV: reference only')], results={})
    for name in ('water-bhc', 'bone-bhc'):
        volume = mapped(output / name / 'reconstruction.raw', shape)
        sums = {key: [0, 0., 0.] for key in masks}
        rows = []
        for z in range(shape[0]):
            error = (volume[z].astype(np.float64) - reference[z]) * (1000/mu)
            if not np.isfinite(error).all():
                raise ValueError(f'Nonfinite reconstruction {name}, z={z}')
            for key, mask in masks.items():
                e = error[mask[z]]
                sums[key][0] += e.size
                sums[key][1] += float(e.sum())
                sums[key][2] += float(np.dot(e, e))
            low = error
            for axis in (0, 1):
                low = np.apply_along_axis(lambda row: np.convolve(np.pad(row, radius, mode='edge'), kernel, mode='valid'), axis, low)
            mask = artifact_mask[z]
            entry = dict(z=z, world_z_mm=float(origin[2] + z*spacing[2]), voxels=int(mask.sum()))
            if mask.any():
                e, l = error[mask], low[mask]
                entry.update(rmse_HU=float(np.sqrt(np.mean(e*e))), lowfreq_rmse_HU=float(np.sqrt(np.mean(l*l))),
                             lowfreq_std_HU=float(l.std()), highfreq_std_HU=float((e-l).std()),
                             abs_error_gt_40_HU_fraction=float(np.mean(np.abs(e)>40)))
            rows.append(entry)
        result = {'artifact_by_slice': rows}
        for key, (n, total, squares) in sums.items():
            result[key] = dict(voxels=n, rmse_HU=float(np.sqrt(squares/n)), mean_error_HU=total/n,
                               std_error_HU=float(np.sqrt(max(0, squares/n - (total/n)**2))))
        metrics['results'][name] = result
    seg = mapped(output / 'segmentation/segmented-u8.raw', shape, 'u1')
    predicted, truth = seg == 2, labels == 7
    metrics['bone_segmentation_dice'] = float(2*np.count_nonzero(predicted & truth)/(predicted.sum()+truth.sum()))
    metrics['regional_artifacts'] = {}
    for region, bounds in (('whole_volume', (-1e6, 1e6)), ('inferior_skull_base', (-82, -42))):
        summary = {'world_z_range_mm': list(bounds)}
        for name, result in metrics['results'].items():
            rows = [r for r in result['artifact_by_slice'] if bounds[0] <= r['world_z_mm'] < bounds[1] and r['voxels']]
            n = sum(r['voxels'] for r in rows)
            summary[name] = {'voxels': n}
            for key in ('lowfreq_rmse_HU', 'lowfreq_std_HU', 'highfreq_std_HU'):
                summary[name][key] = float(np.sqrt(sum(r['voxels']*r[key]**2 for r in rows)/n)) if n else None
            summary[name]['abs_error_gt_40_HU_fraction'] = sum(r['voxels']*r['abs_error_gt_40_HU_fraction'] for r in rows)/n if n else None
        metrics['regional_artifacts'][region] = summary
    metrics['timings_seconds'] = times
    metrics_path = output / 'metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    if args.native_images:
        run('native-bone-window', [args.plotter.resolve(), '--bone-window', output])
        run('native-artifact-window', [args.plotter.resolve(), '--artifact-window', output])
    else:
        run('matplot', [args.plotter.resolve(), output])
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics['regional_artifacts'], indent=2), flush=True)


if __name__ == '__main__':
    main()
