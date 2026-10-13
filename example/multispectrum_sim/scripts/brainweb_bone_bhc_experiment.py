"""Reproducible BrainWeb bone-BHC experiment, optionally with full physics.

Uses production CUDA projection/FDK and CUDA HU/segmentation stages. Python only
orchestrates files, adds a model residual, and measures/plots the comparison.
Not a clinical calibration or a new UI option. Requires numpy; plots use Matplot++.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import tomllib

import numpy as np


def toml_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return '[' + ', '.join(toml_value(v) for v in value) + ']'
    if isinstance(value, dict):
        return '{' + ', '.join(f'{k} = {toml_value(v)}' for k, v in value.items()) + '}'
    return repr(value)


def dump_toml(config):
    lines = []

    def table(data, prefix=''):
        if prefix:
            lines.extend(['', f'[{prefix}]'])
        for key, value in data.items():
            if not isinstance(value, dict):
                lines.append(f'{key} = {toml_value(value)}')
        for key, value in data.items():
            if isinstance(value, dict):
                table(value, f'{prefix}.{key}' if prefix else key)

    table(config)
    return '\n'.join(lines) + '\n'


def read_raw(path, shape):
    data = np.fromfile(path, dtype='<f4')
    if data.size != int(np.prod(shape)) or not np.isfinite(data).all():
        raise ValueError(f'Invalid float32 data: {path}')
    return data.reshape(shape)


def erode(mask):
    # One voxel 26-neighbour erosion, no SciPy dependency.
    padded = np.pad(mask, 1, constant_values=False)
    result = np.ones_like(mask)
    z, y, x = mask.shape
    for dz in range(3):
        for dy in range(3):
            for dx in range(3):
                result &= padded[dz:dz+z, dy:dy+y, dx:dx+x]
    return result


def summarize_regions(metrics):
    """Aggregate existing slice statistics; do not smooth or modify RAW images."""
    nz = len(metrics['results']['bone-bhc']['artifact_by_slice'])
    regions = {'whole_volume': (0, nz)}
    if nz > 96:
        regions['inferior_skull_base_z8_47'] = (8, 48)
    summary = {}
    for region, (begin, end) in regions.items():
        summary[region] = {'z_range': [begin, end]}
        for name in ('water-bhc', 'bone-bhc'):
            rows = [r for r in metrics['results'][name]['artifact_by_slice'] if begin <= r['z'] < end and r['voxels']]
            n = sum(r['voxels'] for r in rows)
            result = {'voxels': n}
            for key in ('lowfreq_rmse_HU', 'lowfreq_std_HU', 'highfreq_std_HU'):
                result[key] = float(np.sqrt(sum(r['voxels']*r[key]**2 for r in rows)/n)) if n else None
            result['abs_error_gt_40_HU_fraction'] = sum(r['voxels']*r['abs_error_gt_40_HU_fraction'] for r in rows)/n if n else None
            summary[region][name] = result
    metrics['regional_artifacts'] = summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--prepare', type=Path, required=True)
    parser.add_argument('--plotter', type=Path, required=True, help='multispectrum_plot_bone_bhc executable')
    parser.add_argument('--physics-config', type=Path, help='Use projection physics and scatter correction from this TOML; retain the experiment circular geometry')
    parser.add_argument('--labels', type=Path, required=True, help='181x217xN uint8 BrainWeb; use the complete volume by default')
    parser.add_argument('--center-z', type=int, help='Explicitly extract only a 33-slice slab centered here; omit for full volume')
    parser.add_argument('--display-z', type=int, help='Slice for the main comparison, without restricting computation')
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing data is never overwritten')
    args = parser.parse_args()
    exe, prepare, labels_path, output = (p.resolve() for p in (args.exe, args.prepare, args.labels, args.output))
    if output.exists():
        raise ValueError('Choose a new output directory')
    source_labels_path = labels_path
    source_labels = np.fromfile(labels_path, np.uint8).reshape(-1, 217, 181)
    begin, end = 0, len(source_labels)
    if args.center_z is not None:
        begin, end = args.center_z-16, args.center_z+17
        if begin < 0 or end > len(source_labels):
            raise ValueError('Selected center must have sixteen source slices on both sides')
    labels = source_labels[begin:end].copy()
    display_z = args.display_z if args.display_z is not None else len(labels)//2
    if not 0 <= display_z < len(labels):
        raise ValueError('Display slice outside reconstructed volume')
    output.mkdir(parents=True)
    if args.center_z is not None:
        labels_path = output / 'source-labels-u8.raw'
        labels.tofile(labels_path)
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[3]
    env['PATH'] = os.pathsep.join([str(exe.parents[2]), str(root / 'out/vtk-env/Lib/site-packages/vtk.libs'), env['PATH']])
    base_path = Path(__file__).resolve().parents[1] / 'configs/ui/brainweb-center33-fullchain.toml'
    base = tomllib.loads(base_path.read_text(encoding='utf-8'))
    base['geometry'].update(views=360, views_per_turn=360, detector_u=512, pixel_u_mm=.945)
    detector_v = max(96, int(np.ceil(len(labels)*base['geometry']['sdd_mm']/base['geometry']['sid_mm']/1.26*1.3/32))*32)
    base['geometry'].update(volume_z=len(labels), reconstruction_volume_z=len(labels), detector_v=detector_v)
    base['projection']['quantum_noise']['enabled'] = False
    base['projection']['label_volume'] = labels_path.as_posix()
    base['projection']['save_metadata'] = True
    base['reconstruction']['save_slices'] = False
    base['visualization'].update(enabled=False, interactive=False, snapshot_directory='')
    full_physics = args.physics_config is not None
    if full_physics:
        physical = tomllib.loads(args.physics_config.read_text(encoding='utf-8'))
        for key in ('spectrum', 'scatter', 'quantum_noise', 'focal_spot', 'sampling',
                    'detector_response', 'detector_postprocess', 'geometry_flux'):
            if key in physical['projection']:
                base['projection'][key] = copy.deepcopy(physical['projection'][key])
        base['reconstruction']['preprocessing']['scatter'] = copy.deepcopy(physical['reconstruction']['preprocessing']['scatter'])
        for section in (base['projection']['scatter'], base['reconstruction']['preprocessing']['scatter']):
            path = Path(section['model_file'])
            if not path.is_absolute():
                path = args.physics_config.resolve().parent / path
            section['model_file'] = path.resolve().as_posix()
        if not base['projection']['scatter']['enabled'] or not base['reconstruction']['preprocessing']['scatter']['enabled']:
            raise ValueError('Full-physics experiment requires scatter simulation and correction enabled')
    # Keep water BHC identical in measured and segmented-model paths.
    times = {}
    mono_file = output / 'mono70.spec'
    mono_file.write_text('# Format: YK-SPECTRUM 1.0\n# Energy[keV] N[keV cm^2 mAs]^-1 @ 1 meter bin_width[keV]\n70 1e8 1\n')

    def run(name, command):
        start = time.perf_counter()
        print(f'Running {name}', flush=True)
        with (output / f'{name}.log').open('wb') as log:
            subprocess.run([str(x) for x in command], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        times[name] = time.perf_counter() - start
        (output / 'timings.json').write_text(json.dumps(times, indent=2))

    def config(name, *, mono=False, bhc=True, segmented=False, mode='project_and_reconstruct', input_file=None, scatter_correction=None):
        c = copy.deepcopy(base)
        d = output / name
        d.mkdir()
        c['workflow']['mode'] = mode
        p, r = c['projection'], c['reconstruction']
        p.update(output_file=(d / 'projection.raw').as_posix(), energy_output_file=(d / 'energy.raw').as_posix(),
                 air_output_file=(d / 'air.raw').as_posix())
        p['spectrum']['export_file'] = (d / 'spectrum.spec').as_posix()
        r.update(input_projection_file=str(input_file or d / 'projection.raw').replace('\\', '/'),
                 output_volume_file=(d / 'reconstruction.raw').as_posix(), slice_prefix=(d / 'slice').as_posix())
        r['preprocessing']['enabled'] = bhc
        if full_physics:
            r['preprocessing']['scatter']['air_projection_file'] = (output / 'water-bhc/air.raw').as_posix()
        if scatter_correction is not None:
            r['preprocessing']['scatter']['enabled'] = scatter_correction
        # Reprojection excludes scatter/noise, but retains the measurement's
        # deterministic blur/lag in the full-physics case. Otherwise a sharp
        # correction residual overcorrects blurred bone boundaries.
        if mono or segmented:
            p['scatter']['enabled'] = False
            p['quantum_noise']['enabled'] = False
            if segmented and full_physics:
                p['detector_postprocess']['das']['electronic_noise_std_electrons'] = 0.
            else:
                p['focal_spot'].update(enabled=False, defocus_enabled=False)
                p['detector_response']['enabled'] = False
                for effect in p['detector_postprocess'].values():
                    effect['enabled'] = False
            r['preprocessing']['scatter']['enabled'] = False
        if mono:
            p['spectrum'] = {'source': 'file'}
            p['spectrum_file'] = mono_file.as_posix()
        if segmented:
            p['label_volume'] = (output / 'segmentation/segmented-u8.raw').as_posix()
            p['materials'] = [
                {'label': 0, 'name': 'air', 'preset': 'mixture.air', 'density_g_cm3': .001205},
                {'label': 1, 'name': 'soft-water', 'preset': 'compound.water', 'density_g_cm3': 1.03},
                {'label': 2, 'name': 'bone', 'preset': 'tissue.ncat_skull', 'density_g_cm3': 1.61}]
        path = d / 'run.toml'
        path.write_text(dump_toml(c), encoding='utf-8')
        return path

    measured_config = config('water-bhc')
    run('water-bhc', [exe, measured_config])
    run('segment', [prepare, measured_config, output / 'water-bhc/reconstruction.raw', output / 'segmentation'])
    run('mono-reference', [exe, config('mono-reference', mono=True, bhc=False)])
    run('uncorrected', [exe, config('uncorrected', mode='reconstruct', bhc=False,
                                   input_file=output / 'water-bhc/projection.raw')])
    if full_physics:
        # Separate input filename prevents overwriting the scatter+water cache.
        import shutil
        water_only_input = output / 'water-only-input.raw'
        shutil.copyfile(output / 'water-bhc/projection.raw', water_only_input)
        run('water-only', [exe, config('water-only', mode='reconstruct', scatter_correction=False, input_file=water_only_input)])
    model_config = config('model-poly', segmented=True, mode='project')
    run('model-poly', [exe, model_config])
    run('model-water-bhc', [exe, '--profile-preprocess', model_config])
    run('model-mono', [exe, config('model-mono', mono=True, bhc=False, segmented=True, mode='project')])
    shape = (360, detector_v, 512)
    water = read_raw(output / 'water-bhc/projection.raw.corrected.raw', shape)
    model_water = read_raw(output / 'model-poly/projection.raw.profile-preprocessed.raw', shape)
    model_mono = read_raw(output / 'model-mono/projection.raw', shape)
    # q is the SAME water correction in both terms: no repeated water correction.
    # p_corrected = q(p_measured) + p_mono(segmentation) - q(p_poly(segmentation)).
    residual = model_mono - model_water
    corrected = water + residual
    if not np.isfinite(corrected).all():
        raise ValueError('Nonfinite corrected projection')
    residual.astype('<f4').tofile(output / 'model-residual.raw')
    corrected.astype('<f4').tofile(output / 'bone-corrected.raw')
    run('bone-bhc', [exe, config('bone-bhc', mode='reconstruct', bhc=False, input_file=output / 'bone-corrected.raw')])
    reference_info = json.loads((output / 'segmentation/segmentation.json').read_text())
    mu_water = reference_info['mu_water_mm_inv']
    reference = read_raw(output / 'mono-reference/reconstruction.raw', labels.shape)
    masks = {'soft_tissue': erode(np.isin(labels, [1, 2, 3, 8])), 'skull': erode(labels == 7),
             'all_tissue_including_interfaces': labels > 0, 'skull_including_interfaces': labels == 7}
    for mask in masks.values():
        mask[:8] = False
        mask[-8:] = False
        if not mask.any():
            raise ValueError('Empty evaluation mask')
    metrics = {'reference': '70 keV monochromatic FDK using original labels (evaluation ONLY)',
               'simulation_profile': 'full-physics' if full_physics else 'noiseless-primary',
               'physics_config': str(args.physics_config.resolve()) if full_physics else None,
               'source_volume': str(source_labels_path), 'source_z_range': [begin, end], 'display_z': display_z,
               'evaluation_z_indices': [8, len(labels)-8],
               'slice_artifact_policy': 'non-air non-bone mask eroded twice; Gaussian sigma=2 pixels XY on error for analysis only; no smoothing of reconstructions',
               'mask_policy': 'soft_tissue and skull eroded one voxel; including_interfaces masks not eroded; all masks exclude first/last eight slices',
               'mu_water_mm_inv': mu_water, 'segmentation_thresholds_hu': [-500, 300], 'results': {}}
    images = {}
    soft_all = erode(erode((labels > 0) & (labels != 7)))
    t = np.arange(-6, 7, dtype=np.float64)
    kernel = np.exp(-t*t/8)
    kernel /= kernel.sum()
    ref_proj = read_raw(output / 'mono-reference/projection.raw', shape)
    ray_mask = ref_proj > .1
    cases = [('uncorrected', read_raw(output / 'water-bhc/projection.raw', shape)),
             ('water-bhc', water), ('bone-bhc', corrected)]
    if full_physics:
        cases.insert(1, ('water-only', read_raw(output / 'water-only-input.raw.corrected.raw', shape)))
    for name, projection in cases:
        volume = read_raw(output / name / 'reconstruction.raw', labels.shape)
        images[name] = volume
        data = {'projection_rmse': float(np.sqrt(np.mean((projection[ray_mask] - ref_proj[ray_mask]) ** 2)))}
        error_hu = (volume - reference) * (1000 / mu_water)
        smooth = error_hu.astype(np.float64)
        for axis in (1, 2):
            smooth = np.apply_along_axis(lambda row: np.convolve(np.pad(row, 6, mode='edge'), kernel, mode='valid'), axis, smooth)
        data['artifact_by_slice'] = []
        for z in range(len(labels)):
            mask = soft_all[z]
            entry = {'z': z, 'voxels': int(mask.sum())}
            if mask.any():
                e = error_hu[z][mask].astype(np.float64)
                low = smooth[z][mask]
                entry.update(rmse_HU=float(np.sqrt(np.mean(e*e))), lowfreq_rmse_HU=float(np.sqrt(np.mean(low*low))),
                             lowfreq_std_HU=float(low.std()), highfreq_std_HU=float((e-low).std()),
                             abs_error_gt_40_HU_fraction=float(np.mean(np.abs(e)>40)))
            data['artifact_by_slice'].append(entry)
        data['full_volume_including_z_edges'] = {'rmse_HU': float(np.sqrt(np.mean(error_hu[labels>0].astype(np.float64)**2)))}
        for region, mask in masks.items():
            e = error_hu[mask].astype(np.float64)
            data[region] = {'voxels': int(e.size), 'rmse_HU': float(np.sqrt(np.mean(e**2))),
                            'mean_error_HU': float(e.mean()), 'std_error_HU': float(e.std())}
        metrics['results'][name] = data
    seg = np.fromfile(output / 'segmentation/segmented-u8.raw', np.uint8).reshape(labels.shape)
    truth_bone, predicted = labels == 7, seg == 2
    metrics['bone_segmentation_dice'] = float(2 * np.count_nonzero(truth_bone & predicted) / (truth_bone.sum() + predicted.sum()))
    metrics['source_sha256'] = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    metrics['source_labels'] = labels_path.as_posix()
    metrics['timings_seconds'] = times
    summarize_regions(metrics)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    run('matplot', [args.plotter.resolve(), output])
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
