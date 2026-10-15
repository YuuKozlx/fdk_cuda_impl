"""Generate a local PMMA/pure-Al phantom and a production simulator TOML.

No downloaded phantom, external source code or Python plotting dependency.
This first experiment uses known CAD labels, ideal primary measurements and
one sampled ray per pixel; it does not claim to validate measured scatter removal.
"""
import argparse
import copy
import json
from pathlib import Path
import tomllib
import numpy as np
from brainweb_bone_bhc_experiment import dump_toml

def evaluate(root):
    shape=(32,320,320)
    before=np.fromfile(root/'uncorrected.raw',dtype='<f4').reshape(shape)
    after=np.fromfile(root/'corrected.raw',dtype='<f4').reshape(shape)
    model=json.loads((root/'applied/calibration.json').read_text())
    labels=np.fromfile(root/'phantom-u8.raw',dtype='u1').reshape(shape)
    z,y,x=np.mgrid[:32,:320,:320];x=x-159.5;y=y-159.5
    # Only the central slab has complete angular detector coverage.
    mid=(z>=14)&(z<18)
    radius=np.sqrt((x+45)**2+y*y)
    masks={'PMMA_center':mid&(radius<30),'PMMA_rim':mid&(radius>85)&(radius<95),
           'Al_interior':mid&((x-100)**2+y*y<18**2)}
    report={'target':'zero-thickness tangent, not mono70 or HU','evaluation_z_slices':[14,15,16,17],'results':{}}
    volumes=[('uncorrected',before),('corrected_CAD',after)]
    if (root/'corrected-segmentation.raw').exists():
        volumes.append(('corrected_reconstructed_segmentation',np.fromfile(root/'corrected-segmentation.raw',dtype='<f4').reshape(shape)))
    for name,vol in volumes:
        data={k:{'mean_mu_mm':float(vol[m].mean()),'std_mu_mm':float(vol[m].std())} for k,m in masks.items()}
        data['PMMA_cupping_percent']=100*(data['PMMA_rim']['mean_mu_mm']-data['PMMA_center']['mean_mu_mm'])/data['PMMA_rim']['mean_mu_mm']
        report['results'][name]=data
    # A second, less ideal calibration trial: recover labels from the actual FDK reconstruction.
    segmentation=np.zeros(shape,dtype='u1');segmentation[before>.009]=1;segmentation[before>.04]=2
    segmentation.tofile(root/'reconstructed-segmentation-u8.raw')
    c=tomllib.loads((root/'scan.toml').read_text())
    c['projection']['label_volume']=(root/'reconstructed-segmentation-u8.raw').as_posix()
    (root/'segmentation-paths.toml').write_text(dump_toml(c),encoding='utf-8')
    report['segmentation_thresholds_mu_mm']=[.009,.04]
    report['central_slab_segmentation_label_agreement']=float((segmentation[mid]==labels[mid]).mean())
    report['target_coefficients']=[model['target_water_mu_mm'],model['target_bone_mu_mm']]
    (root/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('baseline',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--evaluate',action='store_true')
    args=p.parse_args()
    root=args.output.resolve()
    if args.evaluate:
        evaluate(root);return
    root.mkdir(parents=True,exist_ok=False)
    y,x=np.mgrid[:320,:320].astype(float);x-=159.5;y-=159.5
    image=np.zeros((320,320),dtype='u1')
    image[(x+45)**2+y**2<=110**2]=1
    image[(x-100)**2+y**2<=25**2]=2
    np.broadcast_to(image,(32,320,320)).copy().tofile(root/'phantom-u8.raw')
    c=tomllib.loads(args.baseline.read_text(encoding='utf-8-sig'))
    c['workflow']={'mode':'project_and_reconstruct'}
    g=c['geometry']
    g.update(kind='flat_cbct',views=360,views_per_turn=360,scan_range_rad=2*np.pi,
             pitch_mm_per_turn=0.,start_z_mm=0.,detector_u=1024,detector_v=32,
             reconstruction_detector_u=1024,reconstruction_detector_v=32)
    for axis,size in zip('xyz',(320,320,32)):
        g[f'volume_{axis}']=g[f'reconstruction_volume_{axis}']=size
        g[f'voxel_{axis}_mm']=g[f'reconstruction_voxel_{axis}_mm']=1.
        g[f'phantom_offset_{axis}_mm']=g[f'reconstruction_offset_{axis}_mm']=0.
        g[f'phantom_rotation_{axis}_rad']=0.
    q=c['projection']
    q.update(label_volume=(root/'phantom-u8.raw').as_posix(),
             output_file=(root/'projection.raw').as_posix(),
             energy_output_file=(root/'energy.raw').as_posix(),
             air_output_file=(root/'air.raw').as_posix())
    q['materials']=[dict(label=0,name='near-vacuum',preset='mixture.air',density_g_cm3=1e-9),
        dict(label=1,name='PMMA',preset='compound.pmma',density_g_cm3=1.19),
        dict(label=2,name='pure-aluminum',preset='gate.aluminium',density_g_cm3=2.7)]
    for section in ('quantum_noise','scatter','detector_response'):
        q[section]={'enabled':False}
    for section in q['detector_postprocess'].values():
        if isinstance(section,dict): section['enabled']=False
    q['focal_spot'].update(enabled=False,defocus_enabled=False)
    q['sampling']['samples_per_pixel']=1
    q['spectrum']['export_file']=(root/'spectrum.spec').as_posix()
    c['visualization']['enabled']=False
    c['reconstruction']={'type':'analytic','input_projection_file':q['output_file'],
        'output_volume_file':(root/'uncorrected.raw').as_posix(),'save_slices':False,
        'preprocessing':{'enabled':False},'analytic':{'pipeline':'fdk','filter':'ramlak','chunk_views':64}}
    (root/'scan.toml').write_text(dump_toml(c),encoding='utf-8')
    corrected=copy.deepcopy(c);corrected['workflow']['mode']='reconstruct'
    corrected['reconstruction'].update(input_projection_file=(root/'applied/bone-corrected.raw').as_posix(),output_volume_file=(root/'corrected.raw').as_posix())
    (root/'reconstruct-corrected.toml').write_text(dump_toml(corrected),encoding='utf-8')
    (root/'experiment.json').write_text(json.dumps({'shape_zyx':[32,320,320],
        'voxel_mm':1,'materials':['near-vacuum','PMMA 1.19 g/cm3','pure Al 2.7 g/cm3'],
        'phantom':'two separated cylinders, PMMA radius 110 mm at x=-45; Al radius 25 mm at x=100',
        'limitations':['CAD thickness, not reconstructed segmentation','ideal scatter-free calibration',
                       'pure aluminum is not AL6082 or validated equivalent human bone',
                       'tangent target has no assigned monoenergetic reference']},indent=2),encoding='utf-8')
    print(root/'scan.toml')

if __name__=='__main__':main()
