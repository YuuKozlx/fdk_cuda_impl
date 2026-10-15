"""Evaluate all reconstructed slices; original labels are used only for metrics.
Images are exported separately by the native C++ plotting utility.
"""
import argparse,json
from pathlib import Path
import numpy as np
from brainweb_bone_bhc_experiment import erode

def main():
    ap=argparse.ArgumentParser();ap.add_argument('experiment',type=Path);ap.add_argument('--cases',nargs='+',default=['water-bhc-shared','bone-bhc-wfbp-shared-ramlak','2dcalbh-full']);ap.add_argument('--output',type=Path);args=ap.parse_args();r=args.experiment
    metadata=json.loads((r/'metrics.json').read_text());shape=tuple(metadata['dimensions_xyz'][::-1]);mu=metadata['mu_water_mm_inv']
    labels=np.memmap(r/'evaluation-labels-reconstruction-grid-u8.raw',dtype='u1',mode='r',shape=shape)
    mask=(labels>0)&(labels!=7)
    for _ in range(4):mask=erode(mask)
    bone=erode(erode(labels==7))
    reference=np.memmap(r/'mono-reference-spp50/reconstruction.raw',dtype='<f4',mode='r',shape=shape)
    sigma=4.;radius=12;t=np.arange(-radius,radius+1);kernel=np.exp(-.5*(t/sigma)**2);kernel/=kernel.sum()
    report={'reference':'70keV / 50 pixel rays / same WFBP RamLak; evaluation only','mu_water_mm_inv':mu,'analysis':'2mm XY Gaussian on error only, 2mm eroded non-bone mask; original images unfiltered','dimensions_xyz':list(shape[::-1]),'interbone_roi_xyz':[[200,312],[230,360],[36,116]],'results':{}}
    for name in args.cases:
        volume=np.memmap(r/name/'reconstruction.raw',dtype='<f4',mode='r',shape=shape)
        acc={region:np.zeros(6) for region in ['whole_volume','inferior_skull_base','interbone_roi','bone']};rows=[]
        for z in range(shape[0]):
            error=(volume[z].astype('f8')-reference[z])*1000/mu
            if not np.isfinite(error).all():raise ValueError(f'nonfinite {name} z={z}')
            low=error
            for axis in [0,1]:low=np.apply_along_axis(lambda row:np.convolve(np.pad(row,radius,mode='edge'),kernel,mode='valid'),axis,low)
            masks={'whole_volume':mask[z],'bone':bone[z]}
            if 36<=z<116:
                masks['inferior_skull_base']=mask[z]
                roi=np.zeros(shape[1:],bool);roi[230:360,200:312]=mask[z,230:360,200:312];masks['interbone_roi']=roi
            for region,m in masks.items():
                e,l=error[m],low[m];n=e.size
                if n:acc[region]+=np.array([n,e.sum(),e@e,l@l,np.minimum(l,0)@np.minimum(l,0),(e-l)@(e-l)])
            m=mask[z];rows.append({'z':z,'voxels':int(m.sum()),'lowfreq_rmse_HU':float(np.sqrt(np.mean(low[m]**2))) if m.any() else None})
        result={}
        for region,(n,total,ss,lowss,negss,highss) in acc.items():
            result[region]={'voxels':int(n),'mean_error_HU':total/n,'rmse_HU':float(np.sqrt(ss/n)),'lowfreq_rmse_HU':float(np.sqrt(lowss/n)),'dark_error_rms_HU':float(np.sqrt(negss/n)),'highfreq_rmse_HU':float(np.sqrt(highss/n))}
        result['by_slice']=rows;report['results'][name]=result
        print(name,json.dumps({k:v for k,v in result.items() if k!='by_slice'}),flush=True)
        (args.output or r/'2dcalbh-full/evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if __name__=='__main__':main()
