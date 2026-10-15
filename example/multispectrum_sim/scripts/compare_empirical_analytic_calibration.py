"""Evaluate empirical calibration against spectral integration, without refitting."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('case',type=Path);a=p.parse_args();root=a.case
reference=json.loads((root/'analytic-reference.json').read_text())
weights=np.array(reference['energy_weights'])
m1=np.array(reference['first']['mu_mm']);m2=np.array(reference['second']['mu_mm'])
x,y=np.meshgrid(np.arange(881)*.25,np.arange(501)*.1)
signal=np.zeros_like(x)
for w,u,v in zip(weights,m1,m2):signal+=w*np.exp(-u*x-v*y)
truth=-np.log(signal)
analytic_slopes=np.array([weights@m1,weights@m2])
coverage=np.array(json.loads((root/'coverage.json').read_text())['counts_training'])
counts=coverage[(y/2).astype(int),(x/5).astype(int)]

def summary(error,mask):
    z=error[mask];ix=np.flatnonzero(mask)[np.argmax(np.abs(z))];b,w=np.unravel_index(ix,x.shape)
    return dict(count=len(z),bias=float(z.mean()),rmse=float(np.sqrt(np.mean(z*z))),
                p95_abs=float(np.percentile(np.abs(z),95)),max_abs=float(np.abs(z).max()),
                worst_thickness_mm=[float(x[b,w]),float(y[b,w])])

report={'reference':'Same 120 kVp / 6 mm Al spectrum, energy-integrating weights, PMMA 1.19 / pure Al 2.7 g/cc',
        'grid':{'first_range_mm':[0,220],'second_range_mm':[0,50],'step_mm':[.25,.1],'points':int(x.size)},
        'analytic_zero_thickness_slopes_mm':analytic_slopes.tolist(),'fits':{}}
for name in ['fitted-matched','fitted-segmentation']:
    fit=json.loads((root/name/'fit.json').read_text());model=json.loads((root/name/'calibration.json').read_text())
    aa,b,c,d,f=fit['coefficients_a_b_c_d_f']
    prediction=-np.logaddexp(np.log(aa)-b*x-c*y,np.log1p(-aa)-d*x-f*y)
    delta=prediction-truth
    target=model['target_water_mu_mm']*x+model['target_bone_mu_mm']*y
    q=y/model['bone_step_mm'];row=np.minimum(q.astype(int),model['bone_rows']-2);t=q-row
    table=np.array(model['target_log']).reshape(model['bone_rows'],model['projection_columns'])
    low=np.array(model['minimum_log']);step=np.array(model['log_step'])
    def interpolate(r):
        index=(truth-low[r])/step[r];i=np.clip(index,0,model['projection_columns']-2).astype(int)
        return table[r,i]+(index-i)*(table[r,i+1]-table[r,i])
    corrected=interpolate(row)*(1-t)+interpolate(row+1)*t
    residual=corrected-target
    analytic_target=analytic_slopes[0]*x+analytic_slopes[1]*y
    # Same target coefficients isolate the calibration surface from target normalization.
    valid=truth>=.1
    masks={'whole':np.ones_like(x,dtype=bool),'covered_bins':counts>0,'empty_bins':counts==0,
           'weak_bins_1_to_3':(counts>0)&(counts<=3),'thin_al_0_to_10mm':y<10,
           'al_10_to_30mm':(y>=10)&(y<30),'al_30_to_50mm':y>=30}
    result={'response_log_error':{k:summary(delta,v) for k,v in masks.items()},
            'correction_error_same_linear_target':{k:summary(residual,v) for k,v in masks.items()},
            'relative_response_percent_p_ge_0p1':summary(100*delta/np.maximum(truth,1e-30),valid),
            'relative_transmission_percent':summary(100*np.expm1(-delta),masks['whole']),
            'relative_correction_percent_target_ge_0p1':summary(100*residual/np.maximum(target,1e-30),target>=.1),
            'fitted_target_slopes_mm':[model['target_water_mu_mm'],model['target_bone_mu_mm']],
            'target_slope_bias_percent':((np.array([model['target_water_mu_mm'],model['target_bone_mu_mm']])/analytic_slopes-1)*100).tolist()}
    result['correction_error_analytic_tangent_target']=summary(corrected-analytic_target,masks['whole'])
    result['relative_correction_percent_analytic_target_ge_0p1']=summary(100*(corrected-analytic_target)/np.maximum(analytic_target,1e-30),analytic_target>=.1)
    report['fits'][name]=result
    if name=='fitted-segmentation':
        plot={'x_mm':x[::10,::4].tolist(),'y_mm':y[::10,::4].tolist(),
              'response_error':delta[::10,::4].tolist(),'correction_error':residual[::10,::4].tolist()}
        (root/'analytic-error-map.json').write_text(json.dumps(plot),encoding='utf-8')
(root/'analytic-comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
