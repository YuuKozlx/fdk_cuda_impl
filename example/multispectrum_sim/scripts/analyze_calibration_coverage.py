"""Count actual calibration rays; do not infer coverage from the fitted surface."""
import argparse
import json
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('samples')
p.add_argument('output')
a=p.parse_args()
d=np.loadtxt(a.samples,delimiter=',',skiprows=1)
d=d[d[:,1]+d[:,2]>=.5]
train=d[d[:,0].astype(int)%5!=0]
shape=(int(d[:,2].max()//2)+1,int(d[:,1].max()//5)+1)

def histogram(rows,dx=5,dy=2):
    result=np.zeros((int(d[:,2].max()//dy)+1,int(d[:,1].max()//dx)+1),dtype=int)
    np.add.at(result,(np.floor(rows[:,2]/dy).astype(int),np.floor(rows[:,1]/dx).astype(int)),1)
    return result

def stats(h):
    occupied=h[h>0]
    return dict(occupied=int((h>0).sum()),total_cells=h.size,empty=int((h==0).sum()),
                occupied_percent=float(100*(h>0).mean()),one_to_three=int(((h>0)&(h<=3)).sum()),
                counts_percentiles=dict(zip(['min','p25','median','p75','max'],np.percentile(occupied,[0,25,50,75,100]).tolist())))

all_h=histogram(d);train_h=histogram(train)
bins=[]
for low,high in [(0,2),(2,10),(10,20),(20,30),(30,40),(40,50),(50,52)]:
    h=train_h[low//2:high//2]
    missing=np.argwhere(h==0)
    bins.append(dict(al_range_mm=[low,high],**stats(h),empty_centers_mm=[[float((j+.5)*5),float((i+.5)*2+low)] for i,j in missing]))
report=dict(source=a.samples,samples=len(d),training_candidates=len(train),heldout=len(d)-len(train),
            ranges_mm=dict(pmma=[float(d[:,1].min()),float(d[:,1].max())],al=[float(d[:,2].min()),float(d[:,2].max())]),
            bin_mm=[5,2],grid_shape_al_pmma=shape,all=stats(all_h),training=stats(train_h),
            selected_training_samples=int(np.minimum(train_h,12).sum()),
            ray_types=dict(pmma_only=int(((d[:,1]>0)&(d[:,2]==0)).sum()),al_only=int(((d[:,1]==0)&(d[:,2]>0)).sum()),mixed=int(((d[:,1]>0)&(d[:,2]>0)).sum())),
            training_by_al_band=bins,counts_all=all_h.tolist(),counts_training=train_h.tolist(),
            training_1mm_grid=stats(histogram(train,1,1)))
with open(a.output,'w',encoding='utf-8') as f:json.dump(report,f,indent=2)
print(json.dumps({k:v for k,v in report.items() if not k.startswith('counts_') and k!='training_by_al_band'},indent=2))
for b in bins:print({k:v for k,v in b.items() if k!='empty_centers_mm'})
