"""Build an inline diagnostic visualization for the Cho RAW calibration run."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np


ROOT = Path(r"G:\Code\fanproj\fdk-test")
RESULT = ROOT / "out" / "cho_raw_fixed_result.json"
POINTS = ROOT / "out" / "cho_raw_fixed_result.points.npy"
OUTPUT = Path(r"C:\Users\zlx\.codex\visualizations\2026\09\24\01a0d148-c68e-7a01-b5af-e14f723c2d74\cho-calibration-diagnostics.html")


def compact(values, digits=4):
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def main():
    report = json.loads(RESULT.read_text(encoding="utf-8"))
    points = np.load(POINTS)
    diag = report["frame_diagnostics"]
    payload = {
        "points": compact(points, 3),
        "split": [bool(x.get("split_used")) for x in diag],
        "components": [x["raw_component_count"] for x in diag],
        "areaMax": [x["component_area_max"] for x in diag],
        "prediction": compact([x.get("prediction_max_error_px") or 0 for x in diag], 4),
        "reprojection": compact(report["per_view_reprojection_rmse_px"], 5),
        "estimate": {
            "offsetU": abs(report["shared_offset_px"][0]),
            "offsetV": abs(report["shared_offset_px"][1]),
            "sdd": report["shared_sdd_mm"],
            "rmse": report["joint_reprojection_rmse_px"],
        },
        "truth": {"offsetU": 10.0, "offsetV": 20.0, "sdd": 770.0},
    }
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    fragment = r'''<div id="cho-calibration-viz">
  <h2>Cho 双圆环标定诊断</h2>
  <div class="viz-grid cho-summary">
    <div class="card viz-stat"><div class="text-muted">有效帧 / 分割帧</div><div class="viz-stat-value tabular-nums" id="summary-frames"></div><div class="text-small text-muted">24 个质心全部保留</div></div>
    <div class="card viz-stat"><div class="text-muted">联合重投影 RMSE</div><div class="viz-stat-value tabular-nums" id="summary-rmse"></div><div class="text-small text-muted">像素</div></div>
    <div class="card viz-stat"><div class="text-muted">SDD 估计 / 真值</div><div class="viz-stat-value tabular-nums" id="summary-sdd"></div><div class="text-small text-destructive">深度尺度未恢复</div></div>
  </div>
  <div class="viz-controls">
    <label class="form-label" for="frame-slider">帧 <span class="tabular-nums" id="frame-value">0</span>
      <input class="form-range" id="frame-slider" type="range" min="0" max="359" value="0" step="1">
    </label>
    <span class="viz-badge" id="frame-status">普通检测</span>
  </div>
  <section><h3>钢珠轨迹与当前帧索引</h3><div id="trajectory-chart"></div></section>
  <section><h3>逐帧检测与拟合诊断</h3><div id="diagnostic-chart"></div></section>
  <section><h3>固定几何参数：估计值与真值</h3><div id="parameter-chart"></div></section>
  <div class="sr-only" id="cho-description">360 帧双圆环钢珠轨迹、粘连分割状态、逐帧预测与重投影误差，以及 offset 和 SDD 参数真值对比。</div>
</div>
<style>
#cho-calibration-viz{width:100%;color:var(--foreground)}
#cho-calibration-viz .cho-summary{margin-bottom:16px}
#cho-calibration-viz section{margin-top:22px}
#cho-calibration-viz .viz-controls{margin-top:14px}
#cho-calibration-viz .form-label{min-width:min(100%,420px)}
#cho-calibration-viz .chart-svg{display:block;width:100%;height:auto;color:var(--foreground)}
#cho-calibration-viz .axis text,#cho-calibration-viz .axis-title,#cho-calibration-viz .direct-label{fill:var(--foreground);font-size:12px}
#cho-calibration-viz .axis path,#cho-calibration-viz .axis line,#cho-calibration-viz .frame{stroke:var(--border)}
#cho-calibration-viz .grid line{stroke:var(--border);stroke-opacity:.45}
#cho-calibration-viz .grid path{display:none}
#cho-calibration-viz .bead-path{fill:none;stroke-width:1;opacity:.28}
#cho-calibration-viz .current-point{stroke:var(--card);stroke-width:1.5}
#cho-calibration-viz .split-band{fill:var(--viz-series-4);opacity:.10}
#cho-calibration-viz .legend-row{display:flex;flex-wrap:wrap;gap:12px;margin:4px 0 8px}
#cho-calibration-viz .legend-row button{background:transparent;border:0;color:var(--foreground);padding:4px;display:inline-flex;align-items:center;gap:6px}
#cho-calibration-viz .swatch{display:inline-block;width:18px;height:3px;background:var(--swatch)}
#cho-calibration-viz .param-label{fill:var(--foreground);font-size:12px}
#cho-calibration-viz .param-track{fill:var(--muted)}
#cho-calibration-viz .param-truth{stroke:var(--foreground);stroke-width:2}
@media(max-width:560px){#cho-calibration-viz .cho-summary{grid-template-columns:1fr}}
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {
const root=document.getElementById('cho-calibration-viz');
const data=__DATA__;
const colors=['var(--viz-series-1)','var(--viz-series-2)','var(--viz-series-3)','var(--viz-series-4)','var(--viz-series-5)','var(--viz-series-6)'];
root.querySelector('#summary-frames').textContent=`360 / ${data.split.filter(Boolean).length}`;
root.querySelector('#summary-rmse').textContent=data.estimate.rmse.toFixed(4);
root.querySelector('#summary-sdd').textContent=`${data.estimate.sdd.toFixed(1)} / ${data.truth.sdd.toFixed(0)} mm`;
const slider=root.querySelector('#frame-slider'), frameValue=root.querySelector('#frame-value'), frameStatus=root.querySelector('#frame-status');
function baseSvg(container,height,label){
  const node=root.querySelector(container); node.innerHTML='';
  const width=Math.max(320,node.clientWidth||720);
  return {width,svg:d3.select(node).append('svg').attr('class','chart-svg').attr('viewBox',`0 0 ${width} ${height}`).attr('role','img').attr('aria-label',label)};
}
function drawTrajectory(frame=+slider.value){
  const {width,svg}=baseSvg('#trajectory-chart',360,'24 颗钢珠在探测器上的完整轨迹及选中帧位置');
  const m={t:12,r:18,b:48,l:64}, w=width-m.l-m.r,h=360-m.t-m.b;
  const all=data.points.flat();
  const x=d3.scaleLinear().domain(d3.extent(all,d=>d[0])).nice().range([m.l,m.l+w]);
  const y=d3.scaleLinear().domain(d3.extent(all,d=>d[1])).nice().range([m.t,m.t+h]);
  svg.append('rect').attr('class','frame').attr('x',m.l).attr('y',m.t).attr('width',w).attr('height',h).attr('fill','none');
  svg.append('g').attr('class','axis').attr('transform',`translate(0,${m.t+h})`).call(d3.axisBottom(x).ticks(width<500?4:7));
  svg.append('g').attr('class','axis').attr('transform',`translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6));
  svg.append('text').attr('class','axis-title').attr('x',m.l+w/2).attr('y',350).attr('text-anchor','middle').text('u / pixel');
  svg.append('text').attr('class','axis-title').attr('transform',`translate(16,${m.t+h/2}) rotate(-90)`).attr('text-anchor','middle').text('v / pixel');
  const line=d3.line().x(d=>x(d[0])).y(d=>y(d[1]));
  for(let bead=0;bead<24;bead++){
    const series=data.points.map(p=>p[bead]);
    svg.append('path').datum(series).attr('class','bead-path').attr('stroke',colors[bead%6]).attr('d',line);
  }
  svg.selectAll('.current-point').data(data.points[frame].map((p,i)=>({p,i}))).enter().append('circle')
    .attr('class','current-point').attr('cx',d=>x(d.p[0])).attr('cy',d=>y(d.p[1])).attr('r',4).attr('fill',d=>colors[d.i%6]);
  svg.selectAll('.index-label').data(data.points[frame].map((p,i)=>({p,i}))).enter().append('text')
    .attr('class','direct-label').attr('x',d=>x(d.p[0])+5).attr('y',d=>y(d.p[1])-5).text(d=>d.i);
}
function drawDiagnostics(frame=+slider.value){
  const {width,svg}=baseSvg('#diagnostic-chart',330,'360 帧预测误差、重投影误差和粘连分割区间');
  const m={t:14,r:18,b:48,l:64}, w=width-m.l-m.r,h=330-m.t-m.b;
  const x=d3.scaleLinear().domain([0,359]).range([m.l,m.l+w]);
  const ymax=d3.max([...data.prediction,...data.reprojection])*1.08;
  const y=d3.scaleLinear().domain([0,ymax]).nice().range([m.t+h,m.t]);
  let start=null;
  data.split.forEach((v,i)=>{if(v&&start===null)start=i;if(start!==null&&(!v||i===359)){const end=v&&i===359?i:i-1;svg.append('rect').attr('class','split-band').attr('x',x(start)).attr('y',m.t).attr('width',Math.max(1,x(end+1)-x(start))).attr('height',h);start=null;}});
  svg.append('g').attr('class','grid').attr('transform',`translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-w).tickFormat(''));
  svg.append('rect').attr('class','frame').attr('x',m.l).attr('y',m.t).attr('width',w).attr('height',h).attr('fill','none');
  svg.append('g').attr('class','axis').attr('transform',`translate(0,${m.t+h})`).call(d3.axisBottom(x).ticks(width<500?4:8));
  svg.append('g').attr('class','axis').attr('transform',`translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5));
  svg.append('text').attr('class','axis-title').attr('x',m.l+w/2).attr('y',320).attr('text-anchor','middle').text('frame');
  svg.append('text').attr('class','axis-title').attr('transform',`translate(16,${m.t+h/2}) rotate(-90)`).attr('text-anchor','middle').text('error / pixel');
  const line=d3.line().x((d,i)=>x(i)).y(d=>y(d));
  svg.append('path').datum(data.prediction).attr('fill','none').attr('stroke',colors[0]).attr('stroke-width',1.5).attr('d',line);
  svg.append('path').datum(data.reprojection).attr('fill','none').attr('stroke',colors[2]).attr('stroke-width',1.5).attr('d',line);
  svg.append('line').attr('x1',x(frame)).attr('x2',x(frame)).attr('y1',m.t).attr('y2',m.t+h).attr('stroke','var(--foreground)').attr('stroke-width',1);
  const legend=d3.select(root.querySelector('#diagnostic-chart')).insert('div',':first-child').attr('class','legend-row');
  [['预测最大偏差',colors[0]],['重投影 RMSE',colors[2]],['粘连分割区间',colors[3]]].forEach(([label,c])=>legend.append('button').attr('type','button').attr('aria-pressed','true').html(`<span class="swatch" style="--swatch:${c}"></span>${label}`));
}
function drawParameters(){
  const {width,svg}=baseSvg('#parameter-chart',250,'探测器横向偏移、纵向偏移和源探测器距离的估计值与真值对比');
  const rows=[
    {name:'|offset U|',est:data.estimate.offsetU,truth:data.truth.offsetU,unit:'pixel'},
    {name:'|offset V|',est:data.estimate.offsetV,truth:data.truth.offsetV,unit:'pixel'},
    {name:'SDD',est:data.estimate.sdd,truth:data.truth.sdd,unit:'mm'}];
  const left=90,right=78,top=20,rowH=70,w=width-left-right;
  rows.forEach((d,i)=>{
    const y=top+i*rowH, scale=d3.scaleLinear().domain([0,Math.max(d.est,d.truth)*1.12]).range([left,left+w]);
    svg.append('text').attr('class','param-label').attr('x',left-10).attr('y',y+25).attr('text-anchor','end').text(d.name);
    svg.append('rect').attr('class','param-track').attr('x',left).attr('y',y+12).attr('width',w).attr('height',18);
    svg.append('rect').attr('x',left).attr('y',y+12).attr('width',Math.max(1,scale(d.est)-left)).attr('height',18).attr('fill',i===2?'var(--viz-series-4)':'var(--viz-series-1)');
    svg.append('line').attr('class','param-truth').attr('x1',scale(d.truth)).attr('x2',scale(d.truth)).attr('y1',y+7).attr('y2',y+35);
    svg.append('text').attr('class','param-label').attr('x',left).attr('y',y+50).text(`估计 ${d.est.toFixed(i===2?1:3)} ${d.unit}`);
    svg.append('text').attr('class','param-label').attr('x',width-right+8).attr('y',y+25).text(`真值 ${d.truth.toFixed(i===2?0:1)}`);
  });
}
function update(){const frame=+slider.value;frameValue.textContent=frame;frameStatus.textContent=data.split[frame]?'轨迹引导粘连分割':'普通检测';drawTrajectory(frame);drawDiagnostics(frame);drawParameters();}
slider.addEventListener('input',update);
new ResizeObserver(()=>update()).observe(root);
update();
})();
</script>'''.replace("__DATA__", data)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(fragment, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
