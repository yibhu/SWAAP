#!/usr/bin/env python3
"""Three-task main alpha comparison from closed audited grid CSVs; CPU only."""
from pathlib import Path
import argparse,csv,hashlib,json,math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from .grid import ranges,align
BLUE,RED='#4c72b0','#c44e52'
TASKS=[('humanoid-walk','Humanoid-Walk',1000),('myo-pen-twirl-hard','Myo-Pen-Twirl-Hard',4000),('mw-soccer','MW-Soccer',2000)]
STEM='dp_baseline_ours_comparison_plot_table1_9'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path(__file__).resolve().parent.parent/'data');p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv);a.output.mkdir(parents=True,exist_ok=True)
 bars=pd.read_csv(a.data/'GRID_BAR_VALUES.csv');pools=pd.read_csv(a.data/'GRID_POOL_VALUES.csv')
 reference_table=pd.read_csv(a.data/'table1_all_cells_rounding_validation.csv')
 bars=bars[bars.task.isin([t[0] for t in TASKS])&np.isclose(bars.rp,.1)]
 pools=pools[pools.task.isin([t[0] for t in TASKS])&np.isclose(pools.rp,.1)]
 assert len(bars)==36 and len(pools)==54
 plt.rcParams.update({'font.family':'sans-serif','font.size':12,'axes.titlesize':14,'axes.labelsize':13,'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':12,'axes.facecolor':'white','figure.facecolor':'white','axes.edgecolor':'.8','axes.grid':True,'grid.color':'.82','grid.linestyle':'--','grid.linewidth':.6,'axes.axisbelow':True,'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42})
 fig,axes=plt.subplots(1,3,figsize=(13.2,4.45));fig.subplots_adjust(left=.065,right=.945,bottom=.17,top=.74,wspace=.34)
 checks=[];refs=[];plotted=[]
 for col,((task,title,preferred_return_max),ax) in enumerate(zip(TASKS,axes)):
  rr=ax.twinx();rr.grid(False)
  tb=bars[bars.task.eq(task)];tp=pools[pools.task.eq(task)];clean=tp.drop_duplicates('pool_id')
  assert len(clean)==3 and tp.groupby('pool_id')[['Clean_Return','Clean_delta_m_ref']].nunique().le(1).all().all()
  table_clean=reference_table[reference_table.task.eq(task)&reference_table.method.eq('Clean Reference')]
  return_cell=table_clean[table_clean.metric.eq('Return')];ref_cell=table_clean[table_clean.metric.ne('Return')]
  assert len(return_cell)==len(ref_cell)==1
  cr=float(return_cell.iloc[0].recovered_mean);cm=float(ref_cell.iloc[0].recovered_mean)
  assert math.isclose(cr,float(clean.Clean_Return.mean()),rel_tol=1e-11,abs_tol=1e-11)
  rt=tb[tb.metric.eq('Return')];dm=tb[tb.metric.eq('delta_m')]
  left=ranges(tp.Return.to_numpy(),rt['mean'].to_numpy(),rt.sample_sd.to_numpy(),clean.Clean_Return.to_numpy())
  right=ranges(tp.delta_m.to_numpy(),dm['mean'].to_numpy(),dm.sample_sd.to_numpy(),np.asarray([cm]))
  left=(left[0],max(float(preferred_return_max),left[1]));right=(right[0],max(.5,right[1]));left,right=align(left,right)
  for ai,alpha in enumerate([.1,.5,.9]):
   for mi,method in enumerate(['SWAAP','SWAAP_LEGACY_RANDOM']):
    actual=tp[tp.method.eq(method)&np.isclose(tp.alpha,alpha)]
    assert len(actual)==3 and actual.pool_id.nunique()==3
    for metric,target,color,x in [('Return',ax,BLUE,ai-.27+mi*.18),('delta_m',rr,RED,ai+.09+mi*.18)]:
     cell=tb[tb.internal_method.eq(method)&np.isclose(tb.alpha,alpha)&tb.metric.eq(metric)];assert len(cell)==1
     v=cell.iloc[0];mu=float(v['mean']);sd=float(v['sample_sd'])
     assert math.isclose(float(actual[metric].mean()),mu,rel_tol=1e-11,abs_tol=1e-11)
     assert math.isclose(float(actual[metric].std(ddof=1)),sd,rel_tol=1e-10,abs_tol=1e-11)
     target.bar(x,mu,width=.16,color=color,alpha=.85 if mi==0 else .30,zorder=2)
     target.errorbar(x,mu,yerr=sd,fmt='none',ecolor='black',elinewidth=1.2,capsize=3,zorder=4)
     lim=left if metric=='Return' else right;assert lim[0]<=mu-sd<=mu+sd<=lim[1]
     plotted.append(dict(v))
  ax.axhline(cr,color=BLUE,ls=':',lw=1.4);rr.axhline(cm,color=RED,ls='--',lw=1.4)
  ax.set_title(title);ax.set_xticks(range(3),['0.1','0.5','0.9']);ax.set_xlabel(r'$\alpha$')
  ax.set_xlim(-.5,2.5);ax.set_ylim(*left);rr.set_ylim(*right)
  ax.tick_params(axis='y',colors=BLUE);rr.tick_params(axis='y',colors=RED)
  if col==0:ax.set_ylabel('Return',color=BLUE)
  if col==2:rr.set_ylabel(r'$\delta_m$',color=RED)
  for aa in (ax,rr):aa.spines['top'].set_visible(False)
  refs.append(dict(task=task,Clean_Return=cr,Clean_delta_m_ref=cm,reference_source='Main Table1 Clean Reference: Original-reference predictions on matched held-out Clean trajectories; not fitted Clean delta_m',independent_pools=3,orders_per_pool=3))
  checks.append(dict(task=task,conditions=6,bars=12,independent_pools=3,Return_limits=left,delta_m_limits=right,all_SD_caps_in_bounds=True))
 handles=[Line2D([],[],color=BLUE,ls=':',lw=1.4,label='Clean Return'),Line2D([],[],color=RED,ls='--',lw=1.4,label=r'Clean $\delta_{m,\mathrm{ref}}$'),Patch(facecolor=BLUE,alpha=.85,label='SWAAP Return'),Patch(facecolor=RED,alpha=.85,label=r'SWAAP $\delta_m$'),Patch(facecolor=BLUE,alpha=.30,label='SWAAP (Random) Return'),Patch(facecolor=RED,alpha=.30,label=r'SWAAP (Random) $\delta_m$')]
 fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.5,.99),columnspacing=2,handlelength=2)
 fig.savefig(a.output/(STEM+'.pdf'),metadata={'CreationDate':None,'ModDate':None});fig.savefig(a.output/(STEM+'.png'),dpi=300);plt.close(fig)
 pd.DataFrame(plotted).to_csv(a.output/'MAIN_ALPHA_THREE_BARS.csv',index=False,float_format='%.17g')
 report=dict(status='PASS_THREE_TASK_MAIN_ALPHA_FIGURE',tasks=[t[0] for t in TASKS],alpha=[.1,.5,.9],rp=.1,conditions=18,bars=36,pool_conditions=54,independent_pools=3,orders_per_pool=3,statistical_unit='Average three held-out orders within each pool; mean and sample SD across three independent Stage1 pools.',Pen_pool_scope='All three pools; the two-pool selection is restricted to the separate illustrative scatter.',Clean_references=refs,checks=checks,canvas_inches=[13.2,4.45],png_dpi=300,existing_TeX_scale=.36,predicted_TeX_size_pt=[13.2*72*.36,4.45*72*.36],style_reference='Original three-panel manuscript figure supplied by the author; muted blue/red bars, pale Random bars, black SD caps, shared top legend, no hollow overlays.',source_files={n:sha(a.data/n) for n in ['GRID_BAR_VALUES.csv','GRID_POOL_VALUES.csv','table1_all_cells_rounding_validation.csv']},outputs={ext:sha(a.output/(STEM+ext)) for ext in ['.pdf','.png']},scientific_calls=0)
 (a.output/'MAIN_ALPHA_THREE_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:report[k] for k in ['status','tasks','conditions','bars','scientific_calls']}))
if __name__=='__main__':main()
