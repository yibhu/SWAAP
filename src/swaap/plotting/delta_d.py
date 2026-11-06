"""Retained figure style and numerical plotting, explicit invocation only."""
import argparse,csv,statistics
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def main(argv=None):
    matplotlib.use('Agg')
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    OUT=a.output
    OUT.mkdir(parents=True,exist_ok=True)
    points=[]
    checks=[]
    rows=list(csv.DictReader((a.data/'delta_d_clean_poison_alpha.csv').open()))
    def read(task,family):
     ids=sorted({r['pool_id'] for r in rows if r['task']==task and r['record']=='pool'})
     return [dict(pool_id=pool,method='SWAAP',alpha=alpha,rp=.1,**{m:next(r['y'] for r in rows if r['task']==task and r['pool_id']==pool and r['record']=='pool' and float(r['alpha'])==alpha and r['method']==m) for m in ['delta_d_clean','delta_d_poison']}) for pool in ids for alpha in [.1,.5,.9]]
    def stats(rows,key):
     v=[float(r[key]) for r in rows];assert len(v)==3 and len({r['pool_id'] for r in rows})==3;return statistics.fmean(v),statistics.stdev(v)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'mathtext.fontset':'dejavusans','font.size':11,'axes.titlesize':13,'axes.labelsize':12,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'grid.linestyle':'--','grid.color':'#d0d0d0','grid.alpha':.65,'grid.linewidth':.7})
    BLUE='#4c72b0'
    RED='#c44e52'
    GREEN='#55a868'
    ORANGE='#dd8452'
    ALPHAS=(.1,.5,.9)
    TASKS=['humanoid-walk','humanoid-run','dog-walk','dog-run','cheetah-run','myo-pen-twirl-hard','myo-reach-hard','mw-push','mw-soccer']
    NAMES={'humanoid-walk':'Humanoid-Walk','humanoid-run':'Humanoid-Run','dog-walk':'Dog-Walk','dog-run':'Dog-Run','cheetah-run':'Cheetah-Run','myo-pen-twirl-hard':'Myo-Pen-Twirl-Hard','myo-reach-hard':'Myo-Reach-Hard','mw-push':'MW-Push','mw-soccer':'MW-Soccer'}
    fig,axs=plt.subplots(3,3,figsize=(12.7,9.3))
    deviation_rows=[]
    for ax,task in zip(axs.flat,TASKS):
     grid=read(task,'grid')
     for j,alpha in enumerate(ALPHAS):
      rr=[r for r in grid if r['method']=='SWAAP' and float(r['alpha'])==alpha and float(r['rp'])==.1];assert len(rr)==3
      for m,color,shift in [('delta_d_clean',BLUE,-.18),('delta_d_poison',RED,.18)]:
       mu,sd=stats(rr,m);x=j+shift;ax.bar(x,mu,width=.32,color=color,alpha=.84,zorder=2);ax.errorbar(x,mu,yerr=sd,fmt='none',ecolor='black',elinewidth=1.15,capsize=3,zorder=4)
       ax.scatter(x+np.linspace(-.065,.065,3),[float(r[m]) for r in rr],s=18,facecolors='none',edgecolors='black',linewidths=.8,zorder=5)
       for r in rr:points.append({'figure':'delta_d_clean_poison','record':'pool','task':task,'method':m,'alpha':alpha,'rp':.1,'pool_id':r['pool_id'],'x_metric':'alpha','x':alpha,'y_metric':m,'y':r[m],'x_sd':'','y_sd':''})
       points.append({'figure':'delta_d_clean_poison','record':'mean_sample_sd','task':task,'method':m,'alpha':alpha,'rp':.1,'pool_id':'','x_metric':'alpha','x':alpha,'y_metric':m,'y':mu,'x_sd':0,'y_sd':sd})
     ax.set_title(NAMES[task]);ax.set_xticks(range(3),[f'{x:g}' for x in ALPHAS]);ax.set_xlabel(r'$\alpha$');ax.set_ylim(bottom=0);ax.grid(axis='y');ax.set_axisbelow(True)
     checks.append({'task':task,'deviation_alpha_cells':3,'independent_pools_per_cell':3,'scope':'mean over all stored buffer rows'})
    fig.legend(handles=[Patch(facecolor=BLUE,alpha=.84,label=r'Clean reference $\delta_{d,\mathrm{ref}}$'),Patch(facecolor=RED,alpha=.84,label=r'SWAAP $\delta_d$')],loc='upper center',ncol=2,frameon=False,fontsize=12)
    fig.supylabel(r'Normalized data-level deviation $\delta_d$',x=.01,fontsize=13)
    fig.tight_layout(rect=(.025,0,1,.955))
    for ext in ['pdf','png']:fig.savefig(OUT/('appendix_delta_norm_grid_1.'+ext),dpi=160,bbox_inches='tight')
    plt.close(fig)
