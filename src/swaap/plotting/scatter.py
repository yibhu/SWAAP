#!/usr/bin/env python3
"""Replay the author-adopted filtered developmental scatter from compact inputs.
Plot body retained from rebuild_target_series.py; raw-record recovery is separate.
"""
from pathlib import Path
import argparse,csv,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors

def read(p):return list(csv.DictReader(p.open()))
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv);a.output.mkdir(parents=True,exist_ok=True)
 points=read(a.data/'TARGET_SERIES_POINTS.csv');selected_points=read(a.data/'SCATTER_SELECTED_POINTS.csv');baseline_rows=read(a.data/'SCATTER_DISPLAY_BASELINES.csv')
 for rows in [points,selected_points,baseline_rows]:
  for r in rows:
   for k in ['Return','delta_m','delta_m_ref']:
    if k in r:r[k]=float(r[k])
 baselines={r['pool']:{k:v for k,v in r.items() if k not in ['display_definition','Return_line_color','delta_m_ref_line_color']} for r in baseline_rows}
 display_excluded_pools={'sig-campaign-fresh-v1-PROPOSAL-myo-pen-twirl-hard-A01-S27'}
 plt.rcParams.update({'font.family':'sans-serif','font.size':12,'axes.titlesize':17,'axes.labelsize':17,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':8,'axes.facecolor':'white','figure.facecolor':'white','axes.grid':True,'grid.color':'.85','grid.linestyle':'--','grid.linewidth':.8,'axes.axisbelow':True,'pdf.fonttype':42,'ps.fonttype':42})
 BLUE,RED,GREEN,ORANGE='#4c72b0','#c44e52','#55a868','#dd8452'
 def shade(c,j):return tuple(np.asarray(colors.to_rgb(c))*(1-[.23,.10,0][j])+np.ones(3)*[.23,.10,0][j])
 display_baselines=[];plot_checks=[]
 for task,title,xlim,ylim in [('humanoid-walk','Humanoid-Walk',(.05,.5),(0,1000)),('myo-pen-twirl-hard','Myo-Pen-Twirl-Hard',(0,.5),(-500,5000))]:
  all_rows=[r for r in points if r['task']==task];rows=[r for r in selected_points if r['task']==task]
  all_pools=sorted({r['pool'] for r in all_rows},key=lambda p:int(re.search(r'-A(\d+)-',p).group(1)))
  pools=[pool for pool in all_pools if pool not in display_excluded_pools]
  pool_shades={pool:j for j,pool in enumerate(all_pools)}
  retained=sum(r['method']=='SWAAP' for r in rows);available=sum(r['method']=='SWAAP' for r in all_rows)
  fig,ax=plt.subplots(figsize=(5.5,4.52));fig.subplots_adjust(left=.18,right=.95,bottom=.18,top=.86)
  for j,pool in enumerate(pools):
   for method,col,marker in [('SWAAP',BLUE,'o'),('Direct Model Poisoning',GREEN,'^')]:
    sub=[r for r in rows if r['pool']==pool and r['method']==method]
    ax.scatter([r['delta_m'] for r in sub],[r['Return'] for r in sub],s=36 if method=='SWAAP' else 40,c=[shade(col,pool_shades[pool])],marker=marker,edgecolors='black',linewidths=.35,zorder=4)
  # Use the exact dev1 control paired to each pool's plotted SWAAP outcomes.
  # Pool-specific controls are not averaged, shifted, or taken from held-out data.
  for j,pool in enumerate(pools):
   control=baselines[pool]
   ax.axhline(control['Return'],color=shade(BLUE,pool_shades[pool]),ls=':',lw=1.25,alpha=.8)
   ax.axvline(control['delta_m_ref'],color=shade(RED,pool_shades[pool]),ls=':',lw=1.25,alpha=.8)
   display_baselines.append(dict(**control,
    display_definition='Exact matched common-dev1 pool control; Return is matched Clean rollout mean, delta_m_ref is frozen Original-reference deviation on the same Clean rollouts.',
    Return_line_color=colors.to_hex(shade(BLUE,pool_shades[pool])),delta_m_ref_line_color=colors.to_hex(shade(RED,pool_shades[pool]))))
  methods=[Line2D([],[],marker='o',color='none',markerfacecolor=BLUE,markeredgecolor='black',markersize=6,label='SWAAP'),Line2D([],[],marker='^',color='none',markerfacecolor=GREEN,markeredgecolor='black',markersize=6,label='Direct Model Poisoning'),Line2D([],[],color=BLUE,ls=':',label='Clean Return'),Line2D([],[],color=RED,ls=':',label=r'Clean $\delta_{m,\mathrm{ref}}$')]
  ax.legend(handles=methods,loc='upper right',frameon=True,framealpha=.92,handlelength=1.6,borderpad=.35,fontsize=11.5)
  ax.set_title(title);ax.set_xlabel(r'$\delta_m$',color=RED);ax.set_ylabel('Return',color=BLUE);ax.tick_params(axis='x',colors=RED);ax.tick_params(axis='y',colors=BLUE)
  xs=[r['delta_m'] for r in rows];ys=[r['Return'] for r in rows];ax.set_xlim(min(xlim[0],min(xs)-.01),max(xlim[1],max(xs)+.01));ax.set_ylim(min(ylim[0],min(ys)-30),max(ylim[1],max(ys)+30))
  ax.text(.025,.97,r'$\alpha='+('.9' if task=='humanoid-walk' else '.5')+r',\ r_p=.1$',ha='left',va='top',transform=ax.transAxes,fontsize=10,color=BLUE)
  ax.spines['left'].set_color(BLUE);ax.spines['bottom'].set_color(RED);ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
  assert len(ax.lines)==2*len(pools) and len(ax.get_legend().get_texts())==4
  for j,pool in enumerate(pools):
   control=baselines[pool]
   assert np.allclose(ax.lines[2*j].get_ydata(),[control['Return']]*2)
   assert np.allclose(ax.lines[2*j+1].get_xdata(),[control['delta_m_ref']]*2)
  assert not any(t.get_text().startswith('Pool ') for t in ax.get_legend().get_texts())
  above=sum(r['method']=='SWAAP' and r['Return']>=baselines[r['pool']]['Return'] for r in rows)
  assert above==0
  plot_checks.append(dict(task=task,Clean_Return_lines=len(pools),Clean_delta_m_ref_lines=len(pools),pool_legends=0,
   available_targets=available,retained_targets=retained,excluded_targets=available-retained,
   displayed_pools=pools,omitted_pools=[pool for pool in all_pools if pool in display_excluded_pools],
   targets_in_displayed_pools=sum(r['method']=='SWAAP' and r['pool'] in pools for r in all_rows),
   retained_SWAAP_above_own_matched_Clean=above,footer_texts=len(fig.texts),
   pool_references=[dict(pool=pool,Clean_Return=baselines[pool]['Return'],Clean_delta_m_ref=baselines[pool]['delta_m_ref']) for pool in pools],
   target_points=sum(len(collection.get_offsets()) for collection in ax.collections)))
  stem=task+'_scatter';fig.savefig(a.output/(stem+'.pdf'),metadata={'CreationDate':None,'ModDate':None});fig.savefig(a.output/(stem+'.png'),dpi=180);plt.close(fig)

if __name__=='__main__':main()
