"""Retained figure style and numerical plotting, explicit invocation only."""
from pathlib import Path
import argparse,csv,shutil
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.image as mpimg

def main(argv=None):
    matplotlib.use('Agg')
    P=argparse.ArgumentParser()
    P.add_argument('--data',type=Path,default=Path(__file__).parent.parent/'data/TRIM_FOUR_CATEGORY_METRICS.csv')
    P.add_argument('--output',type=Path,default=Path(__file__).parent)
    P.add_argument('--original',type=Path,default=Path(__file__).parent.parent/'originals/trim_original_display.png')
    P.add_argument('--no-contact',action='store_true')
    a=P.parse_args(argv)
    O=a.output
    O.mkdir(parents=True,exist_ok=True)
    bars=list(csv.DictReader(a.data.open()))
    assert [b['category'] for b in bars]==['Clean (No Trim)','Clean (Trim)','SWAAP (No Trim)','SWAAP (Trim)']
    if not a.no_contact and a.original.resolve()!=(O/'original_trim.png').resolve():shutil.copy2(a.original,O/'original_trim.png')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'mathtext.fontset':'dejavusans','font.size':12,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(6.4,4.5))
    right=ax.twinx()
    x=list(range(4))
    w=.34
    blue='#4c72b0'
    red='#c44e52'
    for i,b in enumerate(bars):
     ax.bar(i-w/2,float(b['return_mean']),w,color=blue,alpha=.8,edgecolor='white',linewidth=.8,zorder=2);ax.errorbar(i-w/2,float(b['return_mean']),yerr=float(b['return_sample_sd']),fmt='none',ecolor='black',capsize=3.5,elinewidth=1.3,zorder=4)
     if b['delta_m_mean']!='':right.bar(i+w/2,float(b['delta_m_mean']),w,color=red,alpha=.8,edgecolor='white',linewidth=.8,zorder=2);right.errorbar(i+w/2,float(b['delta_m_mean']),yerr=float(b['delta_m_sample_sd']),fmt='none',ecolor='black',capsize=3.5,elinewidth=1.3,zorder=4)
     else:right.text(i+w/2,.02,'Not recorded',ha='center',va='bottom',rotation=90,color=red,fontsize=9)
    ax.set_xticks(x,[b['category'] for b in bars],rotation=20,ha='center')
    ax.set_ylabel('Return',color=blue,fontsize=15)
    right.set_ylabel(r'$\delta_m$',color=red,fontsize=15)
    ax.tick_params(axis='y',labelcolor=blue)
    right.tick_params(axis='y',labelcolor=red)
    ax.set_ylim(0,1000)
    right.set_ylim(0,.5)
    ax.set_title(r'Humanoid Walk ($\alpha = 0.9$)',fontsize=15,pad=10)
    ax.set_axisbelow(True)
    ax.grid(axis='y',color='#cccccc',linewidth=.8)
    ax.grid(axis='x',color='#cccccc',linestyle='--',linewidth=.6)
    fig.legend(handles=[Patch(facecolor=blue,alpha=.8,label='Return'),Patch(facecolor=red,alpha=.8,label=r'$\delta_m$')],loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.03),fontsize=14)
    fig.tight_layout(rect=(0,0,1,.9))
    stem='humanoid-walk_trim_defense_comparison_corrected'
    for ext in ['pdf','png']:fig.savefig(O/(stem+'.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)
    if a.no_contact:raise SystemExit(0)
    width=.31*5.5
    height=4/2.54
    fig=plt.figure(figsize=(2*width+.5,height+.55))
    for i,(im,label) in enumerate([(O/'original_trim.png','Original (historical values)'),(O/(stem+'.png'),'Repaired (fresh values)')]):
     ax=fig.add_axes([(.08+i*(width+.22))/(2*width+.5),.08,width/(2*width+.5),height/(height+.55)]);ax.imshow(mpimg.imread(im),aspect='auto');ax.axis('off');ax.set_title(label,fontsize=7,pad=5)
    for ext in ['pdf','png']:fig.savefig(O/('TRIM_original_repaired_contact_sheet.'+ext),dpi=300)
    plt.close(fig)
