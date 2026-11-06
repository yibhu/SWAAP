#!/usr/bin/env python3
"""Rebuild the original-named Block-G appendix figure from local audited CSVs.

CPU only. The companion audit resolves all original source identities separately.
No scientific input tensor, model, remote service, fitting or rollout is used.
"""
from pathlib import Path
import argparse, csv, hashlib, json, math, subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parent.parent/'data'
OUTPUT=Path(__file__).resolve().parent.parent
BLUE,RED='#4c72b0','#c44e52'
TASKS=[('humanoid-walk','Humanoid-Walk'),('humanoid-run','Humanoid-Run'),
       ('dog-walk','Dog-Walk'),('dog-run','Dog-Run'),('cheetah-run','Cheetah-Run'),
       ('myo-pen-twirl-hard','Myo-Pen-Twirl-Hard'),('myo-reach-hard','Myo-Reach-Hard'),
       ('mw-soccer','MW-Soccer'),('mw-push','MW-Push')]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def ranges(values, means, sds, references):
    low=min(0.,float(min(values)),float(min(means-sds)),float(min(references)))
    high=max(0.,float(max(values)),float(max(means+sds)),float(max(references)))
    pad=.055*(high-low if high>low else 1.)
    return (low-pad if low<0 else 0.,high+pad)


def align(left,right):
    fraction=max(0.,-left[0]/(left[1]-left[0]),-right[0]/(right[1]-right[0]))
    return ((-fraction*left[1]/(1-fraction),left[1]),(-fraction*right[1]/(1-fraction),right[1]))


def main(argv=None):
    global ROOT,OUTPUT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,default=ROOT);parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args(argv);ROOT=args.data;OUTPUT=args.output;OUTPUT.mkdir(parents=True,exist_ok=True)
    bars=pd.read_csv(ROOT/'GRID_BAR_VALUES.csv')
    pools=pd.read_csv(ROOT/'GRID_POOL_VALUES.csv')
    selected=bars[np.isclose(bars.rp,.1)|np.isclose(bars.rp,.2)|np.isclose(bars.rp,.3)]
    assert len(selected)==324
    plt.rcParams.update({'font.family':'sans-serif','font.size':12,'axes.titlesize':14,
        'axes.labelsize':12,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':11,
        'axes.facecolor':'white','figure.facecolor':'white','axes.edgecolor':'.8',
        'axes.grid':True,'grid.color':'.82','grid.linestyle':'--','grid.linewidth':.5,
        'axes.axisbelow':True,'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42})
    # Preserve the aspect and the nine-task row order of the original PNG.
    fig,axes=plt.subplots(9,3,figsize=(3729/300,7149/300))
    # Some Matplotlib/float combinations truncate7149/300*300 to7148.
    # A one-ULP canvas adjustment preserves the intended original pixel size.
    fig.set_size_inches(np.nextafter(3729/300,np.inf),
                        np.nextafter(7149/300,np.inf),forward=False)
    limits=[];point_checks=[];plotted=[]
    for (task,title),rowaxes in zip(TASKS,axes):
        taskbars=selected[selected.task.eq(task)]
        common_rp=np.isclose(pools.rp,.1)|np.isclose(pools.rp,.2)|np.isclose(pools.rp,.3)
        taskpools=pools[pools.task.eq(task)&common_rp]
        clean=taskpools.drop_duplicates('pool_id')
        assert len(clean)==3
        for metric in ['Clean_Return','Clean_delta_m_ref']:
            assert taskpools.groupby('pool_id')[metric].nunique().le(1).all()
        r=taskbars[taskbars.metric.eq('Return')]
        m=taskbars[taskbars.metric.eq('delta_m')]
        left=ranges(taskpools.Return.to_numpy(),r['mean'].to_numpy(),r.sample_sd.to_numpy(),clean.Clean_Return.to_numpy())
        right=ranges(taskpools.delta_m.to_numpy(),m['mean'].to_numpy(),m.sample_sd.to_numpy(),clean.Clean_delta_m_ref.to_numpy())
        left,right=align(left,right)
        limits.append(dict(task=task,Return_min=left[0],Return_max=left[1],delta_m_min=right[0],delta_m_max=right[1],
            Return_pool_min=taskpools.Return.min(),Return_pool_max=taskpools.Return.max(),
            delta_m_pool_min=taskpools.delta_m.min(),delta_m_pool_max=taskpools.delta_m.max(),
            Return_bar_SD_min=(r['mean']-r.sample_sd).min(),Return_bar_SD_max=(r['mean']+r.sample_sd).max(),
            delta_m_bar_SD_min=(m['mean']-m.sample_sd).min(),delta_m_bar_SD_max=(m['mean']+m.sample_sd).max()))
        for column,(rp,ax) in enumerate(zip([.1,.2,.3],rowaxes),1):
            rightax=ax.twinx();rightax.grid(False)
            for ai,alpha in enumerate([.1,.5,.9]):
                for mi,method in enumerate(['SWAAP','SWAAP_LEGACY_RANDOM']):
                    cell=taskbars[taskbars.internal_method.eq(method)&np.isclose(taskbars.alpha,alpha)&np.isclose(taskbars.rp,rp)]
                    assert len(cell)==2
                    actual=pools[pools.task.eq(task)&pools.method.eq(method)&np.isclose(pools.alpha,alpha)&np.isclose(pools.rp,rp)]
                    assert len(actual)==3 and actual.pool_id.nunique()==3
                    for metric,target,color,x in [('Return',ax,BLUE,ai-.27+mi*.18),('delta_m',rightax,RED,ai+.09+mi*.18)]:
                        value=cell[cell.metric.eq(metric)].iloc[0]
                        assert math.isclose(actual[metric].mean(),value['mean'],rel_tol=1e-11,abs_tol=1e-11)
                        assert math.isclose(actual[metric].std(ddof=1),value.sample_sd,rel_tol=1e-10,abs_tol=1e-11)
                        target.bar(x,value['mean'],width=.16,color=color,alpha=.85 if mi==0 else .30,zorder=2)
                        target.errorbar(x,value['mean'],yerr=value.sample_sd,fmt='none',ecolor='black',elinewidth=1.1,capsize=3,zorder=4)
                        interval=left if metric=='Return' else right
                        assert interval[0]<=value['mean']-value.sample_sd<=value['mean']+value.sample_sd<=interval[1]
                        assert actual[metric].min()>=interval[0] and actual[metric].max()<=interval[1]
                        point_checks.append(dict(bar_id=value.bar_id,task=task,rp=rp,alpha=alpha,metric=metric,
                            independent_pools=3,mean=value['mean'],sample_sd=value.sample_sd,
                            axis_min=interval[0],axis_max=interval[1],all_pool_points_and_SD_inside=True))
                        plotted.append(dict(bar_id=value.bar_id,task=task,rp=rp,alpha=alpha,metric=metric,
                            method='SWAAP' if mi==0 else 'SWAAP (Random)',row=TASKS.index((task,title))+1,column=column,x=x,
                            mean=value['mean'],sample_sd=value.sample_sd,source_pool_ids=value.pool_ids,
                            source_target_SHAs=value.target_sha256s,source_poison_SHAs=value.poison_sha256s,
                            source_target_SHA_role=value.target_sha256_role,
                            source_support_SHAs=value.support_sha256s,actual_support_counts=value.actual_support_counts))
            ax.axhline(clean.Clean_Return.mean(),color=BLUE,linestyle=':',linewidth=1.2)
            rightax.axhline(clean.Clean_delta_m_ref.mean(),color=RED,linestyle='--',linewidth=1.2)
            ax.set_ylim(*left);rightax.set_ylim(*right)
            ax.set_xticks([0,1,2],['0.1','0.5','0.9']);ax.set_xlabel(r'$\alpha$')
            ax.set_ylabel('Return',color=BLUE);rightax.set_ylabel(r'$\delta_m$',color=RED)
            ax.tick_params(axis='y',colors=BLUE);rightax.tick_params(axis='y',colors=RED)
            ax.set_title(title+rf' ($r_p={rp:g}$)',pad=10)
    fig.legend(handles=[Line2D([],[],color=BLUE,linestyle=':',label='Clean Return'),
                        Patch(facecolor=BLUE,alpha=.85,label='SWAAP Return'),
                        Patch(facecolor=BLUE,alpha=.30,label='SWAAP (Random) Return')],
               loc='upper center',bbox_to_anchor=(.5,.996),ncol=3)
    # main18/PDF16 adopt the own-rollout definition and this legend. Values
    # retain the historical CSV column name; no reference data are replaced.
    fig.legend(handles=[Line2D([],[],color=RED,linestyle='--',label=r'Clean $\delta_m$'),
                        Patch(facecolor=RED,alpha=.85,label=r'SWAAP $\delta_m$'),
                        Patch(facecolor=RED,alpha=.30,label=r'SWAAP (Random) $\delta_m$')],
               loc='upper center',bbox_to_anchor=(.5,.975),ncol=3)
    fig.subplots_adjust(left=.07,right=.93,top=.938,bottom=.025,wspace=.65,hspace=.66)
    target=OUTPUT/'dp_appendix_full_results'
    # The original TeX uses scale=.38. Preserve its exact native canvas; a tight
    # bounding-box crop changes the physical page size and can overflow the page.
    fig.savefig(target.with_suffix('.pdf'),metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(target.with_suffix('.png'),dpi=300)
    plt.close(fig)
    pd.DataFrame(limits).to_csv(OUTPUT/'GRID_TASK_AXIS_LIMITS.csv',index=False,float_format='%.17g')
    pd.DataFrame(plotted).to_csv(OUTPUT/'GRID_PLOTTED_BARS.csv',index=False,float_format='%.17g')
    assert len(point_checks)==324
    proof=dict(status='PASS_ORIGINAL_STYLE_TASK_SHARED_LIMITS_NO_CLIPPING',plotted_bars=324,plotted_conditions=162,
        source_conditions=166,unplotted_supplement='Four H-Walk alpha=.9,rp=.01/.05 conditions retained in all audit/source CSVs, outside original three common-fraction columns.',
        task_order=[x[1] for x in TASKS],columns_rp=[.1,.2,.3],x_alpha=[.1,.5,.9],
        Return_label='Return',delta_m_label=r'$\delta_m$',
        legend=['Clean Return',r'Clean $\delta_m$','SWAAP Return',r'SWAAP $\delta_m$','SWAAP (Random) Return',r'SWAAP (Random) $\delta_m$'],
        Clean_red_line_definition='Clean-fitted model prediction error on its own rollouts; main18 caption1188 and PDF16 adopted legend. Historical CSV column Clean_delta_m_ref is retained without numerical changes.',
        SD_definition='Sample SD across three independent Stage1 pool means, each averaging three nested100-episode held-out orders.',
        hollow_overlay_count=0,task_shared_axis_limits=limits,coverage_checks=point_checks,
        native_canvas_points=[894.96,1715.76],png_dimensions=[3729,7149],png_dpi=300,
        original_TeX_scale=.38,display_size_at_original_scale_points=[340.0848,651.9888],
        outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [target.with_suffix('.pdf'),target.with_suffix('.png')]},
        numerical_inputs={n:sha(ROOT/n) for n in ['GRID_BAR_VALUES.csv','GRID_POOL_VALUES.csv']},scientific_calls=0)
    (OUTPUT/'GRID_FIGURE_VALIDATION.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({k:proof[k] for k in ['status','plotted_bars','plotted_conditions','hollow_overlay_count','outputs']},indent=2))


if __name__=='__main__':main()
