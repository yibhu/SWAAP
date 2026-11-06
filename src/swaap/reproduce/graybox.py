"""Exact saved gray-box numeric checks, invoked explicitly; no import side effects."""
import argparse,csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np

def main(argv=None):
    P=argparse.ArgumentParser(description=__doc__)
    P.add_argument('--package-root',required=True,type=Path)
    P.add_argument('--output',required=True,type=Path)
    P.add_argument('--rebuild-figure',action='store_true',help='Also regenerate the original two-page PDF/PNGs; requires Matplotlib')
    a=P.parse_args(argv)
    ROOT=a.package_root.resolve()
    OUT=a.output.resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    checks=[]
    failures=[]
    inputs={}
    def read(name):
     p=ROOT/name;inputs[name]=hashlib.sha256(p.read_bytes()).hexdigest()
     return json.loads(p.read_text()) if p.suffix=='.json' else list(csv.DictReader(p.open()))
    def check(scope,key,got,want,tol=1e-9):
     if want in ('',None):
      if got is not None and math.isfinite(float(got)):failures.append(dict(scope=scope,key=key,regenerated=got,adopted=want))
      return
     x,y=float(got),float(want);d=abs(x-y);checks.append((scope,key,d))
     if not math.isfinite(x) or d>tol+1e-11*abs(y):failures.append(dict(scope=scope,key=key,regenerated=x,adopted=y,difference=d))
    def write(name,rows):
     with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    def array(rows,k):return np.array([float(r[k]) for r in rows],dtype=float)
    episodes=read('GRAYBOX_EPISODES.csv')
    groups=defaultdict(list)
    for r in episodes:groups[(r['panel'],r['task'],r['setting'],r['LR'])].append(r)
    for k,rs in groups.items():
     rs.sort(key=lambda r:int(r['evaluation_random_seed']));expected=100 if k[0]=='detailed100' else 50
     assert len(rs)==expected and [int(r['evaluation_random_seed']) for r in rs]==list(range(expected)),k
     assert np.isfinite(array(rs,'Return')).all() and np.isfinite(array(rs,'delta_m')).all()
    for panel,task in {(k[0],k[1]) for k in groups}:
     original=groups[(panel,task,'Original','')]
     identity=lambda rs:[tuple(r[k] for k in ['environment_seed','planner_seed','initial_observation_hash']) for r in rs]
     for k,rs in groups.items():
      if k[:2]==(panel,task):assert identity(original)==identity(rs),k
    D100=np.random.default_rng(429723).integers(0,100,(10000,100))
    D50=np.random.default_rng(429723).integers(0,50,(10000,50))
    efficacy=read('GRAYBOX_EFFICACY.csv')
    effout=[]
    for r in efficacy:
     task,lr=r['task'],r['LR'];o,c,p=[array(groups[('detailed100',task,s,'' if s=='Original' else lr)],'Return') for s in ['Original','Natural','SWAAP']];row=dict(task=task,LR=lr)
     for name,setting,v in [('Original','Original',o),('Clean','Natural',c),('Poison','SWAAP',p)]:
      dm=array(groups[('detailed100',task,setting,'' if setting=='Original' else lr)],'delta_m')
      for field,value in [(name+'_return',v.mean()),(name+'_return_sample_SD',v.std(ddof=1)),('rollout_delta_m_'+name,dm.mean()),('rollout_delta_m_'+name+'_sample_SD',dm.std(ddof=1))]:row[field]=float(value);check('detailed100',task+'/'+lr+'/'+field,value,r[field])
     cb,pb=c[D100].mean(1),p[D100].mean(1);ci=np.quantile(100*(cb-pb)/cb,[.025,.975]);units=np.quantile(cb-pb,[.025,.975])
     values={'Clean_Original_retention_pct':100*c.mean()/o.mean(),'benign_Clean_Original_change_units':(c-o).mean(),'benign_Clean_Original_change_pct':100*(c.mean()/o.mean()-1),'poison_specific_harm_units':(c-p).mean(),'matched_Clean_reduction_pct':100*(c.mean()-p.mean())/c.mean(),'Original_relative_Poison_change_pct':100*(p.mean()/o.mean()-1),'paired_reduction_pct_CI95_lower':ci[0],'paired_reduction_pct_CI95_upper':ci[1],'paired_harm_units_CI95_lower':units[0],'paired_harm_units_CI95_upper':units[1]}
     for field,v in values.items():row[field]=float(v);check('detailed100',task+'/'+lr+'/'+field,v,r[field])
     effout.append(row)
    write('GRAYBOX_REPRODUCED_EFFICACY.csv',effout)
    metrics=read('GRAYBOX_BREADTH160.csv')
    wide=read('GRAYBOX_BREADTH80.csv')
    summary=read('GRAYBOX_BREADTH_SUMMARY.csv')
    breadth=[]
    for r in metrics:
     task,lr=r['task'],r['LR'];o,c,p=[array(groups[('breadth50',task,s,'' if s=='Original' else lr)],'Return') for s in ['Original','Natural','SWAAP']];threshold=max(1e-8,1e-6*max(1.,float(np.abs(c).mean())));cb,pb,ob=c[D50].mean(1),p[D50].mean(1),o[D50].mean(1);invalid=int((np.abs(cb)<=threshold).sum())
     vals={'Original_return':o.mean(),'Clean_return':c.mean(),'Poison_return':p.mean(),'Clean_gain':c.mean()-o.mean(),'Poison_gain':p.mean()-o.mean(),'poison_specific_harm':c.mean()-p.mean(),'matched_reduction':(c.mean()-p.mean())/c.mean(),'matched_reduction_pct':100*(c.mean()-p.mean())/c.mean(),'Clean_Original_retention_pct':100*c.mean()/o.mean() if abs(o.mean())>threshold else None,'bootstrap_invalid_denominators':invalid}
     for prefix,v in [('paired_Clean_gain',cb-ob),('paired_Poison_gain',pb-ob),('paired_harm',cb-pb)]:
      lo,hi=np.quantile(v,[.025,.975]);vals[prefix+'_CI95_lower']=lo;vals[prefix+'_CI95_upper']=hi
     vals['paired_reduction_pct_CI95_lower'],vals['paired_reduction_pct_CI95_upper']=np.quantile(100*(cb-pb)/cb,[.025,.975]) if invalid==0 else (None,None)
     row=dict(task_index=r['task_index'],task=task,LR=lr)
     for field,v in vals.items():check('breadth50',task+'/'+lr+'/'+field,v,r[field]);row[field]=None if v is None else float(v)
     assert ('UNDEFINED'in r['percentage_status'])==(invalid>0)
     if r['native_success_available']=='True':
      for field,s in [('Original_success','Original'),('Clean_success','Natural'),('Poison_success','SWAAP')]:check('breadth_success',task+'/'+lr+'/'+field,np.mean([x['success_any']=='True' for x in groups[('breadth50',task,s,'' if s=='Original' else lr)]]),r[field])
     w=next(x for x in wide if x['task']==task);suffix='LR25' if float(lr)==2.5e-5 else'LR50'
     for field in ['Clean_return','Poison_return','Clean_gain','Poison_gain','poison_specific_harm','paired_harm_CI95_lower','paired_harm_CI95_upper','paired_reduction_pct_CI95_lower','paired_reduction_pct_CI95_upper']:check('wide80',task+'/'+field+'_'+suffix,vals[field],w[field+'_'+suffix])
     check('wide80',task+'/Original_return',o.mean(),w['Original_return']);check('wide80',task+'/matched_reduction_pct_'+suffix,vals['matched_reduction_pct'],w['matched_reduction_pct_'+suffix]);breadth.append(row)
    write('GRAYBOX_REPRODUCED_BREADTH160.csv',breadth)
    for r in summary:
     rs=[x for x in breadth if x['LR']==r['LR']];v=np.array([x['matched_reduction_pct'] for x in rs]);vals={'declared_tasks':80,'complete_tasks':len(rs),'failed_tasks':0,'unmeasured_tasks':0,'incomplete_tasks':0,'tasks_with_positive_absolute_harm':sum(x['poison_specific_harm']>0 for x in rs),'tasks_with_defined_percentage':80,'tasks_with_undefined_percentage':0,'tasks_reduction_ge5pct':sum(v>=5),'tasks_reduction_ge10pct':sum(v>=10),'reduction_pct_Q1':np.quantile(v,.25),'reduction_pct_median':np.median(v),'reduction_pct_Q3':np.quantile(v,.75)}
     for field,x in vals.items():check('breadth_summary',r['LR']+'/'+field,x,r[field])
    def ranking(c,p):
     scores=np.r_[c,p];labels=np.r_[np.zeros(len(c)),np.ones(len(p))];ix=np.argsort(-scores,kind='mergesort');s=scores[ix];y=labels[ix];ends=np.r_[np.flatnonzero(np.diff(s)),len(s)-1];tp=np.cumsum(y==1)[ends];fp=np.cumsum(y==0)[ends];tpr=np.r_[0.,tp/len(p)];fpr=np.r_[0.,fp/len(c)]
     def area(m):
      if m==1:return float(np.trapezoid(tpr,fpr))
      stop=int(np.searchsorted(fpr,m,side='right'));y=tpr[stop-1]+(m-fpr[stop-1])/(fpr[stop]-fpr[stop-1])*(tpr[stop]-tpr[stop-1]);area=float(np.trapezoid(np.r_[tpr[:stop],y],np.r_[fpr[:stop],m]));return .5*(1+(area-.5*m*m)/(m-.5*m*m))
     return np.array([area(1),area(.05),area(.1)])
    def wilson(k,n):
     z=1.959963984540054;q=k/n;den=1+z*z/n;mid=(q+z*z/(2*n))/den;rad=z*math.sqrt(q*(1-q)/n+z*z/(4*n*n))/den;return mid-rad,mid+rad
    incoming=read('GRAYBOX_INCOMING.csv')[0]
    di=read('GRAYBOX_DETECTOR_INPUTS.json')
    clean=np.array(di['independent_clean'])
    poison=np.array(di['selected_poison'])
    roc=ranking(clean,poison)
    for field,x in zip(['AUC','pAUC5','pAUC10'],roc):check('incoming',field,x,incoming[field])
    boot=np.empty((di['bootstrap_resamples'],3))
    rng=np.random.default_rng(di['bootstrap_seed'])
    for i in range(len(boot)):boot[i]=ranking(clean[rng.integers(0,len(clean),len(clean))],poison[rng.integers(0,len(poison),len(poison))])
    for i,field in enumerate(['AUC','pAUC5','pAUC10']):
     for s,x in zip(['lower','upper'],np.quantile(boot[:,i],[.025,.975])):check('incoming_bootstrap',field+'/'+s,x,incoming[field+'_CI95_'+s])
    for suffix,nominal in [('5','0.05'),('10','0.10')]:
     t=di['thresholds'][nominal];th=float(t['threshold']) if isinstance(t,dict) else float(t);k=int((poison>th).sum());n=int((clean>th).sum());check('incoming','alarms'+suffix,k,incoming['alarms'+suffix]);check('incoming','Clean_alarms'+suffix,n,incoming['Clean_alarms'+suffix]);check('incoming','TPR'+suffix,k/50,incoming['TPR'+suffix]);check('incoming','realized_Clean_FPR'+suffix,n/1000,incoming['realized_Clean_FPR'+suffix])
     for metric,count,total in [('TPR',k,50),('Clean_FPR',n,1000)]:
      for side,x in zip(['lower','upper'],wilson(count,total)):check('incoming_wilson',metric+suffix+'/'+side,x,incoming[metric+suffix+'_CI95_'+side])
    for i,trace in enumerate(np.asarray(di['raw_residuals'],dtype=float)):
     run=maximum=0.
     for value in trace:run=max(0.,run+value-di['median']-di['drift_factor']*di['mad']);maximum=max(maximum,run)
     check('saved_residual_recurrence',str(i),maximum,poison[i])
    def esc(s):return s.replace('_',r'\_')
    def lr(s):return r'$2.5\times10^{-5}$' if float(s)==2.5e-5 else r'$5\times10^{-5}$'
    def f(x,n=2):return f'{float(x):.{n}f}'
    def pm(x,y,n=1):return '$'+f(x,n)+r'\pm'+f(y,n)+'$'
    T=[r'% Insert table units within the existing Gray-Box Attack on TD-MPC2 Multi-Task World Model subsection (app:graybox).',r'\begin{table}[htbp]\centering\scriptsize\setlength{\tabcolsep}{3pt}',r'\caption{Gray-box attack on TD-MPC2 multi-task world model. Source Task: MW-Soccer. Natural denotes matched Clean fine-tuning; Original is the pretrained MT80 model. Return and $\delta_m$ are mean $\pm$ sample SD over 100 paired evaluation random seeds for fixed models. SD and paired 95\% bootstrap intervals describe evaluation-seed variation; each LR has one fine-tuning run of the same source-selected poison.}\label{tab:gray-box}',r'\begin{tabular}{@{}lllrrr@{}}\toprule Target Task & LR & Setting & Return & $\delta_m$ & Reduction (\%) [95\% CI] \\ \midrule']
    for task in ['mw-soccer','mw-push']:
     for s,learning in [('Original',''),('Natural','2.5e-05'),('SWAAP','2.5e-05'),('Natural','5e-05'),('SWAAP','5e-05')]:
      rs=groups[('detailed100',task,s,learning)];v=array(rs,'Return');dm=array(rs,'delta_m');red=''
      if s=='SWAAP':r=next(r for r in efficacy if r['task']==task and r['LR']==learning);red='$'+f(r['matched_Clean_reduction_pct'])+r'\;['+f(r['paired_reduction_pct_CI95_lower'])+','+f(r['paired_reduction_pct_CI95_upper'])+']$'
      T.append(' & '.join([task.upper().replace('MW-','MW-').replace('SOCCER','Soccer').replace('PUSH','Push'),lr(learning) if learning else'',s,pm(v.mean(),v.std(ddof=1)),pm(dm.mean(),dm.std(ddof=1),4),red])+r' \\')
     T.append(r'\midrule')
    T[-1]=r'\bottomrule\end{tabular}\end{table}'
    T += [r'\begin{table}[htbp]\centering\scriptsize',r'\caption{One incoming Soccer poison buffer, screened under Original MT80 with Soccer conditioning before adaptation: 50 poison trajectories and 1,000 independent Clean trajectories. The same measurement applies to both LRs and all downstream tasks. Brackets give 95\% bootstrap intervals for AUC/pAUC and Wilson intervals for TPR/FPR.}\label{tab:graybox-incoming-stealth}',r'\begin{tabular}{@{}rr@{}}\toprule $\delta_{d,\mathrm{ref}}$ & $\delta_d$ \\ \midrule '+f(incoming['delta_d_ref'],8)+' & '+f(incoming['delta_d'],8)+r' \\ \bottomrule\end{tabular}\quad\begin{tabular}{@{}lrr@{}}\toprule Metric & Estimate & 95\% CI \\ \midrule']
    for x,label in [('AUC','AUC'),('pAUC5',r'pAUC$_5$'),('pAUC10',r'pAUC$_{10}$')]:T.append(label+' & '+f(incoming[x],5)+' & $['+f(incoming[x+'_CI95_lower'],6)+','+f(incoming[x+'_CI95_upper'],6)+r']$ \\')
    T += [r'\bottomrule\end{tabular}\medskip\begin{tabular}{@{}lrr@{}}\toprule Nominal FPR & Poison alarms/50; TPR (\%) [CI] & Clean alarms/1000; FPR (\%) [CI] \\ \midrule']
    for x in ['5','10']:
     cells=[x+r'\%']
     for count,metric in [('alarms','TPR'),('Clean_alarms','Clean_FPR')]:
      point=incoming['TPR'+x] if metric=='TPR' else incoming['realized_Clean_FPR'+x];cells.append(str(incoming[count+x])+r'; $'+f(100*float(point))+r'\;['+f(100*float(incoming[metric+x+'_CI95_lower']))+','+f(100*float(incoming[metric+x+'_CI95_upper']))+']$')
     T.append(' & '.join(cells)+r' \\')
    T += [r'\bottomrule\end{tabular}\end{table}',r'\begin{table}[htbp]\centering\small',r'\caption{Complete MT80 breadth using 50 paired evaluation random seeds per task for fixed models. Positive harm is $R_{\mathrm{Natural}}-R_{\mathrm{SWAAP}}>0$. Reductions and quantiles are percentages; raw returns are not averaged across tasks.}\label{tab:graybox-breadth-summary}',r'\begin{tabular}{@{}lrrrrr@{}}\toprule LR & Positive harm & $\geq5\%$ & $\geq10\%$ & Median & IQR \\ \midrule']
    for r in summary:T.append(' & '.join([lr(r['LR']),r['tasks_with_positive_absolute_harm']+'/80',r['tasks_reduction_ge5pct']+'/80',r['tasks_reduction_ge10pct']+'/80','$'+f(r['reduction_pct_median'],6)+'$','$['+f(r['reduction_pct_Q1'],6)+','+f(r['reduction_pct_Q3'],6)+']$'])+r' \\')
    T += [r'\bottomrule\end{tabular}\end{table}']
    for page in range(2):
     T += [r'\begin{table}[p]\centering\scriptsize\setlength{\tabcolsep}{2.8pt}\renewcommand{\arraystretch}{1.04}',r'\caption{Complete signed 80-task gray-box evaluation (tasks '+str(page*40+1)+'--'+str(page*40+40)+r'). O is Original; N is Natural (matched Clean); S is SWAAP. Columns 25/50 use LR $2.5\times10^{-5}$/$5\times10^{-5}$. Reduction is $100(N-S)/N$. The denominator is 50 paired evaluation random seeds per task.}'+(r'\label{tab:gray-box-breadth}' if page==0 else''),r'\begin{tabular}{@{}lrrrrrrr@{}}\toprule Task & O & N25 & S25 & Red.25 (\%) & N50 & S50 & Red.50 (\%) \\ \midrule']
     for r in wide[40*page:40*(page+1)]:T.append(' & '.join([r'\texttt{'+esc(r['task'])+'}',f(r['Original_return']),f(r['Clean_return_LR25']),f(r['Poison_return_LR25']),f(float(r['matched_reduction_pct_LR25'])),f(r['Clean_return_LR50']),f(r['Poison_return_LR50']),f(float(r['matched_reduction_pct_LR50']))])+r' \\')
     T += [r'\bottomrule\end{tabular}\end{table}']
    (OUT/'GRAYBOX_TABLES.tex').write_text('\n'.join(T)+'\n')
    SIX=['mw-handle-pull-side','mw-hammer','mw-pick-place','mw-push','mw-handle-pull','cheetah-run-front']
    FOUR=['mw-hand-insert','cheetah-run','mw-peg-unplug-side','mw-pick-place-wall']
    BLUE,RED='#4c72b0','#c44e52'
    def csvout(path,rows):
     with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def figures(out,wide,metrics):
        # Sorting is presentation only, never selection. Both LRs use one ordering.
        ordered=sorted(wide,key=lambda r:(-(float(r['matched_reduction_pct_LR25'])+
            float(r['matched_reduction_pct_LR50']))/2,int(r['task_index'])))
        lookup={(r['task'],float(r['LR'])):r for r in metrics}
        values=[];extent=[]
        for rank,r in enumerate(ordered,1):
            for lr in [2.5e-5,5e-5]:
                m=lookup[r['task'],lr];x=float(m['matched_reduction_pct'])
                lo=float(m['paired_reduction_pct_CI95_lower'])if m['paired_reduction_pct_CI95_lower']else None
                hi=float(m['paired_reduction_pct_CI95_upper'])if m['paired_reduction_pct_CI95_upper']else None
                assert np.isfinite(x) and ((lo is None and hi is None)or(lo<=hi and np.isfinite([lo,hi]).all()))
                extent.extend([v for v in [x,lo,hi]if v is not None])
                values.append(dict(sort_rank=rank,page=1+(rank-1)//40,task=r['task'],task_index=r['task_index'],
                    LR=lr,reduction_pct=x,CI95_lower=lo,CI95_upper=hi,
                    highlight='ge10_both'if r['task']in SIX else'additional_ge5_both'if r['task']in FOUR else'all_other_tasks',
                    percentage_status=m['percentage_status']))
        xmin=min(extent)*1.10;xmax=max(extent)*1.16
        assert xmin<0<xmax
        plt.rcParams.update({'font.family':'sans-serif','font.size':8,'axes.labelsize':8,'axes.titlesize':9,
            'xtick.labelsize':7,'ytick.labelsize':6.9,'legend.fontsize':7.2,'axes.facecolor':'white',
            'figure.facecolor':'white','axes.edgecolor':'.8','axes.grid':True,'grid.color':'.82',
            'grid.linestyle':'--','grid.linewidth':.55,'axes.axisbelow':True,'legend.frameon':False,
            'pdf.fonttype':42,'ps.fonttype':42})
        with PdfPages(out/'graybox_breadth_sorted.pdf',metadata={'CreationDate':None,'ModDate':None,
                'Title':'Frozen Soccer SWAAP poison: complete MT80 breadth'})as pdf:
            for page in [1,2]:
                subset=ordered[(page-1)*40:page*40]
                fig,ax=plt.subplots(figsize=(5.5,7.25))
                fig.subplots_adjust(left=.325,right=.98,bottom=.16,top=.925)
                ax.set_xscale('symlog',linthresh=10,linscale=1,base=10)
                ax.set_xlim(xmin,xmax);ax.set_ylim(39.65,-.65)
                ax.set_yticks(np.arange(40),[r['task']+('*'if r['task']=='hopper-hop'else'')for r in subset])
                for text,r in zip(ax.get_yticklabels(),subset):
                    if r['task']in SIX:text.set_fontweight('bold')
                    elif r['task']in FOUR:text.set_fontstyle('italic')
                for lr,color,marker,offset in [(2.5e-5,BLUE,'o',-.16),(5e-5,RED,'s',.16)]:
                    for i,r in enumerate(subset):
                        m=lookup[r['task'],lr];x=float(m['matched_reduction_pct']);y=i+offset
                        if m['paired_reduction_pct_CI95_lower']:
                            lo=float(m['paired_reduction_pct_CI95_lower']);hi=float(m['paired_reduction_pct_CI95_upper'])
                            assert xmin<=lo<=hi<=xmax
                            ax.hlines(y,lo,hi,color=color,alpha=.48,lw=.72)
                            ax.vlines([lo,hi],y-.055,y+.055,color=color,alpha=.65,lw=.7)
                        assert xmin<=x<=xmax
                        ax.plot(x,y,marker=marker,ms=3.0,color=color,ls='none',markeredgewidth=.35)
                ax.axvline(0,color='.2',lw=.9)
                ax.set_xticks([-1000,-100,-10,0,10,100],['−1000','−100','−10','0','10','100'])
                ax.grid(axis='y',visible=False);ax.spines[['top','right']].set_visible(False)
                ax.tick_params(axis='y',length=0,pad=3)
                ax.set_xlabel('Matched return reduction (%)\nSymmetric log scale outside [−10%, 10%]')
                ax.set_title('All MT80 tasks: sorted ranks {}–{}'.format(1+(page-1)*40,page*40),pad=5)
                fig.legend(handles=[Line2D([],[],color=BLUE,marker='o',ms=3,ls='-',lw=.7,label=r'LR $2.5\times10^{-5}$'),
                    Line2D([],[],color=RED,marker='s',ms=3,ls='-',lw=.7,label=r'LR $5\times10^{-5}$')],
                    loc='upper center',bbox_to_anchor=(.61,.992),ncol=2,columnspacing=1.1,handlelength=1.2)
                fig.text(.5,.026,'Bold: ≥10% at both LRs; italic: additional ≥5% at both.\n'
                    '* Percentage CI unavailable; absolute-harm CI retained.',ha='center',va='bottom',fontsize=6.6)
                pdf.savefig(fig);fig.savefig(out/f'graybox_breadth_sorted_{page}.png',dpi=300)
                plt.close(fig)
        csvout(out/'breadth_plot_values.csv',values)
        return dict(points=160,finite_percentage_intervals=sum(v['CI95_lower']is not None for v in values),
            tasks=80,pages=2,sort='descending mean of the two LR point reductions; canonical index ties',
            ordered_tasks=[r['task']for r in ordered],scale='symlog; linear region [-10,10] percent',
            point_and_interval_extent=[min(extent),max(extent)],axis_limits=[xmin,xmax],clipped_points=0,
            clipped_finite_intervals=0,canvas_inches=[5.5,7.25],style='Author-supplied blue/red sans-serif, dashed gray grid, embedded Type42 fonts')
    figure_path=ROOT/'graybox_breadth_sorted.pdf'
    if figure_path.exists():
     inputs[figure_path.name]=hashlib.sha256(figure_path.read_bytes()).hexdigest()
     figure_reproduction={'mode':'retained verified original PDF;not regenerated by default','source_pdf_sha256':inputs[figure_path.name]}
    else:
     figure_reproduction={'mode':'not included in current author manuscript; numerical table reproduction only'}
    if a.rebuild_figure:
     import matplotlib
     matplotlib.use('Agg')
     import matplotlib.pyplot as plt
     from matplotlib.backends.backend_pdf import PdfPages
     from matplotlib.lines import Line2D
     figure_reproduction=figures(OUT,wide,metrics);figure_reproduction['mode']='exact original plotting function and style;CPU regeneration';figure_reproduction['byte_identical_to_delivered_pdf']=hashlib.sha256((OUT/figure_path.name).read_bytes()).hexdigest()==inputs.get(figure_path.name)
    report={'status':'PASS' if not failures else'FAIL','checks':len(checks),'maximum_absolute_difference':max((r[2] for r in checks),default=0),'failures':failures,'input_hashes':inputs,'contexts':{'detailed100':10,'breadth50':400},'episode_rows':len(episodes),'breadth_rows':len(metrics),'undefined_percentage_intervals':[{'task':r['task'],'LR':r['LR'],'invalid_bootstrap_denominators':r['bootstrap_invalid_denominators']} for r in metrics if 'UNDEFINED'in r['percentage_status']],'paired_bootstrap':{'seed':429723,'resamples':10000},'detector_bootstrap':{'seed':20260925,'resamples':5000},'figure_reproduction':figure_reproduction,'science_calls':0,'portability':'Only package-local CSV/JSON inputs;NumPy and Pythonstdlib. No externalrepo,SSH,models,tensors,simulation or calibration.'}
    (OUT/'GRAYBOX_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','checks','maximum_absolute_difference','episode_rows','breadth_rows','undefined_percentage_intervals']}))
    raise SystemExit(0 if not failures else 1)
