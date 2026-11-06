"""Recompute adopted results from genuine, checksum-verified numerical inputs.

This path performs no model forwards, simulator work, fitting or calibration.
Each output identifies its statistical unit. Missing observations are errors.
"""
from __future__ import annotations
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

TASKS = ['humanoid-walk', 'humanoid-run', 'dog-walk', 'dog-run', 'cheetah-run',
         'myo-pen-twirl-hard', 'myo-reach-hard', 'mw-push', 'mw-soccer',
         'lift-cube', 'pick-cube', 'stack-cube']


def read(root, name):
    with (root / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def group(rows, *keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[k] for k in keys)].append(row)
    return result


def stats(values):
    values = [float(v) for v in values]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError('Missing/nonfinite numerical observation; no imputation is permitted.')
    return mean(values), stdev(values) if len(values) > 1 else None


def near(a, b, *, tol=1e-9):
    if not math.isclose(float(a), float(b), rel_tol=2e-12, abs_tol=tol):
        raise ValueError(f'Numerical source mismatch: {a} versus {b}')


def verify_inputs(root):
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    for entry in manifest['files']:
        file = root / entry['file']
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        expected = entry.get('public_export_sha256', entry.get('sha256'))
        if digest != expected:
            raise ValueError('Saved input checksum mismatch: ' + file.name)
    return {e['file']: hashlib.sha256((root/e['file']).read_bytes()).hexdigest()
            for e in manifest['files']}


def nested(rows, field, run='pool_id', fit='order_id', expected_orders=3):
    """Scalar observations -> fit mean -> Stage-1 mean, equal weighting."""
    fitted = []
    for (identity, order), rr in group(rows, run, fit).items():
        fitted.append({run: identity, fit: order, 'value': stats(r[field] for r in rr)[0]})
    values = {}
    for (identity,), rr in group(fitted, run).items():
        if len(rr) != expected_orders:
            raise ValueError('Incomplete nested fitting panel: ' + identity)
        values[identity] = stats(r['value'] for r in rr)[0]
    return values


def check_maniskill_norms(root, references, dmp_rows):
    """Retained metric-completion arithmetic; no predictions are regenerated."""
    import numpy as np
    with np.load(root/'MANISKILL_DMP_COMPACT_NORMS.npz',allow_pickle=False) as arrays:
        values=arrays['delta_d_star_values']
        residual=arrays['DMP_residual_norm'];denominator=arrays['Original_prediction_norm']
        if values.shape!=residual.shape or values.shape!=denominator.shape or values.shape!=(8,5000):
            raise ValueError('Incomplete final-target DMP norm vectors.')
        if not np.array_equal(values,residual/(denominator+1e-8)):
            raise ValueError('DMP denominator differs from the frozen Original prediction.')
        for i,(task,run) in enumerate(zip(arrays['task'],arrays['stage1_run'])):
            row=next(r for r in dmp_rows if (r['task'],r['stage1_run'])==(str(task),str(run)))
            near(values[i].mean(),row['delta_d_star'],tol=2e-12)
            ordered=np.asarray(arrays['support'][i],dtype='<i8')
            if len(set(ordered.tolist()))!=500 or hashlib.sha256(ordered.tobytes()).hexdigest()!=row['ordered_support_int64_SHA256']:
                raise ValueError('Final DMP support identity differs.')
    with np.load(root/'MANISKILL_CLEAN_REFERENCE_NORMS.npz',allow_pickle=False) as arrays:
        residual=arrays['Original_residual_norm'];denominator=arrays['realized_successor_norm']
        if residual.shape!=denominator.shape or residual.shape!=(2400,100):
            raise ValueError('Incomplete matched Clean reference norms.')
        values=(residual/(denominator+1e-8)).mean(axis=1)
        used=set()
        for row in references:
            i=int(row['norm_row'])
            if i in used or not 0<=i<2400:raise ValueError('Duplicate or invalid reference norm-row join.')
            used.add(i);near(values[i],row['delta_m_ref'],tol=2e-12)
        if len(used)!=2400:raise ValueError('A matched Clean reference episode was dropped.')


def main_table(root):
    out = []
    def add(task, method, metric, values, source, alpha='', unit='independent Stage-1 run'):
        values = dict(values)
        mu, sd = stats(values.values())
        out.append(dict(task=task, method=method, metric=metric, mean=mu, sample_sd=sd,
                        independent_stage1_runs=len(values), nested_fit_orders=3,
                        alpha=alpha, rp=.1, statistical_unit=unit, source=source,
                        stage1_means=json.dumps(values, sort_keys=True)))

    panels = json.loads((root/'MAIN_AF_RETURN_VECTORS.json').read_text())['return_panels']
    if len(panels) != 405 or any(len(r['return_vector']) != 100 for r in panels):
        raise ValueError('Incomplete main return vectors.')
    original = []
    for row in panels:
        original.append(dict(row, value=stats(row['return_vector'])[0]))
    episodes = read(root, 'MAIN_ALPHA09_EPISODES.csv')
    if len(episodes) != 16200:
        raise ValueError('Incomplete uniform-alpha episode panel.')
    grid = read(root, 'FIGURE_GRID_POOL_VALUES.csv')
    for task in TASKS[:9]:
        metrics = read(root, 'MAIN_METRICS_'+task+'.csv')
        orders = [r for r in metrics if r['level'] == 'order']
        for name, source_method in [('Clean Reference','Clean'), ('Direct Model Poisoning','Direct')]:
            rr = [r for r in original if r['task']==task and r['method']==source_method]
            add(task,name,'Return',nested(rr,'value'),'MAIN_AF_RETURN_VECTORS.json')
        for name, method in [('SWAAP','SWAAP'), ('SWAAP (Random)','SWAAP_LEGACY_RANDOM')]:
            rr = [r for r in episodes if r['task']==task and r['method']==method]
            if len(rr)!=900:
                raise ValueError('Incomplete 3 x 3 x 100 episode panel.')
            add(task,name,'Return',nested(rr,'return_value'),'MAIN_ALPHA09_EPISODES.csv',.9)
            add(task,name,'delta_m',nested(rr,'delta_m'),'MAIN_ALPHA09_EPISODES.csv',.9)
            buffers = [r for r in grid if r['task']==task and r['method']==method
                       and float(r['alpha'])==.9 and float(r['rp'])==.1]
            if len(buffers)!=3 or len({r['pool_id'] for r in buffers})!=3:
                raise ValueError('A fixed buffer must occur once per Stage-1 run.')
            add(task,name,'delta_d',{r['pool_id']:r['delta_d'] for r in buffers},
                'FIGURE_GRID_POOL_VALUES.csv',.9,'one fixed buffer per independent Stage-1 run')
        clean = [r for r in orders if r['method']=='Clean']
        add(task,'Clean Reference','delta_d',nested(clean,'delta_d'),
            'MAIN_METRICS_'+task+'.csv',unit='one clean bank; repeated identity across Stage-1 runs')
        ref = [r for r in episodes if r['task']==task and r['method']=='SWAAP']
        add(task,'Clean Reference','delta_m',nested(ref,'Original_on_Clean_delta_m'),
            'MAIN_ALPHA09_EPISODES.csv:Original_on_Clean_delta_m')
        direct = [r for r in orders if r['method']=='Direct']
        add(task,'Direct Model Poisoning','delta_m',nested(direct,'delta_m'),
            'MAIN_METRICS_'+task+'.csv:Direct')
        raw = [r for r in orders if r['method'] in ('RAW_DMP_Data','RAW_DMP_DATA','raw DMP-data')]
        if len(raw)!=9:
            raise ValueError('Missing final-target raw-DMP support measurements: '+task)
        dd = nested(raw,'delta_d')
        expected = next(r for r in read(root,'TABLE1_ADDED_DMP_DATA.csv') if r['task']==task.removeprefix('myo-').removeprefix('mw-'))
        near(stats(dd.values())[0],expected['delta_d_DMP_data'])
        near(stats(dd.values())[1],expected['sample_SD'])
        add(task,'Direct Model Poisoning','delta_d',dd,'MAIN_METRICS_'+task+'.csv:RAW_DMP_Data',
            unit='one target-copy buffer per independent Stage-1 run, not Direct rollouts')

    fits = read(root,'MANISKILL_HIERARCHY_FITS.csv')
    ms_episodes=read(root,'MANISKILL_EPISODES.csv')
    by_fit=group(ms_episodes,'task','stage1_run','fine_tuning_run','method')
    for row in fits:
        for method,prefix in [('Clean','clean'),('SWAAP (Random)','random'),
                              ('Direct Model Poisoning','direct'),('SWAAP','swaap')]:
            rr=by_fit[(row['task'],row['stage1_run'],row['fine_tuning_run'],method)]
            if len(rr)!=100 or len({r['evaluation_episode'] for r in rr})!=100:
                raise ValueError('Incomplete final ManiSkill episode panel.')
            near(stats(r['return_value'] for r in rr)[0],row[prefix+'_return'])
            if prefix!='clean':
                near(stats(r['delta_m_own_model'] for r in rr)[0],row[prefix+'_delta_m'])
    refs = read(root,'MANISKILL_CLEAN_REFERENCE_EPISODES.csv')
    dmp = read(root,'MANISKILL_DMP_PER_RUN.csv')
    if len(fits)!=24 or len(refs)!=2400 or len(dmp)!=8:
        raise ValueError('Incomplete final ManiSkill measurements.')
    check_maniskill_norms(root,refs,dmp)
    for task in TASKS[9:]:
        rr = [r for r in fits if r['task']==task]
        count = 2 if task=='lift-cube' else 3
        if len(rr)!=count*3:
            raise ValueError('ManiSkill contributing-run count changed.')
        for name, prefix in [('Clean Reference','clean'),('SWAAP (Random)','random'),
                             ('Direct Model Poisoning','direct'),('SWAAP','swaap')]:
            for metric in ['Return','delta_d','delta_m']:
                field = prefix+'_'+('return' if metric=='Return' else metric)
                source = 'MANISKILL_HIERARCHY_FITS.csv'
                if name=='Clean Reference' and metric=='delta_m':
                    values = nested([r for r in refs if r['task']==task], 'delta_m_ref',
                                    run='stage1_run',fit='fine_tuning_run')
                    source = 'MANISKILL_CLEAN_REFERENCE_EPISODES.csv:Original_on_matched_Clean'
                elif name=='Direct Model Poisoning' and metric=='delta_d':
                    values = {r['stage1_run']:r['delta_d_star'] for r in dmp if r['task']==task}
                    source = 'MANISKILL_DMP_PER_RUN.csv:final_target_and_support'
                else:
                    if name=='Clean Reference' and metric=='delta_d':
                        field='swaap_delta_d_ref'
                    values = nested(rr,field,run='stage1_run',fit='fine_tuning_run')
                add(task,name,metric,values,source,alpha=float(rr[0]['alpha']) if name.startswith('SWAAP') else '',
                    unit='contributing independent Stage-1 run; Lift conditional on 2/3 stealth-qualified runs')
    if len(out)!=144:
        raise ValueError('Expected 144 main-table numerical cells.')
    return out


def cusum_table(root):
    result = []
    raw = read(root,'alpha_cusum_pool.csv')
    if len(raw)!=54 or any(int(r['fit_order_repeats'])!=0 for r in raw):
        raise ValueError('CUSUM is one incoming buffer, not repeated post-fit rollouts.')
    for (task, alpha), rows in sorted(group(raw,'task','alpha').items()):
        if len(rows)!=3 or len({r['pool_id'] for r in rows})!=3:
            raise ValueError('Incomplete incoming-buffer detector panel.')
        record=dict(task=task,alpha=alpha,independent_stage1_runs=3,
                    poison_trajectories_each=int(rows[0]['poison_trajectories']),clean_test_trajectories=1000,
                    statistical_unit='one fixed incoming buffer per Stage-1 run; common clean test cohort')
        for field in ['auc','pauc_05','pauc_10','tpr_05','tpr_10',
                      'realized_clean_test_fpr_05','realized_clean_test_fpr_10']:
            record[field], record[field+'_sample_sd'] = stats(r[field] for r in rows)
        result.append(record)
    final = [r for r in read(root,'MANISKILL_detectors.csv') if r['record_type']=='FIXED_BUFFER'
             and r['role']=='FINAL_SELECTED' and r['method']=='SWAAP']
    if len(final)!=8:
        raise ValueError('Expected only 2/3/3 final ManiSkill SWAAP buffers.')
    names={'AUC':'auc','pAUC5':'pauc_05','pAUC10':'pauc_10','TPR5':'tpr_05','TPR10':'tpr_10',
           'realized_Clean_FPR5':'realized_clean_test_fpr_05','realized_Clean_FPR10':'realized_clean_test_fpr_10'}
    for (task,alpha), rr in sorted(group(final,'task','alpha').items()):
        record=dict(task=task,alpha=alpha,independent_stage1_runs=len(rr),poison_trajectories_each=50,
                    clean_test_trajectories=1000,statistical_unit='final-selected fixed incoming buffers; 2/3/3 contributing Stage-1 runs')
        for key, public in names.items():
            record[public],record[public+'_sample_sd']=stats(r[key] for r in rr)
        result.append(record)
    return result


def grouped_summary(rows, keys, fields):
    out=[]
    for key, rr in sorted(group(rows,*keys).items()):
        r=dict(zip(keys,key));r['independent_stage1_runs']=len(rr)
        for field in fields:
            r[field+'_mean'],r[field+'_sample_sd']=stats(x[field] for x in rr)
        out.append(r)
    return out


def summarize_grid(root):
    rows=read(root,'FIGURE_GRID_POOL_VALUES.csv')
    if len(rows)!=498:
        raise ValueError('The retained grid has 498 fixed-buffer conditions including reuse.')
    return grouped_summary(rows,['task','method','alpha','rp'],['Return','delta_d','delta_m'])


def per_order_ablations(root):
    """Retained PGD/support/MPC/defense means from closed per-order inputs."""
    outputs={}
    for filename,keys,vector_pairs,scalar_fields,name in [
        ('J1_ONLINE_PGD_PER_ORDER.csv',['task','arm','epsilon'],
         [('return_vector','return_mean'),('clean_return_vector','clean_return_mean'),
          ('paired_harm_vector','matched_harm'),('canonical_true_current_delta_m_vector','canonical_true_current_delta_m'),
          ('paper_online_delta_m_vector','paper_online_delta_m')],
         ['return_mean','clean_return_mean','matched_harm','relative_reduction_pct','delta_d_poison',
          'delta_d_clean','canonical_true_current_delta_m','paper_online_delta_m'],'pgd'),
        ('H_RETAINED_PER_ORDER.csv',['task','block_h_case','condition','num_samples','planning_horizon'],
         [('poison_return_vector','return_mean'),('clean_return_vector','clean_return_mean'),
          ('paired_harm_vector','matched_harm'),('delta_m_vector','delta_m'),
          ('clean_delta_m_vector','delta_m_clean'),('delta_m_ref_vector','delta_m_ref'),
          ('clean_delta_m_ref_vector','delta_m_ref_clean')],
         ['return_mean','clean_return_mean','matched_harm','relative_reduction_pct','delta_d_poison',
          'delta_d_clean','delta_m','delta_m_clean','delta_m_ref','delta_m_ref_clean',
          'original_clean_return_mean','primary_current_return_mean','raw_no_trim_return_mean',
          'trim_current_return_mean','raw_poison_recovery','clean_utility_cost',
          'trim_current_poison_recovery','trim_current_defended_harm'],'defense_and_mpc')]:
        data=read(root,filename);out=[]
        for key,rr in sorted(group(data,*keys).items()):
            record=dict(zip(keys,key));record.update(statistical_unit='three Stage-1 means; three nested fitting orders',
                   independent_stage1_runs=3,fit_orders=3,episodes_per_order=100)
            for row in rr:
                for vector,scalar in vector_pairs:
                    if not row[vector]:continue
                    values=json.loads(row[vector])
                    if len(values)!=100:raise ValueError('Incomplete ablation return/deviation vector.')
                    near(stats(values)[0],row[scalar])
                near(100*(float(row['clean_return_mean'])-float(row['return_mean']))/float(row['clean_return_mean']),
                     row['relative_reduction_pct'])
            for field in scalar_fields:
                present=[r[field]!='' for r in rr]
                if not any(present):record[field+'_mean']=record[field+'_sample_sd']='';continue
                if not all(present):raise ValueError('Partially missing ablation metric: '+field)
                values=nested(rr,field)
                if len(values)!=3:raise ValueError('Incomplete three-run ablation.')
                record[field+'_mean'],record[field+'_sample_sd']=stats(values.values())
            if name=='pgd':
                expected=next(r for r in read(root,'CORE_J_TASK_MEAN_SD.csv') if r['panel']=='J1'
                              and all(r[k]==record[k] for k in keys))
                near(record['return_mean_mean'],expected['return_mean_mean'])
                near(record['return_mean_sample_sd'],expected['return_mean_sample_sd'])
                near(record['relative_reduction_pct_mean'],expected['relative_reduction_pct_mean'])
            out.append(record)
        outputs[name]=out
    rows=read(root,'TRIM_CURRENT_PER_ORDER.csv');dedup={}
    fields=['original_clean','current','trim_clean','trim_current','poison_recovery','clean_utility_cost','defended_paired_harm']
    for row in rows:
        key=tuple(row[k] for k in ('task','pool_id','order_id'))
        if key in dedup:
            for field in fields:near(row[field],dedup[key][field])
        else:dedup[key]=row
    if len(dedup)!=27:raise ValueError('TRIM matched conditions must deduplicate to 27 order panels.')
    outputs['trim']=[]
    for (task,),rr in sorted(group(dedup.values(),'task').items()):
        record=dict(task=task,independent_stage1_runs=3,fit_orders=3,
                    statistical_unit='Stage-1 mean after three matched fitting orders')
        for field in fields:
            values=nested(rr,field);record[field+'_mean'],record[field+'_sample_sd']=stats(values.values())
        outputs['trim'].append(record)
    return outputs


def further_retained_tables(root):
    out={}
    sizes=[r for r in read(root,'CORE_buffer25k.csv') if r['ordinary_attacker_controlled_SWAAP']=='True']
    fixed=[]
    for (task,),rr in sorted(group(sizes,'task').items()):
        if len(rr)!=3 or any(r['heldout_orders']!='3' for r in rr):
            raise ValueError('Fixed-ratio 25k is three Stage-1 runs with three orders each.')
        row=dict(task=task,method='SWAAP',rows=25000,modified_rows=2500,alpha=.9,rp=.1,
                 independent_stage1_runs=3,fit_orders=3,optimizer_updates=980)
        expected=next(x for x in read(root,'BUFFER_FIXED_RATIO_SUMMARY.csv') if x['task']==task)
        for field in ['Clean_mean','poison_mean','reduction_percent','delta_d_poison','delta_m']:
            mu,sd=stats(r[field] for r in rr)
            near(mu,expected[field]);near(sd,expected[field+'_sd'])
            row[field]=mu;row[field+'_sample_sd']=sd
        fixed.append(row)
    out['buffer25k']=fixed
    matched=[]
    for (task,method),rr in sorted(group(read(root,'DMP_MATCHED_FIT_INPUTS.csv'),'task','method').items()):
        if len(rr)!=3:raise ValueError('Older matched-deviation comparison has three nested fits.')
        row=dict(task=task,method=method,independent_stage1_runs=1,fitting_runs=3,
                 statistical_unit='three fitting runs of one representative target; distinct older matched-deviation study')
        for r in rr:near(100*(float(r['Clean_return'])-float(r['poison_return']))/float(r['Clean_return']),r['paired_reduction_percent'])
        for field in ['Clean_return','poison_return','paired_reduction_percent']:
            row[field],row[field+'_sample_sd']=stats(r[field] for r in rr)
        expected=next(r for r in read(root,'DMP_MATCHED_COMPARISON.csv') if r['task']==task)
        prefix='SWAAP' if method=='SWAAP' else 'DMP'
        near(row['poison_return'],expected[prefix+'_return'])
        near(row['paired_reduction_percent'],expected[prefix+'_reduction_percent'])
        matched.append(row)
    out['matched_deviation_dmp']=matched
    out['trajectory_consistency']=[{k:r[k] for k in ['task','stage1_run','role','clean_return',
          'consistent_clean_init','clean_init_reduction','delta_d','delta_m']}
          for r in read(root,'TRAJECTORY_RUN_MEANS.csv')]
    return out


def table_tex(rows):
    methods=['Clean Reference','SWAAP (Random)','Direct Model Poisoning','SWAAP']
    lines=[r'% Generated from released scalar inputs; SD is across contributing Stage-1 runs.',
           r'\begin{tabular}{l'+'ccc'*4+'}',r'\toprule',
           'Task & '+(r'Clean Return & $\delta_{d,\mathrm{ref}}$ & $\delta_{m,\mathrm{ref}}$ & '
           r'SWAAP (Random) Return & $\delta_d$ & $\delta_m$ & '
           r'Direct Return & $\delta_d^\ast$ & $\delta_m$ & '
           r'SWAAP Return & $\delta_d$ & $\delta_m$')+r' \\',r'\midrule']
    for task in TASKS:
        cells=[]
        for method in methods:
            for metric in ['Return','delta_d','delta_m']:
                row=next(r for r in rows if r['task']==task and r['method']==method and r['metric']==metric)
                n=0 if metric=='Return' else 3
                a=f"{row['mean']:.{n}f}";b=f"{row['sample_sd']:.{n}f}"
                dagger=r'^{\dagger}' if row['sample_sd']>0 and float(b)==0 else ''
                cells.append('$'+a+r'\pm'+b+dagger+'$')
        lines.append(task+' & '+' & '.join(cells)+r' \\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}'])+'\n'


def reproduce(data_root, output, figures=False):
    root=Path(data_root).expanduser().resolve();output=Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError('Output exists; choose a new directory. Saved inputs are never overwritten.')
    hashes=verify_inputs(root)
    output.mkdir(parents=True)
    tables=output/'tables';tables.mkdir()
    main=main_table(root);write(tables/'main_results.csv',main)
    (tables/'main_results.tex').write_text(table_tex(main))
    cusum=cusum_table(root);write(tables/'incoming_cusum.csv',cusum)
    grid=summarize_grid(root);write(tables/'alpha_poison_fraction.csv',grid)
    j2=grouped_summary(read(root,'CORE_J2B_PER_POOL.csv'),['task','arm'],
                       ['return_mean','matched_harm','relative_reduction_pct','delta_d_poison'])
    write(tables/'random_target_shared_support.csv',j2)
    sizes=[]
    for r in read(root,'CORE_BUFFER_FIXED500_TABLE11_METRICS.csv'):
        values=json.loads(r['pool_Return_means_json']);dm=json.loads(r['pool_delta_m_means_json'])
        mu,sd=stats(values.values());dmu,dsd=stats(dm.values())
        for a,b in [(mu,r['Return_mean']),(sd,r['Return_sample_sd']),
                    (dmu,r['delta_m_mean']),(dsd,r['delta_m_sample_sd'])]:near(a,b)
        sizes.append(dict(rows=r['rows'],return_mean=mu,return_sample_sd=sd,delta_m=dmu,
                          delta_m_sample_sd=dsd,independent_stage1_runs=3,fit_orders=3,
                          updates_per_fit=r['updates_per_fit']))
    write(tables/'fixed500_buffer_size.csv',sizes)
    for name,rr in {**per_order_ablations(root),**further_retained_tables(root)}.items():
        # Archived matched-deviation and raw-DMP/TRIM checks remain verifiable
        # references, but are not restored into the current manuscript's tables.
        if name=='matched_deviation_dmp':continue
        if name=='defense_and_mpc':
            rr=[r for r in rr if r['block_h_case']!='H1_RAW_DMP_TRIM']
        write(tables/(name+'.csv'),rr)
    # Source tables whose finest retained units are condition summaries remain
    # explicitly identified; do not pretend to reconstruct their raw episodes.
    for name, source in [('stage1_dispositions','MANISKILL_Stage1_runs.csv')]:
        write(tables/(name+'.csv'),read(root,source))
    from . import dino
    payload=json.loads((root/'DINO_inputs.json').read_text())
    for name, rr in [('dino_efficacy',dino.efficacy(payload)),
                     ('dino_deviation',dino.deviations(payload)),('dino_incoming_cusum',dino.detector(payload))]:
        write(tables/(name+'.csv'),rr)
    from . import graybox
    try:
        graybox.main(['--package-root',str(root),'--output',str(tables)])
    except SystemExit as e:
        if e.code not in (None,0):raise ValueError('Saved gray-box numerical check failed.') from e
    plot_report=None
    if figures:
        from ..plotting.reproduce import plot_all
        plot_report=plot_all(root,output/'figures',main)
    outputs={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in output.rglob('*') if p.is_file()}
    report=dict(status='PASS_SAVED_NUMERICAL_REPRODUCTION',main_table_cells=len(main),
                grid_conditions=498,cusum_task_conditions=len(cusum),source_hashes=hashes,
                outputs=outputs,figures=plot_report,model_forwards=0,environment_episodes=0,victim_fits=0,
                caveats=['Saved-result reproduction is distinct from a new attack.',
                         'Some retained ablations have condition-summary inputs only; no fabricated episode arrays.',
                         'Final ManiSkill has 2/3/3 contributing Stage-1 runs; selection failures remain recorded.',
                         'Gray-box detailed100 and breadth50 panels remain separate.',
                         'DINO is one Stage-1 run with nested fitting and goal variation.'])
    (output/'reproduction.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return dict(status=report['status'],output=str(output),main_table_cells=len(main),
                new_science=False,figures=figures)
