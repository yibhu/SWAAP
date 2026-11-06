"""Reaggregate the retained DINO-WM experiment. Excluded controls are not replayed."""
import csv, math
from collections import defaultdict
import numpy as np
from swaap.evaluation.cusum import robust_stats,cusum_score,split_conformal_threshold,evaluate_score_samples,TARGET_FPRS

def read_table(data,key):
    t=data['source_tables'][key]
    return [dict(zip(t['columns'],r),_source_id=t['source_id'],_CSV_row=i+2) for i,r in enumerate(t['rows'])]

def ms(values):
    v=list(map(float,values));return float(np.mean(v)),float(np.std(v,ddof=1)) if len(v)>1 else ''

def group(rows,*keys):
    out=defaultdict(list)
    for r in rows:out[tuple(r[k] for k in keys)].append(r)
    return out

def near(a,b):
    assert math.isclose(float(a),float(b),rel_tol=2e-12,abs_tol=2e-12),(a,b)

def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def efficacy(data):
    rows=read_table(data,'final/EFFICACY_GOALS.csv');runs=[]
    for (method,panel,run),rr in sorted(group(rows,'method','panel','fitting_run').items()):
        assert len(rr)==20 and len({r['pair_identity'] for r in rr})==20
        n=sum(int(r['endpoint_success']) for r in rr);c=sum(int(r['Clean_success']) for r in rr)
        runs.append(dict(method=method,panel=panel,fitting_run=run,successes=n,Clean_successes=c,loss_count=c-n,loss_pp=100*(c-n)/20,
            distance=ms(r['endpoint_state_distance'] for r in rr)[0],distance_change=ms(r['paired_state_distance_change'] for r in rr)[0]))
    recorded=read_table(data,'final/EFFICACY_AGGREGATES.csv');out=[]
    for (method,panel),rr in sorted(group(runs,'method','panel').items()):
        assert len(rr)==3
        count,sd=ms(r['successes'] for r in rr);loss,lsd=ms(r['loss_pp'] for r in rr);dist,dsd=ms(r['distance'] for r in rr);dc,dcsd=ms(r['distance_change'] for r in rr)
        expected=next(r for r in recorded if r['method']==method and r['panel']==panel)
        for a,b in [(count,expected['native_successes_mean']),(sd,expected['native_successes_sample_SD']),(loss/100,expected['success_harm_fraction_mean']),(lsd/100,expected['success_harm_fraction_sample_SD']),(dist,expected['mean_state_distance_mean']),(dsd,expected['mean_state_distance_sample_SD'])]:near(a,b)
        out.append(dict(record_type='nested_fitting_runs',method=method,panel=panel,success_counts='/'.join(str(r['successes']) for r in rr),goals_per_run=20,fitting_runs=3,independent_Stage1_runs=1,success_count_mean=count,success_count_sample_SD=sd,success_fraction_mean=count/20,success_fraction_sample_SD=sd/20,paired_loss_pp_mean=loss,paired_loss_pp_sample_SD=lsd,paired_baseline='matched Clean fitting run',mean_state_distance=dist,state_distance_sample_SD=dsd,paired_state_distance_change=dc,paired_state_distance_change_sample_SD=dcsd,source_id=expected['_source_id'],source_CSV_row=expected['_CSV_row']))
    for (method,),rr in sorted(group(read_table(data,'final/DIRECT_GOALS.csv'),'method').items()):
        assert len(rr)==20;count=sum(int(r['native_success']) for r in rr);oc=sum(int(r['Original_success']) for r in rr)
        out.append(dict(record_type='single_model_final_panel',method=method,panel='final',success_counts=str(count),goals_per_run=20,fitting_runs=0,independent_Stage1_runs=(0 if method=='Original' else 1),success_count_mean=count,success_count_sample_SD='',success_fraction_mean=count/20,success_fraction_sample_SD='',paired_loss_pp_mean=100*(oc-count)/20,paired_loss_pp_sample_SD='',paired_baseline='Original, not fitted Clean',mean_state_distance=ms(r['state_distance'] for r in rr)[0],state_distance_sample_SD='',paired_state_distance_change=ms(r['paired_state_distance_change'] for r in rr)[0],paired_state_distance_change_sample_SD='',source_id=rr[0]['_source_id'],source_CSV_row='all rows for method'))
    return out

def deviations(data):
    out=[];expected=read_table(data,'final/DELTA_D.csv')
    for (method,domain,subset),rr in sorted(group(read_table(data,'final/DELTA_D_WINDOWS.csv'),'method','precision_domain','subset').items()):
        n=320 if subset=='whole320' else 32;assert len(rr)==n
        value=ms(r['normalized_delta_d'] for r in rr)[0]
        ref=next(r for r in expected if r['method']==method and r['precision_domain']==domain and r['subset']==subset)
        # Preserve the recorded device FP32 reduction; NumPy float64 reaggregation
        # is checked with the exact closed helper's tolerance, not substituted.
        assert math.isclose(value,float(ref['delta_d']),rel_tol=2e-7,abs_tol=1e-8)
        recorded=float(ref['delta_d'])
        out.append(dict(method=method,precision_domain=domain,subset=subset,windows=n,delta_d=recorded,Clean_delta_d_ref=ref['Clean_delta_d_ref'],poison_minus_Clean=recorded-float(ref['Clean_delta_d_ref']),per_window_float64_mean=value,recorded_reduction_minus_float64_mean=recorded-value,denominator='per-window norm of frozen Original prediction +1e-8',features='196 x394 successor-state entries; fixed action tail excluded',CUSUM_input=False,source_id=ref['_source_id'],source_CSV_row=ref['_CSV_row']))
    return out

def detector(data):
    raw=read_table(data,'final/RESIDUALS.csv');mat={};ids={}
    for (role,),rr in group(raw,'role').items():
        trajectories=sorted(group(rr,'trajectory_ordinal').items(),key=lambda item:int(item[0][0]))
        mat[role]=np.asarray([[float(r['residual_FP32_raw_L2']) for r in sorted(v,key=lambda r:int(r['local_window']))] for _,v in trajectories]);ids[role]=[v[0]['record_index'] for _,v in trajectories]
    assert [len(mat[r]) for r in ['statistics','threshold','clean_test','matched_Clean40','alpha0','alpha09']]==[200,200,400,40,40,40]
    assert len(set(ids['statistics']+ids['threshold']+ids['clean_test']+ids['matched_Clean40']))==840
    median,mad=robust_stats(mat['statistics'].astype(np.float32).reshape(-1));lock=data['detector_lock'];near(median,lock['median']);near(mad,lock['mad'])
    scores={r:np.asarray([cusum_score(t,median=median,mad=mad,drift_factor=6.) for t in m]) for r,m in mat.items() if r!='statistics'}
    for r in read_table(data,'final/SCORES.csv'):near(scores[r['role']][int(r['trajectory_ordinal'])],r['score'])
    thresholds={}
    for level in TARGET_FPRS:
        cutoff,rank=split_conformal_threshold(scores['threshold'],level);key=f'{level:.2f}';near(cutoff,lock['thresholds'][key]['threshold']);assert rank==lock['thresholds'][key]['rank_one_based'];thresholds[key]=cutoff
    out=[];cache={};recorded_rank=read_table(data,'final/CUSUM_RANKING.csv');recorded_op=read_table(data,'final/CUSUM_OPERATING.csv')
    for method in ['alpha0','alpha09']:
        ps=scores[method];cs=scores['clean_test'];mscores=scores['matched_Clean40'];key=tuple(ps)
        if key not in cache:cache[key]=evaluate_score_samples(cs,ps,thresholds,bootstrap_resamples=5000,bootstrap_seed=20261104,confidence=.95)
        metric=cache[key];rank=next(r for r in recorded_rank if r['method']==method)
        near(metric['auc'],rank['AUC'])
        for edge,suffix in [('lower','low'),('upper','high')]:near(metric['intervals']['auc'][edge],rank['AUC_CI95_'+suffix])
        for level in TARGET_FPRS:
            k=f'{level:.2f}';suffix='5' if level==.05 else '10';p=metric['operating_points'][k];cut=thresholds[k];ca=mscores>cut;pa=ps>cut
            expected=next(r for r in recorded_op if r['method']==method and float(r['nominal_FPR'])==level)
            for field,recorded in [('tpr','TPR'),('realized_fpr','realized_Clean_FPR')]:near(p[field],expected[recorded])
            for metric_name in ['standardized_pauc']:
                near(metric[metric_name][k],rank['standardized_pAUC'+suffix])
                for edge,suf in [('lower','low'),('upper','high')]:near(metric['intervals'][metric_name][k][edge],rank['standardized_pAUC'+suffix+'_CI95_'+suf])
            out.append(dict(method=method,nominal_FPR=level,threshold=cut,threshold_rank=lock['thresholds'][k]['rank_one_based'],statistics_trajectories=200,threshold_trajectories=200,median=median,MAD=mad,drift_factor=6,poison_alarms=int(pa.sum()),incoming_trajectories=40,TPR=p['tpr'],TPR_Wilson95_low=p['intervals']['tpr']['lower'],TPR_Wilson95_high=p['intervals']['tpr']['upper'],independent_Clean_alarms=p['num_clean_alarms'],independent_Clean_trajectories=400,realized_Clean_FPR=p['realized_fpr'],FPR_Wilson95_low=p['intervals']['realized_fpr']['lower'],FPR_Wilson95_high=p['intervals']['realized_fpr']['upper'],matched_Clean_alarms=int(ca.sum()),newly_acquired_alarms=int((~ca&pa).sum()),resolved_alarms=int((ca&~pa).sum()),AUC=metric['auc'],AUC_CI95_low=metric['intervals']['auc']['lower'],AUC_CI95_high=metric['intervals']['auc']['upper'],standardized_pAUC=metric['standardized_pauc'][k],pAUC_CI95_low=metric['intervals']['standardized_pauc'][k]['lower'],pAUC_CI95_high=metric['intervals']['standardized_pauc'][k]['upper'],interval_unit='independent-by-class trajectory bootstrap, 5000, seed20261104; no detector refit',primary_attack_replications=1))
    return out

