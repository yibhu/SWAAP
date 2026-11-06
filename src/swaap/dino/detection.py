"""Exact fixed10 Original-policy DINO detector collection and saved-array handoff."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from swaap.latent.io import asset,output,write_json,require,digest
from .metrics import residuals_from_predictions

COUNTS={'statistics':200,'threshold':200,'clean_test':400}


def collect(config):
    from .pipeline import _runtime,_predict,_eval_frozen
    from .collection import collect as collect_record
    from swaap.evaluation.detector import calibrate
    root=output(config);rt=_runtime(config,planning=True);reference=_eval_frozen(deepcopy(rt.original))
    require(all(p.dtype==rt.torch.float32 for p in reference.parameters()),'Incoming DINO reference must remainFP32')
    records=rt.torch.load(asset(config,'records' if config.get('detector',{}).get('role') and config.get('assets',{}).get('records') else 'detector_records'),map_location='cpu',weights_only=False)
    smoke=int(config.get('collection',{}).get('maximum_trajectories',0))
    require(len(records)==800 or bool(smoke and records),'DINO detector needs fixed200/200/400 record manifest')
    require(bool(smoke) or {role:sum(r['role']==role for r in records) for role in COUNTS}==COUNTS,'DINO record roles changed')
    ids=[int(r['record_index']) for r in records];require(len(ids)==len(set(ids)),'DINO detector record IDs overlap')
    smoke=int(config.get('collection',{}).get('maximum_trajectories',0));used=0;result={};original_sha=digest(asset(config,'original'))
    requested=config.get('detector',{}).get('role')
    if requested=='stats':requested='statistics'
    require(requested is None or requested in COUNTS,'Unknown detector role')
    roles=[requested] if requested else list(COUNTS)
    for role in roles:
        total=COUNTS[role]
        if role=='clean_test' and not smoke:
            import json
            calibration=asset(config,'calibration') if requested else root/'calibration.json'
            require(calibration.is_file(),'Thresholds must lock before clean-test collection')
            require(json.loads(calibration.read_text())['original_sha256']==original_sha,'Frozen detector and DINO Original differ')
        rows=[];record_ids=[]
        for row in [r for r in records if r['role']==role]:
            bank,metadata=collect_record(rt,row,root,root)
            prediction=_predict(reference,bank)
            origins=[{'record_index':row['record_index'],'local_window':i,'last_current_grouped_step':i+2,'successor_grouped_step':i+3} for i in range(8)]
            residual,joined_ids=residuals_from_predictions(bank[3],prediction,origins)
            require(residual.shape==(1,8) and joined_ids==[row['record_index']],'DINO trajectory/window identity join changed')
            rows.append(residual[0]);record_ids.append(str(row['record_index']));used+=1
            if smoke and used>=smoke:break
        path=root/(role+'.npz')
        with path.open('xb') as stream:np.savez_compressed(stream,residuals=np.stack(rows).astype(np.float32,copy=False),trajectory_ids=np.asarray(record_ids),original_sha256=np.asarray(original_sha))
        result[role]={'trajectories':len(rows),'residual_file':str(path)}
        if smoke and used>=smoke:break
        if role=='threshold' and not requested:calibrate(root/'statistics.npz',root/'threshold.npz',root/'calibration.json',family='dino-wm')
    summary={'status':'COLLECTION_SMOKE_ONLY' if smoke else 'COMPLETE_CLEAN_COHORTS','environment_trajectories':used,'windows_per_trajectory':8,'precision':'FP32 Original','roles':result,'calibration_frozen':(root/'calibration.json').is_file()}
    write_json(root/'detector_collection.json',summary);return summary


def prepare_saved(config):
    """Preserve released FP32 residual values and exact record IDs without models."""
    root=output(config);reference=digest(asset(config,'original'));result={}
    for role,count in COUNTS.items():
        with np.load(asset(config,'detector_'+role),allow_pickle=False) as packet:
            residuals=packet['residuals'].copy();ids=packet['record_ids'].astype(str)
        require(residuals.shape==(count,8) and len(ids)==count and len(set(ids))==count,'Released DINO cohort shape/IDs changed')
        path=root/(role+'.npz')
        with path.open('xb') as stream:np.savez_compressed(stream,residuals=residuals,trajectory_ids=ids,original_sha256=np.asarray(reference))
        result[role]=str(path)
    write_json(root/'saved_cohorts.json',{'derived_schema_only':True,'residual_values_unchanged':True,'cohorts':result});return result
