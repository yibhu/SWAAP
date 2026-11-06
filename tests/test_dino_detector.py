from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch


def test_dino_collector_writes_one_fp32_eight_window_trajectory(tmp_path,monkeypatch):
    from swaap.dino import pipeline,detection,collection
    from swaap.evaluation.detector import load
    original=torch.nn.Linear(2,2).float()
    rt=SimpleNamespace(torch=torch,original=original)
    monkeypatch.setattr(pipeline,'_runtime',lambda c,planning=False:rt)
    bank=(torch.zeros(8,3,196,404),torch.zeros(8,10),torch.zeros(8,1),torch.zeros(8,196,404),torch.zeros(8,1))
    monkeypatch.setattr(collection,'collect',lambda rt,row,log,data:(bank,{'record_index':row['record_index']}))
    monkeypatch.setattr(pipeline,'_predict',lambda model,bank:torch.ones(8,196,404,dtype=torch.float32))
    records=tmp_path/'records.pt';torch.save([{'record_index':42,'role':'statistics'}],records)
    checkpoint=tmp_path/'original.pt';checkpoint.write_bytes(b'exact-reference-fixture')
    root=tmp_path/'collected'
    result=detection.collect({'output_dir':str(root),'assets':{'records':str(records),'original':str(checkpoint)},'detector':{'role':'statistics'},'collection':{'maximum_trajectories':1}})
    got=load(root/'statistics.npz')
    assert got['residuals'].shape==(1,8) and got['ids']==['42']
    assert np.allclose(got['residuals'],np.sqrt(196*394))
    assert result['status']=='COLLECTION_SMOKE_ONLY' and not result['calibration_frozen']


def test_dino_evaluation_does_not_silently_fall_back_to_original(monkeypatch):
    from swaap.dino import pipeline
    monkeypatch.setattr(pipeline,'_runtime',lambda *a,**kw:pytest.fail('runtime should not start'))
    with pytest.raises(ValueError,match='requires assets.checkpoint'):
        pipeline.evaluate({'method':'swaap','assets':{}})


def test_normalization_tensors_are_exact_source_assignments_and_float32():
    from swaap.dino import stats
    expected={
      'ACTION_MEAN':[-0.0087,0.0068],'ACTION_STD':[0.2019,0.2002],
      'STATE_MEAN':[236.6155,264.5674,255.1307,266.3721,1.9584,-2.93032027,2.54307914],
      'STATE_STD':[101.1202,87.0112,52.7054,57.4971,1.7556,74.84556075,74.14009094],
      'PROPRIO_MEAN':[236.6155,264.5674,-2.93032027,2.54307914],
      'PROPRIO_STD':[101.1202,87.0112,74.84556075,74.14009094]}
    for key,values in expected.items():
        assert getattr(stats,key).dtype==torch.float32
        assert torch.equal(getattr(stats,key),torch.tensor(values,dtype=torch.float32))


def test_new_bank_origins_follow_controlled_then_untouched_tensor_order():
    from swaap.dino.pipeline import _fitting_origins
    rows=[{'record_index':2,'role':'reference'}, {'record_index':3,'role':'untouched'},
          {'record_index':5,'role':'controlled'}, {'record_index':7,'role':'controlled'}]
    origins=_fitting_origins(rows)
    assert len(origins)==24
    assert [origins[i]['record_index'] for i in (0,8,16)]==[5,7,3]
    for start in range(0,24,8):
        block=origins[start:start+8]
        assert [r['local_window'] for r in block]==list(range(8))
        assert [r['last_current_grouped_step'] for r in block]==list(range(2,10))
        assert [r['successor_grouped_step'] for r in block]==list(range(3,11))


def test_incoming_metrics_rejects_prediction_packet_from_different_new_bank(tmp_path):
    import torch
    import pytest
    from swaap.dino.pipeline import _incoming_metrics
    clean=tmp_path/'clean.pt';prediction=tmp_path/'predictions.pt'
    torch.save({'fitting':(None,None,None,None,None)},clean)
    torch.save({'clean_sha256':'0'*64,'Original_FP32_fitting':None},prediction)
    with pytest.raises(ValueError,match='different fitting bank'):
        _incoming_metrics({'output_dir':str(tmp_path/'out'),'assets':{'clean':str(clean),'reference_predictions':str(prediction)}})
