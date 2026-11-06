"""Exact ordered support handoffs, with no target search or model execution."""
import hashlib,json
from pathlib import Path
import numpy as np
import pytest
import torch
from swaap.latent.io import load_support
ROOT=Path(__file__).resolve().parents[1]


def test_all_five_retained_tied_supports_preserve_source_order(tmp_path):
    folder=ROOT/'configs/tdmpc2'
    data=json.loads((folder/'trajectory_conditions.json').read_text())
    assert len(data['conditions'])==5
    assert sum(r['task']=='humanoid-walk'for r in data['conditions'])==3
    for row in data['conditions']:
        path=folder/row['support_file'];value=load_support(path,rows=5000,count=250)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['support_file_sha256']
        assert hashlib.sha256(value.numpy().astype('<i8',copy=False).tobytes()).hexdigest()==row['ordered_int64_sha256']
        assert not any(int(i)%500==499 for i in value)
        if row['original_source_support_format']=='json':
            assert row['original_source_support_sha256']==row['support_file_sha256']
        # Original closed dispatch accepts these same values as a CPU int64
        # tensor. Moving between the public text format and Torch cannot sort.
        binary=tmp_path/(row['task']+str(row['run_index'])+'.pt');torch.save(value,binary)
        assert torch.equal(value,load_support(binary,rows=5000,count=250))
        assert row['alpha']==.9 and row['poison_fraction']==.05
        assert row['ordered_source_tensor_equality_verified'] and not row['support_reselected']


def test_support_format_never_coerces_bool_or_repairs_duplicates(tmp_path):
    p=tmp_path/'bad.json';p.write_text('[true, 2]')
    with pytest.raises(ValueError,match='integer'):load_support(p,rows=5,count=2)
    p.write_text('[2,2]')
    with pytest.raises(ValueError,match='duplicated'):load_support(p,rows=5,count=2)
