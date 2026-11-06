"""Executed FP32 predictor-only replacement; preserve Original module modes."""
from collections.abc import Mapping
from copy import deepcopy

def predictor_replacement(rt, target_state):
    """Return (FP32 Original with target predictor, JSON-compatible proof).

    This function does not set eval/train, instantiate a new architecture,
    initialize an optimizer, read an asset or write a checkpoint. The caller
    retains responsibility for the target file SHA and task/goal bindings.
    """
    torch = rt.torch
    assert isinstance(target_state, Mapping)
    original_state = rt.original.state_dict()
    original_predictor = rt.original.predictor.state_dict()
    assert len(original_state) == 295 and len(original_predictor) == 69
    assert set(target_state) == set(original_predictor), 'Strict predictor architecture/key binding'
    for name, value in original_state.items():
        assert torch.is_tensor(value)
        if value.is_floating_point():
            assert value.dtype == torch.float32, f'Original must remain canonical FP32: {name}'
    for name, value in target_state.items():
        assert torch.is_tensor(value) and value.shape == original_predictor[name].shape, name
        assert value.dtype == original_predictor[name].dtype == torch.float32, name
        assert bool(value.isfinite().all()), name

    original_modes = {name: module.training for name, module in rt.original.named_modules()}
    original_flags = {name: value.requires_grad for name, value in rt.original.named_parameters()}
    model = deepcopy(rt.original)
    loaded = model.predictor.load_state_dict(target_state, strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    actual_state = model.state_dict()
    assert set(actual_state) == set(original_state)
    assert {name: module.training for name, module in model.named_modules()} == original_modes
    assert {name: value.requires_grad for name, value in model.named_parameters()} == original_flags
    nonpredictor = [name for name in actual_state if not name.startswith('predictor.')]
    assert len(nonpredictor) == 226
    assert all(torch.equal(actual_state[name], original_state[name]) for name in nonpredictor)
    changed = []
    for name, value in actual_state.items():
        if value.is_floating_point():
            assert value.dtype == torch.float32, name
        if not torch.equal(value, original_state[name]):
            changed.append(name)
    assert changed and all(name.startswith('predictor.') for name in changed), 'Only a nonempty predictor subset may change'
    for name, value in model.predictor.state_dict().items():
        assert torch.equal(value, target_state[name].to(device=value.device)), name

    verification = dict(
        status='PASS_FP32_ORIGINAL_PREDICTOR_ONLY_REPLACEMENT',
        original_state_tensors=295, predictor_state_tensors=69,
        nonpredictor_state_tensors_exact=226,
        changed_state_keys=changed, changed_state_tensor_count=len(changed),
        changed_state_is_nonempty_predictor_subset=True,
        all_floating_state_FP32=True, target_predictor_exact=True,
        module_mode_count=len(original_modes), module_modes_exactly_Original=True,
        parameter_requires_grad_flags_exactly_Original=True,
        BF16_conversion=False, optimizer_updates=0,
        architecture_initializations=0, saved_new_weights=False,
        target_binding='Caller supplies pinned Stage1 update25 predictor state only',
        evaluation_binding='Caller uses the unchanged qualified base.goal MPC/CEM10 path')
    return model, verification
