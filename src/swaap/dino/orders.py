import hashlib
import json
ROWS,EPOCHS,BATCH_SIZE,UPDATES=320,10,16,200
def slate_hash(slate):
    return hashlib.sha256(json.dumps(slate, separators=(',', ':')).encode()).hexdigest()

def make_source_slate(torch, seed=99, data_loader=None):
    """Derive the native N320 frozen-seed CPU shuffle without advancing caller RNG.

    No explicit DataLoader generator is supplied: that would change the source
    RandomSampler's CPU RNG consumption. CUDA/dropout consumes a separate RNG.
    This helper neither loads a model nor creates a CUDA context.
    """
    assert isinstance(seed, int) and seed in (99, 100, 101), 'Only the three prospectively frozen fit seeds'
    state = torch.get_rng_state()
    try:
        isolated = torch.Generator(device='cpu').manual_seed(seed)
        torch.set_rng_state(isolated.get_state())
        loader_type = data_loader or torch.utils.data.DataLoader
        loader = loader_type(torch.utils.data.TensorDataset(torch.arange(ROWS)),
                             batch_size=BATCH_SIZE, shuffle=True)
        slate = []
        for _ in range(EPOCHS):
            epoch = []
            for (indices,) in loader:
                epoch.extend(int(value) for value in indices.tolist())
            assert sorted(epoch) == list(range(ROWS))
            slate.append(epoch)
        return slate
    finally:
        torch.set_rng_state(state)
