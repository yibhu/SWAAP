"""Unchanged saved-permutation extension used for the retained E25/E50 fits."""
from .io import require as need

class _EndpointReached(Exception):
    pass

def continue_exact_orders(native_loop, agent, dynamics, optimizer, buffer, schedule, torch, *, start_updates, target_updates, epoch_callback=None):
    """Call the unchanged numerical loop; offset/rotate only saved epoch orders.

    Adam and RNG remain continuous. The only early exit is after a complete
    epoch, including the original 136-row last minibatch, at the exact endpoint.
    """
    need(start_updates % 20 == target_updates % 20 == 0 and 0 <= start_updates < target_updates, 'Continuation endpoints must be complete native epochs')
    need(len(schedule['permutations']) == 10, 'Exactly ten saved epoch orders are required')
    history = []
    completed = start_updates
    while completed < target_updates:
        base = completed
        offset = base // 20 % 10
        rotated = dict(schedule)
        rotated['permutations'] = schedule['permutations'][offset:] + schedule['permutations'][:offset]

        def completed_epoch(epoch, steps, row):
            nonlocal completed
            completed = base + steps
            adjusted = dict(row, epoch=base // 20 + epoch, optimizer_updates=completed)
            history.append(adjusted)
            if epoch_callback is not None:
                epoch_callback(adjusted['epoch'], completed, adjusted)
            need(completed <= target_updates, 'Continuation exceeded its fixed update endpoint')
            if completed == target_updates and steps < 200:
                raise _EndpointReached()
        try:
            _, steps = native_loop(agent, dynamics, optimizer, buffer, rotated, torch, epoch_callback=completed_epoch)
            need(steps == 200, 'Shared native chunk did not close 200 updates')
        except _EndpointReached:
            need(completed == target_updates, 'Unexpected shortened native chunk')
    return (history, completed)
