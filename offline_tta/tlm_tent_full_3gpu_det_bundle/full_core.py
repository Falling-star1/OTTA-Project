"""Same generated-token entropy, without projecting the entire long prompt."""
import inspect
import math
import torch
from tent_text import generated_entropy, generated_positions, snapshot_adapters, trainable_parameters
from gpu_layout import logical_devices


def validate_preflight_update(report, device_count=2):
    gradients = report['gradient_l1_by_device']
    if report['changed_elements'] <= 0 or set(gradients) != logical_devices(device_count) or not all(
            math.isfinite(value) and value > 0 for value in gradients.values()):
        raise ValueError('Preflight must change adapters and have positive gradients on every selected GPU')


def suffix_entropy(model, ids, prompt_length, stop_ids, pad_id):
    generated_positions(ids, prompt_length, stop_ids, pad_id)
    base = model.get_base_model()
    if 'num_logits_to_keep' not in inspect.signature(base.forward).parameters:
        raise ValueError('Pinned model must support num_logits_to_keep')
    # Keep last prompt position + all generated positions. The final position
    # predicts beyond the supplied sequence and is excluded by generated_entropy.
    keep = ids.shape[1] - prompt_length + 1
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                return_dict=True, num_logits_to_keep=keep)
    cropped_ids = ids[:, prompt_length - 1:]
    return generated_entropy(out.logits, cropped_ids, 1, stop_ids, pad_id)


@torch.enable_grad()
def entropy_step(model, optimizer, ids, prompt_length, stop_ids, pad_id, measure_after=False):
    pairs = trainable_parameters(model)
    actual = [p for group in optimizer.param_groups for p in group['params']]
    if len(actual) != len(pairs) or {id(p) for p in actual} != {id(p) for _, p in pairs}:
        raise ValueError('Adam must contain exactly the q/v adapter parameters')
    ids = ids.detach()
    count = int(generated_positions(ids, prompt_length, stop_ids, pad_id).sum())
    before_params = snapshot_adapters(model)
    optimizer.zero_grad(set_to_none=True)
    loss = suffix_entropy(model, ids, prompt_length, stop_ids, pad_id)
    before = float(loss.detach())
    loss.backward()
    gradients = {}
    for name, parameter in pairs:
        if parameter.grad is None:
            raise ValueError('Missing adapter gradient: ' + name)
        if not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError('Nonfinite adapter gradient: ' + name)
        device = str(parameter.device)
        gradients[device] = gradients.get(device, 0.) + float(parameter.grad.detach().abs().sum())
    trainable_parameters(model)
    optimizer.step()
    changed, maximum = 0, 0.
    for name, parameter in pairs:
        if not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError('Nonfinite adapter after update: ' + name)
        difference = parameter.detach().cpu() - before_params[name]
        changed += int(torch.count_nonzero(difference))
        maximum = max(maximum, float(difference.abs().max()))
    for values in optimizer.state.values():
        for value in values.values():
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                raise FloatingPointError('Nonfinite Adam state')
    del loss
    after = None
    if measure_after:
        with torch.no_grad():
            after = float(suffix_entropy(model, ids, prompt_length, stop_ids, pad_id))
    return {'entropy_before': before, 'entropy_after': after, 'valid_generated_tokens': count,
            'gradient_l1_by_device': gradients, 'changed_elements': changed,
            'max_parameter_change': maximum, 'zero_update': changed == 0,
            'optimizer_steps': 1, 'logits_strategy': 'last_generated_length_plus_one'}
