"""Tent-text-LoRA engineering adaptation; no reference-answer argument."""
import hashlib
import json
import re

import torch
from peft import LoraConfig, TaskType, get_peft_model


def prompt_text(row):
    instruction, context = row['instruction'], row['input']
    if not isinstance(instruction, str) or not isinstance(context, str):
        raise ValueError('instruction/input must be strings')
    return instruction + ('\n' + context if context else '')


def encode_prompt(row, tokenizer, template, system, max_input):
    """Match Source: trim only user content, then apply the complete template."""
    original = prompt_text(row)
    used = original

    def encode(content):
        return template.encode_oneturn(tokenizer, [
            {'role': 'user', 'content': content},
            {'role': 'assistant', 'content': ''}], system=system)[0]

    ids = encode(used)
    original_length = len(ids)
    if original_length > max_input:
        content_ids = tokenizer.encode(original, add_special_tokens=False)
        while len(ids) > max_input:
            content_ids = content_ids[:-max(1, len(ids) - max_input)]
            if not content_ids:
                raise ValueError('Prompt budget too small for the TLM template')
            used = tokenizer.decode(content_ids, skip_special_tokens=False)
            ids = encode(used)
    if not ids:
        raise ValueError('Empty encoded prompt')
    return used, ids, {'original_prompt_tokens': original_length,
                      'used_prompt_tokens': len(ids),
                      'input_truncated': original_length > len(ids),
                      'original_prompt_sha256': hashlib.sha256(original.encode('utf-8')).hexdigest()}


def generated_positions(input_ids, prompt_length, stop_ids, pad_id):
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.dtype != torch.long:
        raise ValueError('Expected batch-one int64 token IDs')
    length = input_ids.shape[1]
    if not 1 <= prompt_length < length:
        raise ValueError('Expected a nonempty prompt and generated span')
    targets = input_ids[:, prompt_length:]
    special = torch.zeros_like(targets, dtype=torch.bool)
    ignored = set(stop_ids)
    if pad_id is not None:
        ignored.add(pad_id)
    for token_id in ignored:
        special |= targets == token_id
    # Exclude the first stop/pad token and every position after it.
    valid = (~special).to(torch.int64).cumprod(dim=1).bool()
    mask = torch.zeros((1, length - 1), dtype=torch.bool, device=input_ids.device)
    mask[:, prompt_length - 1:] = valid
    if not bool(mask.any()):
        raise ValueError('No valid generated tokens for entropy minimization')
    return mask


def generated_entropy(logits, input_ids, prompt_length, stop_ids, pad_id):
    if logits.ndim != 3 or tuple(logits.shape[:2]) != tuple(input_ids.shape):
        raise ValueError('Logits and token sequence lengths must match')
    mask = generated_positions(input_ids, prompt_length, stop_ids, pad_id)
    selected = logits[:, :-1, :][mask.to(logits.device)].float()
    if not bool(torch.isfinite(selected).all()):
        raise FloatingPointError('Nonfinite logits at a selected generation position')
    log_probability = selected.log_softmax(dim=-1)
    entropy = -(log_probability.exp() * log_probability).sum(dim=-1).mean()
    if not bool(torch.isfinite(entropy)):
        raise FloatingPointError('Nonfinite entropy')
    return entropy


def is_adapter_name(name):
    return re.search(r'(?:^|\.)(q_proj|v_proj)\.lora_[AB]\.default\.weight$', name) is not None


def trainable_parameters(model):
    pairs = []
    for name, parameter in model.named_parameters():
        expected = is_adapter_name(name)
        if parameter.requires_grad != expected:
            raise ValueError('Unexpected trainability: ' + name)
        if expected:
            if parameter.dtype != torch.float32:
                raise ValueError('Adapter must be FP32: ' + name)
            pairs.append((name, parameter))
        elif parameter.grad is not None:
            raise ValueError('Frozen base parameter has a gradient: ' + name)
    if not pairs:
        raise ValueError('No q/v LoRA parameters found')
    return pairs


def install_lora(model, rank, alpha):
    model.requires_grad_(False)
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha,
                        lora_dropout=0., target_modules=['q_proj', 'v_proj'],
                        bias='none', init_lora_weights='gaussian', inference_mode=False)
    model = get_peft_model(model, config)
    for name, parameter in model.named_parameters():
        if is_adapter_name(name):
            parameter.data = parameter.data.to(dtype=torch.float32)
    trainable_parameters(model)
    model.train()
    return model


def make_optimizer(model, learning_rate):
    return torch.optim.Adam([p for _, p in trainable_parameters(model)],
                            lr=learning_rate, betas=(.9, .999), eps=1e-8, weight_decay=0.)


def snapshot_adapters(model):
    return {name: p.detach().cpu().clone() for name, p in trainable_parameters(model)}


def reset_adapters(model, optimizer, initial):
    pairs = trainable_parameters(model)
    if set(initial) != {name for name, _ in pairs}:
        raise ValueError('Adapter reset names do not match')
    with torch.no_grad():
        for name, parameter in pairs:
            parameter.copy_(initial[name].to(parameter.device))
    optimizer.state.clear()
    optimizer.zero_grad(set_to_none=True)


def frozen_digest(model, chunk_bytes=4 * 1024 * 1024):
    """Hash ALL frozen base tensors with bounded CPU transfer, no model clone."""
    if chunk_bytes < 1:
        raise ValueError('chunk_bytes must be positive')
    digest = hashlib.sha256()
    count = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if is_adapter_name(name):
                continue
            if parameter.requires_grad or not parameter.is_contiguous():
                raise ValueError('Expected contiguous frozen parameter: ' + name)
            digest.update(json.dumps([name, list(parameter.shape), str(parameter.dtype)]).encode())
            flat = parameter.detach().view(-1)
            step = max(1, chunk_bytes // parameter.element_size())
            for start in range(0, flat.numel(), step):
                block = flat[start:start + step].cpu().contiguous().view(torch.uint8)
                digest.update(block.numpy().tobytes())
            count += 1
    if not count:
        raise ValueError('No frozen base tensors found')
    return digest.hexdigest()


def _entropy_forward(model, input_ids, prompt_length, stop_ids, pad_id):
    outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                    use_cache=False, return_dict=True)
    return generated_entropy(outputs.logits, input_ids, prompt_length, stop_ids, pad_id)


@torch.enable_grad()
def adapt_once(model, optimizer, input_ids, prompt_length, stop_ids, pad_id):
    """One step on fixed detached generated IDs; no labels or task scores."""
    pairs = trainable_parameters(model)
    expected = {id(p) for _, p in pairs}
    actual = [p for group in optimizer.param_groups for p in group['params']]
    if len(actual) != len(expected) or {id(p) for p in actual} != expected:
        raise ValueError('Optimizer must contain exactly the trainable adapter parameters')
    input_ids = input_ids.detach()
    selected = generated_positions(input_ids, prompt_length, stop_ids, pad_id)
    initial = snapshot_adapters(model)
    optimizer.zero_grad(set_to_none=True)
    loss = _entropy_forward(model, input_ids, prompt_length, stop_ids, pad_id)
    before = float(loss.detach())
    loss.backward()
    gradient_l1 = {}
    for name, parameter in pairs:
        grad = parameter.grad
        if grad is None:
            raise ValueError('Missing adapter gradient: ' + name)
        if not bool(torch.isfinite(grad).all()):
            raise FloatingPointError('Nonfinite adapter gradient: ' + name)
        device = str(parameter.device)
        gradient_l1[device] = gradient_l1.get(device, 0.) + float(grad.detach().abs().sum())
    if not gradient_l1 or any(value <= 0 for value in gradient_l1.values()):
        raise ValueError('Expected nonzero aggregate adapter gradients on every participating device')
    trainable_parameters(model)
    optimizer.step()
    changed_elements = 0
    max_change = 0.
    for name, parameter in pairs:
        if not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError('Nonfinite updated adapter: ' + name)
        delta = parameter.detach().cpu() - initial[name]
        changed_elements += int(torch.count_nonzero(delta))
        max_change = max(max_change, float(delta.abs().max()))
    if changed_elements == 0:
        raise ValueError('Optimizer did not change any adapter parameter')
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                raise FloatingPointError('Nonfinite Adam state')
    del loss
    with torch.no_grad():
        after = float(_entropy_forward(model, input_ids, prompt_length, stop_ids, pad_id))
    return {'entropy_before': before, 'entropy_after': after, 'entropy_unit': 'nat',
            'valid_generated_tokens': int(selected.sum()),
            'gradient_l1_by_device': gradient_l1, 'changed_elements': changed_elements,
            'max_parameter_change': max_change, 'optimizer_steps': 1}
