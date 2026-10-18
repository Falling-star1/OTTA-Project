"""CPU tests; no downloaded model, network, reference-answer training or pytest."""
import math
import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

try:
    import tent_text as tt
except ModuleNotFoundError as exc:
    if exc.name != 'tent_text':
        raise
    tt = None


class ImplementationTest(unittest.TestCase):
    def test_implementation_exists(self):
        self.assertIsNotNone(tt, 'Tent core has not been implemented yet')


@unittest.skipIf(tt is None, 'waiting for core implementation')
class EntropyTest(unittest.TestCase):
    def test_uniform_and_peaked_entropy(self):
        ids = torch.tensor([[1, 2, 3]])
        uniform = torch.zeros(1, 3, 2, requires_grad=True)
        loss = tt.generated_entropy(uniform, ids, 2, [], None)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        peaked = torch.tensor([[[0., 0.], [8., -8.], [0., 0.]]])
        self.assertLess(tt.generated_entropy(peaked, ids, 2, [], None).item(), .001)
        loss.backward()
        self.assertIsNotNone(uniform.grad)

    def test_causal_alignment_excludes_prompt_stop_pad_and_trailing_tokens(self):
        # Prompt tokens 1,2,3; generated 4,5,EOS=9,PAD=0,7.
        ids = torch.tensor([[1, 2, 3, 4, 5, 9, 0, 7]])
        logits = torch.full((1, 8, 2), float('nan'))
        logits[0, 2] = torch.tensor([0., 0.])
        logits[0, 3] = torch.tensor([2., -2.])
        logits.requires_grad_()
        expected = (math.log(2) + float(torch.distributions.Categorical(logits=torch.tensor([2., -2.])).entropy())) / 2
        loss = tt.generated_entropy(logits, ids, 3, [9], 0)
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        self.assertGreater(logits.grad[0, 3].abs().sum().item(), 0)
        self.assertEqual(logits.grad[0, :2].abs().sum().item(), 0)
        self.assertEqual(logits.grad[0, 4:].abs().sum().item(), 0)

    def test_nonfinite_selected_logits_fail(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            logits = torch.zeros(1, 3, 2)
            logits[0, 1, 0] = value
            with self.assertRaises(FloatingPointError):
                tt.generated_entropy(logits, torch.tensor([[1, 2, 3]]), 2, [], None)

    def test_empty_and_invalid_spans_fail(self):
        logits = torch.zeros(1, 3, 2)
        with self.assertRaises(ValueError):
            tt.generated_entropy(logits, torch.tensor([[1, 2, 9]]), 2, [9], None)
        for prompt in (0, 3, 4):
            with self.assertRaises(ValueError):
                tt.generated_entropy(logits, torch.tensor([[1, 2, 3]]), prompt, [], None)
        with self.assertRaises(ValueError):
            tt.generated_entropy(logits[:, :2], torch.tensor([[1, 2, 3]]), 2, [], None)

    def test_half_logits_accumulate_in_float32(self):
        result = tt.generated_entropy(torch.zeros(1, 3, 2, dtype=torch.float16),
                                      torch.tensor([[1, 2, 3]]), 2, [], None)
        self.assertEqual(result.dtype, torch.float32)
        self.assertAlmostEqual(result.item(), math.log(2), places=6)


class GuardedRow(dict):
    def __getitem__(self, key):
        if key in ('output', 'label', 'answer'):
            raise AssertionError('Reference answer was accessed')
        return super().__getitem__(key)


@unittest.skipIf(tt is None, 'waiting for core implementation')
class UpdateTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=2, num_attention_heads=2,
                          num_key_value_heads=1, attention_dropout=0.)
        self.model = tt.install_lora(LlamaForCausalLM(cfg), rank=2, alpha=4)
        self.optimizer = tt.make_optimizer(self.model, learning_rate=1e-3)
        self.ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

    def test_only_qv_adapters_update_and_reset(self):
        before = tt.snapshot_adapters(self.model)
        self.assertTrue(any('.lora_A.' in n and p.abs().sum() > 0 for n, p in before.items()))
        self.assertTrue(all(p.count_nonzero() == 0 for n, p in before.items() if '.lora_B.' in n))
        digest = tt.frozen_digest(self.model, chunk_bytes=64)
        report = tt.adapt_once(self.model, self.optimizer, self.ids, 3, [], None)
        self.assertGreater(report['changed_elements'], 0)
        self.assertGreater(report['gradient_l1_by_device']['cpu'], 0)
        self.assertTrue(math.isfinite(report['entropy_before']))
        self.assertTrue(math.isfinite(report['entropy_after']))
        self.assertEqual(digest, tt.frozen_digest(self.model, chunk_bytes=128))
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self.assertTrue(tt.is_adapter_name(name))
                self.assertEqual(parameter.dtype, torch.float32)
            else:
                self.assertIsNone(parameter.grad)
        self.assertTrue(self.optimizer.state)
        tt.reset_adapters(self.model, self.optimizer, before)
        self.assertFalse(self.optimizer.state)
        for name, value in tt.snapshot_adapters(self.model).items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_reference_answer_isolation_and_deterministic_update(self):
        a = GuardedRow(instruction='Question', input='Context', output='CANARY_A')
        b = GuardedRow(instruction='Question', input='Context', output='CANARY_B')
        self.assertEqual(tt.prompt_text(a), 'Question\nContext')
        self.assertEqual(tt.prompt_text(a), tt.prompt_text(b))
        initial = tt.snapshot_adapters(self.model)
        first = tt.adapt_once(self.model, self.optimizer, self.ids, 3, [], None)
        updated = tt.snapshot_adapters(self.model)
        tt.reset_adapters(self.model, self.optimizer, initial)
        second = tt.adapt_once(self.model, self.optimizer, self.ids, 3, [], None)
        self.assertEqual(first['entropy_before'], second['entropy_before'])
        for name, value in tt.snapshot_adapters(self.model).items():
            self.assertTrue(torch.equal(value, updated[name]), name)

    def test_optimizer_with_frozen_parameter_is_rejected(self):
        base = next(p for n, p in self.model.named_parameters() if not tt.is_adapter_name(n))
        self.optimizer.add_param_group({'params': [base]})
        with self.assertRaises(ValueError):
            tt.adapt_once(self.model, self.optimizer, self.ids, 3, [], None)

    def test_nonfinite_gradient_aborts_before_optimizer_step(self):
        initial = tt.snapshot_adapters(self.model)
        parameter = next(p for n, p in self.model.named_parameters() if '.lora_B.' in n)
        hook = parameter.register_hook(lambda grad: torch.full_like(grad, float('nan')))
        try:
            with self.assertRaises(FloatingPointError):
                tt.adapt_once(self.model, self.optimizer, self.ids, 3, [], None)
        finally:
            hook.remove()
        self.assertFalse(self.optimizer.state)
        for name, value in tt.snapshot_adapters(self.model).items():
            self.assertTrue(torch.equal(value, initial[name]), name)

    def test_nonreentrant_checkpointing_preserves_adapter_gradients(self):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        report = tt.adapt_once(self.model, self.optimizer, self.ids, 3, [], None)
        self.assertGreater(report['changed_elements'], 0)
        self.assertGreater(report['gradient_l1_by_device']['cpu'], 0)

    def test_frozen_digest_detects_change(self):
        initial = tt.frozen_digest(self.model, chunk_bytes=64)
        base = next(p for n, p in self.model.named_parameters() if not tt.is_adapter_name(n))
        with torch.no_grad():
            base.view(-1)[0] += 1
        self.assertNotEqual(initial, tt.frozen_digest(self.model, chunk_bytes=64))


if __name__ == '__main__':
    unittest.main(verbosity=2)
