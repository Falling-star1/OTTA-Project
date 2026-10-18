import copy
import unittest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from tent_text import install_lora, make_optimizer, snapshot_adapters, _entropy_forward

try:
    import full_core as core
except ModuleNotFoundError as exc:
    if exc.name != 'full_core':
        raise
    core = None


def tiny_model():
    torch.manual_seed(42)
    cfg = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                      attention_dropout=0., max_position_embeddings=8192)
    return install_lora(Qwen2ForCausalLM(cfg), rank=2, alpha=4)


class ImplementationTest(unittest.TestCase):
    def test_suffix_core_exists(self):
        self.assertIsNotNone(core, 'Full-run suffix core is not implemented')


@unittest.skipIf(core is None, 'waiting for suffix implementation')
class SuffixTest(unittest.TestCase):
    def test_preflight_requires_real_changes_on_both_devices(self):
        report = {'changed_elements': 1, 'gradient_l1_by_device': {'cuda:0': 1., 'cuda:1': 2.}}
        core.validate_preflight_update(report)
        for invalid in ({**report, 'changed_elements': 0},
                        {**report, 'gradient_l1_by_device': {'cuda:0': 0., 'cuda:1': 2.}},
                        {**report, 'gradient_l1_by_device': {'cuda:0': 1.}}):
            with self.assertRaises(ValueError):
                core.validate_preflight_update(invalid)

    def test_suffix_loss_and_gradients_equal_full_logits(self):
        for ids, prompt in [(torch.tensor([[1, 2, 3, 4, 5, 6]]), 3),
                            (torch.tensor([[1, 2, 3, 4, 5, 30, 0, 7]]), 3)]:
            a = tiny_model()
            b = copy.deepcopy(a)
            full = _entropy_forward(a, ids, prompt, [30], 0)
            suffix = core.suffix_entropy(b, ids, prompt, [30], 0)
            self.assertAlmostEqual(full.item(), suffix.item(), places=6)
            full.backward()
            suffix.backward()
            for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
                if p.requires_grad:
                    torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-7, msg=name)

    def test_suffix_step_matches_reference_adam_step(self):
        a = tiny_model()
        b = copy.deepcopy(a)
        oa, ob = make_optimizer(a, 1e-3), make_optimizer(b, 1e-3)
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        loss = _entropy_forward(a, ids, 3, [], None)
        loss.backward()
        oa.step()
        report = core.entropy_step(b, ob, ids, 3, [], None)
        self.assertEqual(report['optimizer_steps'], 1)
        for name, value in snapshot_adapters(a).items():
            torch.testing.assert_close(value, snapshot_adapters(b)[name], rtol=1e-5, atol=1e-7)

    def test_finite_zero_parameter_change_is_valid_for_full_run(self):
        model = tiny_model()
        report = core.entropy_step(model, make_optimizer(model, 0.),
                                   torch.tensor([[1, 2, 3, 4, 5]]), 3, [], None)
        self.assertTrue(report['zero_update'])
        self.assertEqual(report['optimizer_steps'], 1)

    def test_nonfinite_gradient_does_not_step(self):
        model = tiny_model()
        optimizer = make_optimizer(model, 1e-3)
        before = snapshot_adapters(model)
        p = next(p for n, p in model.named_parameters() if '.lora_B.' in n)
        hook = p.register_hook(lambda g: torch.full_like(g, float('nan')))
        try:
            with self.assertRaises(FloatingPointError):
                core.entropy_step(model, optimizer, torch.tensor([[1, 2, 3, 4, 5]]), 3, [], None)
        finally:
            hook.remove()
        self.assertFalse(optimizer.state)
        for name, value in snapshot_adapters(model).items():
            self.assertTrue(torch.equal(value, before[name]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
