"""CPU + GPU together (--cpu-share). The arithmetic and the command line are tested everywhere; the PyTorch checks are skipped
(not failed) when PyTorch is not installed. They need no GPU: both 'devices' are the CPU, which is enough to prove that splitting
the micro-batches gives the same gradients as not splitting them.
Run on your own machine:  python -m unittest tests.test_hybrid -v"""
import importlib.util
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HAVE_TORCH = importlib.util.find_spec("torch") is not None


class ChooseShare(unittest.TestCase):
    def setUp(self):
        from persona_lm.torch_backend import choose_cpu_share
        self.choose = choose_cpu_share

    def test_slow_cpu_gets_nothing(self):
        n, t, base = self.choose(t_gpu=1.0, t_cpu=10.0, accum=4)
        self.assertEqual(n, 0)
        self.assertEqual(t, base)

    def test_cpu_three_times_slower_takes_one_of_four(self):
        n, t, base = self.choose(t_gpu=1.0, t_cpu=3.0, accum=4)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(t, 3.0)
        self.assertAlmostEqual(base, 4.0)

    def test_fast_cpu_takes_half(self):
        self.assertEqual(self.choose(1.0, 1.0, 4)[0], 2)

    def test_copy_overhead_can_cancel_the_gain(self):
        self.assertEqual(self.choose(1.0, 3.0, 4, overhead=1.5)[0], 0)

    def test_gpu_always_keeps_one_micro_batch(self):
        self.assertLessEqual(self.choose(1.0, 0.01, 4)[0], 3)

    def test_needs_at_least_two_micro_batches(self):
        self.assertEqual(self.choose(1.0, 0.5, 1)[0], 0)


class CommandLine(unittest.TestCase):
    def test_flags_exist_and_default_to_off(self):
        from persona_lm.cli import build_parser
        a = build_parser().parse_args(["train", "--run", "r", "--data-dir", "d"])
        self.assertEqual(a.cpu_share, "0")
        self.assertIsNone(a.cpu_threads)
        a = build_parser().parse_args(["train", "--run", "r", "--data-dir", "d", "--cpu-share", "auto", "--cpu-threads", "4"])
        self.assertEqual((a.cpu_share, a.cpu_threads), ("auto", 4))


@unittest.skipUnless(HAVE_TORCH, "PyTorch is not installed (these checks are UNTESTED until run where PyTorch exists)")
class HybridMatchesPlain(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch
        from persona_lm import torch_backend as tb
        from persona_lm.model import Config, init_params
        from persona_lm.train import step_denominators
        self.torch, self.tb, self.np = torch, tb, np
        self.cfg = Config(vocab_size=64, n_layer=2, d_model=32, n_head=4, d_ff=64, ctx=16)
        self.params_np = init_params(self.cfg, 0)
        rng = np.random.Generator(np.random.PCG64(5))
        self.batches = []
        for i in range(4):
            x = rng.integers(0, 64, size=(2, 16)).astype(np.int64)
            y = rng.integers(0, 64, size=(2, 16)).astype(np.int64)
            m = (rng.random((2, 16)) > 0.4).astype(np.float32) if i % 2 else None
            self.batches.append((x, y, m))
        self.denoms = step_denominators(self.batches, "batch")
        self.dev = torch.device("cpu")

    def grads(self, n_cpu, scale=1.0):
        p = self.tb.params_to_torch(self.params_np, self.dev)
        p_cpu = self.tb.make_cpu_params(p)
        loss = self.tb.hybrid_accumulate_step(p, p_cpu, self.cfg, self.batches, self.denoms, self.dev, "fp32", n_cpu, scale=scale)
        return loss, {k: w.grad.detach().clone() for k, w in p.items()}

    def test_split_gives_the_same_gradients_and_loss(self):
        loss0, g0 = self.grads(0)
        for n in (1, 2, 3):
            loss, g = self.grads(n)
            self.assertAlmostEqual(loss, loss0, places=5, msg="loss differs with n_cpu=%d" % n)
            for k in g0:
                self.assertTrue(self.torch.allclose(g[k], g0[k], rtol=1e-4, atol=1e-6), "gradient %s differs with n_cpu=%d" % (k, n))

    def test_loss_scale_is_applied_to_both_parts(self):
        _, g1 = self.grads(2, scale=1.0)
        _, g8 = self.grads(2, scale=8.0)
        for k in g1:
            self.assertTrue(self.torch.allclose(g8[k], g1[k] * 8.0, rtol=1e-4, atol=1e-6))

    def test_n_cpu_is_clamped_so_the_main_device_keeps_a_micro_batch(self):
        loss0, g0 = self.grads(0)
        loss, g = self.grads(99)
        self.assertAlmostEqual(loss, loss0, places=5)
        for k in g0:
            self.assertTrue(self.torch.allclose(g[k], g0[k], rtol=1e-4, atol=1e-6))

    def test_error_in_the_cpu_thread_is_raised(self):
        p = self.tb.params_to_torch(self.params_np, self.dev)
        p_cpu = self.tb.make_cpu_params(p)
        bad = [(self.batches[0][0], self.batches[0][1], None)] * 3 + [(self.batches[0][0][:, :4].copy() * 0 + 999, self.batches[0][1], None)]
        with self.assertRaises(Exception):
            self.tb.hybrid_accumulate_step(p, p_cpu, self.cfg, bad, [1.0] * 4, self.dev, "fp32", 1)


if __name__ == "__main__":
    unittest.main()
