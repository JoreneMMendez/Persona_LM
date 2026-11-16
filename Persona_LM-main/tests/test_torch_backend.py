"""PyTorch backend checks against the NumPy reference. Skipped (not failed) when PyTorch is not installed.
Run on your own machine:  python -m unittest tests.test_torch_backend -v     (or simply:  python -m persona_lm check-gpu)"""
import importlib.util
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HAVE_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAVE_TORCH, "PyTorch is not installed (these checks are UNTESTED until run where PyTorch exists)")
class TorchMatchesNumpy(unittest.TestCase):
    def test_all_selfchecks_pass_on_cpu(self):
        from persona_lm import torch_backend as tb
        results = tb.selfcheck("cpu", log=lambda *_: None)
        failed = [r for r in results if not r[1]]
        self.assertEqual(failed, [], "\n".join("%s: %s" % (n, d) for n, _, d in failed))

    @unittest.skipUnless(HAVE_TORCH and __import__("torch").cuda.is_available(), "no CUDA GPU")
    def test_all_selfchecks_pass_on_gpu(self):
        from persona_lm import torch_backend as tb
        results = tb.selfcheck("cuda", log=lambda *_: None)
        failed = [r for r in results if not r[1]]
        self.assertEqual(failed, [], "\n".join("%s: %s" % (n, d) for n, _, d in failed))


if __name__ == "__main__":
    unittest.main()
