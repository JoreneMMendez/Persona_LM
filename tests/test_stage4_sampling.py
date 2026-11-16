import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from persona_lm import cli  # noqa: E402
from persona_lm.fastgen import sample_ids_cached  # noqa: E402
from persona_lm.generate import is_looping, pick_token, sample_ids  # noqa: E402
from persona_lm.model import PRESETS, Config, init_params  # noqa: E402


class Sampling(unittest.TestCase):
    def setUp(self):
        self.cfg = Config.from_dict(dict(PRESETS["tiny"]))
        self.params = init_params(self.cfg, 3)

    def test_defaults_unchanged(self):
        a = sample_ids(self.params, self.cfg, [5, 6, 7], max_new=12, seed=4, stop_ids=())
        b = sample_ids(self.params, self.cfg, [5, 6, 7], max_new=12, seed=4, stop_ids=(), rep_penalty=1.0, stop_repeat=False)
        self.assertEqual(a, b)

    def test_cached_matches_plain_with_new_options(self):
        kw = dict(max_new=15, seed=2, stop_ids=(), rep_penalty=1.3, temperature=0.9)
        self.assertEqual(sample_ids(self.params, self.cfg, [5, 6, 7], **kw), sample_ids_cached(self.params, self.cfg, [5, 6, 7], **kw))

    def test_rep_penalty_lowers_seen_token(self):
        lg = np.array([3.0, 2.9, -1.0, 0.0])
        rng = np.random.Generator(np.random.PCG64(0))
        self.assertEqual(pick_token(lg, rng, temperature=0), 0)
        self.assertEqual(pick_token(lg, rng, temperature=0, rep_penalty=2.0, recent=[0]), 1)
        self.assertEqual(pick_token(lg, rng, temperature=0, rep_penalty=1.0, recent=[0]), 0)     # off
        n = np.array([-2.0, -1.0, -3.0])                                                            # negative logits get pushed down too
        self.assertEqual(pick_token(n, rng, temperature=0, rep_penalty=3.0, recent=[1]), 0)

    def test_is_looping(self):
        self.assertEqual(is_looping([9, 1, 2, 3, 1, 2, 3, 1, 2, 3]), 3)
        self.assertEqual(is_looping([1, 2, 3, 1, 2, 3, 4]), 0)
        self.assertEqual(is_looping([1, 2, 3, 1, 2, 3]), 0)       # only twice

    def test_stop_repeat_ends_a_loop_and_keeps_one_copy(self):
        # an untrained tiny model at temperature 0 falls into a loop quickly
        free = sample_ids(self.params, self.cfg, [5, 6, 7], max_new=60, temperature=0, stop_ids=())
        cut = sample_ids(self.params, self.cfg, [5, 6, 7], max_new=60, temperature=0, stop_ids=(), stop_repeat=True)
        if is_looping(free):
            self.assertLess(len(cut), len(free))
            self.assertEqual(cut, free[:len(cut)])
        cut2 = sample_ids_cached(self.params, self.cfg, [5, 6, 7], max_new=60, temperature=0, stop_ids=(), stop_repeat=True)
        self.assertEqual(cut, cut2)

    def test_cli_flags_only_set_when_given(self):
        self.assertEqual(cli._extra_sampling(SimpleNamespace(top_p=None, rep_penalty=1.0, stop_repeat=False)), {})
        self.assertEqual(cli._extra_sampling(SimpleNamespace(top_p=0.9, rep_penalty=1.2, stop_repeat=True)),
                         {"top_p": 0.9, "rep_penalty": 1.2, "stop_repeat": True})
        p = cli.build_parser().parse_args(["chat", "--top-p", "0.8", "--rep-penalty", "1.2", "--stop-repeat"])
        self.assertEqual((p.top_p, p.rep_penalty, p.stop_repeat), (0.8, 1.2, True))
        p = cli.build_parser().parse_args(["generate", "--ckpt", "c", "--tokenizer", "t", "--prompt", "x"])
        self.assertEqual((p.top_p, p.rep_penalty, p.stop_repeat), (None, 1.0, False))


if __name__ == "__main__":
    unittest.main()
