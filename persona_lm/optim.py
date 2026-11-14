"""AdamW, gradient clipping and the learning-rate schedule, from scratch."""
import math

import numpy as np


class AdamW:
    def __init__(self, params, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    @staticmethod
    def _decays(name, arr):
        return arr.ndim == 2 and name != "tok"      # not norms, not the embedding

    def step(self, params, grads, lr):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for k, p in params.items():
            g = grads[k]
            m, v = self.m[k], self.v[k]
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * g * g
            upd = (m / bc1) / (np.sqrt(v / bc2) + self.eps)
            if self._decays(k, p):
                p *= 1.0 - lr * self.wd
            p -= lr * upd

    def state(self):
        out = {"opt_t": np.array(self.t, dtype=np.int64)}
        for k in self.m:
            out["m." + k] = self.m[k]
            out["v." + k] = self.v[k]
        return out

    def load_state(self, arrays):
        self.t = int(arrays["opt_t"])
        for k in self.m:
            self.m[k][...] = arrays["m." + k]
            self.v[k][...] = arrays["v." + k]


def clip_grad_norm(grads, max_norm):
    total = math.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in grads.values()))
    if max_norm and total > max_norm:
        f = max_norm / (total + 1e-6)
        for g in grads.values():
            g *= f
    return total


def lr_at(step, total, lr, min_lr, warmup):
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * prog))
