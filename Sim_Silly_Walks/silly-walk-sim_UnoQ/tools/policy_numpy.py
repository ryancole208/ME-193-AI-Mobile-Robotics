"""The actor network re-written with plain NumPy (no PyTorch). Used to check the exports.

    obs (7) -> normalize -> Linear(7,64) -> ELU -> Linear(64,64) -> ELU -> Linear(64,2) -> action (2)

normalize = (obs - mean) / (std + EPS), exactly like rsl_rl's EmpiricalNormalization.
"""
import numpy as np

EPS = 0.01   # rsl_rl EmpiricalNormalization default eps (checked against rsl_rl in export_policy.py)


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


class NumpyPolicy:
    def __init__(self, p):
        """p: dict of numpy arrays with keys mean, std, W0, b0, W1, b1, W2, b2 (W shaped [out, in])."""
        self.p = {k: np.asarray(v, dtype=np.float64) for k, v in p.items()}

    @classmethod
    def from_actor_state_dict(cls, sd):
        """sd: the checkpoint's actor_state_dict, values as numpy arrays (or anything np.asarray accepts)."""
        g = lambda k: np.asarray(sd[k], dtype=np.float64)
        return cls({"mean": g("obs_normalizer._mean").reshape(-1), "std": g("obs_normalizer._std").reshape(-1),
                    "W0": g("mlp.0.weight"), "b0": g("mlp.0.bias"),
                    "W1": g("mlp.2.weight"), "b1": g("mlp.2.bias"),
                    "W2": g("mlp.4.weight"), "b2": g("mlp.4.bias")})

    def __call__(self, obs):
        p = self.p
        x = (np.asarray(obs, dtype=np.float64) - p["mean"]) / (p["std"] + EPS)
        h = elu(x @ p["W0"].T + p["b0"])
        h = elu(h @ p["W1"].T + p["b1"])
        return h @ p["W2"].T + p["b2"]
