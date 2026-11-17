"""STEP 5 (run on the TRAINING machine, it needs PyTorch): export the actor + observation normalizer to ONNX
and check that the original checkpoint, the ONNX file and the NumPy re-implementation all agree.

Needs: torch, numpy, onnx, onnxruntime (pip install onnx onnxruntime). rsl_rl is optional but recommended:
if it is installed, the normalization is checked against rsl_rl's own EmpiricalNormalization class.

Copy the whole tools/ folder and model_499.pt to the training machine, then:
    python tools/export_policy.py model_499.pt            -> writes policy.onnx next to the checkpoint

What it checks, on 1000 random observations plus the 20 fixed test vectors in tools/policy_test_vectors.json
(those are the same inputs the STM32 was tested with in step 5, so this links PyTorch -> Python -> STM32):
  1. our PyTorch actor (normalizer + MLP) vs rsl_rl's normalizer (if available)  -> is eps = 0.01 right?
  2. PyTorch vs ONNX (onnxruntime)
  3. PyTorch vs NumPy (tools/policy_numpy.py)
  4. PyTorch vs the expected actions stored in policy_test_vectors.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from policy_numpy import EPS, NumpyPolicy  # noqa: E402

TOL = 1e-5


class ActorWithNormalizer(nn.Module):
    """obs -> (obs - mean) / (std + eps) -> Linear(7,64) -> ELU -> Linear(64,64) -> ELU -> Linear(64,2)."""

    def __init__(self, sd):
        super().__init__()
        self.register_buffer("mean", sd["obs_normalizer._mean"].reshape(-1).float().clone())
        self.register_buffer("std", sd["obs_normalizer._std"].reshape(-1).float().clone())
        w = [sd[f"mlp.{i}.weight"] for i in (0, 2, 4)]
        self.mlp = nn.Sequential(nn.Linear(w[0].shape[1], w[0].shape[0]), nn.ELU(),
                                 nn.Linear(w[1].shape[1], w[1].shape[0]), nn.ELU(),
                                 nn.Linear(w[2].shape[1], w[2].shape[0]))
        mlp_sd = {k[len("mlp."):]: v for k, v in sd.items() if k.startswith("mlp.")}
        self.mlp.load_state_dict(mlp_sd, strict=True)   # fails loudly if the layout is not 7-64-64-2

    def normalize(self, obs):
        return (obs - self.mean) / (self.std + EPS)

    def forward(self, obs):
        return self.mlp(self.normalize(obs))


def rsl_rl_normalizer(sd, n_obs):
    """rsl_rl's own EmpiricalNormalization loaded with the checkpoint stats, or None if rsl_rl is missing."""
    cls = None
    for mod in ("rsl_rl.networks", "rsl_rl.modules.normalizer", "rsl_rl.modules"):
        try:
            cls = getattr(__import__(mod, fromlist=["EmpiricalNormalization"]), "EmpiricalNormalization")
            break
        except (ImportError, AttributeError):
            pass
    if cls is None:
        return None
    norm = cls(shape=n_obs)
    norm.load_state_dict({k[len("obs_normalizer."):]: v for k, v in sd.items() if k.startswith("obs_normalizer.")})
    norm.eval()   # eval = use the stored stats, don't update them
    return norm


def report(name, a, b):
    err = float(np.abs(np.asarray(a) - np.asarray(b)).max())
    ok = err < TOL
    print(f"  {name:45s} max error {err:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ckpt_path = Path(sys.argv[1] if len(sys.argv) > 1 else "model_499.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)   # our own checkpoint, trusted
    sd = ckpt["actor_state_dict"]
    model = ActorWithNormalizer(sd).eval()
    n_obs = model.mean.numel()
    print(f"loaded {ckpt_path} (iteration {ckpt.get('iter')}), actor input {n_obs}")

    # ---- export ----
    onnx_path = ckpt_path.with_name("policy.onnx")
    dummy = torch.zeros(1, n_obs)
    kwargs = dict(input_names=["obs"], output_names=["action"], dynamic_axes={"obs": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(model, dummy, str(onnx_path), dynamo=False, **kwargs)   # classic exporter (torch >= 2.5)
    except TypeError:
        torch.onnx.export(model, dummy, str(onnx_path), **kwargs)                 # older torch
    print(f"wrote {onnx_path}  (input 'obs' [batch,{n_obs}] raw observation -> output 'action' [batch,2] raw action)")

    # ---- test inputs ----
    rng = np.random.default_rng(0)
    mean, std = model.mean.numpy(), model.std.numpy()
    obs = (mean + std * rng.uniform(-2, 2, size=(1000, n_obs))).astype(np.float32)
    vectors = json.loads((TOOLS / "policy_test_vectors.json").read_text())
    vec_obs = np.array([v["obs"] for v in vectors], dtype=np.float32)
    vec_act = np.array([v["action"] for v in vectors])
    all_obs = np.concatenate([obs, vec_obs])

    with torch.no_grad():
        out_torch = model(torch.from_numpy(all_obs)).numpy()

    import onnxruntime as ort
    out_onnx = ort.InferenceSession(str(onnx_path)).run(None, {"obs": all_obs})[0]
    out_numpy = NumpyPolicy.from_actor_state_dict({k: v.numpy() for k, v in sd.items()})(all_obs)

    print("checks:")
    ok = True
    norm = rsl_rl_normalizer(sd, n_obs)
    if norm is None:
        print("  rsl_rl not importable: skipped the normalizer check (eps = 0.01 is assumed)")
    else:
        with torch.no_grad():
            ok &= report("normalizer: ours vs rsl_rl EmpiricalNormalization",
                         model.normalize(torch.from_numpy(all_obs)).numpy(), norm(torch.from_numpy(all_obs)).numpy())
    ok &= report("checkpoint (PyTorch) vs ONNX", out_torch, out_onnx)
    ok &= report("checkpoint (PyTorch) vs NumPy", out_torch, out_numpy)
    ok &= report("checkpoint vs test vectors (= STM32 reference)", out_torch[-len(vectors):], vec_act)
    print("ALL PASS" if ok else "SOMETHING FAILED - tell Claude the output above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
