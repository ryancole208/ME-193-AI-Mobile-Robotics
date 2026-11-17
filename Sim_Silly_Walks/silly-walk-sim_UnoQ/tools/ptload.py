"""Load an rsl_rl PyTorch checkpoint (.pt) WITHOUT PyTorch, using only the Python standard library.

A .pt file is a zip: data.pkl describes the structure (dicts of names -> tensors) and data/<n> holds
each tensor's raw float32 bytes. We replace PyTorch's "rebuild tensor" functions with our own that
return a small Tensor object (shape + flat list of floats).
"""
import pickle
import struct
import sys
import zipfile
from collections import OrderedDict


class Tensor:
    def __init__(self, data, shape):
        self.data = data          # flat list of floats, row-major
        self.shape = tuple(shape)

    def rows(self):
        """2-D tensor -> list of rows; 1-D -> the list itself."""
        if len(self.shape) == 1:
            return list(self.data)
        n = self.shape[1]
        return [self.data[i * n:(i + 1) * n] for i in range(self.shape[0])]

    def __repr__(self):
        return f"Tensor{self.shape}"


_DTYPE = {"FloatStorage": ("f", 4), "DoubleStorage": ("d", 8), "LongStorage": ("q", 8),
          "IntStorage": ("i", 4), "BoolStorage": ("?", 1)}


def load(path):
    z = zipfile.ZipFile(path)
    prefix = z.namelist()[0].split("/")[0]

    class _Storage:                     # stands in for torch.FloatStorage etc.
        def __init__(self, name):
            self.name = name

    def _rebuild_tensor_v2(storage, offset, size, stride, *_):
        fmt, nbytes = storage
        n = 1
        for s in size:
            n *= s
        flat = storage_vals[storage]
        if n == 0:
            return Tensor([], size)
        # only plain contiguous tensors appear in rsl_rl checkpoints; check that
        exp, st = 1, []
        for s in reversed(size):
            st.insert(0, exp)
            exp *= s
        assert list(stride) == st or n == 1, f"non-contiguous tensor {size} {stride}"
        return Tensor(flat[offset:offset + n], size)

    storage_vals = {}

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == "torch._utils" and name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if mod == "torch._utils" and name == "_rebuild_parameter":
                return lambda t, *_: t
            if mod == "torch" and name in _DTYPE:
                return name
            if mod == "collections" and name == "OrderedDict":
                return OrderedDict
            if mod == "torch" and name == "Size":
                return tuple
            # anything else (e.g. numpy scalars in infos) -> harmless placeholder
            return lambda *a, **k: f"<{mod}.{name}>"

        def persistent_load(self, pid):
            # pid = ('storage', storage_type, key, location, numel)
            _, stype, key, _, numel = pid
            fmt, nb = _DTYPE[stype]
            raw = z.read(f"{prefix}/data/{key}")
            vals = list(struct.unpack(f"<{len(raw) // nb}{fmt}", raw))
            k = (fmt, key)
            storage_vals[k] = vals
            return k

    with z.open(f"{prefix}/data.pkl") as f:
        return U(f).load()


def walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    else:
        yield path, obj


if __name__ == "__main__":
    ck = load(sys.argv[1])
    for p, v in walk(ck):
        if isinstance(v, Tensor):
            extra = f"  {[round(x, 4) for x in v.data]}" if len(v.data) <= 8 else ""
            print(f"{p:55s} {v.shape}{extra}")
        else:
            print(f"{p:55s} {v!r}"[:150])
