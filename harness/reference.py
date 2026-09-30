import math

import torch
import torch.nn.functional as F


def reference(q, k, v):
    att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    return att @ v


def compare(out, ref, atol=1e-2):
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"forward returned {type(out).__name__}, expected torch.Tensor")
    if tuple(out.shape) != tuple(ref.shape):
        raise ValueError(f"shape mismatch: got {tuple(out.shape)}, expected {tuple(ref.shape)}")
    max_diff = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, rtol=0, atol=atol)
    return ok, max_diff
