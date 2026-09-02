import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from core.hcm import HCM
import torch

h = HCM(768, max_patterns=512, n_clusters=32, min_age=0)

# Write a pattern with target token 42
s = torch.randn(768)
h.write(s, 2.0, target_token=42)

# Write another pattern with target token 99
h.write(torch.randn(768), 2.0, target_token=99)

# Read and get stored targets
retrieved, sim, stored_targets = h.read(s)
print(f"retrieved={retrieved is not None} sim={sim:.3f}")
print(f"stored_targets={stored_targets}")

# Test crutch loss — if model predicts token 42 (exact match)
logits_match = torch.randn(8192)
logits_match[42] = 10.0  # Model strongly predicts token 42
loss_match = h.compute_crutch_loss(logits_match, stored_targets, w_crutch=0.1)
print(f"crutch loss (match): {loss_match:.4f}")

# Test crutch loss — if model predicts token 7 (no match)
logits_nomatch = torch.randn(8192)
logits_nomatch[7] = 10.0  # Model predicts different token
loss_nomatch = h.compute_crutch_loss(logits_nomatch, stored_targets, w_crutch=0.1)
print(f"crutch loss (no match): {loss_nomatch:.4f}")

print("OK")
