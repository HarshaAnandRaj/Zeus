import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from core.hcm import HCM
import torch

h = HCM(768, max_patterns=512, n_clusters=32, min_age=0)

# Simulate: write with target embedding
state = torch.randn(768)
target_emb = torch.randn(768)  # embedding of next token
h.write(state, 2.0, target_token=42, target_embed=target_emb)

# Read — should return target embedding, not raw state
retrieved, sim, stored_targets, ids = h.read(state)
print(f"retrieved matches target_embed: {torch.allclose(retrieved, target_emb, atol=1e-3)}")
print(f"retrieved matches raw state: {torch.allclose(retrieved, state, atol=1e-3)}")
print(f"stored_targets: {stored_targets}")
print("OK")
