import torch, glob, os

files = sorted(glob.glob('experiments/p5_night6/zeus_step*.pt'))
d = torch.load(files[-1], map_location='cpu', weights_only=False)
print(f"Checkpoint step: {d['step']}")
print(f"Old HCM patterns: {d['hcm']['n_patterns']}")

# Clear HCM — old patterns have no target_embed, useless with new injection
d['hcm'] = None
torch.save(d, files[-1])
print(f"HCM cleared in {files[-1]}")

# Remove older checkpoints to avoid confusion
for f in files[:-1]:
    os.remove(f)
    print(f"Removed: {f}")
