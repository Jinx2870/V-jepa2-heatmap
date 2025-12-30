import torch

feature_path = "/data3/lg2/human_wm/vjepa2/features/cam01_vjepa2_features.pt"
features = torch.load(feature_path)

print("Loaded feature shape:", features.shape)
print("Data type:", features.dtype)
print("Mean / Std:", features.mean().item(), features.std().item())