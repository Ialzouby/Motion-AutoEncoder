import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset
from MotionAE import Autoencoder, pad_or_truncate_motion, MAX_LEN, FEATURE_DIM, BATCH_SIZE, LATENT_DIM

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*40)
print(f"✅ Using device: {device}")
print("="*40)

# Dataset wrapper
class HumanML3DAEDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds):
        self.ds = hf_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        motion = self.ds[idx]["motion"]
        padded = pad_or_truncate_motion(motion)
        return padded.view(-1)

# Load dataset
print("Loading dataset for validation...")
ds = load_dataset("TeoGchx/HumanML3D", split="train")
dataset = HumanML3DAEDataset(ds)

# Use a small split for validation
val_size = int(0.1 * len(dataset))
_, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - val_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Load model
model = Autoencoder(input_dim=MAX_LEN * FEATURE_DIM, latent_dim=LATENT_DIM).to(device)
model.load_state_dict(torch.load("autoencoder_humanml3d.pth", map_location=device))
model.eval()

# Metrics
mse_fn = nn.MSELoss(reduction='none')
mae_fn = nn.L1Loss(reduction='none')

total_mse = 0
total_mae = 0
total_l2 = 0
num_samples = 0

with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(device)
        recon = model(batch)

        # Element-wise errors
        mse = mse_fn(recon, batch).mean(dim=1)      # (batch_size,)
        mae = mae_fn(recon, batch).mean(dim=1)
        l2 = torch.sqrt(((recon - batch) ** 2).sum(dim=1))  # Euclidean per sample

        total_mse += mse.sum().item()
        total_mae += mae.sum().item()
        total_l2 += l2.sum().item()
        num_samples += batch.size(0)

# Normalize over total samples
avg_mse = total_mse / num_samples
avg_mae = total_mae / num_samples
avg_l2 = total_l2 / num_samples

print(f"\nValidation Metrics:")
print(f"  ▸ MSE: {avg_mse:.6f}")
print(f"  ▸ MAE: {avg_mae:.6f}")
print(f"  ▸ L2  (Euclidean Distance): {avg_l2:.6f}")
