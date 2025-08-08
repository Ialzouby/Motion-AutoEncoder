import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader
from MotionAE import Autoencoder, pad_or_truncate_motion, HumanML3DAEDataset
import matplotlib
matplotlib.use('Agg')

# ---- Config ----
MAX_LEN = 196
FEATURE_DIM = 263
LATENT_DIM = 512
JOINTS = 22
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load Model ----
model = Autoencoder(input_dim=MAX_LEN * FEATURE_DIM, latent_dim=LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load("autoencoder_humanml3d.pth", map_location=DEVICE))
model.eval()

# ---- Load Data ----
print("Loading dataset...")
ds = load_dataset("TeoGchx/HumanML3D", split="train")
dataset = HumanML3DAEDataset(ds)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---- Run Inference on One Sample ----
with torch.no_grad():
    for sample in loader:
        sample = sample.to(DEVICE)
        recon = model(sample)
        break

# ---- Unflatten and Extract Joint Coordinates ----
motion_raw = sample[0].cpu()
recon_tensor = recon[0].cpu()

JOINT_FEATURES = JOINTS * 3
motion_orig = motion_raw.view(MAX_LEN, FEATURE_DIM)[:, :JOINT_FEATURES].view(MAX_LEN, JOINTS, 3).numpy()
motion_recon = recon_tensor.view(MAX_LEN, FEATURE_DIM)[:, :JOINT_FEATURES].view(MAX_LEN, JOINTS, 3).numpy()

# ---- Visualize Joint 0's X Position Over Time ----
joint_id = 0  # Change this to visualize another joint
x_orig = motion_orig[:, joint_id, 0]
x_recon = motion_recon[:, joint_id, 0]

plt.figure(figsize=(10, 5))
plt.plot(x_orig, label="Original", linewidth=2)
plt.plot(x_recon, label="Reconstructed", linestyle="--")
plt.title(f"Joint {joint_id} - X Position Over Time")
plt.xlabel("Frame")
plt.ylabel("X Position")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
