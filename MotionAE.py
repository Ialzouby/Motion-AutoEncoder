import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from datasets import load_dataset
import random
from tqdm import tqdm

# ---- Config ----
MAX_LEN = 196
FEATURE_DIM = 263
BATCH_SIZE = 32
EPOCHS = 10
LATENT_DIM = 512
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1
LR = 1e-3
SEED = 42

# ---- Setup ----
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Dataset ----
def pad_or_truncate_motion(motion, max_len=MAX_LEN, dim=FEATURE_DIM):
    motion = np.array(motion)
    length = motion.shape[0]
    if length >= max_len:
        return torch.tensor(motion[:max_len], dtype=torch.float32)
    else:
        padding = np.zeros((max_len - length, dim))
        return torch.tensor(np.concatenate([motion, padding], axis=0), dtype=torch.float32)

class HumanML3DAEDataset(Dataset):
    def __init__(self, hf_ds):
        self.ds = hf_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        motion = self.ds[idx]["motion"]
        padded = pad_or_truncate_motion(motion)
        return padded.view(-1)  # Flatten to (196 * 263,)

# ---- Load + Split ----
print("Loading dataset...")
ds = load_dataset("TeoGchx/HumanML3D", split="train")
dataset = HumanML3DAEDataset(ds)

n = len(dataset)
test_len = int(TEST_SPLIT * n)
valid_len = int(VALID_SPLIT * n)
train_len = n - test_len - valid_len

train_set, valid_set, test_set = random_split(dataset, [train_len, valid_len, test_len])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE)

# ---- Autoencoder Model ----
class Autoencoder(nn.Module):
    def __init__(self, input_dim=MAX_LEN * FEATURE_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

model = Autoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# ---- Training Loop ----
if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = loss_fn(recon, batch)
                val_loss += loss.item() * batch.size(0)
        val_loss /= len(valid_loader.dataset)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # Optional: Save model
    torch.save(model.state_dict(), "autoencoder_humanml3d.pth")