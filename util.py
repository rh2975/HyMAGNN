import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class StandardScaler:
    """
    Fit on TRAIN only.
    Data shape for fit/transform: (T, N)
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        self.mean = data.mean(axis=0, keepdims=True)
        self.std = data.std(axis=0, keepdims=True) + 1e-8

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean


# Data Loading
def read_timeseries_matrix(path: str) -> np.ndarray:
    """
    Returns float32 numpy array with shape (T, N).
    Supports: .csv, .txt, .npz
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npz":
        npz = np.load(path)              # optionally: np.load(path, mmap_mode="r")
        data = npz["data"].astype(np.float32)   # (T, N, 1)
    
        # (T, N, 1) -> (T, N)
        if data.ndim == 3 and data.shape[-1] == 1:
            data = data[..., 0]
        elif data.ndim == 3:
            # if future datasets have more features, pick the first or aggregate
            data = data[..., 0]          # or: data = data.mean(axis=-1)
        elif data.ndim != 2:
            raise ValueError(f"Unexpected npz shape: {data.shape}")
    
        return data  # (T, N)

    if ext == ".csv":
        df = pd.read_csv(path)
        for c in list(df.columns):
            if not np.issubdtype(df[c].dtype, np.number):
                try:
                    pd.to_datetime(df[c])
                    df = df.drop(columns=[c])
                except Exception:
                    df = df.drop(columns=[c])
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.ffill().bfill()
        data = df.to_numpy(dtype=np.float32)

    else:
        try:
            data = np.loadtxt(path, delimiter=",", dtype=np.float32)
        except Exception:
            data = np.loadtxt(path, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]

    assert data.ndim == 2
    return data



# Dataset
class SlidingWindowDatasetScaled(Dataset):
    """
    X: (1, N, Tin)
    Y: (H, N, 1)
    """
    def __init__(self, data_TN, seq_in_len, horizon, scaler):
        self.data = data_TN.astype(np.float32)
        self.seq_in_len = seq_in_len
        self.horizon = horizon
        self.scaler = scaler
        self.T, self.N = self.data.shape
        self.max_t = self.T - (seq_in_len + horizon) + 1

    def __len__(self):
        return self.max_t

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_in_len]
        y = self.data[idx+self.seq_in_len:idx+self.seq_in_len+self.horizon]

        x = self.scaler.transform(x)
        y = self.scaler.transform(y)

        x = torch.from_numpy(x.T).unsqueeze(0)
        y = torch.from_numpy(y).unsqueeze(-1)
        return x, y


def make_loaders(data_path, seq_in_len, horizon,
                 batch_size, num_workers=2,
                 split=(0.6,0.2,0.2), pin_memory=True):

    raw = read_timeseries_matrix(data_path)
    T = raw.shape[0]

    n_train = int(T * split[0])
    n_val = int(T * split[1])

    train_raw = raw[:n_train]
    val_raw = raw[n_train:n_train+n_val]
    test_raw = raw[n_train+n_val:]

    scaler = StandardScaler()
    scaler.fit(train_raw)

    train_ds = SlidingWindowDatasetScaled(train_raw, seq_in_len, horizon, scaler)
    val_ds = SlidingWindowDatasetScaled(val_raw, seq_in_len, horizon, scaler)
    test_ds = SlidingWindowDatasetScaled(test_raw, seq_in_len, horizon, scaler)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader, scaler, raw.shape[1]


# Metrics
def rse(pred, true):
    pred = pred.reshape(-1)
    true = true.reshape(-1)
    return np.sqrt(np.sum((true - pred)**2)) / (np.sqrt(np.sum((true - true.mean())**2)) + 1e-8)

def corr(pred, true):
    pred = pred.reshape(-1)
    true = true.reshape(-1)
    pred -= pred.mean()
    true -= true.mean()
    return np.sum(pred*true) / ((np.sqrt(np.sum(pred**2))*np.sqrt(np.sum(true**2))) + 1e-8)

def mae(pred, true):
    return np.mean(np.abs(pred - true))

def rmse(pred, true):
    return np.sqrt(np.mean((pred - true)**2))


class GlobalCoreFusion(nn.Module):
    """
    Pools global context across nodes and time (or scales),
    refines it, and fuses back via gated residual.

    Expects x: (B, C, N, T)
    """
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        # gate over channels, per node (and per time=1)
        self.gate = nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=(1, 1))

    def forward(self, x):
        # Accept either (B, C, N) or (B, C, N, T)
        if x.dim() == 3:
            x = x.unsqueeze(-1)  # (B, C, N, 1)
    
        B, C, N, T = x.shape
    
        # global core: (B, C)
        core = x.mean(dim=(2, 3))          # best             # (B, C)
        core = self.mlp(core).view(B, C, 1, 1)           # (B, C, 1, 1)
    
        # node summary: (B, C, N, 1)
        node_mean = x.mean(dim=3, keepdim=True)          # (B, C, N, 1)
    
        # broadcast core per node (B, C, N, 1)
        core_for_gate = core.expand(-1, -1, N, 1)
    
        # gate over channels
        gate = torch.sigmoid(self.gate(torch.cat([node_mean, core_for_gate], dim=1)))  # (B, C, N, 1)
    
        # broadcast core across time for residual add
        core_full = core.expand(-1, -1, N, T)            # (B, C, N, T)
    
        return x + gate * core_full
