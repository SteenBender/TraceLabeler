"""1D-CNN encoder with masked-autoencoder pretraining for feature extraction.

Provides ``TraceEncoder`` which pretrains a 1D-CNN on all traces via a masked
autoencoder objective, then extracts fixed-length embedding vectors that can be
used as features for any downstream classifier (e.g. HistGradientBoosting).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── helpers ──────────────────────────────────────────────────────────────


def get_available_devices():
    """Return a list of available torch device strings.

    Always includes ``'cpu'``.  Adds ``'cuda'`` / ``'cuda:N'`` for each
    NVIDIA GPU and ``'mps'`` on Apple Silicon when supported.
    """
    devices = ["cpu"]
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if n == 1:
            devices.append("cuda")
        else:
            for i in range(n):
                devices.append(f"cuda:{i}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def _traces_to_tensor(sequences, t_max=None):
    """Pad a list of (C, T_i) arrays to (N, C, T_max) float32 tensor + mask."""
    if t_max is None:
        t_max = max(s.shape[1] for s in sequences)
    n_ch = sequences[0].shape[0]
    X = np.zeros((len(sequences), n_ch, t_max), dtype=np.float32)
    mask = np.zeros((len(sequences), t_max), dtype=np.float32)
    for i, s in enumerate(sequences):
        T = min(s.shape[1], t_max)
        X[i, :, :T] = s[:, :T].astype(np.float32)
        mask[i, :T] = 1.0
    return torch.from_numpy(X), torch.from_numpy(mask)


def _downsample_mask(mask, target_len):
    """Downsample a (B, T) binary mask to (B, target_len) using max-pool logic."""
    if mask.shape[1] == target_len:
        return mask
    return F.adaptive_max_pool1d(mask.unsqueeze(1), target_len).squeeze(1)


# ── 1D-CNN encoder ──────────────────────────────────────────────────────


class _Encoder(nn.Module):
    """Lightweight 1D-CNN encoder for multi-channel time-series."""

    def __init__(self, n_channels, embed_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, embed_dim, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(embed_dim)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x, mask=None):
        """x: (B, C, T), mask: (B, T) with 1=valid."""
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.pool(F.relu(self.bn2(self.conv2(h))))
        h = self.pool(F.relu(self.bn3(self.conv3(h))))
        if mask is not None:
            m = _downsample_mask(mask, h.shape[2])
            m = m.unsqueeze(1)
            h = (h * m).sum(dim=2) / (m.sum(dim=2) + 1e-8)
        else:
            h = h.mean(dim=2)
        return h  # (B, embed_dim)


# ── Masked autoencoder ──────────────────────────────────────────────────


class _MaskedAutoencoder(nn.Module):
    """Encode → masked reconstruction for self-supervised pretraining."""

    def __init__(self, n_channels, embed_dim=64):
        super().__init__()
        self.encoder = _Encoder(n_channels, embed_dim)
        self.expand = nn.Linear(embed_dim, embed_dim * 4)
        self.decoder = nn.Sequential(
            nn.Conv1d(embed_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, n_channels, kernel_size=3, padding=1),
        )
        self.n_channels = n_channels
        self.embed_dim = embed_dim

    def forward(self, x, mask=None, mask_ratio=0.3):
        B, C, T = x.shape
        recon_mask = torch.rand(B, T, device=x.device) < mask_ratio
        if mask is not None:
            recon_mask = recon_mask & mask.bool()

        x_masked = x.clone()
        rm_expanded = recon_mask.unsqueeze(1).expand_as(x)
        x_masked[rm_expanded] = 0.0

        emb = self.encoder(x_masked, mask)

        h = F.relu(self.expand(emb))
        h = h.view(B, self.embed_dim, 4)
        h = F.interpolate(h, size=T, mode="linear", align_corners=False)
        pred = self.decoder(h)

        diff = (pred - x) ** 2
        rm_weight = recon_mask.unsqueeze(1).float()
        n_masked = rm_weight.sum() + 1e-8
        loss = (diff * rm_weight).sum() / (n_masked * C)
        return loss


# ── Feature extractor ───────────────────────────────────────────────────


class TraceEncoder:
    """Pretrain a 1D-CNN encoder via MAE, then extract embeddings as features.

    Parameters
    ----------
    n_channels : int
        Number of signal channels.
    embed_dim : int
        Dimensionality of the output embedding per trace.
    pretrain_epochs : int
        MAE pretraining epochs.
    lr : float
        Learning rate.
    batch_size : int
        Mini-batch size.
    device : str
        ``'cpu'``, ``'cuda'``, or ``'mps'``.
    """

    def __init__(
        self,
        n_channels=3,
        embed_dim=64,
        pretrain_epochs=30,
        lr=1e-3,
        batch_size=128,
        device="cpu",
    ):
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.pretrain_epochs = pretrain_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

        self._encoder = None
        self._pretrained = False
        self._t_max = None
        self._mean = None
        self._std = None

    @property
    def is_pretrained(self):
        return self._pretrained

    def pretrain(self, sequences):
        """Self-supervised MAE pretraining on all traces.

        Parameters
        ----------
        sequences : list[np.ndarray]
            Each element has shape ``(n_channels, T_i)``.
        """
        self._mean = np.zeros(self.n_channels, dtype=np.float32)
        self._std = np.ones(self.n_channels, dtype=np.float32)
        for c in range(self.n_channels):
            vals = np.concatenate([s[c] for s in sequences])
            self._mean[c] = vals.mean()
            self._std[c] = vals.std() + 1e-8

        X, mask = _traces_to_tensor(sequences)
        self._t_max = X.shape[2]
        X = self._normalise_tensor(X)

        mae = _MaskedAutoencoder(self.n_channels, self.embed_dim).to(self.device)
        opt = torch.optim.AdamW(mae.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.pretrain_epochs
        )
        dataset = TensorDataset(X, mask)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=len(dataset) > self.batch_size,
        )

        mae.train()
        for epoch in range(self.pretrain_epochs):
            total_loss = 0.0
            n_samples = 0
            for xb, mb in loader:
                xb, mb = xb.to(self.device), mb.to(self.device)
                loss = mae(xb, mb, mask_ratio=0.3)
                if torch.isnan(loss):
                    continue
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
                opt.step()
                total_loss += loss.item() * xb.size(0)
                n_samples += xb.size(0)
            scheduler.step()
            if (epoch + 1) % 5 == 0:
                avg = total_loss / max(n_samples, 1)
                print(
                    f"  [pretrain] epoch {epoch+1}/{self.pretrain_epochs}  "
                    f"loss={avg:.4f}"
                )

        # Move encoder to CPU after pretraining
        self._encoder = mae.encoder.to("cpu")
        self._pretrained = True

    def _normalise_tensor(self, X):
        mean_t = torch.from_numpy(self._mean).view(1, -1, 1)
        std_t = torch.from_numpy(self._std).view(1, -1, 1)
        return (X - mean_t) / std_t

    def extract_embeddings(self, sequences):
        """Extract embedding vectors for a list of traces.

        Parameters
        ----------
        sequences : list[np.ndarray]
            Each element has shape ``(n_channels, T_i)``.

        Returns
        -------
        embeddings : np.ndarray, shape (N, embed_dim)
        """
        if not self._pretrained:
            raise RuntimeError("Call pretrain() before extract_embeddings().")

        X, mask = _traces_to_tensor(sequences, t_max=self._t_max)
        X = self._normalise_tensor(X)
        dataset = TensorDataset(X, mask)
        loader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False)

        self._encoder = self._encoder.to(self.device)
        self._encoder.eval()

        parts = []
        with torch.no_grad():
            for xb, mb in loader:
                xb, mb = xb.to(self.device), mb.to(self.device)
                emb = self._encoder(xb, mb)
                parts.append(emb.cpu().numpy())

        self._encoder = self._encoder.to("cpu")
        return np.concatenate(parts, axis=0)
