# FINCH/DLPS — Plan B (4-step protocol, physics consistency) trainer
# ---------------------------------------------------------------
# 입력 : I0 (0° 간섭무늬, 1×256×256)
# 출력 : (I90, I180, I270) — 3채널 intensity 직접 회귀
# 손실 : 메인 = 채널별 RMSE 합 (논문 지표 공정성 유지)
#        보조 = 4-스텝 물리 일관성 |I0 + I180 − I90 − I270| (아주 작게)
#        선택 = DC(A) 음수 억제: A_est = (I0 + I90 + I180 + I270)/4 >= 0 (작게)
#        ※ 평가/저장은 항상 RMSE(val)로만 판단
# 안정화 : UpSample+Conv, GroupNorm, LeakyReLU, AdamW, AMP, GradClip, OneCycleLR(완화)


import os, glob, math, random
from dataclasses import dataclass
from typing import Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------
# Utils
# ----------------------

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def to_tensor01(img: Image.Image) -> torch.Tensor:
    """PIL → float tensor [0,1], shape (H,W). Support 8/16-bit."""
    if img.mode in ("I;16", "I;16B", "I"):
        arr = np.array(img, dtype=np.uint16).astype(np.float32) / 65535.0
    else:
        arr = np.array(img.convert('L'), dtype=np.uint8).astype(np.float32) / 255.0
    return torch.from_numpy(arr)

# ----------------------
# Dataset (256×256 고정)
# ----------------------
class HologramDataset(Dataset):
    def __init__(self, root: str, split: str):
        self.data_dir = os.path.join(root, split)
        self.sample_dirs = sorted(glob.glob(os.path.join(self.data_dir, "sample_*")))
        if len(self.sample_dirs) == 0:
            raise FileNotFoundError(f"No samples under {self.data_dir}")
        print(f"[{split}] {len(self.sample_dirs)} samples @ {self.data_dir}")

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sd = self.sample_dirs[idx]
        paths = [os.path.join(sd, f"phase_{i}.png") for i in range(4)]
        exists = [os.path.exists(p) for p in paths]
        if not all(exists):
            miss = [i for i,b in enumerate(exists) if not b]
            raise FileNotFoundError(f"Missing phase file(s) {miss} in {sd}")
        imgs = [to_tensor01(Image.open(p)) for p in paths]  # list of (H,W)
        # 256×256 고정 확인
        H, W = imgs[0].shape
        if (H, W) != (256, 256):
            raise ValueError(f"Image size must be 256×256, got {H}×{W} at {sd}")
        # x: I0, y: (I90,I180,I270)
        x = imgs[0].unsqueeze(0)  # (1,256,256)
        y = torch.stack([imgs[1], imgs[2], imgs[3]], dim=0)  # (3,256,256)
        return x.float(), y.float()

class Holo3PhaseDataset(Dataset):
    def __init__(self, root: str, split: str, degree_to_index: Dict[int,int]):
        self.base = HologramDataset(root, split)
        self.d2i = dict(degree_to_index)
        assert self.d2i == {0:0, 90:1, 180:2, 270:3}, \
            f"degree_to_index must be {{0:0,90:1,180:2,270:3}}, got {self.d2i}"
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        return self.base[idx]

# ----------------------
# Model (U-Net shared encoder + 3 decoders)
# ----------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            nn.GroupNorm(1, out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.GroupNorm(1, out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 2, 1, bias=False),
            nn.GroupNorm(1, out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ThreeHeadDecoder(nn.Module):
    def __init__(self, base_c=(32,64,128,256)):
        super().__init__()
        c1, c2, c3, c4 = base_c
        # shared encoder
        self.stem = ConvBlock(1, c1)
        self.d1 = Down(c1, c2)
        self.d2 = Down(c2, c3)
        self.d3 = Down(c3, c4)
        # three decoders (heads)
        self.heads = nn.ModuleList()
        for _ in range(3):
            dec = nn.ModuleDict({
                'u1': Up(c4 + c3, c3),
                'u2': Up(c3 + c2, c2),
                'u3': Up(c2 + c1, c1),
                'out': nn.Conv2d(c1, 1, 3, 1, 1)
            })
            self.heads.append(dec)

    def forward(self, x):
        s0 = self.stem(x)   # 256
        s1 = self.d1(s0)    # 128
        s2 = self.d2(s1)    # 64
        s3 = self.d3(s2)    # 32
        outs = []
        for dec in self.heads:
            y = dec['u1'](s3, s2)
            y = dec['u2'](y, s1)
            y = dec['u3'](y, s0)
            y = dec['out'](y)
            y = torch.sigmoid(y)  # [0,1] intensity
            outs.append(y)
        return torch.cat(outs, dim=1)  # (B,3,256,256)

# ----------------------
# Metrics / Loss pieces
# ----------------------
class RMSEThree(nn.Module):
    def __init__(self, variance_stabilize: bool = False):
        super().__init__(); self.vs = variance_stabilize
    def _vs(self, t):
        return torch.sqrt(t.clamp_min(0.0) + 1e-3)
    def forward(self, pred3: torch.Tensor, tgt3: torch.Tensor):
        assert pred3.shape == tgt3.shape
        x = self._vs(pred3) if self.vs else pred3
        y = self._vs(tgt3)  if self.vs else tgt3
        rmses = []
        for c in range(3):
            mse = F.mse_loss(x[:, c], y[:, c], reduction='mean')
            rmses.append(torch.sqrt(mse + 1e-8))
        loss = torch.stack(rmses).sum()
        return loss, {f"rmse_ch{c}": float(rmses[c].detach().cpu()) for c in range(3)} | {"rmse_sum": float(loss.detach().cpu())}

# ----------------------
# Train / Validate
# ----------------------
@dataclass
class TrainConfig:
    dataset_root: str = "./hologram_dataset_images_clean"
    degree_to_index: Dict[int,int] = None
    batch_size: int = 16
    epochs: int = 120
    lr: float = 1e-3
    weight_decay: float = 5e-4      # ↑ 일반화 강화
    max_grad_norm: float = 1.0
    num_workers: int = 4
    amp: bool = True
    seed: int = 42
    patience: int = 20
    pct_start: float = 0.1
    save_path: str = "finch_planB_4step_physics_best.pth"
    # Plan B losses
    physics_weight: float = 0.01     # |I0+I180 − I90 − I270|
    dc_nonneg_weight: float = 0.001  # ReLU(-A_est)
    # variance stabilize only for TRAIN (검증/평가는 raw RMSE)
    variance_stabilize_train: bool = False


def make_loaders(cfg: TrainConfig):
    if cfg.degree_to_index is None:
        cfg.degree_to_index = {0:0, 90:1, 180:2, 270:3}
    train_ds = Holo3PhaseDataset(cfg.dataset_root, "train", cfg.degree_to_index)
    val_ds   = Holo3PhaseDataset(cfg.dataset_root, "validation", cfg.degree_to_index)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, drop_last=False, pin_memory=True)
    return train_loader, val_loader

@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = True) -> float:
    model.eval()
    metric = RMSEThree(variance_stabilize=False)  # 평가: raw RMSE
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=amp):
            pred = model(x)
            loss, _ = metric(pred, y)
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(1, n)


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    train_loader, val_loader = make_loaders(cfg)
    steps = len(train_loader)
    print(f"[Data] steps/epoch={steps}  batch={cfg.batch_size}")

    model = ThreeHeadDecoder(base_c=(32,64,128,256)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.99))
    if steps > 0:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg.lr, epochs=cfg.epochs, steps_per_epoch=steps,
            pct_start=cfg.pct_start, div_factor=10.0, final_div_factor=50.0,
            anneal_strategy='cos'
        )
    else:
        sched = None

    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp)
    metric_train = RMSEThree(variance_stabilize=cfg.variance_stabilize_train)

    best_val, no_imp = float('inf'), 0

    for ep in range(1, cfg.epochs + 1):
        model.train()
        run = 0.0
        for bi, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=cfg.amp):
                pred = model(x)  # (B,3,256,256) = (I90,I180,I270)
                L_rmse, det = metric_train(pred, y)  # 메인: RMSE

                # Plan B: physics consistency for 4-step
                resid = (x + pred[:,1:2]) - (pred[:,0:1] + pred[:,2:3])  # (I0+I180) − (I90+I270)
                L_phys = resid.abs().mean()

                # DC(A) >= 0 (아주 약하게)
                A_est = (x + pred.sum(1, keepdim=True)) / 4.0  # (B,1,H,W)
                L_dc = F.relu(-A_est).mean()

                loss = L_rmse + cfg.physics_weight * L_phys + cfg.dc_nonneg_weight * L_dc

            scaler.scale(loss).backward()

            if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            scaler.step(opt); scaler.update()
            if sched is not None: sched.step()
            run += loss.item()

            if bi % 1 == 0:
                lr = opt.param_groups[0]['lr']
                print(
                    f"   [Ep {ep:03d}/{cfg.epochs} | {bi:04d}/{steps}] "
                    f"loss={loss.item():.6f} rmse_sum={det['rmse_sum']:.6f} "
                    f"phys={L_phys.item():.5f} dc={L_dc.item():.5f} lr={lr:.6f}"
                )

        avg_train = run / max(1, steps)
        val = validate(model, val_loader, device, cfg.amp)
        print(f"\n✅ Epoch {ep:03d} | train={avg_train:.6f}  val={val:.6f}  lr={opt.param_groups[0]['lr']:.6f}")

        if val < best_val - 1e-6:
            best_val, no_imp = val, 0
            torch.save(model.state_dict(), cfg.save_path)
            print(f"   🏆 NEW BEST saved → {cfg.save_path}")
        else:
            no_imp += 1
            print(f"   ⏳ no improve {no_imp}/{cfg.patience}")
            if no_imp >= cfg.patience:
                print("\n🛑 Early stop: patience exceeded")
                break

    print(f"\n🎉 Done. Best val RMSE-sum: {best_val:.6f}")


if __name__ == "__main__":
    cfg = TrainConfig(
        dataset_root = "./hologram_dataset_images_clean",  # ⛳️ 실제 경로
        degree_to_index = {0:0, 90:1, 180:2, 270:3},
        batch_size = 16,
        epochs = 120,
        lr = 1e-3,
        weight_decay = 5e-4,
        max_grad_norm = 1.0,
        num_workers = 0,
        amp = True,
        seed = 42,
        patience = 20,
        pct_start = 0.1,
        save_path = "finch_planB_4step_physics_best.pth",
        physics_weight = 0.01,
        dc_nonneg_weight = 0.001,
        variance_stabilize_train = False,  # 필요 시 True로 바꿔도 '평가'는 raw RMSE 유지
    )
    print(cfg)
    train(cfg)
