# FINCH/DLPS — Best-Practice Trainer (Direct 3-Intensity Heads, RMSE-only)
# ------------------------------------------------------------------------
# 목표
#  - 입력: 0° 간섭무늬 I0 (세기 이미지, 1×H×W)
#  - 출력: 3채널 (I90, I180, I270) — Y-Net 스타일(공유 백본 + 3 디코더 헤드)
#  - 학습/평가: **채널별 RMSE의 합**만 사용 (논문과 공정 비교 유지)
#  - 안정화: UpSample+Conv(체커보드 방지), GroupNorm, LeakyReLU, AMP, Clip, OneCycleLR
#  - 옵션: resize/crop, I0 consistency(기본 off), variance-stabilizing(sqrt) 학습(기본 off)
#
# 사용법
#  1) dataset_root를 실제 경로로 설정
#  2) python3 FINCH_DLPS_best_v2_intensity_RMSE.py
#  3) best .pth는 cfg.save_path로 저장

import os, glob, math, random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --------------------
# Utils
# --------------------

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def to_tensor01(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to float tensor in [0,1], shape (H,W). Support 8/16-bit."""
    if img.mode in ("I;16", "I;16B", "I"):
        arr = np.array(img, dtype=np.uint16).astype(np.float32) / 65535.0
    else:
        arr = np.array(img.convert('L'), dtype=np.uint8).astype(np.float32) / 255.0
    return torch.from_numpy(arr)

# --------------------
# Dataset
# --------------------
class HologramDataset(Dataset):
    """Expect folder layout:
    root/
      train|validation|test/
        sample_XXXX/phase_0.png  # 0°
                       /phase_1.png  # 90°
                       /phase_2.png  # 180°
                       /phase_3.png  # 270°
    """
    def __init__(self, root: str, split: str, ensure_all_four: bool = True,
                 resize_to: Optional[int] = None, random_crop: Optional[int] = None,
                 crop_same_for_all=True):
        self.data_dir = os.path.join(root, split)
        self.sample_dirs = sorted(glob.glob(os.path.join(self.data_dir, "sample_*")))
        self.ensure_all_four = ensure_all_four
        self.resize_to = resize_to
        self.random_crop = random_crop
        self.crop_same = crop_same_for_all
        print(f"[{split}] {len(self.sample_dirs)} samples @ {self.data_dir}")

    def __len__(self): return len(self.sample_dirs)

    def _load_all(self, paths):
        imgs = [to_tensor01(Image.open(p)) for p in paths]
        return imgs

    def _resize(self, t: torch.Tensor, size: int) -> torch.Tensor:
        # t: (H,W) float
        t = t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        t = F.interpolate(t, size=(size, size), mode='bilinear', align_corners=False)
        return t[0,0]

    def _rand_crop(self, t: torch.Tensor, size: int, top: int, left: int) -> torch.Tensor:
        H, W = t.shape
        return t[top:top+size, left:left+size]

    def __getitem__(self, idx):
        sd = self.sample_dirs[idx]
        paths = [os.path.join(sd, f"phase_{i}.png") for i in range(4)]
        exists = [os.path.exists(p) for p in paths]
        if self.ensure_all_four and not all(exists):
            missing = [i for i,b in enumerate(exists) if not b]
            raise FileNotFoundError(f"Missing phase file(s) {missing} in {sd}")
        imgs = self._load_all(paths)  # list of 4 tensors (H,W) in [0,1]

        # Optional resize (keep aspect square assumed)
        if self.resize_to is not None:
            imgs = [self._resize(im, self.resize_to) for im in imgs]

        # Optional random crop (same crop for all channels)
        if self.random_crop is not None:
            H, W = imgs[0].shape
            ch = self.random_crop
            if self.crop_same:
                if H < ch or W < ch:
                    raise ValueError(f"Crop size {ch} exceeds image size {(H,W)}")
                top = random.randint(0, H - ch)
                left = random.randint(0, W - ch)
                imgs = [self._rand_crop(im, ch, top, left) for im in imgs]
            else:
                imgs = [self._rand_crop(im, ch,
                                        random.randint(0, H - ch),
                                        random.randint(0, W - ch)) for im in imgs]

        # x: I0, y: [I90, I180, I270]
        x = imgs[0].unsqueeze(0)                 # (1,H,W)
        y = torch.stack([imgs[1], imgs[2], imgs[3]], dim=0)  # (3,H,W)
        return x, y

class Holo3PhaseDataset(Dataset):
    def __init__(self, root: str, split: str, degree_to_index: Dict[int,int],
                 resize_to: Optional[int] = None, random_crop: Optional[int] = None):
        self.base = HologramDataset(root, split, resize_to=resize_to, random_crop=random_crop)
        self.d2i = dict(degree_to_index)
        assert self.d2i == {0:0, 90:1, 180:2, 270:3}, \
            f"degree_to_index must be {{0:0,90:1,180:2,270:3}}, got {self.d2i}"
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x.float(), y.float()

# --------------------
# Model (UNet backbone + 3 decoder heads)
# --------------------
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
    def __init__(self, base_c=(64,128,256,512)):
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
        s0 = self.stem(x)   # H
        s1 = self.d1(s0)    # H/2
        s2 = self.d2(s1)    # H/4
        s3 = self.d3(s2)    # H/8
        outs = []
        for dec in self.heads:
            y = dec['u1'](s3, s2)
            y = dec['u2'](y, s1)
            y = dec['u3'](y, s0)
            y = dec['out'](y)
            # Clamp to [0,1] softly via sigmoid if 원 데이터가 [0,1]인 경우 유리
            y = torch.sigmoid(y)
            outs.append(y)
        return torch.cat(outs, dim=1)  # (B,3,H,W)

# --------------------
# Loss & Metrics (RMSE per channel)
# --------------------
class RMSEThree(nn.Module):
    def __init__(self, variance_stabilize: bool = False):
        super().__init__()
        self.vs = variance_stabilize
    def _vs(self, t):
        # Anscombe-like sqrt for stability (optional); keep evaluation in raw space
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

# --------------------
# Train / Val
# --------------------
@dataclass
class TrainConfig:
    dataset_root: str = "./hologram_dataset_images_clean"
    degree_to_index: Dict[int,int] = None
    batch_size: int = 8
    epochs: int = 120
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    num_workers: int = 4
    amp: bool = True
    seed: int = 42
    patience: int = 15
    pct_start: float = 0.1
    save_path: str = "finch_3head_intensity_best.pth"
    resize_to: Optional[int] = None     # e.g., 512 or 256 (None이면 원본)
    random_crop: Optional[int] = None   # e.g., 512 (None이면 크롭 없음)
    i0_consistency_weight: float = 0.0  # 0.0=off (공정 비교 기본값)
    variance_stabilize: bool = False    # True면 sqrt-space로 학습(평가는 원공간 RMSE)


def make_loaders(cfg: TrainConfig):
    if cfg.degree_to_index is None:
        cfg.degree_to_index = {0:0, 90:1, 180:2, 270:3}
    train_ds = Holo3PhaseDataset(cfg.dataset_root, "train", cfg.degree_to_index,
                                 resize_to=cfg.resize_to, random_crop=cfg.random_crop)
    val_ds   = Holo3PhaseDataset(cfg.dataset_root, "validation", cfg.degree_to_index,
                                 resize_to=cfg.resize_to, random_crop=None)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, drop_last=False, pin_memory=True)
    return train_loader, val_loader

@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device,
             amp: bool = True, variance_stabilize: bool = False) -> float:
    model.eval()
    metric = RMSEThree(variance_stabilize=variance_stabilize)
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

    model = ThreeHeadDecoder().to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.99))
    if steps > 0:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg.lr, epochs=cfg.epochs, steps_per_epoch=steps,
            pct_start=cfg.pct_start, div_factor=25.0, final_div_factor=100.0,
            anneal_strategy='cos'
        )
    else:
        sched = None

    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp)
    metric = RMSEThree(variance_stabilize=cfg.variance_stabilize)

    best_val, no_imp = float('inf'), 0

    for ep in range(1, cfg.epochs + 1):
        model.train()
        run = 0.0
        for bi, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=cfg.amp):
                pred = model(x)  # (B,3,H,W)
                loss, det = metric(pred, y)  # RMSE-only (논문과 동일 지표)

                # Optional: I0 consistency (합성 방식 없이, 단순 monotonic constraint)
                if cfg.i0_consistency_weight > 0.0:
                    # 약한 smoothness/consistency: 예측한 3장의 평균이 입력 I0와 비슷하도록
                    mean_pred = pred.mean(dim=1, keepdim=True)  # (B,1,H,W)
                    mse0 = F.mse_loss(mean_pred, x, reduction='mean')
                    rmse0 = torch.sqrt(mse0 + 1e-8)
                    loss = loss + cfg.i0_consistency_weight * rmse0

            scaler.scale(loss).backward()

            if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            scaler.step(opt); scaler.update();
            if sched is not None: sched.step()
            run += loss.item()

            if bi % 1 == 0:
                lr = opt.param_groups[0]['lr']
                print(f"   [Ep {ep:03d}/{cfg.epochs} | {bi:04d}/{steps}] loss={loss.item():.6f} rmse_sum={det['rmse_sum']:.6f} lr={lr:.6f}")

        avg_train = run / max(1, steps)
        val = validate(model, val_loader, device, cfg.amp, variance_stabilize=cfg.variance_stabilize)
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
        batch_size = 8,
        epochs = 120,
        lr = 1e-3,
        weight_decay = 1e-4,
        max_grad_norm = 1.0,
        num_workers = 4,
        amp = True,
        seed = 42,
        patience = 15,
        pct_start = 0.1,
        save_path = "finch_3head_intensity_best.pth",
        resize_to = None,          # 원 해상도 유지가 논문 비교엔 가장 공정
        random_crop = None,        # 메모리 아끼려면 예: 512
        i0_consistency_weight = 0.0,   # 공정 비교 기본값: off
        variance_stabilize = False,    # 켜면 수렴 빨라질 수 있으나 비교는 raw RMSE로
    )
    print(cfg)
    train(cfg)
