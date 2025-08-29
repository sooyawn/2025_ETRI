# FINCH/DLPS Training (3-Head, 0° → 90°/180°/270°) — Save `.pth`
# 
# 목표: **0°** 입력에서 **90°/180°/270°** 간섭무늬를 예측하는 네트워크를 **훈련만** 해서,
# 최적 가중치를 **`.pth` 체크포인트**로 저장합니다.
# 
# - 인코더: `Conv(3×3, s=2)` + **LayerNorm**(= `GroupNorm(1,C)`) + `LeakyReLU(0.2)`  
# - 디코더(헤드 3개): `LeakyReLU → Deconv(5×5, s=2)` × 3, **skip + FuseConv(3×3)**  
# - 출력: 3채널(90°, 180°, 270°)  
# - 손실: 채널별 **RMSE 합**  
# - 저장: `save_path`에 **best model**을 `state_dict()`로 저장 (`.pth`)

import os, random
from dataclasses import dataclass
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
import glob
from PIL import Image
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def exists(x): return x is not None

print("PyTorch:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------------------------
# Model blocks
# ---------------------------
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, out_c),                 # LayerNorm(채널 전체)와 동등
            nn.LeakyReLU(0.2, inplace=False),
        )
    def forward(self, x): return self.block(x)

class FuseConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(1, out_c),
            nn.LeakyReLU(0.2, inplace=False),
        )
    def forward(self, x): return self.block(x)

class DeconvUp(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.act = nn.LeakyReLU(0.2, inplace=False)
        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True)
    def forward(self, x): return self.deconv(self.act(x))

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class DecoderHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1   = DeconvUp(256, 128)       # H/8 -> H/4
        self.fuse1 = FuseConv(128+128, 128)   # concat e2 -> 128
        self.att1  = ChannelAttention(128)    # Attention 추가
        
        self.up2   = DeconvUp(128, 64)        # -> H/2
        self.fuse2 = FuseConv(64+64, 64)      # concat e1 -> 64
        self.att2  = ChannelAttention(64)     # Attention 추가
        
        self.up3   = DeconvUp(64, 64)         # -> H
        self.out   = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, e1, e2, e3, out_hw: Tuple[int,int]):
        H, W = out_hw
        y = self.up1(e3)
        if y.shape[2:] != e2.shape[2:]:
            y = F.interpolate(y, size=e2.shape[2:], mode='bilinear', align_corners=False)
        y = self.fuse1(torch.cat([y, e2], dim=1))
        y = self.att1(y)  # Attention 적용

        y = self.up2(y)
        if y.shape[2:] != e1.shape[2:]:
            y = F.interpolate(y, size=e1.shape[2:], mode='bilinear', align_corners=False)
        y = self.fuse2(torch.cat([y, e1], dim=1))
        y = self.att2(y)  # Attention 적용

        y = self.up3(y)
        if y.shape[2:] != (H, W):
            y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
        return self.out(y)

class FINCH_DLPS_3Head(nn.Module):
    def __init__(self, in_channels=1, num_heads=3):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, 64)  # (B,64,H/2,W/2)
        self.enc2 = EncoderBlock(64, 128)          # (B,128,H/4,W/4)
        self.enc3 = EncoderBlock(128, 256)         # (B,256,H/8,W/8)
        self.heads = nn.ModuleList([DecoderHead() for _ in range(num_heads)])

    def forward(self, x):
        H, W = x.shape[2:]
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2)
        outs = [h(e1, e2, e3, (H, W)) for h in self.heads]  # 3 × (B,1,H,W)
        return torch.cat(outs, dim=1)                       # (B,3,H,W)

# ---------------------------
# Loss: sum of per-channel RMSE
# ---------------------------
def rmse_sum_per_channel(pred: torch.Tensor, tgt: torch.Tensor, channel_weights=None):
    assert pred.shape == tgt.shape, f"shape mismatch: {pred.shape} vs {tgt.shape}"
    
    # 채널별 가중치 (90°, 180°, 270°에 대해 조금씩 다른 중요도)
    if channel_weights is None:
        channel_weights = [1.0, 1.1, 1.0]  # 180도를 조금 더 중요하게
    
    rmses = []
    weighted_loss = 0.0
    
    for c in range(pred.shape[1]):
        mse = F.mse_loss(pred[:, c], tgt[:, c], reduction='mean')
        rmse = torch.sqrt(mse + 1e-8)
        rmses.append(rmse)
        weighted_loss += rmse * channel_weights[c]
    
    details = {f"rmse_ch{c}": float(v.detach().cpu()) for c, v in enumerate(rmses)}
    details["rmse_sum"] = float(weighted_loss.detach().cpu())
    return weighted_loss, details

# Dataset wrapper (0° → 90°/180°/270°)
# 
# **전제**: 당신의 `HologramDataset(root, split)`이 (x, y)를 반환  
#   - `x`: 0° 간섭무늬 — `(1,H,W)` 또는 `(H,W)`  
#   - `y`: 스택 — `(C,H,W)` (채널 순서 예: `[0°, 90°, 180°, 270°]`)  
# 아래 `degree_to_index`를 이용해 **(90°,180°,270°) → (3,H,W)**만 뽑아서 사용.
# **import 경로만 수정**하면 됩니다.

# ⛳️ 실제 데이터셋 import 경로로 바꾸세요!
# 예: from data.holo_dataset import HologramDataset
# from your_package.your_dataset_module import HologramDataset  # <-- 여기 수정

# 간단한 HologramDataset 클래스 생성

class HologramDataset(Dataset):
    def __init__(self, root: str, split: str):
        super().__init__()
        self.root = root
        self.split = split
        
        # split에 따라 폴더 경로 설정
        if split == "train":
            self.data_dir = os.path.join(root, "train")
        elif split == "validation":
            self.data_dir = os.path.join(root, "validation")
        elif split == "test":
            self.data_dir = os.path.join(root, "test")
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # sample 폴더들 찾기
        self.sample_dirs = sorted(glob.glob(os.path.join(self.data_dir, "sample_*")))
        print(f"[{split}] Found {len(self.sample_dirs)} samples in {self.data_dir}")
    
    def __len__(self):
        return len(self.sample_dirs)
    
    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        
        # phase_0.png (0°), phase_1.png (90°), phase_2.png (180°), phase_3.png (270°) 로드
        phases = []
        for i in range(4):
            phase_path = os.path.join(sample_dir, f"phase_{i}.png")
            if os.path.exists(phase_path):
                img = Image.open(phase_path).convert('L')  # 그레이스케일로 변환
                img_array = np.array(img).astype(np.float32) / 255.0  # 0-1 정규화
                phases.append(img_array)
            else:
                # 파일이 없으면 0으로 채움
                phases.append(np.zeros((1024, 1024), dtype=np.float32))
        
        # x: phase_0.png (0° 입력)
        # y: [phase_1.png, phase_2.png, phase_3.png] (90°, 180°, 270° 타겟)
        x = phases[0]  # 0° 입력
        y = np.stack(phases[1:], axis=0)  # 90°, 180°, 270° 스택
        
        return x, y

class Holo3PhaseDataset(Dataset):
    def __init__(self, root: str, split: str, degree_to_index: Dict[int,int],
                 target_degrees=(90,180,270), augment=True):
        super().__init__()
        self.base = HologramDataset(root, split)
        self.degree_to_index = dict(degree_to_index)  # 예: {0:0, 90:1, 180:2, 270:3}
        self.target_degrees = tuple(target_degrees)
        self.augment = augment and (split == "train")  # 훈련 시에만 증강

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]                 # x: (1,H,W)/(H,W), y: (C,H,W)/(H,W)
        x = torch.as_tensor(x).float()
        y = torch.as_tensor(y).float()

        if x.dim() == 2: x = x.unsqueeze(0)   # (1,H,W) 강제
        assert x.dim()==3 and x.shape[0]==1, f"x must be (1,H,W), got {tuple(x.shape)}"

        if y.dim() == 2:
            # 매우 예외적: 단일장 라벨이면 3장 복제
            y = y.unsqueeze(0).repeat(3,1,1)
        else:
            idxs = [self.degree_to_index[d] for d in self.target_degrees]
            y = torch.stack([y[i] for i in idxs], dim=0)  # (3,H,W)

        # 데이터 증강 (훈련 시에만)
        if self.augment:
            # 수평 플리핑 (50% 확률)
            if random.random() > 0.5:
                x = TF.hflip(x)
                y = TF.hflip(y)
            
            # 수직 플리핑 (50% 확률)
            if random.random() > 0.5:
                x = TF.vflip(x)
                y = TF.vflip(y)
            
            # 90도 회전 (25% 확률)
            if random.random() > 0.75:
                angle = random.choice([90, 180, 270])
                x = TF.rotate(x, angle)
                y = TF.rotate(y, angle)

        return x, y

@dataclass
class TrainConfig:
    dataset_root: str = "./hologram_dataset_images_clean"     # ⛳️ 실제 경로
    degree_to_index: Dict[int,int] = None               # {0:0, 90:1, 180:2, 270:3}
    target_degrees: Tuple[int,int,int] = (90,180,270)   # 예측 대상
    batch_size: int = 8
    epochs: int = 80
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    pct_start: float = 0.1
    num_workers: int = 4
    amp: bool = True
    seed: int = 42
    patience: int = 12
    save_path: str = "finch_3head_90_180_270.pth"       # ⛳️ 저장할 .pth 경로

def make_loaders(cfg: TrainConfig):
    if cfg.degree_to_index is None:
        cfg.degree_to_index = {0:0, 90:0, 180:1, 270:2}  # 90°->0, 180°->1, 270°->2 (3채널)

    TrainDS = Holo3PhaseDataset
    train_ds = TrainDS(cfg.dataset_root, "train", cfg.degree_to_index, target_degrees=cfg.target_degrees)
    val_ds = TrainDS(cfg.dataset_root, "validation", cfg.degree_to_index, target_degrees=cfg.target_degrees)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, drop_last=False, pin_memory=True)
    return train_loader, val_loader

@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = True) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=amp):
            pred = model(x)
            loss, _ = rmse_sum_per_channel(pred, y)
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(1, n)

def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    train_loader, val_loader = make_loaders(cfg)
    steps = len(train_loader)
    print(f"[Data] steps/epoch={steps}  batch={cfg.batch_size}")
    
    # 훈련 상황 브리핑을 위한 설정
    print("=" * 60)
    print("=== FINCH/DLPS 3-Head Training Configuration ===")
    print(f"Dataset Root: {cfg.dataset_root}")
    print(f"Target Degrees: {cfg.target_degrees}")
    print(f"Batch Size: {cfg.batch_size}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Learning Rate: {cfg.lr}")
    print(f"Weight Decay: {cfg.weight_decay}")
    print(f"Max Gradient Norm: {cfg.max_grad_norm}")
    print(f"Patience: {cfg.patience}")
    print(f"Save Path: {cfg.save_path}")
    print("=" * 60)

    model = FINCH_DLPS_3Head(in_channels=1, num_heads=3).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp)

    best_val, no_improve = float("inf"), 0
    
    # 훈련 통계 초기화
    total_train_loss = 0.0
    total_val_loss = 0.0

    for ep in range(1, cfg.epochs + 1):
        model.train()
        run = 0.0
        epoch_losses = []
        epoch_rmse_details = []
        
        print(f"\n📊 Epoch {ep:03d}/{cfg.epochs} 시작...")
        print(f"   🔄 총 {steps} 배치 처리 중...")
        
        for bi, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            # Forward pass
            with torch.amp.autocast('cuda', enabled=cfg.amp):
                pred = model(x)  # (B,3,H,W)
                loss, details = rmse_sum_per_channel(pred, y)
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if exists(cfg.max_grad_norm) and cfg.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            scaler.step(optimizer); scaler.update(); scheduler.step()
            run += loss.item()
            
            # 배치별 손실 정보 저장
            epoch_losses.append(loss.item())
            epoch_rmse_details.append(details)

            # 상세한 훈련 상황 브리핑 (모든 배치마다 출력)
            if bi % 1 == 0 or bi == steps:  # 모든 배치마다 출력
                current_lr = optimizer.param_groups[0]['lr']
                avg_loss_so_far = sum(epoch_losses) / len(epoch_losses)
                
                # RMSE 상세 정보 계산
                avg_rmse_sum = sum(d['rmse_sum'] for d in epoch_rmse_details) / len(epoch_rmse_details)
                avg_rmse_ch0 = sum(d['rmse_ch0'] for d in epoch_rmse_details) / len(epoch_rmse_details)
                avg_rmse_ch1 = sum(d['rmse_ch1'] for d in epoch_rmse_details) / len(epoch_rmse_details)
                avg_rmse_ch2 = sum(d['rmse_ch2'] for d in epoch_rmse_details) / len(epoch_rmse_details)
                
                print(f"   📈 [Batch {bi:04d}/{steps}] "
                      f"Loss: {loss.item():.5f} | "
                      f"Avg Loss: {avg_loss_so_far:.5f} | "
                      f"RMSE Sum: {details['rmse_sum']:.5f} | "
                      f"LR: {current_lr:.6f}")
                
                if exists(cfg.max_grad_norm) and cfg.max_grad_norm > 0:
                    print(f"      📊 RMSE Details - Ch0: {avg_rmse_ch0:.5f}, Ch1: {avg_rmse_ch1:.5f}, Ch2: {avg_rmse_ch2:.5f} | "
                          f"Grad Norm: {grad_norm:.3f}")
                else:
                    print(f"      📊 RMSE Details - Ch0: {avg_rmse_ch0:.5f}, Ch1: {avg_rmse_ch1:.5f}, Ch2: {avg_rmse_ch2:.5f}")

        # 에포크 완료 후 검증 및 요약
        avg_train = run / max(1, steps)
        val = validate(model, val_loader, device, cfg.amp)
        
        # 훈련 통계 업데이트
        total_train_loss += avg_train
        total_val_loss += val
        
        print(f"\n✅ Epoch {ep:03d} 완료!")
        print(f"   📈 훈련 Loss: {avg_train:.6f}")
        print(f"   🧪 검증 Loss: {val:.6f}")
        print(f"   ⚙️  학습률: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 에포크별 진행률 표시
        progress_bar = "█" * (ep * 20 // cfg.epochs) + "░" * (20 - (ep * 20 // cfg.epochs))
        print(f"   📊 진행률: [{progress_bar}] {ep}/{cfg.epochs} ({ep*100//cfg.epochs}%)")
        
        # 평균 손실 추이
        avg_train_so_far = total_train_loss / ep
        avg_val_so_far = total_val_loss / ep
        print(f"   📊 전체 평균 - 훈련: {avg_train_so_far:.6f}, 검증: {avg_val_so_far:.6f}")

        if val < best_val - 1e-6:
            best_val, no_improve = val, 0
            torch.save(model.state_dict(), cfg.save_path)   # ← best를 .pth로 저장
            print(f"   🏆 NEW BEST! 검증 Loss: {val:.6f} → {cfg.save_path}")
        else:
            no_improve += 1
            print(f"   ⏳ 개선 없음 ({no_improve}/{cfg.patience})")
            
            if no_improve >= cfg.patience:
                print(f"\n🛑 조기 종료! {cfg.patience} 에포크 동안 개선 없음")
                break

        print("-" * 60)

    # 최종 훈련 결과 요약
    print(f"\n🎉 훈련 완료!")
    print(f"   🏆 최고 검증 Loss: {best_val:.6f}")
    print(f"   💾 모델 저장: {cfg.save_path}")
    print(f"   📊 전체 평균 훈련 Loss: {total_train_loss/cfg.epochs:.6f}")
    print(f"   📊 전체 평균 검증 Loss: {total_val_loss/cfg.epochs:.6f}")
    print(f"   ⏱️  총 훈련 에포크: {ep}")
    print("=" * 60)

if __name__ == "__main__":
    # ==== Configure & Train ====
    cfg = TrainConfig(
        dataset_root="./hologram_dataset_images_clean",  # ⛳️ 실제 경로로 변경
        degree_to_index={0:0, 90:0, 180:1, 270:2},     # 90°->0, 180°->1, 270°->2 (3채널)
        target_degrees=(90,180,270),                     # 0° 입력 → 90/180/270 예측
        batch_size=16,                                   # 배치 크기 증가 (성능 향상)
        epochs=120,                                      # 에포크 증가 (더 많은 학습)
        lr=2e-3,                                         # 학습률 증가 (빠른 수렴)
        weight_decay=5e-5,                               # 가중치 감쇠 감소 (과적합 방지하면서 성능 향상)
        max_grad_norm=0.5,                               # 그래디언트 클리핑 강화
        pct_start=0.1,                                   # CosineAnnealingWarmRestarts에서는 사용안함
        num_workers=6,                                   # 워커 수 증가
        amp=True,
        seed=42,
        patience=20,                                     # 인내심 증가 (더 많은 기회)
        save_path="finch_3head_optimized.pth",          # 최적화된 모델명
    )
    print(cfg)
    train(cfg)
