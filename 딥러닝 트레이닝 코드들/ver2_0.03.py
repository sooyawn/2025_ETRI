# =============================================================================
# FINCH/DLPS Training Script - Converted from Jupyter Notebook
# =============================================================================

# =============================================================================
# 1. Import Libraries
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image
from torch.optim.lr_scheduler import OneCycleLR

# 데이터 증강 및 전처리 설정 (수정)
# 노이즈가 있는 환경에 강건하도록 다양한 증강 기법 추가
phase_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(degrees=5),
    transforms.RandomHorizontalFlip(),  # 좌우 반전 추가
    transforms.RandomVerticalFlip(),    # 상하 반전 추가
    transforms.ToTensor()
])

# =============================================================================
# 2. Device Setup
# =============================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Available device: {device}')

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# =============================================================================
# 3. Custom Dataset Class
# =============================================================================
class HologramDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = os.path.join(root_dir, split)
        self.samples = sorted([
            os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir)
            if f.startswith('sample_')
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_path = self.samples[idx]

        phases = []
        for i in range(4):
            p = os.path.join(sample_path, f'phase_{i}.png')
            img = Image.open(p)

            # (선택) 크기 통일이 필요하면 여기서 리사이즈
            img = img.resize((256, 256), resample=Image.BILINEAR)

            arr = np.array(img)
    
            # 만약 HxWx3 형태(실수로 컬러 저장)면 첫 채널만 사용
            if arr.ndim == 3:
                arr = arr[..., 0]

            # dtype별 정규화
            if arr.dtype == np.uint16:
                arr = arr.astype(np.float32) / 65535.0
            elif arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            else:
                arr = arr.astype(np.float32)  # 이미 0~1이면 그대로

            phases.append(arr)

        phases = np.stack(phases, axis=0).astype(np.float32)  # (4, H, W)

        input_phase   = torch.from_numpy(phases[0:1])  # (1, H, W)
        target_phases = torch.from_numpy(phases[1:])   # (3, H, W)
        return input_phase, target_phases

# =============================================================================
# 4. Model Architecture Components (stride=2, kernel=3x3 반영)
# =============================================================================
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32,out_c),
            nn.LeakyReLU(0.2, inplace=False)
        )

    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 업샘플 먼저
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)  # Conv2d로 후처리
        )

    def forward(self, x):
        return self.block(x)

# =============================================================================
# 5. Main Model Architecture (stride=2 유지, 출력 크기 복원 포함)
# =============================================================================
class FINCH_DLPS_Net(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1, bias=False),
            nn.GroupNorm(32, 64), nn.LeakyReLU(0.2, inplace=False)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.GroupNorm(32, 128), nn.LeakyReLU(0.2, inplace=False)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.GroupNorm(32, 256), nn.LeakyReLU(0.2, inplace=False)
        )

        self.dec1 = self._make_decoder()
        self.dec2 = self._make_decoder()
        self.dec3 = self._make_decoder()

    def _make_decoder(self):
        return nn.Sequential(
            DecoderBlock(256, 128),                     # up1
            EncoderBlock(256, 128),                     # conv1 (with skip)
            DecoderBlock(128, 64),                      # up2
            EncoderBlock(128, 64),                      # conv2 (with skip)
            DecoderBlock(64, 64),                       # up3
            nn.Conv2d(64, 1, kernel_size=3, padding=1), # final conv
        )

    def forward(self, x):
        e1 = self.enc1(x)  # (B, 64, H/2, W/2)
        e2 = self.enc2(e1) # (B, 128, H/4, W/4)
        e3 = self.enc3(e2) # (B, 256, H/8, W/8)

        input_shape = x.shape[2:]  # (H, W)

        d1 = self._decode(self.dec1, e1, e2, e3, input_shape)  # predict phase_1
        d2 = self._decode(self.dec2, e1, e2, e3, input_shape)  # predict phase_2
        d3 = self._decode(self.dec3, e1, e2, e3, input_shape)  # predict phase_3

        return d1, d2, d3  # Each is (B, 1, H, W)

    def _decode(self, dec, e1, e2, e3, input_shape):
        x = dec[0](e3)                            # up to H/4
        if x.shape[2:] != e2.shape[2:]:
            x = F.interpolate(x, size=e2.shape[2:], mode='bilinear', align_corners=False)
        x = dec[1](torch.cat([x, e2], dim=1))     # fuse

        x = dec[2](x)                             # up to H/2
        if x.shape[2:] != e1.shape[2:]:
            x = F.interpolate(x, size=e1.shape[2:], mode='bilinear', align_corners=False)
        x = dec[3](torch.cat([x, e1], dim=1))     # fuse

        x = dec[4](x)                             # up to H
        x = dec[5](x)                             # final 1ch
        if x.shape[2:] != input_shape:
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x                                  # linear 출력

# =============================================================================
# 6. Loss Functions - SSIM + RMSE + 위상 일관성 손실 함수
# =============================================================================
def ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    mu1 = F.avg_pool2d(img1, 3, 1, 1)
    mu2 = F.avg_pool2d(img2, 3, 1, 1)
    sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1 ** 2
    sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2 ** 2
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
    
def improved_physics_loss(output1, target1, output2, target2, output3, target3, epoch=0):
    eps = 1e-8

    # --- RMSE 계산 ---
    rmse1 = torch.sqrt(torch.mean((output1 - target1) ** 2) + eps)
    rmse2 = torch.sqrt(torch.mean((output2 - target2) ** 2) + eps)
    rmse3 = torch.sqrt(torch.mean((output3 - target3) ** 2) + eps)
    base_rmse = (rmse1 + rmse2 + rmse3) /3

    if epoch<10:
        return base_rmse, {'rmse': base_rmse.item(), 'consistency':0, 'ssim':0,
                           'weight_consistency':0, 'weight_ssim':0}

    # --- 위상 일관성 손실 (phase consistency) ---
    pred_diff_1 = output1 - output3
    pred_diff_2 = output2 - output3
    target_diff_1 = target1 - target3
    target_diff_2 = target2 - target3
    consistency_loss = F.mse_loss(pred_diff_1, target_diff_1) + F.mse_loss(pred_diff_2, target_diff_2)

    # --- SSIM 손실 계산 ---
    ssim_loss1 = 1 - ssim(output1, target1)
    ssim_loss2 = 1 - ssim(output2, target2)
    ssim_loss3 = 1 - ssim(output3, target3)
    ssim_total = (ssim_loss1 + ssim_loss2 + ssim_loss3) / 3

    consistency_weight = 0.01
    ssim_weight = 0.1

    # --- 총 loss ---
    total_loss = base_rmse + consistency_weight * consistency_loss + ssim_weight * ssim_total

    # --- NaN 방지 ---
    if torch.isnan(total_loss):
        total_loss = base_rmse

    # --- 리턴 (loss + 상세정보) ---
    return total_loss, {
        'rmse': base_rmse.item(),
        'consistency': consistency_loss.item(),
        'ssim': ssim_total.item(),
        'weight_consistency': consistency_weight,
        'weight_ssim': ssim_weight
    }

# =============================================================================
# 7. Training Function
# =============================================================================
def train_model():
    # 데이터셋 설정
    dataset_root = 'hologram_dataset_images_clean'
    batch_size = 10  # 홀로그램 데이터는 크기가 크므로 배치 크기 줄임
    
    # 데이터셋 로드
    train_dataset = HologramDataset(dataset_root, 'train')
    val_dataset = HologramDataset(dataset_root, 'validation')
    test_dataset = HologramDataset(dataset_root, 'test')

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    print(f"데이터 로드 완료:")
    print(f"  - Train: {len(train_dataset)}개 샘플")
    print(f"  - Validation: {len(val_dataset)}개 샘플")
    print(f"  - Test: {len(test_dataset)}개 샘플")

    # 모델 초기화
    model = FINCH_DLPS_Net().to(device)
    print(f"GPU 사용: {next(model.parameters()).device}")

    # 훈련 설정
    learning_rate = 0.001
    training_epochs = 80
    weight_decay = 0.0001

    print("=== 훈련 설정 ===")
    print(f"학습률: {learning_rate}")
    print(f"훈련 에포크: {training_epochs}")
    print(f"가중치 감소: {weight_decay}")

    # 옵티마이저 및 스케줄러
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
        amsgrad=False
    )

    max_batches = min(100, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=max_batches,
        epochs=training_epochs,
        pct_start=0.1
    )

    print(f"총 샘플 수: {len(train_dataset)}")
    print(f"배치 크기: {batch_size}")
    print(f"총 배치 개수: {max_batches}")
    print("=" * 60)

    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 15

    # 훈련 루프
    for epoch in range(training_epochs):
        model.train()
        running_loss = 0.0
        epoch_rmse_sum = 0.0

        print(f"\n📊 Epoch {epoch + 1}/{training_epochs} 시작...")

        scaler = torch.amp.GradScaler('cuda')  # 최신 버전 사용
        
        for i, (input_phase, target_phases) in enumerate(train_loader):
            if i >= max_batches:
                break
            input_phase = input_phase.to(device)         # shape: (B, 1, H, W)
            target_phase1 = target_phases[:, 0:1].to(device)  # shape: (B, 1, H, W)
            target_phase2 = target_phases[:, 1:2].to(device)
            target_phase3 = target_phases[:, 2:3].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):  # 최신 버전 사용
                # 모델 예측
                output1, output2, output3 = model(input_phase)
                # 손실 계산
                loss, loss_details = improved_physics_loss(
                    output1, target_phase1,
                    output2, target_phase2,
                    output3, target_phase3,
                    epoch=epoch
                )

            # 역전파 및 업데이트
            scaler.scale(loss).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()

            running_loss += loss.item()
            epoch_rmse_sum += loss_details['rmse']

            # 로그 출력
            print(f"[Epoch {epoch + 1}, Batch {i + 1}/{max_batches}] "
                  f"Loss: {loss.item():.4f} | "
                  f"RMSE: {loss_details['rmse']:.4f} | "
                  f"Consistency: {loss_details['consistency']:.4f} | "
                  f"SSIM: {loss_details['ssim']:.4f} | "
                  f"Grad: {grad_norm:.3f}")

        # 에포크 마무리
        current_lr = optimizer.param_groups[0]['lr']
        avg_loss = running_loss / max_batches
        avg_rmse = epoch_rmse_sum / max_batches

        print(f"✅ Epoch {epoch + 1} 완료!")
        print(f"   📈 평균 Loss: {avg_loss:.6f}")
        print(f"   📊 평균 RMSE: {avg_rmse:.6f}")
        print(f"   ⚙️  학습률: {current_lr:.6f}")

        # 모델 저장 조건
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_noise.pth')
            print(f"   🏆 NEW BEST! RMSE: {avg_rmse:.6f}")
        else:
            patience_counter += 1

        # 조기 종료 조건
        if patience_counter >= patience_limit:
            print(f"🛑 조기 종료! {patience_limit} 에포크 개선 없음")
            break

        if avg_rmse <= 0.0036:
            print(f"🎊 목표 달성! RMSE: {avg_rmse:.6f} <= 0.0036")
            break

        print("-" * 60)
    
    print("모델이 'best_model_noise.pth'로 저장되었습니다.")
    return model

# =============================================================================
# 8. Main Execution
# =============================================================================
if __name__ == "__main__":
    try:
        model = train_model()
        print("\n✅ 모든 작업이 성공적으로 완료되었습니다!")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
