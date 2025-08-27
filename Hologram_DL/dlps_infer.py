import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

# ---------- Model Architecture ----------
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 2, 1, bias=False),
            nn.GroupNorm(32, out_c),
            nn.LeakyReLU(0.2, inplace=False),
        )
    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
        )
    def forward(self, x):
        return self.block(x)

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
            DecoderBlock(256, 128),
            EncoderBlock(256, 128),
            DecoderBlock(128, 64),
            EncoderBlock(128, 64),
            DecoderBlock(64, 64),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def _decode(self, dec, e1, e2, e3, input_shape):
        x = dec[0](e3)  # up to H/4
        if x.shape[2:] != e2.shape[2:]:
            x = F.interpolate(x, size=e2.shape[2:], mode='bilinear', align_corners=False)
        x = dec[1](torch.cat([x, e2], dim=1))
        x = dec[2](x)   # up to H/2
        if x.shape[2:] != e1.shape[2:]:
            x = F.interpolate(x, size=e1.shape[2:], mode='bilinear', align_corners=False)
        x = dec[3](torch.cat([x, e1], dim=1))
        x = dec[4](x)   # up to H
        x = dec[5](x)   # final 1ch
        if x.shape[2:] != input_shape:
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        input_shape = x.shape[2:]
        d1 = self._decode(self.dec1, e1, e2, e3, input_shape)
        d2 = self._decode(self.dec2, e1, e2, e3, input_shape)
        d3 = self._decode(self.dec3, e1, e2, e3, input_shape)
        return d1, d2, d3  # (B,1,H,W)

# 파라미터
lambda_ = 633e-9      # 파장 (633nm)
z_nominal = 0.25      # 복원할 거리 (m)
pixel_size = 10e-6    # 픽셀 크기 (10um)
N = 256               # 이미지 크기

# 주파수 좌표계는 함수 내에서 동적으로 생성

def load_model(model_path, device):
    # 모델 구조 생성
    model = FINCH_DLPS_Net().to(device)
    
    # state_dict 로드
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def infer_and_save_pngs(model_path, phase0_path, out_dir, N_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)

    # 입력 로드
    I0 = np.array(Image.open(phase0_path)).astype(np.float32) / 65535.0
    inp = torch.from_numpy(I0).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)

    # 모델 예측 (phase1~3)
    with torch.no_grad():
        y1, y2, y3 = model(inp)  # 각각 (1,1,H,W)
        preds = torch.cat([y1, y2, y3], dim=1).cpu().numpy()  # shape: (1, 3, H, W)

    # 저장 폴더
    os.makedirs(out_dir, exist_ok=True)

    # 원본 phase_0 저장 (다시 저장 필요 없으면 생략 가능)
    Image.fromarray((I0*65535).astype(np.uint16)).save(os.path.join(out_dir, 'phase_0.png'))

    # 위상 이동 간섭 패턴(intensity) 저장
    phase_preds = []
    for i in range(3):
        pred_img = np.clip(preds[0, i], 0, 1)
        phase_preds.append(pred_img)
        Image.fromarray((pred_img*65535).astype(np.uint16)).save(os.path.join(out_dir, f'phase_{i+1}_pred.png'))
    
    # 딥러닝 예측 결과로 4-PSH 복원 (classical_reconstruction과 동일한 방식)
    I1p, I2p, I3p = phase_preds
    
    # 실제 이미지 크기에 맞는 주파수 좌표계 생성
    actual_N = I0.shape[0]  # 실제 이미지 크기
    fx = np.arange(-actual_N//2, actual_N//2) / (actual_N * pixel_size)
    fy = np.arange(-actual_N//2, actual_N//2) / (actual_N * pixel_size)
    fx, fy = np.meshgrid(fx, fy)
    
    # 4-PSH 복원 공식 (딥러닝 예측 간섭 패턴 사용)
    CH = (I0 - I2p) - 1j * (I1p - I3p)
    H_back = np.exp(1j * np.pi * lambda_ * z_nominal * (fx**2 + fy**2))
    F_psi = np.fft.fftshift(np.fft.fft2(CH))
    psi_z0 = np.fft.ifft2(np.fft.ifftshift(F_psi * H_back))
    recon_nom = np.abs(psi_z0)
    recon_nom = recon_nom / (recon_nom.max() + 1e-8)

    # 복원 이미지 저장
    Image.fromarray((recon_nom*65535).astype(np.uint16)).save(os.path.join(out_dir, 'reconstructed_pred.png'))
    
    return out_dir

# ---------- Main Execution ----------
if __name__ == "__main__":
    print("FINCH DLPS Inference Script (Clean Model)")
    print("=" * 50)
    
    # Check if model file exists
    model_path = "best_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please ensure the model file exists in the current directory.")
        exit(1)
    
    # Check if input image exists
    phase0_path = "hologram_dataset_images_clean/test/sample_0001/phase_0.png"
    if not os.path.exists(phase0_path):
        print(f"Error: Input image '{phase0_path}' not found!")
        print("Please ensure the input image exists.")
        exit(1)
    
    # Create output directory
    out_dir = "inference_results_clean"
    
    print(f"Model: {model_path}")
    print(f"Input: {phase0_path}")
    print(f"Output: {out_dir}")
    print(f"Image size: {N}x{N}")
    print(f"Wavelength: {lambda_*1e9:.0f}nm")
    print(f"Reconstruction distance: {z_nominal*1000:.1f}mm")
    print(f"Pixel size: {pixel_size*1e6:.1f}um")
    print("-" * 50)
    
    try:
        print("Loading model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        print("Running inference...")
        result_dir = infer_and_save_pngs(model_path, phase0_path, out_dir, N)
        
        print(f"\nSuccess! Results saved to: {result_dir}")
        print("Generated files:")
        print("  - phase_0.png (input)")
        print("  - phase_1_pred.png")
        print("  - phase_2_pred.png") 
        print("  - phase_3_pred.png")
        print("  - reconstructed_pred.png")
        
        # Check if files were actually created
        import glob
        created_files = glob.glob(os.path.join(result_dir, "*.png"))
        print(f"\nTotal files created: {len(created_files)}")
        for f in sorted(created_files):
            file_size = os.path.getsize(f) / 1024  # KB
            print(f"  {os.path.basename(f)} ({file_size:.1f} KB)")
            
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
