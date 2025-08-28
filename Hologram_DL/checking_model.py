# eval_finch_dlps.py
import os, gc, re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from math import log10

# ============ 모델 (훈련 코드와 동일) ============
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, out_c),
            nn.LeakyReLU(0.2, inplace=False)
        )
    def forward(self, x): return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        )
    def forward(self, x): return self.block(x)

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
            DecoderBlock(256, 128),      # up to H/4
            EncoderBlock(256, 128),      # fuse(e2)
            DecoderBlock(128, 64),       # up to H/2
            EncoderBlock(128, 64),       # fuse(e1)
            DecoderBlock(64, 64),        # up to H
            nn.Conv2d(64, 1, 3, 1, 1)    # linear output
        )

    def forward(self, x):
        e1 = self.enc1(x)  # H/2
        e2 = self.enc2(e1) # H/4
        e3 = self.enc3(e2) # H/8
        H, W = x.shape[2:]
        d1 = self._decode(self.dec1, e1, e2, e3, (H, W))
        d2 = self._decode(self.dec2, e1, e2, e3, (H, W))
        d3 = self._decode(self.dec3, e1, e2, e3, (H, W))
        return d1, d2, d3

    def _decode(self, dec, e1, e2, e3, input_shape):
        x = dec[0](e3)  # up
        if x.shape[2:] != e2.shape[2:]:
            x = F.interpolate(x, size=e2.shape[2:], mode='bilinear', align_corners=False)
        x = dec[1](torch.cat([x, e2], dim=1))  # fuse

        x = dec[2](x)  # up
        if x.shape[2:] != e1.shape[2:]:
            x = F.interpolate(x, size=e1.shape[2:], mode='bilinear', align_corners=False)
        x = dec[3](torch.cat([x, e1], dim=1))  # fuse

        x = dec[4](x)          # up to H
        x = dec[5](x)          # final conv
        if x.shape[2:] != input_shape:
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

# ============ 유틸 ============
def normalize_img(img):
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-8)

def rmse(o, t): return torch.sqrt(torch.mean((o - t) ** 2))

def psnr(o, t, max_val=1.0):
    mse = torch.mean((o - t) ** 2)
    return float('inf') if mse == 0 else 20 * log10(max_val / torch.sqrt(mse))

def classical_reconstruction(phase0, phase1, phase2, phase3):
    # 입력 텐서는 같은 device에 있어야 함(CPU 권장)
    lambda_val = 633e-9
    z = 0.25
    pixel_size = 10e-6
    N = phase0.shape[-1]
    device = phase0.device
    fx = torch.arange(-N//2, N//2, device=device) / (N * pixel_size)
    fy = torch.arange(-N//2, N//2, device=device) / (N * pixel_size)
    FX, FY = torch.meshgrid(fx, fy, indexing='ij')
    CH = (phase0 - phase2) - 1j * (phase1 - phase3)
    H_back = torch.exp(1j * torch.pi * lambda_val * z * (FX**2 + FY**2))
    F_psi = torch.fft.fftshift(torch.fft.fft2(CH))
    psi_z0 = torch.fft.ifft2(torch.fft.ifftshift(F_psi * H_back))
    reconstructed = torch.abs(psi_z0)
    reconstructed = reconstructed / (reconstructed.max() + 1e-8)
    return reconstructed

# === 정규화 없이 재구성 amplitude 계산 ===
def classical_reconstruction_raw(phase0, phase1, phase2, phase3):
    """
    4-step PSH 복원된 복소장의 amplitude(절댓값)를 '정규화 없이' 반환.
    입력 텐서 device는 동일해야 함.
    """
    lambda_val = 633e-9
    z = 0.25
    pixel_size = 10e-6
    N = phase0.shape[-1]
    device = phase0.device
    fx = torch.arange(-N//2, N//2, device=device) / (N * pixel_size)
    fy = torch.arange(-N//2, N//2, device=device) / (N * pixel_size)
    FX, FY = torch.meshgrid(fx, fy, indexing='ij')
    CH = (phase0 - phase2) - 1j * (phase1 - phase3)
    H_back = torch.exp(1j * torch.pi * lambda_val * z * (FX**2 + FY**2))
    F_psi = torch.fft.fftshift(torch.fft.fft2(CH))
    psi_z0 = torch.fft.ifft2(torch.fft.ifftshift(F_psi * H_back))
    return torch.abs(psi_z0)  # ← 정규화 없음!

# === 수정된 위상 계산 유틸리티 ===
def classical_reconstruction_phase_raw(phase0, phase1, phase2, phase3):
    """
    4-step PSH 복원된 복소장의 phase(위상)를 '정규화 없이' 반환.
    입력 텐서 device는 동일해야 함.
    """
    CH = (phase0 - phase2) - 1j * (phase1 - phase3)
    psi_phase_reconstructed = torch.angle(CH)
    return psi_phase_reconstructed

def get_available_sample_numbers(root):
    # root: e.g., '.../hologram_dataset_images_clean/test'
    nums = []
    if not os.path.isdir(root): return nums
    for name in os.listdir(root):
        if name.startswith('sample_') and os.path.isdir(os.path.join(root, name)):
            m = re.match(r'^sample_(\d+)$', name)
            if m: nums.append(int(m.group(1)))
    return sorted(nums)

# ============ 샘플 로드 ============
def load_single_sample(sample_num, root, resize=None):
  
    sample_name = f'sample_{sample_num:04d}'
    sample_path = os.path.join(root, sample_name)
    assert os.path.isdir(sample_path), f"not found: {sample_path}"

    phases = []
    for i in range(4):
        p = os.path.join(sample_path, f'phase_{i}.png')
        with Image.open(p) as img:
            if resize is not None:
                img = img.resize(resize, resample=Image.BILINEAR)
            arr = np.array(img)

        if arr.ndim == 3:  # 실수로 컬러 저장된 경우 첫 채널만
            arr = arr[..., 0]
        if arr.dtype == np.uint16: arr = arr.astype(np.float32) / 65535.0
        elif arr.dtype == np.uint8: arr = arr.astype(np.float32) / 255.0
        else: arr = arr.astype(np.float32)
        phases.append(arr)

    phases = np.stack(phases, axis=0).astype(np.float32)  # [4,H,W]
    return torch.from_numpy(phases).unsqueeze(0)          # [1,4,H,W]

# ============ 메인 평가 ============
def evaluate(sample_num=1, weights='best_model_noise.pth',
             dataset_root='hologram_dataset_images',
             split='test',
             save_dir='evaluation_results_noise',
             resize=None):
    """
    split: 'train' | 'validation' | 'test'
    resize: (256,256) 등으로 주면 학습과 동일 크기에서 평가
    """
    os.makedirs(save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:", device)

    # 모델 로드
    model = FINCH_DLPS_Net().to(device)
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"loaded: {weights}")

    # 데이터 로드 (요청 split)
    split_root = os.path.join(dataset_root, split)
    avail = get_available_sample_numbers(split_root)
    if not avail:
        raise RuntimeError(f"No samples found under: {split_root}")
    if sample_num not in avail:
        print(f"[warn] sample_{sample_num:04d} not found in {split}. Using {avail[0]:04d}.")
        sample_num = avail[0]

    holograms = load_single_sample(sample_num, root=split_root, resize=resize).to(device)  # [1,4,H,W]
    inp = holograms[:, 0:1]  # phase0
    t1  = holograms[:, 1:2]
    t2  = holograms[:, 2:3]
    t3  = holograms[:, 3:4]

    with torch.no_grad():
        p1, p2, p3 = model(inp)

    # ----- 재구성은 CPU에서 수행(메모리 절약) -----
    inp_cpu = inp.squeeze(0).cpu()  # [1,H,W]
    t1_cpu  = t1.squeeze(0).cpu()
    t2_cpu  = t2.squeeze(0).cpu()
    t3_cpu  = t3.squeeze(0).cpu()
    p1_cpu  = p1.squeeze(0).cpu()
    p2_cpu  = p2.squeeze(0).cpu()
    p3_cpu  = p3.squeeze(0).cpu()

    cr = classical_reconstruction(inp_cpu[0], t1_cpu[0], t2_cpu[0], t3_cpu[0])  # torch CPU (정규화됨)
    dr = classical_reconstruction(inp_cpu[0], p1_cpu[0], p2_cpu[0], p3_cpu[0])

    # ----- 시각화 (기존) -----
    in_np = inp_cpu.numpy()
    t1_np = t1_cpu.numpy()
    t2_np = t2_cpu.numpy()
    t3_np = t3_cpu.numpy()
    p1_np = p1_cpu.numpy()
    p2_np = p2_cpu.numpy()
    p3_np = p3_cpu.numpy()

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    # row 1: targets
    axes[0,0].imshow(normalize_img(in_np[0]), cmap='gray', interpolation='nearest'); axes[0,0].set_title("Target (Phase 0°)"); axes[0,0].axis('off')
    axes[0,1].imshow(normalize_img(t1_np[0]), cmap='gray', interpolation='nearest'); axes[0,1].set_title("Target (Phase 90°)"); axes[0,1].axis('off')
    axes[0,2].imshow(normalize_img(t2_np[0]), cmap='gray', interpolation='nearest'); axes[0,2].set_title("Target (Phase 180°)"); axes[0,2].axis('off')
    axes[0,3].imshow(normalize_img(t3_np[0]), cmap='gray', interpolation='nearest'); axes[0,3].set_title("Target (Phase 270°)"); axes[0,3].axis('off')

    # row 2: predictions
    axes[1,0].imshow(normalize_img(in_np[0]), cmap='gray', interpolation='nearest'); axes[1,0].set_title("Input (Phase 0°)"); axes[1,0].axis('off')
    axes[1,1].imshow(normalize_img(p1_np[0]), cmap='gray', interpolation='nearest'); axes[1,1].set_title("Pred (90°)"); axes[1,1].axis('off')
    axes[1,2].imshow(normalize_img(p2_np[0]), cmap='gray', interpolation='nearest'); axes[1,2].set_title("Pred (180°)"); axes[1,2].axis('off')
    axes[1,3].imshow(normalize_img(p3_np[0]), cmap='gray', interpolation='nearest'); axes[1,3].set_title("Pred (270°)"); axes[1,3].axis('off')

    # row 3: reconstructions
    axes[2,0].imshow(normalize_img(cr.numpy()), cmap='gray', interpolation='nearest'); axes[2,0].set_title("4-PSH (Real)"); axes[2,0].axis('off')
    axes[2,1].imshow(normalize_img(dr.numpy()), cmap='gray', interpolation='nearest'); axes[2,1].set_title("4-PSH (Pred)"); axes[2,1].axis('off')
    axes[2,2].axis('off'); axes[2,3].axis('off')

    plt.tight_layout()
    out1 = os.path.join(save_dir, f'phase_and_reconstruction_{split}_s{sample_num:04d}.png')
    plt.savefig(out1, dpi=300, bbox_inches='tight')
    plt.show(); plt.close('all')
    print("saved:", out1)

    # ===== 'Amplitude (RAW)' Figure (3x4) - 축/격자 추가 =====
    # 각 프레임의 amplitude를 '정규화 없이' 시각화.
    # 위상 프레임은 강도 I 이므로 amplitude = sqrt(max(I, 0)).
    def amp_raw_from_intensity(arr2d: np.ndarray) -> np.ndarray:
        return np.sqrt(np.clip(arr2d, 0.0, None))

    # Target amplitude (0°, 90°, 180°, 270°)
    amp_t0 = amp_raw_from_intensity(in_np[0])
    amp_t1 = amp_raw_from_intensity(t1_np[0])
    amp_t2 = amp_raw_from_intensity(t2_np[0])
    amp_t3 = amp_raw_from_intensity(t3_np[0])

    # Pred amplitude (0°=input, 90°, 180°, 270°)
    amp_p0 = amp_raw_from_intensity(in_np[0])
    amp_p1 = amp_raw_from_intensity(p1_np[0])
    amp_p2 = amp_raw_from_intensity(p2_np[0])
    amp_p3 = amp_raw_from_intensity(p3_np[0])

    # 재구성 amplitude (정규화 X)
    cr_raw = classical_reconstruction_raw(inp_cpu[0], t1_cpu[0], t2_cpu[0], t3_cpu[0]).numpy()
    dr_raw = classical_reconstruction_raw(inp_cpu[0], p1_cpu[0], p2_cpu[0], p3_cpu[0]).numpy()

    # 전역 vmax를 잡아 절대 스케일 통일
    global_max = np.max([
        amp_t0.max(), amp_t1.max(), amp_t2.max(), amp_t3.max(),
        amp_p0.max(), amp_p1.max(), amp_p2.max(), amp_p3.max(),
        cr_raw.max(), dr_raw.max()
    ])
    vmin, vmax = 0.0, float(global_max if global_max > 0 else 1.0)

    # 축 범위를 데이터 크기에 맞춤 (256x256이면 0~256)
    H, W = in_np.shape[1], in_np.shape[2]

    fig2, axes2 = plt.subplots(3, 4, figsize=(16, 12))

    def show(ax, img, title):

        im = ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest',
                       origin='upper', extent=[0, W, 0, H])
        ax.set_title(title)
        ax.set_xlabel(f'X ({W} pixels)')
        ax.set_ylabel(f'Y ({H} pixels)')

        step_x = max(1, int(W/5))
        step_y = max(1, int(H/5))
        ax.set_xticks(np.arange(0, W+0.1, step_x))
        ax.set_yticks(np.arange(0, H+0.1, step_y))
        ax.set_xlim(0, W); ax.set_ylim(0, H)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.colorbar(im, ax=ax, shrink=0.6, aspect=20)

    # Row 1: Target amplitude (RAW)
    show(axes2[0,0], amp_t0, "Target (Phase 0°)")
    show(axes2[0,1], amp_t1, "Target (Phase 90°)")
    show(axes2[0,2], amp_t2, "Target (Phase 180°)")
    show(axes2[0,3], amp_t3, "Target (Phase 270°)")

    # Row 2: Pred amplitude (RAW)
    show(axes2[1,0], amp_p0, "Input (Phase 0°)")
    show(axes2[1,1], amp_p1, "Pred (90°)")
    show(axes2[1,2], amp_p2, "Pred (180°)")
    show(axes2[1,3], amp_p3, "Pred (270°)")

    # Row 3: Reconstruction amplitude (RAW)
    show(axes2[2,0], cr_raw, "4-PSH (Real)")
    show(axes2[2,1], dr_raw, "4-PSH (Pred)")
    axes2[2,2].axis('off'); axes2[2,3].axis('off')

    plt.tight_layout()
    out2 = os.path.join(save_dir, f'amplitude_raw_{split}_s{sample_num:04d}.png')
    plt.savefig(out2, dpi=300, bbox_inches='tight')
    plt.show(); plt.close('all')
    print("saved:", out2)

    # ===== 'Phase (RAW)' Figure (3x4) - 축/격자 추가 =====
    # 각 위상 프레임의 위상을 '정규화 없이' 시각화.
    # 복원된 위상 값은 -pi 에서 +pi 범위를 가짐.
    def phase_raw_from_intensity(arr2d: np.ndarray) -> np.ndarray:
        """강도 영상(I)에서 위상 정보는 없으므로 0으로 가정."""
        return np.zeros_like(arr2d)

    # Target phase (0°, 90°, 180°, 270°)
    phase_t0 = phase_raw_from_intensity(in_np[0])
    phase_t1 = phase_raw_from_intensity(t1_np[0])
    phase_t2 = phase_raw_from_intensity(t2_np[0])
    phase_t3 = phase_raw_from_intensity(t3_np[0])

    # Pred phase (0°=input, 90°, 180°, 270°)
    phase_p0 = phase_raw_from_intensity(in_np[0])
    phase_p1 = phase_raw_from_intensity(p1_np[0])
    phase_p2 = phase_raw_from_intensity(p2_np[0])
    phase_p3 = phase_raw_from_intensity(p3_np[0])

    # 재구성 phase (정규화 X)
    cr_raw_phase = classical_reconstruction_phase_raw(inp_cpu[0], t1_cpu[0], t2_cpu[0], t3_cpu[0]).numpy()
    dr_raw_phase = classical_reconstruction_phase_raw(inp_cpu[0], p1_cpu[0], p2_cpu[0], p3_cpu[0]).numpy()

    # 전역 vmin/vmax를 잡아 절대 스케일 통일 (-pi ~ pi)
    vmin_phase, vmax_phase = -np.pi, np.pi

    fig3, axes3 = plt.subplots(3, 4, figsize=(16, 12))

    def show_phase(ax, img, title):
        im = ax.imshow(img, cmap='gray', vmin=vmin_phase, vmax=vmax_phase, interpolation='nearest',
                       origin='upper', extent=[0, W, 0, H])
        ax.set_title(title)
        ax.set_xlabel(f'X ({W} pixels)')
        ax.set_ylabel(f'Y ({H} pixels)')
        step_x = max(1, int(W/5))
        step_y = max(1, int(H/5))
        ax.set_xticks(np.arange(0, W+0.1, step_x))
        ax.set_yticks(np.arange(0, H+0.1, step_y))
        ax.set_xlim(0, W); ax.set_ylim(0, H)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.colorbar(im, ax=ax, shrink=0.6, aspect=20, label='Phase (radians)')

    # Row 1: Target phase (RAW)
    show_phase(axes3[0,0], phase_t0, "Target (Phase 0°)")
    show_phase(axes3[0,1], phase_t1, "Target (Phase 90°)")
    show_phase(axes3[0,2], phase_t2, "Target (Phase 180°)")
    show_phase(axes3[0,3], phase_t3, "Target (Phase 270°)")

    # Row 2: Pred phase (RAW)
    show_phase(axes3[1,0], phase_p0, "Input (Phase 0°)")
    show_phase(axes3[1,1], phase_p1, "Pred (90°)")
    show_phase(axes3[1,2], phase_p2, "Pred (180°)")
    show_phase(axes3[1,3], phase_p3, "Pred (270°)")

    # Row 3: Reconstruction phase (RAW)
    show_phase(axes3[2,0], cr_raw_phase, "4-PSH (Real)")
    show_phase(axes3[2,1], dr_raw_phase, "4-PSH (Pred)")
    axes3[2,2].axis('off'); axes3[2,3].axis('off')

    plt.tight_layout()
    out3 = os.path.join(save_dir, f'phase_raw_{split}_s{sample_num:04d}.png')
    plt.savefig(out3, dpi=300, bbox_inches='tight')
    plt.show(); plt.close('all')
    print("saved:", out3)

    # ----- 지표 (CPU에서 계산) -----
    rmse1 = rmse(p1_cpu, t1_cpu).item()
    rmse2 = rmse(p2_cpu, t2_cpu).item()
    rmse3 = rmse(p3_cpu, t3_cpu).item()
    psnr1 = psnr(p1_cpu, t1_cpu)
    psnr2 = psnr(p2_cpu, t2_cpu)
    psnr3 = psnr(p3_cpu, t3_cpu)
    print(f"RMSE avg: {(rmse1+rmse2+rmse3)/3:.6f} | PSNR avg: {(psnr1+psnr2+psnr3)/3:.2f} dB")

if __name__ == "__main__":
    # 기본 경로/스플릿 설정
    dataset_root = 'hologram_dataset_images'
    split = input("Choose split [train/validation/test] (default: test): ").strip().lower() or 'test'
    if split not in ('train', 'validation', 'test'):
        print("Invalid split. Using 'test'.")
        split = 'test'

    split_root = os.path.join(dataset_root, split)
    avail = get_available_sample_numbers(split_root)
    if not avail:
        raise RuntimeError(f"No samples under {split_root}. Check your dataset path.")

    # 샘플 번호 입력 (유효한 입력까지 반복)
    while True:
        try:
            prompt = f"Enter sample number ({avail[0]}–{avail[-1]}) [{avail[0]}]: "
            sample_num = input(prompt).strip()
            if not sample_num:
                sample_num = avail[0]
                print(f"No input. Using default sample {sample_num}.")
                break
            sample_num = int(sample_num)
            if sample_num in avail:
                break
            else:
                print(f"⚠️ Sample number {sample_num} is not available. Please enter a number between {avail[0]} and {avail[-1]}.")
        except ValueError:
            print("⚠️ Invalid input. Please enter a number.")

    print(f"\nAnalyzing {split} / sample_{sample_num:04d} ...")
    # 학습에서 256x256로 리사이즈했다면 아래 resize를 (256,256)로 맞춰주세요.
    evaluate(sample_num=sample_num,
             weights="best_model_noise.pth",
             dataset_root=dataset_root,
             split=split,
             save_dir="evaluation_results_noise",
             resize=None)  # ← 필요 시 (256, 256)로 변경
