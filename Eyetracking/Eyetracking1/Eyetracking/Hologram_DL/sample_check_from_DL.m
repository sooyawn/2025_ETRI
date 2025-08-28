%% 딥러닝 모델이 예측한 위상 정보로 디지털 홀로그램 재구성 + PSNR 분석 (Approx ASM + AV 보정)
clear; clc; close all;

%% ===== 파라미터 설정 =====
lambda = 633e-9;          % 파장 (633 nm)
z_nominal = 0.25;         % 전파 거리 (25 cm)
pixel_size = 10e-6;       % 픽셀 크기 (10 μm)
N = 256;                  % 이미지 크기
k = 2 * pi / lambda;      % 파수
[fx, fy] = meshgrid((-N/2:N/2-1)/(N*pixel_size));

%% ===== 경로 =====
dataset_folder = 'hologram_dataset_images';
train_folder   = fullfile(dataset_folder, 'train');
model_path     = 'best_model_noise.pth';
if ~isfile(model_path)
    error('best_model.pth 파일이 현재 폴더에 없습니다.');
end

%% ===== Python 환경 준비 =====
pe = pyenv;
if count(py.sys.path, string(pwd)) == 0
    insert(py.sys.path, int32(0), pwd);
end
mod = py.importlib.import_module('dlps_infer');
py.importlib.reload(mod);
disp('dlps_infer.py 로드 완료');

%% ===== 샘플 선택 (sample_1500 + 랜덤 2개) =====
sample_folders = dir(fullfile(train_folder, 'sample_*'));
num_samples    = numel(sample_folders);

fixed_sample = 1500;
all_idx = 1:num_samples;
all_idx(all_idx==fixed_sample) = [];
rng('shuffle');
rand_idx = randsample(all_idx, 2);

samples_to_visualize = [fixed_sample, rand_idx];

%% ===== 깊이 범위 =====
z_min = 0.16; z_max = 0.48; z_step = 0.01;
z_range = z_min:z_step:z_max;
num_depths = numel(z_range);

%% ===== Figure 준비 =====
figure(1); clf; % Phase + Recon
figure(2); clf; % PSNR vs Depth

%% ===== 샘플별 반복 =====
for row = 1:3
    s_idx = samples_to_visualize(row);
    sample_path = fullfile(train_folder, sprintf('sample_%04d', s_idx));
    fprintf('\n=== 샘플 %d: DL 예측 위상/재구성 ===\n', s_idx);

    %% --- DL 모델 실행 ---
    out_dir = fullfile(sample_path, '_pred_from_model');
    if ~isfolder(out_dir), mkdir(out_dir); end
    py.dlps_infer.infer_and_save_pngs(model_path, ...
        fullfile(sample_path, 'phase_0.png'), out_dir, int32(N));

    %% --- phase 불러오기 ---
    I0  = double(imread(fullfile(sample_path, 'phase_0.png'))) / 65535;
    I1p = double(imread(fullfile(out_dir, 'phase_1_pred.png'))) / 65535;
    I2p = double(imread(fullfile(out_dir, 'phase_2_pred.png'))) / 65535;
    I3p = double(imread(fullfile(out_dir, 'phase_3_pred.png'))) / 65535;

    %% --- 복원 (근사 ASM + AV 보정) ---
    EO = I0;  
    AV = (min(EO(:)) + max(EO(:))) / 2;   % 보정된 AV 공식

    CH = (I0 - I2p) - 1i * (I1p - I3p);
    psi_complex = CH / (4 * AV);
    F_psi  = fftshift(fft2(psi_complex));

    % 근사 ASM propagation (nominal z)
    H_back = exp(1i * pi * lambda * z_nominal * (fx.^2 + fy.^2));
    psi_z0 = ifft2(ifftshift(F_psi .* H_back));
    recon_nom = abs(psi_z0);
    recon_nom = recon_nom / max(recon_nom(:));

    %% === Figure 1: Phase + Recon ===
    figure(1);
    start_col = (row - 1) * 5;
    subplot(3, 5, start_col + 1); imshow(I0, []);  axis off; title('Phase 0');
    subplot(3, 5, start_col + 2); imshow(I1p, []); axis off; title('Phase π/2');
    subplot(3, 5, start_col + 3); imshow(I2p, []); axis off; title('Phase π');
    subplot(3, 5, start_col + 4); imshow(I3p, []); axis off; title('Phase 3π/2');
    subplot(3, 5, start_col + 5); imshow(recon_nom, []); axis off;
    title(sprintf('Recon @ %.0fmm', z_nominal*1000));

    %% === Figure 2: PSNR vs Depth ===
    psnr_values = zeros(1, num_depths);
    recon_images = zeros(N, N, num_depths);
    for i = 1:num_depths
        z_test = z_range(i);
        H_depth = exp(1i * pi * lambda * z_test * (fx.^2 + fy.^2));  % 근사 ASM
        psi_d = ifft2(ifftshift(F_psi .* H_depth));
        recon_d = abs(psi_d);
        recon_d = recon_d / max(recon_d(:));

        mse_d = mean((recon_nom(:) - recon_d(:)).^2);
        psnr_values(i) = 10 * log10(1 / max(mse_d, eps));
        recon_images(:,:,i) = recon_d;
    end

    [max_psnr, best_idx] = max(psnr_values);
    best_depth = z_range(best_idx);
    idx_minus = nearest_idx(z_range, best_depth - 0.08);
    idx_plus  = nearest_idx(z_range, best_depth + 0.08);

    figure(2);
    subplot(3, 4, (row - 1)*4 + 1);
    plot(z_range*1000, psnr_values, 'b-', 'LineWidth', 2); hold on;
    plot(best_depth*1000, max_psnr, 'ro','MarkerFaceColor','r');
    xlabel('Depth (mm)'); ylabel('PSNR (dB)');
    title(sprintf('Sample %d | PSNR vs Depth', s_idx)); grid on;

    subplot(3, 4, (row - 1)*4 + 2);
    imshow(recon_images(:,:,best_idx), []); axis off;
    title(sprintf('Best %.0fmm\n%.2f dB', best_depth*1000, max_psnr));

    subplot(3, 4, (row - 1)*4 + 3);
    imshow(recon_images(:,:,idx_minus), []); axis off;
    title(sprintf('-80mm %.0fmm', z_range(idx_minus)*1000));

    subplot(3, 4, (row - 1)*4 + 4);
    imshow(recon_images(:,:,idx_plus), []); axis off;
    title(sprintf('+80mm %.0fmm', z_range(idx_plus)*1000));
end

figure(1);
sgtitle(['Phase-shifted Patterns + Reconstructions (DL Predicted(Noisy))'], 'FontSize', 14);
figure(2);
sgtitle('PSNR vs Depth (DL Predicted Recon(Noisy))', 'FontSize', 14);

%% ===== Helper =====
function idx = nearest_idx(z_range, z0)
    [~, idx] = min(abs(z_range - z0));
end
