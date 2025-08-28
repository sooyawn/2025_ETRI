clear; clc; close all;

%% ===== 파라미터 =====
lambda = 633e-9;
pixel_size = 10e-6;
N = 256;
[fx, fy] = meshgrid((-N/2:N/2-1)/(N*pixel_size));
z_nominal = 0.25;

%% ===== 경로 =====
dataset_folder = 'hologram_dataset_images';
train_folder   = fullfile(dataset_folder, 'train');

%% ===== 샘플 폴더 스캔 =====
sample_folders = dir(fullfile(train_folder, 'sample_*'));
num_samples = numel(sample_folders);
assert(num_samples >= 3, 'train에 sample_* 폴더가 최소 3개 필요합니다.');

% sample_1500 존재 확인 및 인덱스 찾기
fixed_name = 'sample_1500';
fixed_idx_in_dir = find(strcmp({sample_folders.name}, fixed_name), 1);
assert(~isempty(fixed_idx_in_dir), '%s 폴더가 없습니다: %s', fixed_name, fullfile(train_folder, fixed_name));

% 랜덤 2개 선택 (fixed 제외)
rng('shuffle');
candidates = setdiff(1:num_samples, fixed_idx_in_dir);
rand_pick = candidates(randperm(numel(candidates), 2));

% 최종 선택: [고정 1개 + 랜덤 2개]
selected_indices = [fixed_idx_in_dir, rand_pick];

%% ===== Figure 준비 =====
figure(1); clf; % Phase + Recon
figure(2); clf; % PSNR vs Depth

%% ===== 샘플별 처리 =====
for row = 1:3
    % 현재 샘플 경로
    s_dir = sample_folders(selected_indices(row)).name;   % e.g., 'sample_1500'
    sample_path = fullfile(train_folder, s_dir);

    % --- AV 불러오기 ---
    S = load(fullfile(sample_path,'AV.mat')); 
    AV = S.AV;

    % --- 4단계 위상 이미지 로드 ---
    I_list = zeros(N, N, 4);
    for i = 0:3
        fni = fullfile(sample_path, sprintf('phase_%d.png', i));
        assert(exist(fni,'file')==2, '이미지가 없습니다: %s', fni);
        I_list(:,:,i+1) = double(imread(fni)) / 65535;
    end
    I0 = I_list(:,:,1); I1 = I_list(:,:,2);
    I2 = I_list(:,:,3); I3 = I_list(:,:,4);

    % --- 복원 ---
    CH = (I0 - I2) - 1i * (I1 - I3);
    psi_complex = CH / (4 * AV);
    F_psi = fftshift(fft2(psi_complex));

    H_nom = exp(1i * pi * lambda * z_nominal * (fx.^2 + fy.^2));
    psi_nom = ifft2(ifftshift(F_psi .* H_nom));
    recon_nom = abs(psi_nom);
    recon_nom = recon_nom / max(recon_nom(:) + eps);

    % Figure 1: phase + recon
    figure(1);
    start_col = (row - 1) * 5;
    phase_labels = {'0', '\pi/2', '\pi', '3\pi/2'};
    for i = 1:4
        subplot(3, 5, start_col + i);
        imshow(I_list(:,:,i), []); axis off;
        title(sprintf('%s\n | Phase(Noisy) %s', s_dir, phase_labels{i}), 'Interpreter','tex');
    end
    subplot(3, 5, start_col + 5);
    imshow(recon_nom, []); 
    axis off;
    title(sprintf('%s\n | Recon(Nosiy)) @ %.0fmm', s_dir, z_nominal*1000));

    % Figure 2: PSNR vs Depth
    z_range = 0.16:0.01:0.48;
    num_depths = numel(z_range);
    psnr_values = zeros(1, num_depths);
    recon_images = zeros(N, N, num_depths);

    for i = 1:num_depths
        H_depth = exp(1i * pi * lambda * z_range(i) * (fx.^2 + fy.^2));
        psi_d = ifft2(ifftshift(F_psi .* H_depth));
        recon_d = abs(psi_d);
        recon_d = recon_d / max(recon_d(:) + eps);

        mse_d = mean((recon_nom(:) - recon_d(:)).^2);
        psnr_values(i) = 10 * log10(1 / max(mse_d, eps));
        recon_images(:,:,i) = recon_d;
    end

    [max_psnr, best_idx] = max(psnr_values);
    best_depth = z_range(best_idx);
    idx_minus = max(1, best_idx - round(0.08 / 0.02));
    idx_plus  = min(num_depths, best_idx + round(0.08 / 0.02));

    figure(2);
    subplot(3, 4, (row-1)*4 + 1);
    plot(z_range*1000, psnr_values, 'b-', 'LineWidth', 2); hold on;
    plot(best_depth*1000, max_psnr, 'ro', 'MarkerFaceColor', 'r');
    xlabel('Depth (mm)'); ylabel('PSNR (dB)'); grid on;
    title(sprintf('%s | PSNR vs Depth', s_dir));

    subplot(3, 4, (row-1)*4 + 2);
    imshow(recon_images(:,:,best_idx), []); axis off;
    title(sprintf('Best %.0fmm\n%.2f dB', best_depth*1000, max_psnr));

    subplot(3, 4, (row-1)*4 + 3);
    imshow(recon_images(:,:,idx_minus), []); axis off;
    title(sprintf('-80mm %.0fmm\n%.2f dB', z_range(idx_minus)*1000, psnr_values(idx_minus)));

    subplot(3, 4, (row-1)*4 + 4);
    imshow(recon_images(:,:,idx_plus), []); axis off;
    title(sprintf('+80mm %.0fmm\n%.2f dB', z_range(idx_plus)*1000, psnr_values(idx_plus)));
end

figure(1); sgtitle('Phase-shifted Patterns + Reconstructions (Noisy dataset)', 'FontSize', 14);
figure(2); sgtitle('PSNR vs Depth (Noisy dataset)', 'FontSize', 14);
