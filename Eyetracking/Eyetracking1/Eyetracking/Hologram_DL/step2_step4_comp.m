%% 원본 vs 전통적/DL 재구성 PSNR 비교
clear; clc; close all;

%% ===== 파라미터 =====
lambda = 633e-9;          % 파장 (633 nm)
z_nominal = 0.25;         % 전파 거리 (25 cm)
pixel_size = 10e-6;       % 픽셀 크기 (10 μm)
N = 256;                  % 이미지 크기
[fx, fy] = meshgrid((-N/2:N/2-1)/(N*pixel_size));

%% ===== Ground Truth (digit dataset) =====
% MATLAB Deep Learning Toolbox에 포함된 digit dataset
try
    [XTrain, YTrain] = digitTrain4DArrayData;  % (28x28x1xN)
catch
    error('digitTrain4DArrayData를 불러올 수 없습니다. Deep Learning Toolbox 확인하세요.');
end
% 크기 맞추기 (28x28 → 256x256 업샘플링)
gt_imgs = imresize(XTrain, [N N]);

%% ===== 경로 =====
dataset_folder_clean = 'hologram_dataset_images';         % 전통적 방법
dataset_folder_dl    = 'hologram_dataset_images';   % DL 입력용
train_folder_clean   = fullfile(dataset_folder_clean, 'train');
train_folder_dl      = fullfile(dataset_folder_dl, 'train');
model_path           = 'best_model_noise.pth';
if ~isfile(model_path)
    error('best_model.pth 파일이 현재 폴더에 없습니다.');
end

%% ===== Python 환경 준비 (DL Inference) =====
pe = pyenv;
if count(py.sys.path, string(pwd)) == 0
    insert(py.sys.path, int32(0), pwd);
end
mod = py.importlib.import_module('dlps_infer');
py.importlib.reload(mod);

%% ===== 샘플 선택 (1500 + 랜덤 2개) =====
sample_folders = dir(fullfile(train_folder_clean, 'sample_*'));
num_samples    = numel(sample_folders);

fixed_sample = 1500;
all_idx = 1:num_samples;
all_idx(all_idx==fixed_sample) = [];
rng('shuffle');
rand_idx = randsample(all_idx, 2);
samples_to_visualize = [fixed_sample, rand_idx];

%% ===== 결과 저장 변수 =====
PSNR_clean = zeros(1, numel(samples_to_visualize));
PSNR_dl    = zeros(1, numel(samples_to_visualize));

%% ===== 샘플별 반복 =====
for row = 1:3
    s_idx = samples_to_visualize(row);
    s_dir = sprintf('sample_%04d', s_idx);
    sample_path_clean = fullfile(train_folder_clean, s_dir);
    sample_path_dl    = fullfile(train_folder_dl, s_dir);

    fprintf('\n=== Sample %d 비교 ===\n', s_idx);

    %% === Ground Truth (원본 이미지) ===
    GT = double(gt_imgs(:,:,1,s_idx));   % digit 원본
    GT = GT / max(GT(:));                % 0~1 정규화

    %% -----------------------
    %% (1) 전통적 PSH 기반 재구성
    %% -----------------------
    S = load(fullfile(sample_path_clean,'AV.mat')); 
    AV_clean = S.AV;

    I_list = zeros(N, N, 4);
    for i = 0:3
        fni = fullfile(sample_path_clean, sprintf('phase_%d.png', i));
        I_list(:,:,i+1) = double(imread(fni)) / 65535;
    end
    I0 = I_list(:,:,1); I1 = I_list(:,:,2);
    I2 = I_list(:,:,3); I3 = I_list(:,:,4);

    CH = (I0 - I2) - 1i * (I1 - I3);
    psi_complex = CH / (4 * AV_clean);
    F_psi = fftshift(fft2(psi_complex));

    H_nom = exp(1i * pi * lambda * z_nominal * (fx.^2 + fy.^2));
    psi_nom_clean = ifft2(ifftshift(F_psi .* H_nom));
    recon_clean = abs(psi_nom_clean);
    recon_clean = recon_clean / max(recon_clean(:)+eps);

    %% -----------------------
    %% (2) DL 기반 재구성
    %% -----------------------
    out_dir = fullfile(sample_path_dl, '_pred_from_model');
    if ~isfolder(out_dir), mkdir(out_dir); end
    py.dlps_infer.infer_and_save_pngs(model_path, ...
        fullfile(sample_path_dl, 'phase_0.png'), out_dir, int32(N));

    I0_dl  = double(imread(fullfile(sample_path_dl, 'phase_0.png'))) / 65535;
    I1p_dl = double(imread(fullfile(out_dir, 'phase_1_pred.png'))) / 65535;
    I2p_dl = double(imread(fullfile(out_dir, 'phase_2_pred.png'))) / 65535;
    I3p_dl = double(imread(fullfile(out_dir, 'phase_3_pred.png'))) / 65535;

    EO = I0_dl;
    AV_dl = (min(EO(:)) + max(EO(:))) / 2;   % 보정된 AV 공식

    CH_dl = (I0_dl - I2p_dl) - 1i * (I1p_dl - I3p_dl);
    psi_complex_dl = CH_dl / (4 * AV_dl);
    F_psi_dl  = fftshift(fft2(psi_complex_dl));

    H_back = exp(1i * pi * lambda * z_nominal * (fx.^2 + fy.^2));
    psi_z0_dl = ifft2(ifftshift(F_psi_dl .* H_back));
    recon_dl = abs(psi_z0_dl);
    recon_dl = recon_dl / max(recon_dl(:));

    %% -----------------------
    %% (3) PSNR (원본 vs 각각)
    %% -----------------------
    mse_clean = mean((GT(:) - recon_clean(:)).^2);
    mse_dl    = mean((GT(:) - recon_dl(:)).^2);

    PSNR_clean(row) = 10 * log10(1 / max(mse_clean, eps));
    PSNR_dl(row)    = 10 * log10(1 / max(mse_dl, eps));

    fprintf('Sample %d: PSNR(Clean vs GT) = %.2f dB | PSNR(DL vs GT) = %.2f dB\n', ...
            s_idx, PSNR_clean(row), PSNR_dl(row));

    %% -----------------------
    %% (4) 시각화
    %% -----------------------
    figure(1);
    subplot(3, 3, (row-1)*3 + 1);
    imshow(GT, []); axis off; title(sprintf('GT (Sample %d)', s_idx));
    subplot(3, 3, (row-1)*3 + 2);
    imshow(recon_clean, []); axis off;
    title(sprintf('Clean Recon\nPSNR=%.2f dB', PSNR_clean(row)));
    subplot(3, 3, (row-1)*3 + 3);
    imshow(recon_dl, []); axis off;
    title(sprintf('DL Recon\nPSNR=%.2f dB', PSNR_dl(row)));
end

%% 최종 결과 요약
fprintf('\n=== 최종 PSNR 결과 (GT 기준) ===\n');
for row = 1:3
    fprintf('Sample %d : Clean=%.2f dB | DL=%.2f dB\n', ...
        samples_to_visualize(row), PSNR_clean(row), PSNR_dl(row));
end
