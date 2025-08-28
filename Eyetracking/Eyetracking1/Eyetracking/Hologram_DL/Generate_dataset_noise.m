%% dataset_gen_noisy.m
clear; clc; close all;

%% ===== 파라미터 =====
lambda     = 633e-9;     % 파장 (633 nm)
z          = 0.25;       % 전파 거리 (25 cm)
pixel_size = 10e-6;      % 픽셀 크기 (10 μm)
N          = 256;        % 이미지 크기
[fx, fy]   = meshgrid((-N/2:N/2-1)/(N*pixel_size));

% ----- 노이즈 설정 -----
NOISE_MODE = 'variance';   % 'variance' or 'snr'
NOISE_MEAN = 0;            % 가우시안 평균
GAUSS_VAR  = 1.0e-2;       % 분산 σ^2 (NOISE_MODE='variance')
SNR_dB     = 30;           % (NOISE_MODE='snr')에서 사용
SAVE_CLEAN = false;        % 깨끗한 이미지도 같이 저장할지

rng(20250818);             % 재현성

%% ===== 저장 경로/개수 =====
% 요구사항: train/test/validation = 3500 / 500 / 1000
dataset_folder = 'C:\Users\박수연\Desktop\Hologram_DL\hologram_dataset_images';
train_folder   = fullfile(dataset_folder, 'train');
test_folder    = fullfile(dataset_folder, 'test');
val_folder     = fullfile(dataset_folder, 'validation');
cellfun(@(f) ~exist(f,'dir') && mkdir(f), {train_folder, test_folder, val_folder});

counts = struct('train',3500, 'test',500, 'validation',1000);
num_total_samples = counts.train + counts.test + counts.validation;  % = 5000

%% ===== MNIST 로드 =====
try
    [XTrain, YTrain] = digitTrain4DArrayData;
catch
    error('MNIST 데이터셋 로드 실패 (digitTrain4DArrayData 미존재).');
end
if size(XTrain,4) < num_total_samples
    error('MNIST 이미지 수가 부족합니다. 필요: %d, 실제: %d', ...
          num_total_samples, size(XTrain,4));
end

% 필요한 만큼만 슬라이스
XTrain = XTrain(:,:,:,1:num_total_samples);

%% ===== 분할 인덱스 (연속 슬라이스) =====
idx_train = 1 : counts.train;
idx_test  = (idx_train(end)+1) : (idx_train(end)+counts.test);
idx_val   = (idx_test(end)+1)  : (idx_test(end)+counts.validation);

splits = {
    'train',      train_folder, idx_train;
    'test',       test_folder,  idx_test;
    'validation', val_folder,   idx_val
};

%% ===== 노이즈 함수 핸들 =====
add_awgn = @(I) local_add_awgn(I, NOISE_MODE, NOISE_MEAN, GAUSS_VAR, SNR_dB);

%% ===== 데이터 생성 =====
for s = 1:size(splits,1)
    split_name   = splits{s,1};
    split_folder = splits{s,2};
    idx_range    = splits{s,3};
    split_count  = numel(idx_range);

    fprintf('\n=== %s 데이터 생성 (%d개) ===\n', split_name, split_count);

    for n = 1:split_count
        global_idx = idx_range(n);

        % --- 입력(0~1, 256x256) ---
        I = double(XTrain(:,:,1,global_idx)) / 255;
        I = imresize(I, [N, N]);
        I = (I - min(I(:))) / max(1e-12, (max(I(:)) - min(I(:))));
        if mean(I(:)) < 0.1, I = min(I*3, 1); end

        % --- 물체파 전파(ASM) ---
        U_obj  = I .* exp(1i*zeros(N)); % 위상 0
        H      = exp(-1i * pi * lambda * z * (fx.^2 + fy.^2));
        U_objF = fftshift(fft2(U_obj));
        U_obj_z = ifft2(ifftshift(U_objF .* H));

        % --- 참조 진폭 AV ---
        AV = min(abs(U_obj_z(:))) + max(abs(U_obj_z(:)));

        % --- 샘플 폴더 ---
        sample_folder = fullfile(split_folder, sprintf('sample_%04d', n));
        if ~exist(sample_folder,'dir'), mkdir(sample_folder); end
        save(fullfile(sample_folder,'AV.mat'),'AV');

        % --- 4-step PSH 저장 ---
        for k = 0:3
            delta = k * pi/2;
            U_ref = AV * exp(-1i * delta);
            total_field = U_obj_z + U_ref;

            I_clean = abs(total_field).^2;
            I_clean = I_clean / (max(I_clean(:)) + eps);  % 0~1

            if SAVE_CLEAN
                imwrite(uint16(I_clean * 65535), ...
                    fullfile(sample_folder, sprintf('phase_%d_clean.png', k)));
            end

            % 노이즈 추가
            I_noisy = add_awgn(I_clean);
            I_noisy = max(0, min(1, I_noisy));            % 안정화

            imwrite(uint16(I_noisy * 65535), ...
                fullfile(sample_folder, sprintf('phase_%d.png', k)));
        end

        if mod(n,500)==0
            fprintf('진행률: %d / %d\n', n, split_count);
        end
    end
    fprintf('%s 데이터셋 생성 완료\n', split_name);
end

fprintf('\n=== 전체 데이터셋 생성 완료 ===\n');

%% ===== 로컬 함수 =====
function I_noisy = local_add_awgn(I_norm, mode, mu, var_g, SNR_dB)
% I_norm: 0~1 정규화 영상
% mode  : 'variance' 또는 'snr'
% mu    : 평균 (보통 0)
% var_g : 분산 σ^2 (mode='variance')
% SNR_dB: SNR(dB) (mode='snr')
    switch lower(mode)
        case 'variance'
            sigma = sqrt(max(var_g, 0));
        case 'snr'
            P_signal = mean(I_norm(:).^2);
            P_noise  = P_signal / (10^(SNR_dB/10));
            sigma    = sqrt(max(P_noise, 0));
        otherwise
            error('NOISE_MODE must be ''variance'' or ''snr''.');
    end
    noise   = mu + sigma * randn(size(I_norm));
    I_noisy = I_norm + noise;
end
