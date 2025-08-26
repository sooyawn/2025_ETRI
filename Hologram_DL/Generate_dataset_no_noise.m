clear; clc; close all;

%% ===== 파라미터 설정 =====
lambda = 633e-9;     % 파장 (633 nm)
z = 0.25;            % 전파 거리 (25 cm)
pixel_size = 10e-6;  % 픽셀 크기 (10 μm)
N = 256;             % 이미지 크기
[fx, fy] = meshgrid((-N/2:N/2-1)/(N*pixel_size));

%% ===== 저장 경로 =====
num_total_samples = 5000;
dataset_folder = 'hologram_dataset_images_clean';
train_folder = fullfile(dataset_folder, 'train');
val_folder   = fullfile(dataset_folder, 'validation');
test_folder  = fullfile(dataset_folder, 'test');
cellfun(@(f) ~exist(f,'dir') && mkdir(f), {train_folder, val_folder, test_folder});

%% ===== MNIST 데이터 로드 =====
try
    [XTrain, YTrain] = digitTrain4DArrayData;
catch
    error('MNIST 데이터셋 로드 실패');
end
if size(XTrain,4) > num_total_samples
    XTrain = XTrain(:,:,:,1:num_total_samples);
    YTrain = YTrain(1:num_total_samples);
end

%% ===== 데이터 분할 =====
num_train = round(num_total_samples * 0.7);
num_val   = round(num_total_samples * 0.2);
num_test  = num_total_samples - num_train - num_val;
splits        = {'train', 'validation', 'test'};
split_folders = {train_folder, val_folder, test_folder};
split_counts  = [num_train, num_val, num_test];
split_starts  = [1, num_train+1, num_train+num_val+1];

%% ===== 데이터셋 생성 =====
for split_idx = 1:3
    split_folder = split_folders{split_idx};
    split_count  = split_counts(split_idx);
    split_start  = split_starts(split_idx);
    fprintf('\n=== %s 데이터 생성 (%d개) ===\n', splits{split_idx}, split_count);

    for sample_idx = 1:split_count
        global_idx = split_start + sample_idx - 1;

        % --- 입력 이미지(256x256) ---
        I = double(XTrain(:,:,1,global_idx)) / 255;
        I = imresize(I, [N, N]);
        I = (I - min(I(:))) / (max(I(:)) - min(I(:)));
        if mean(I(:)) < 0.1, I = min(I*3, 1); end

        % --- Angluar Spectrum Method(전파 시뮬레이션) ---
        U_obj = I .* exp(1i*zeros(N)); %물체파
        H = exp(-1i * pi * lambda * z * (fx.^2 + fy.^2));
        U_obj_f = fftshift(fft2(U_obj));
        U_obj_z = ifft2(ifftshift(U_obj_f .* H));

        % --- AV 계산 ---
        % 참조파의 진폭을 물체파의 진폭을 기준으로 계산
        AV = min(abs(U_obj_z(:))) + max(abs(U_obj_z(:)));

        % --- 샘플 폴더 ---
        sample_folder = fullfile(split_folder, sprintf('sample_%04d', sample_idx));
        if ~exist(sample_folder,'dir'), mkdir(sample_folder); end
        save(fullfile(sample_folder,'AV.mat'),'AV');

        % --- 4단계 위상 시프팅 이미지 저장 ---
        for i = 0:3
            delta = i * pi/2;
            U_ref = AV * exp(-1i * delta);
            total_field = U_obj_z + U_ref;
            I_clean = abs(total_field).^2;
            I_norm = I_clean / max(I_clean(:));
            imwrite(uint16(I_norm * 65535), fullfile(sample_folder, sprintf('phase_%d.png', i)));
        end

        if mod(sample_idx,500)==0
            fprintf('진행률: %d / %d\n', sample_idx, split_count);
        end
    end
    fprintf('%s 데이터셋 생성 완료\n', splits{split_idx});
end

fprintf('\n=== 전체 데이터셋 생성 완료 ===\n');
