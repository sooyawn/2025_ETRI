import cv2
import numpy as np
import os
import json

# ========================================
# 체크보드 설정 (사용자가 쉽게 수정 가능)
# ========================================
CHESSBOARD_SIZE = (3, 9)      # 체크보드 크기 (가로, 세로) - 내부 코너 개수

# ========================================
# 카메라 인덱스 설정 (Camera_1.py와 동일)
# ========================================
LEFT_CAMERA_INDEX = 2         # 왼쪽 카메라
CENTER_CAMERA_INDEX = 1       # 중앙 카메라
RIGHT_CAMERA_INDEX = 0        # 오른쪽 카메라

# ========================================
# 디스플레이 설정 (Realtime_Video_4.py와 동일)
# ========================================
DISPLAY_SCALE = 0.5          # 화면 표시용 스케일 (0.5 = 50% 크기, 1.0 = 100% 크기)
# ========================================

def load_calibration_config(config_file):
    """캘리브레이션 설정 파일을 로드합니다 (Calibration_New.py 호환)."""
    print(f"\n📁 캘리브레이션 설정 로드: {config_file}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Calibration_New.py가 생성하는 JSON 구조에 맞춰 파라미터 추출
    # 카메라 내부 파라미터
    left_mtx = np.array(config['left_camera']['intrinsic_matrix'])
    left_dist = np.array(config['left_camera']['distortion_coefficients'])
    right_mtx = np.array(config['right_camera']['intrinsic_matrix'])
    right_dist = np.array(config['right_camera']['distortion_coefficients'])
    
    # 스테레오 관계
    R = np.array(config['stereo_calibration']['rotation_matrix'])
    T = np.array(config['stereo_calibration']['translation_vector'])
    
    # 정렬 맵은 별도 NPZ 파일에서 로드해야 함
    # config_file 경로에서 maps_file 경로 생성
    config_dir = os.path.dirname(config_file)
    pair_name = config['pair_name']
    maps_file = os.path.join(config_dir, f"{pair_name}_rectification_maps.npz")
    
    print(f"   🔍 정렬 맵 파일 찾는 중: {maps_file}")
    
    if os.path.exists(maps_file):
        # NPZ 파일에서 정렬 맵 로드
        # left_map1_x와 right_map1_x가 이미 2채널 맵이므로 map2는 사용하지 않음
        maps_data = np.load(maps_file)
        left_map1_x = np.array(maps_data['left_map1_x'], dtype=np.float32)
        right_map1_x = np.array(maps_data['right_map1_x'], dtype=np.float32)
        print(f"   ✅ 정렬 맵 로드 성공")
    else:
        print(f"   ❌ 정렬 맵 파일을 찾을 수 없습니다: {maps_file}")
        print(f"   💡 Calibration_New.py를 먼저 실행하여 캘리브레이션을 완료해주세요.")
        raise FileNotFoundError(f"정렬 맵 파일을 찾을 수 없습니다: {maps_file}")
    
    # 거울모드 데이터는 현재 Calibration_New.py에서 지원하지 않음
    # mirror_data = None
    
    print(f"   ✅ {config['pair_name']} 설정 로드 완료")
    print(f"   맵 크기 - 왼쪽: {left_map1_x.shape}, 오른쪽: {right_map1_x.shape}")
    
    return {
        'left_mtx': left_mtx, 'left_dist': left_dist,
        'right_mtx': right_mtx, 'right_dist': right_dist,
        'R': R, 'T': T,
        'left_map1_x': left_map1_x, 'right_map1_x': right_map1_x
    }

def apply_rectification_maps(left_img, right_img, config_LC, config_CR):
    """캘리브레이션 결과를 사용하여 이미지를 정렬합니다."""
    print(f"\n🔄 이미지 정렬 (캘리브레이션 맵 적용)")
    
    # 맵 데이터 타입과 크기 확인
    print(f"   맵 데이터 정보:")
    print(f"   LC_left_map1_x: {config_LC['left_map1_x'].shape}, dtype: {config_LC['left_map1_x'].dtype}")
    print(f"   CR_right_map1_x: {config_CR['right_map1_x'].shape}, dtype: {config_CR['right_map1_x'].dtype}")
    
    # 입력 이미지 정보
    print(f"   입력 이미지 정보:")
    print(f"   left_img: {left_img.shape}, dtype: {left_img.dtype}")
    print(f"   right_img: {right_img.shape}, dtype: {right_img.dtype}")
    
    try:
        # LC의 Left 카메라 정렬 맵 사용
        # left_map1_x가 이미 2채널 맵이므로 map2는 사용하지 않음
        print(f"   왼쪽 이미지 정렬 시도...")
        left_rectified = cv2.remap(left_img, config_LC['left_map1_x'], None, 
                                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        print(f"   ✅ 왼쪽 이미지 정렬 성공")
        
        # CR의 Right 카메라 정렬 맵 사용
        # right_map1_x가 이미 2채널 맵이므로 map2는 사용하지 않음
        print(f"   오른쪽 이미지 정렬 시도...")
        right_rectified = cv2.remap(right_img, config_CR['right_map1_x'], None, 
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        print(f"   ✅ 오른쪽 이미지 정렬 성공")
        
    except cv2.error as e:
        print(f"   ❌ cv2.remap 오류: {e}")
        print(f"   맵 데이터 검증:")
        print(f"     left_map1_x 범위: {np.min(config_LC['left_map1_x']):.3f} ~ {np.max(config_LC['left_map1_x']):.3f}")
        print(f"     right_map1_x 범위: {np.min(config_CR['right_map1_x']):.3f} ~ {np.max(config_CR['right_map1_x']):.3f}")
        raise e
    
    print(f"   ✅ 이미지 정렬 완료")
    print(f"   왼쪽 정렬: {left_rectified.shape}")
    print(f"   오른쪽 정렬: {right_rectified.shape}")
    
    # 시각화는 파이프라인에서 필요한 3가지만 수행하도록 제한 (여기서는 표시하지 않음)
    
    return left_rectified, right_rectified

def preprocess_image_for_chessboard(gray):
    """체커보드 검출을 위한 이미지 전처리"""
    # 노이즈 제거
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # 대비 향상 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # 가우시안 블러로 노이즈 제거
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # 이진화 (적응적 임계값)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    return enhanced, binary

def validate_chessboard_corners(corners, pattern_size, gray):
    """검출된 코너의 유효성 검증"""
    cols, rows = pattern_size
    
    if corners is None or len(corners) != cols * rows:
        return False
    
    # 코너 간격 일관성 검사
    corners_reshaped = corners.reshape(rows, cols, 2)
    
    # 수평 간격 검사
    for row in range(rows):
        for col in range(1, cols):
            dist = np.linalg.norm(corners_reshaped[row, col] - corners_reshaped[row, col-1])
            if dist < 10:  # 최소 간격
                return False
    
    # 수직 간격 검사
    for col in range(cols):
        for row in range(1, rows):
            dist = np.linalg.norm(corners_reshaped[row, col] - corners_reshaped[row-1, col])
            if dist < 10:  # 최소 간격
                return False
    
    return True

def detect_chessboard_multiscale(gray, pattern_size, max_attempts=3):
    """여러 스케일에서 체커보드 검출 시도"""
    scales = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for scale in scales:
        # 이미지 크기 조정
        h, w = gray.shape
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(gray, (new_w, new_h))
        
        # 검출 시도
        ret, corners = cv2.findChessboardCorners(resized, pattern_size, 
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            # 원본 크기로 좌표 변환
            corners = corners / scale
            print(f"           ✅ 다중 스케일 검출 성공 (스케일: {scale:.1f})")
            return True, corners
    
    print(f"           ❌ 다중 스케일 검출 실패")
    return False, None

def save_debug_images(gray, enhanced, binary, img_name, pattern_size):
    """디버깅을 위한 이미지 저장"""
    debug_dir = './data/debug_images'
    os.makedirs(debug_dir, exist_ok=True)
    
    # 원본 이미지 저장
    cv2.imwrite(os.path.join(debug_dir, f'{img_name}_original.png'), gray)
    
    # 전처리된 이미지들 저장
    cv2.imwrite(os.path.join(debug_dir, f'{img_name}_enhanced.png'), enhanced)
    cv2.imwrite(os.path.join(debug_dir, f'{img_name}_binary.png'), binary)
    
    # 체커보드 패턴 정보 저장
    with open(os.path.join(debug_dir, f'{img_name}_pattern_info.txt'), 'w') as f:
        f.write(f"Pattern Size: {pattern_size}\n")
        f.write(f"Image Shape: {gray.shape}\n")
        f.write(f"Image Type: {gray.dtype}\n")
    
    print(f"         💾 디버그 이미지 저장: {debug_dir}")

def detect_chessboard_corners(gray, img_name, pattern_size):
    """전달된 pattern_size(내부 코너: cols x rows)로 체크보드 코너를 검출하고 중간 행을 반환합니다.

    실패 시, rows는 고정하고 cols를 1씩 줄이며(OpenCV 제약상 >=3 유지) 재시도합니다.
    """
    base_cols, base_rows = int(pattern_size[0]), int(pattern_size[1])
    print(f"       🔍 OpenCV 체크보드 코너 검출 시작... (요청: {base_cols}x{base_rows})")

    # 이미지 전처리
    print(f"         🎨 이미지 전처리 중...")
    enhanced, binary = preprocess_image_for_chessboard(gray)
    
    # 디버그 이미지 저장 (선택적)
    save_debug_images(gray, enhanced, binary, img_name, pattern_size)
    
    # 다양한 검출 플래그 조합
    flags_combinations = [
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_CLUSTERING,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_CLUSTERING
    ]

    # cols만 줄여가며 재시도 (rows는 고정)
    for cols in range(base_cols, 2, -1):  # OpenCV 조건: cols, rows > 2
        rows = base_rows
        print(f"         🔁 시도: {cols}x{rows} (cols 감소 재시도)")
        
        # 다중 스케일 검출 시도
        print(f"           🔍 다중 스케일 검출 시도 중...")
        multiscale_ret, multiscale_corners = detect_chessboard_multiscale(gray, (cols, rows))
        if multiscale_ret:
            print(f"           ✅ 다중 스케일 검출 성공!")
            # 코너 검증 및 처리
            if validate_chessboard_corners(multiscale_corners, (cols, rows), gray):
                corners_xy = multiscale_corners.reshape(-1, 2)
                # 한 열만 사용: 왼쪽=가장 왼쪽 열, 오른쪽=오른쪽 끝 열
                if '왼쪽' in str(img_name):
                    col_index = 0  # 가장 왼쪽 열
                elif '오른쪽' in str(img_name):
                    col_index = cols - 1  # 가장 오른쪽 열
                else:
                    col_index = 0

                indices = [r * cols + col_index for r in range(rows)]
                selected_corners = corners_xy[indices]
                print(f"           ✅ {img_name}: 열 {col_index} 선택, {len(selected_corners)}개 코너 추출 (다중 스케일)")
                return selected_corners
            else:
                print(f"           ⚠️ 다중 스케일 검출 결과 검증 실패")
        else:
            print(f"           ℹ️ 다중 스케일 검출 실패, 표준 방법으로 진행")

        # 1) SB 알고리즘 우선 시도 (전처리된 이미지 사용)
        try:
            sb_flags_list = [
                0,
                cv2.CALIB_CB_NORMALIZE_IMAGE,
                cv2.CALIB_CB_EXHAUSTIVE if hasattr(cv2, 'CALIB_CB_EXHAUSTIVE') else 0,
                (cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE) if hasattr(cv2, 'CALIB_CB_EXHAUSTIVE') else cv2.CALIB_CB_NORMALIZE_IMAGE,
                cv2.CALIB_CB_ACCURACY if hasattr(cv2, 'CALIB_CB_ACCURACY') else 0
            ]
            for sb_flags in sb_flags_list:
                if sb_flags == 0 and sb_flags_list.count(0) > 1:
                    # 중복 0 제거 목적의 continue
                    pass
                print(f"           📍 SB 시도 (flags={sb_flags})")
                
                # 전처리된 이미지들로 검출 시도
                for img, img_name_suffix in [(enhanced, "enhanced"), (binary, "binary"), (gray, "original")]:
                    ret, corners = cv2.findChessboardCornersSB(img, (cols, rows), flags=sb_flags)
                    if ret:
                        print(f"           ✅ SB 성공 ({cols}x{rows}) - {img_name_suffix} 이미지 사용")
                        
                        # 코너 서브픽셀 정밀도 향상 (더 정밀한 파라미터)
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
                        corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criteria)
                        
                        # 코너 검증
                        if validate_chessboard_corners(corners, (cols, rows), gray):
                            corners_xy = corners.reshape(-1, 2)
                            total_corners = corners_xy.shape[0]
                            expected_corners = cols * rows
                            
                            if total_corners >= expected_corners:
                                # 한 열만 사용: 왼쪽=가장 왼쪽 열, 오른쪽=오른쪽 끝 열
                                if '왼쪽' in str(img_name):
                                    col_index = 0  # 가장 왼쪽 열
                                elif '오른쪽' in str(img_name):
                                    col_index = cols - 1  # 가장 오른쪽 열
                                else:
                                    col_index = 0

                                indices = [r * cols + col_index for r in range(rows)]
                                selected_corners = corners_xy[indices]
                                print(f"           ✅ {img_name}: 열 {col_index} 선택, {len(selected_corners)}개 코너 추출")
                                return selected_corners
                            else:
                                print(f"           ⚠️ {img_name}: 코너 개수 부족 ({total_corners}/{expected_corners})")
                        else:
                            print(f"           ⚠️ {img_name}: 코너 검증 실패")
                            continue
        except AttributeError:
            print(f"           ℹ️ SB 알고리즘 미지원(OpenCV 버전)")
        except Exception as e:
            print(f"           ❌ SB 오류: {e}")

        # 2) 표준 알고리즘 플래그 조합 시도 (전처리된 이미지들로)
        for i, flags in enumerate(flags_combinations):
            print(f"           📍 방법 {i+1} 시도 중... (플래그: {flags})")

            # 전처리된 이미지들로 검출 시도
            for img, img_name_suffix in [(enhanced, "enhanced"), (binary, "binary"), (gray, "original")]:
                try:
                    ret, corners = cv2.findChessboardCorners(img, (cols, rows), flags=flags)

                    if ret:
                        print(f"           ✅ {img_name}: 체크보드 검출 성공 ({cols}x{rows}) - {img_name_suffix} 이미지 사용!")

                        # 코너 서브픽셀 정밀도 향상 (더 정밀한 파라미터)
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
                        corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criteria)

                        # 코너 검증
                        if validate_chessboard_corners(corners, (cols, rows), gray):
                            # Nx1x2 -> Nx2
                            corners_xy = corners.reshape(-1, 2)

                            total_corners = corners_xy.shape[0]
                            expected_corners = cols * rows

                            if total_corners >= expected_corners:
                                if '왼쪽' in str(img_name):
                                    col_index = 0  # 가장 왼쪽 열
                                elif '오른쪽' in str(img_name):
                                    col_index = cols - 1  # 가장 오른쪽 열
                                else:
                                    col_index = 0

                                indices = [r * cols + col_index for r in range(rows)]
                                selected_corners = corners_xy[indices]
                                print(f"           ✅ {img_name}: 열 {col_index} 선택, {len(selected_corners)}개 코너 추출")
                                return selected_corners
                            else:
                                print(f"           ⚠️ {img_name}: 코너 개수 부족 ({total_corners}/{expected_corners})")
                                # 이 경우도 다음 cols로 재시도
                        else:
                            print(f"           ⚠️ {img_name}: 코너 검증 실패")
                            continue
                    else:
                        print(f"           ❌ {img_name}: 방법 {i+1} 실패 ({img_name_suffix} 이미지)")

                except Exception as e:
                    print(f"           ❌ {img_name}: 방법 {i+1} 오류 - {e}")
                    continue

    print(f"       ❌ {img_name}: 모든 cols 감소 재시도 실패 (최종 {base_cols}→3)")
    
    # 마지막 대안: 대체 검출 방법 시도
    print(f"       🆘 최종 대안: 대체 검출 방법 시도")
    fallback_ret, fallback_corners, fallback_pattern = fallback_chessboard_detection(gray, img_name)
    
    if fallback_ret:
        print(f"       ✅ 대체 방법으로 검출 성공! 패턴: {fallback_pattern}")
        # 한 열만 사용 (패턴에 맞게 조정)
        cols, rows = fallback_pattern
        if '왼쪽' in str(img_name):
            col_index = 0
        elif '오른쪽' in str(img_name):
            col_index = cols - 1
        else:
            col_index = 0
            
        indices = [r * cols + col_index for r in range(rows)]
        selected_corners = fallback_corners.reshape(-1, 2)[indices]
        print(f"       ✅ {img_name}: 대체 방법으로 열 {col_index} 선택, {len(selected_corners)}개 코너 추출")
        return selected_corners
    
    print(f"       ❌ {img_name}: 모든 검출 방법 실패")
    return None


def visualize_rectified_images(left_original, right_original, left_rectified, right_rectified):
    """정렬 전후 이미지를 시각화하고 저장합니다. (필요시만 사용)"""
    print(f"\n🔄 정렬 전후 이미지 시각화")
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"   ❌ matplotlib이 설치되지 않았습니다. pip install matplotlib")
        return
    
    # BGR을 RGB로 변환
    if len(left_original.shape) == 3:
        left_orig_rgb = cv2.cvtColor(left_original, cv2.COLOR_BGR2RGB)
        right_orig_rgb = cv2.cvtColor(right_original, cv2.COLOR_BGR2RGB)
    else:
        left_orig_rgb = left_original
        right_orig_rgb = right_original
        
    if len(left_rectified.shape) == 3:
        left_rect_rgb = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2RGB)
        right_rect_rgb = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2RGB)
    else:
        left_rect_rgb = left_rectified
        right_rect_rgb = right_rectified
    
    # 2x2 서브플롯 생성
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # 원본 이미지
    axes[0, 0].imshow(left_orig_rgb)
    axes[0, 0].set_title('1. Original Left Image', fontsize=16, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(right_orig_rgb)
    axes[0, 1].set_title('2. Original Right Image', fontsize=16, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 정렬된 이미지
    axes[1, 0].imshow(left_rect_rgb)
    axes[1, 0].set_title('3. Rectified Left Image', fontsize=16, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(right_rect_rgb)
    axes[1, 1].set_title('4. Rectified Right Image', fontsize=16, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # 저장
    output_dir = './data/visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    rectified_plot_file = os.path.join(output_dir, 'rectification_comparison.png')
    plt.savefig(rectified_plot_file, dpi=300, bbox_inches='tight')
    print(f"   💾 정렬 전후 비교 이미지 저장: {rectified_plot_file}")
    
    plt.show()
    
    # 개별 정렬된 이미지 저장
    cv2.imwrite(os.path.join(output_dir, 'left_rectified.png'), left_rectified)
    cv2.imwrite(os.path.join(output_dir, 'right_rectified.png'), right_rectified)
    print(f"   💾 정렬된 이미지 개별 저장 완료")


def visualize_chessboard_corners(left_img, right_img, left_corners, right_corners):
    """검출된 체크보드 코너를 이미지에 표시하고 저장합니다."""
    print(f"\n🎯 체크보드 코너 검출 결과 시각화")
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"   ❌ matplotlib이 설치되지 않았습니다. pip install matplotlib")
        return
    
    # BGR을 RGB로 변환
    if len(left_img.shape) == 3:
        left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
    else:
        left_rgb = left_img
        right_rgb = right_img
    
    # 코너점을 이미지에 그리기
    left_with_corners = left_rgb.copy()
    right_with_corners = right_rgb.copy()
    
    # 왼쪽 이미지에 코너점 표시
    for i, corner in enumerate(left_corners):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(left_with_corners, (x, y), 8, (255, 0, 0), -1)  # 파란색 원
        cv2.circle(left_with_corners, (x, y), 10, (255, 255, 255), 2)  # 흰색 테두리
        cv2.putText(left_with_corners, str(i), (x+15, y+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 오른쪽 이미지에 코너점 표시
    for i, corner in enumerate(right_corners):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(right_with_corners, (x, y), 8, (0, 255, 0), -1)  # 초록색 원
        cv2.circle(right_with_corners, (x, y), 10, (255, 255, 255), 2)  # 흰색 테두리
        cv2.putText(right_with_corners, str(i), (x+15, y+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    ax1.imshow(left_with_corners)
    ax1.set_title(f'1. Left Image - Detected Corners ({len(left_corners)} points)', 
                  fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(right_with_corners)
    ax2.set_title(f'2. Right Image - Detected Corners ({len(right_corners)} points)', 
                  fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    
    # 저장
    output_dir = './data/visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    corners_plot_file = os.path.join(output_dir, 'chessboard_corners_detection.png')
    plt.savefig(corners_plot_file, dpi=300, bbox_inches='tight')
    print(f"   💾 체크보드 코너 검출 결과 저장: {corners_plot_file}")
    
    plt.show()
    
    # 개별 코너 표시 이미지 저장
    cv2.imwrite(os.path.join(output_dir, 'left_corners.png'), 
                cv2.cvtColor(left_with_corners, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, 'right_corners.png'), 
                cv2.cvtColor(right_with_corners, cv2.COLOR_RGB2BGR))
    print(f"   💾 코너 표시 이미지 개별 저장 완료")


def visualize_feature_matching(left_img, right_img, left_corners, right_corners, H=None):
    """특징점 매칭 시각화: 좌우 이미지를 가로로 붙여 대응점 라인을 그립니다."""
    print(f"\n🔗 특징점 매칭 과정 시각화")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"   ❌ matplotlib이 설치되지 않았습니다. pip install matplotlib")
        return

    # BGR->RGB 변환
    left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB) if len(left_img.shape) == 3 else left_img
    right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB) if len(right_img.shape) == 3 else right_img

    # 동일한 높이로 보정 후 가로 연결
    h1, w1 = left_rgb.shape[:2]
    h2, w2 = right_rgb.shape[:2]
    Hh = max(h1, h2)
    pad_left = np.zeros((Hh - h1, w1, 3), dtype=left_rgb.dtype) if h1 < Hh else None
    pad_right = np.zeros((Hh - h2, w2, 3), dtype=right_rgb.dtype) if h2 < Hh else None
    left_pad = np.vstack([left_rgb, pad_left]) if pad_left is not None else left_rgb
    right_pad = np.vstack([right_rgb, pad_right]) if pad_right is not None else right_rgb
    concat = np.hstack([left_pad, right_pad])

    # 그리기
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.imshow(concat)
    ax.set_title('Feature Matching Visualization - Left-Right Correspondence', fontsize=16, fontweight='bold')
    ax.axis('off')

    # 점/라인 표시 (오른쪽 이미지는 x좌표에 w1 오프셋)
    for i, (lp, rp) in enumerate(zip(left_corners, right_corners)):
        lx, ly = float(lp[0]), float(lp[1])
        rx, ry = float(rp[0]) + w1, float(rp[1])
        ax.plot([lx, rx], [ly, ry], '-', color='yellow', linewidth=2, alpha=0.9)
        ax.plot(lx, ly, 'ro', markersize=6)
        ax.plot(rx, ry, 'bo', markersize=6)

    plt.tight_layout()

    # 저장
    output_dir = './data/visualization'
    os.makedirs(output_dir, exist_ok=True)
    matching_plot_file = os.path.join(output_dir, 'feature_matching_process.png')
    plt.savefig(matching_plot_file, dpi=300, bbox_inches='tight')
    print(f"   💾 특징점 매칭 과정 저장: {matching_plot_file}")
    plt.show()


def visualize_homography_matching(left_img, right_img, left_corners, right_corners, H):
    """호모그래피 매칭 결과 시각화: 좌우 이미지를 붙여 변환 좌표와 대응선을 표시합니다."""
    print(f"\n🔗 호모그래피 매칭 관계 시각화")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"   ❌ matplotlib이 설치되지 않았습니다. pip install matplotlib")
        return

    left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB) if len(left_img.shape) == 3 else left_img
    right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB) if len(right_img.shape) == 3 else right_img

    h1, w1 = left_rgb.shape[:2]
    h2, w2 = right_rgb.shape[:2]
    Hh = max(h1, h2)
    pad_left = np.zeros((Hh - h1, w1, 3), dtype=left_rgb.dtype) if h1 < Hh else None
    pad_right = np.zeros((Hh - h2, w2, 3), dtype=right_rgb.dtype) if h2 < Hh else None
    left_pad = np.vstack([left_rgb, pad_left]) if pad_left is not None else left_rgb
    right_pad = np.vstack([right_rgb, pad_right]) if pad_right is not None else right_rgb
    concat = np.hstack([left_pad, right_pad])

    # 오른쪽 코너점을 호모그래피로 왼쪽 좌표계로 변환
    right_corners_float = np.float32(right_corners.reshape(-1, 1, 2))
    right_transformed = cv2.perspectiveTransform(right_corners_float, H).reshape(-1, 2)

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.imshow(concat)
    ax.set_title('Homography Transformation Visualization - Before/After Mapping', fontsize=16, fontweight='bold')
    ax.axis('off')

    # 왼쪽 원본 코너 (왼쪽 영역)과 변환된 오른쪽 코너(왼쪽 영역에 그리기), 그리고 오른쪽 원본 위치와의 대응선
    for i, (lp, rp, rt) in enumerate(zip(left_corners, right_corners, right_transformed)):
        lx, ly = float(lp[0]), float(lp[1])
        rx, ry = float(rp[0]) + w1, float(rp[1])  # 오른쪽 원본 위치(오른쪽 영역)
        rtx, rty = float(rt[0]), float(rt[1])     # 변환된 좌표(왼쪽 영역)

        # 왼쪽-변환오른쪽 매칭선 (왼쪽 영역 내)
        ax.plot([lx, rtx], [ly, rty], '-', color='lime', linewidth=2, alpha=0.9)
        ax.plot(lx, ly, 'ro', markersize=6)
        ax.plot(rtx, rty, 'go', markersize=6)

        # 변환 전/후 표시를 위해 오른쪽 원본 위치도 점으로 표시
        ax.plot(rx, ry, 'bo', markersize=4, alpha=0.6)

    plt.tight_layout()

    output_dir = './data/visualization'
    os.makedirs(output_dir, exist_ok=True)
    homography_plot_file = os.path.join(output_dir, 'homography_matching_relationship.png')
    plt.savefig(homography_plot_file, dpi=300, bbox_inches='tight')
    print(f"   💾 호모그래피 매칭 관계 저장: {homography_plot_file}")
    plt.show()
    
    # 매칭 정확도 분석
    print(f"\n📊 매칭 정확도 분석:")
    print(f"   왼쪽 코너점 개수: {len(left_corners)}")
    print(f"   오른쪽 코너점 개수: {len(right_corners)}")
    print(f"   변환된 오른쪽 코너점 개수: {len(right_transformed)}")
    
    # 좌표 범위 비교
    left_x_range = np.max(left_corners[:, 0]) - np.min(left_corners[:, 0])
    left_y_range = np.max(left_corners[:, 1]) - np.min(left_corners[:, 1])
    right_x_range = np.max(right_transformed[:, 0]) - np.min(right_transformed[:, 0])
    right_y_range = np.max(right_transformed[:, 1]) - np.min(right_transformed[:, 1])
    
    print(f"   왼쪽 좌표 범위: X={left_x_range:.1f}, Y={left_y_range:.1f}")
    print(f"   변환된 오른쪽 좌표 범위: X={right_x_range:.1f}, Y={right_y_range:.1f}")











def detect_overlap_features(rectified_left, rectified_right, pattern_size=(1, 9)):
    """정렬된 이미지에서 중첩 영역의 체크보드 코너를 검출합니다."""
    print(f"\n🎯 중첩 영역 체크보드 코너 검출")
    print(f"   패턴 크기: {pattern_size[0]}x{pattern_size[1]}")
    print(f"   왼쪽 이미지 크기: {rectified_left.shape}")
    print(f"   오른쪽 이미지 크기: {rectified_right.shape}")
    print(f"   💡 중첩 영역: chessboard in overlap area")
    
    def extract_overlap_corners(img, img_name):
        """이미지에서 중첩 영역의 체크보드 코너 추출"""
        print(f"   🔍 {img_name} 이미지 중첩 영역 코너 추출 시작...")
        
        # 그레이스케일 변환
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # 이미지 크기 확인 (OpenCV는 height, width 순서)
        h, w = gray.shape
        print(f"     이미지 크기: {w} x {h} (width x height)")
        
        # 체크보드 OpenCV 표준 검출 (전달된 패턴 사용)
        print(f"     🔍 Chessboard detection start... ({pattern_size[0]}x{pattern_size[1]})")
        
        # OpenCV 체크보드 검출 사용
        corners = detect_chessboard_corners(gray, img_name, pattern_size)
        if corners is not None:
            return corners
        
        print(f"     ❌ {img_name}: 체크보드 검출 실패")
        return None
    
    # 왼쪽 이미지에서 중첩 영역 코너 추출
    left_corners = extract_overlap_corners(rectified_left, "왼쪽")
    
    # 오른쪽 이미지에서 중첩 영역 코너 추출
    right_corners = extract_overlap_corners(rectified_right, "오른쪽")
    
    # 결과 확인
    if left_corners is None or right_corners is None:
        print(f"   ❌ 중첩 영역 코너 추출 실패")
        print(f"   💡 해결 방법:")
        print(f"     1. 체크보드가 정렬된 이미지에 완전히 보이는지 확인")
        print(f"     2. 체크보드 크기 확인: 현재 {pattern_size[0]} x {pattern_size[1]}")
        print(f"     3. 이미지 품질 개선: 선명하게, 충분한 조명으로")
        print(f"     4. 가운데 화살표가 있는 중첩 영역이 양쪽 이미지에 보이는지 확인")
        return None, None
    
    print(f"   ✅ 양쪽 이미지 모두 중첩 영역 코너 추출 성공!")
    print(f"   왼쪽: {len(left_corners)}개 점")
    print(f"   오른쪽: {len(right_corners)}개 점")
    print(f"   💡 중첩 영역의 공통 특징점으로 호모그래피 계산 가능!")
    
    # 체크보드 코너 검출 결과 시각화
    visualize_chessboard_corners(rectified_left, rectified_right, left_corners, right_corners)
    
    return left_corners, right_corners

def visualize_and_save_features(left_pts, right_pts, left_img=None, right_img=None):
    """검출된 특징점들을 matplotlib으로 시각화하고 저장합니다."""
    print(f"   🎨 특징점 시각화 시작...")
    
    # matplotlib import
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print(f"   ❌ matplotlib이 설치되지 않았습니다. pip install matplotlib")
        return
    
    # 특징점 좌표 출력
    print(f"   📍 왼쪽 이미지 특징점 좌표:")
    for i, pt in enumerate(left_pts):
        print(f"     점 {i}: ({pt[0]:.1f}, {pt[1]:.1f})")
    
    print(f"   📍 오른쪽 이미지 특징점 좌표:")
    for i, pt in enumerate(right_pts):
        print(f"     점 {i}: ({pt[0]:.1f}, {pt[1]:.1f})")
    
    # matplotlib으로 시각화
    if left_img is not None and right_img is not None:
        print(f"   🖼️ matplotlib으로 특징점 시각화 중...")
        
        # 이미지가 BGR이면 RGB로 변환
        if len(left_img.shape) == 3:
            left_img_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        else:
            left_img_rgb = left_img
            
        if len(right_img.shape) == 3:
            right_img_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        else:
            right_img_rgb = right_img
        
        # 서브플롯 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 왼쪽 이미지와 특징점
        ax1.imshow(left_img_rgb)
        ax1.set_title('1. Left Image with Detected Features', fontsize=14, fontweight='bold')
        
        # 왼쪽 특징점 표시
        for i, pt in enumerate(left_pts):
            x, y = pt[0], pt[1]
            ax1.plot(x, y, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
            ax1.text(x+10, y+10, f'{i}', color='white', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
        
        # 오른쪽 이미지와 특징점
        ax2.imshow(right_img_rgb)
        ax2.set_title('2. Right Image with Detected Features', fontsize=14, fontweight='bold')
        
        # 오른쪽 특징점 표시
        for i, pt in enumerate(right_pts):
            x, y = pt[0], pt[1]
            ax2.plot(x, y, 'bo', markersize=8, markeredgecolor='white', markeredgewidth=2)
            ax2.text(x+10, y+10, f'{i}', color='white', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7))
        
        # 축 레이블 제거
        ax1.axis('off')
        ax2.axis('off')
        
        # 특징점 연결선 표시 (순서대로)
        if len(left_pts) > 1:
            left_x_coords = left_pts[:, 0]
            left_y_coords = left_pts[:, 1]
            ax1.plot(left_x_coords, left_y_coords, 'r-', linewidth=2, alpha=0.7)
            
        if len(right_pts) > 1:
            right_x_coords = right_pts[:, 0]
            right_y_coords = right_pts[:, 1]
            ax2.plot(right_x_coords, right_y_coords, 'b-', linewidth=2, alpha=0.7)
        
        plt.tight_layout()
        
        # 이미지 저장
        output_dir = './data/feature_analysis'
        os.makedirs(output_dir, exist_ok=True)
        
        features_plot_file = os.path.join(output_dir, 'detected_features_visualization.png')
        plt.savefig(features_plot_file, dpi=300, bbox_inches='tight')
        print(f"   💾 특징점 시각화 이미지 저장: {features_plot_file}")
        
        # 화면에 표시
        plt.show()
        
        # 개별 이미지별 특징점 시각화
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 왼쪽 이미지 상세 분석
        ax3.imshow(left_img_rgb)
        ax3.set_title('3. Left Image - Feature Analysis', fontsize=14, fontweight='bold')
        
        # 왼쪽 특징점과 거리 정보
        for i in range(len(left_pts)-1):
            pt1 = left_pts[i]
            pt2 = left_pts[i+1]
            dist = np.linalg.norm(pt2 - pt1)
            
            # 연결선과 거리 표시
            ax3.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-', linewidth=2)
            mid_x = (pt1[0] + pt2[0]) / 2
            mid_y = (pt1[1] + pt2[1]) / 2
            ax3.text(mid_x, mid_y, f'{dist:.1f}px', color='white', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.8))
        
        # 오른쪽 이미지 상세 분석
        ax4.imshow(right_img_rgb)
        ax4.set_title('4. Right Image - Feature Analysis', fontsize=14, fontweight='bold')
        
        # 오른쪽 특징점과 거리 정보
        for i in range(len(right_pts)-1):
            pt1 = right_pts[i]
            pt2 = right_pts[i+1]
            dist = np.linalg.norm(pt2 - pt1)
            
            # 연결선과 거리 표시
            ax4.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-', linewidth=2)
            mid_x = (pt1[0] + pt2[0]) / 2
            mid_y = (pt1[1] + pt2[1]) / 2
            ax4.text(mid_x, mid_y, f'{dist:.1f}px', color='white', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='blue', alpha=0.8))
        
        ax3.axis('off')
        ax4.axis('off')
        
        plt.tight_layout()
        
        # 상세 분석 이미지 저장
        detailed_plot_file = os.path.join(output_dir, 'feature_analysis_detailed.png')
        plt.savefig(detailed_plot_file, dpi=300, bbox_inches='tight')
        print(f"   💾 상세 분석 이미지 저장: {detailed_plot_file}")
        
        # 화면에 표시
        plt.show()
    
    # 특징점 간 거리 분석
    print(f"   📏 특징점 간 거리 분석:")
    
    # 왼쪽 이미지 특징점 간 거리
    left_distances = []
    for i in range(len(left_pts)-1):
        dist = np.linalg.norm(left_pts[i+1] - left_pts[i])
        left_distances.append(dist)
        print(f"     왼쪽 점 {i}→{i+1}: {dist:.1f} 픽셀")
    
    # 오른쪽 이미지 특징점 간 거리
    right_distances = []
    for i in range(len(right_pts)-1):
        dist = np.linalg.norm(right_pts[i+1] - right_pts[i])
        right_distances.append(dist)
        print(f"     오른쪽 점 {i}→{i+1}: {dist:.1f} 픽셀")
    
    # 거리 패턴 분석
    if len(left_distances) > 0 and len(right_distances) > 0:
        left_avg_dist = np.mean(left_distances)
        right_avg_dist = np.mean(right_distances)
        print(f"   📊 평균 거리:")
        print(f"     왼쪽: {left_avg_dist:.1f} 픽셀")
        print(f"     오른쪽: {right_avg_dist:.1f} 픽셀")
        print(f"     거리 비율: {left_avg_dist/right_avg_dist:.3f}")
    
    # 특징점 분포 분석
    left_x_range = np.max(left_pts[:, 0]) - np.min(left_pts[:, 0])
    left_y_range = np.max(left_pts[:, 1]) - np.min(left_pts[:, 1])
    right_x_range = np.max(right_pts[:, 0]) - np.min(right_pts[:, 0])
    right_y_range = np.max(right_pts[:, 1]) - np.min(right_pts[:, 1])
    
    print(f"   📐 특징점 분포 범위:")
    print(f"     왼쪽: X={left_x_range:.1f}, Y={left_y_range:.1f}")
    print(f"     오른쪽: X={right_x_range:.1f}, Y={right_y_range:.1f}")
    
    # 특징점 정렬 상태 확인
    print(f"   🔍 특징점 정렬 상태:")
    
    # X 좌표 정렬 확인
    left_x_sorted = np.all(np.diff(left_pts[:, 0]) >= 0)
    right_x_sorted = np.all(np.diff(right_pts[:, 0]) >= 0)
    
    print(f"     왼쪽 X 좌표 정렬: {'✅ 정렬됨' if left_x_sorted else '❌ 정렬 안됨'}")
    print(f"     오른쪽 X 좌표 정렬: {'✅ 정렬됨' if right_x_sorted else '❌ 정렬 안됨'}")
    
    # 특징점 저장
    features_data = {
        'left_points': left_pts.tolist(),
        'right_points': right_pts.tolist(),
        'left_distances': [float(d) for d in left_distances],
        'right_distances': [float(d) for d in right_distances],
        'analysis': {
            'left_x_range': float(left_x_range),
            'left_y_range': float(left_y_range),
            'right_x_range': float(right_x_range),
            'right_y_range': float(right_y_range),
            'left_x_sorted': bool(left_x_sorted),
            'right_x_sorted': bool(right_x_sorted)
        }
    }
    
    # 디렉토리 생성
    os.makedirs('./data/feature_analysis', exist_ok=True)
    
    # JSON 파일로 저장
    features_file = './data/feature_analysis/detected_features.json'
    with open(features_file, 'w') as f:
        json.dump(features_data, f, indent=2)
    
    print(f"   💾 특징점 분석 결과 저장: {features_file}")
    print(f"   ✅ 특징점 시각화 완료!")

def calculate_homography_from_overlap_corners(left_corners, right_corners, left_img=None, right_img=None):
    """중첩 영역 한 열 대응점: 유사변환(회전+이동+등방성 스케일) RANSAC.

    한 열(동직선)만으로는 projective H가 기하적으로 불안정하므로, 
    estimateAffinePartial2D로 강체/유사변환을 추정해 3x3 행렬로 승격합니다.
    """
    print(f"\n🔗 Similarity (R,t,s) from overlap chessboard corners (RANSAC)")
    
    # 코너 좌표를 float32로 변환
    left_pts = np.float32(left_corners.reshape(-1, 2))
    right_pts = np.float32(right_corners.reshape(-1, 2))
    
    print(f"   왼쪽 특징점: {left_pts.shape}")
    print(f"   오른쪽 특징점: {right_pts.shape}")
    
    # 특징점 개수 확인 (최소 4개 필요 - 1×10 체크보드)
    if len(left_pts) < 4 or len(right_pts) < 4:
        print(f"   ❌ 특징점이 부족합니다 (최소 4개 필요)")
        print(f"   왼쪽: {len(left_pts)}개, 오른쪽: {len(right_pts)}개")
        return None
    
    print(f"   ✅ 충분한 특징점으로 안정적인 정합 가능!")
    
    # 특징점 매칭 과정 시각화
    print(f"   🔍 특징점 매칭 과정 시각화 중...")
    visualize_feature_matching(left_img, right_img, left_pts, right_pts)

    # 유사변환(회전+이동+등방성 스케일) 추정
    try:
        # RANSAC 파라미터 조정으로 더 안정적인 추정
        A, inliers = cv2.estimateAffinePartial2D(
            right_pts, left_pts, 
            method=cv2.RANSAC, 
            ransacReprojThreshold=2.0,  # 더 엄격한 임계값
            maxIters=2000,  # 더 많은 반복
            confidence=0.99  # 높은 신뢰도
        )
        
        if A is not None and inliers is not None:
            # 인라이어 비율 확인
            inlier_ratio = np.sum(inliers) / len(inliers)
            print(f"   인라이어 비율: {inlier_ratio:.2%}")
            
            if inlier_ratio < 0.7:  # 70% 미만이면 경고
                print(f"   ⚠️ 인라이어 비율이 낮습니다. 결과가 불안정할 수 있습니다.")
            
            H = np.vstack([A, [0, 0, 1]]).astype(np.float64)
            s = np.sqrt(max(1e-12, np.linalg.det(A[:2,:2])))
            angle = np.degrees(np.arctan2(A[1,0], A[0,0]))
            print(f"   ✅ Similarity estimated: scale≈{s:.4f}, angle={angle:.2f}°")
            visualize_homography_matching(left_img, right_img, left_pts, right_pts, H)
            return H, inliers
        else:
            print("   ❌ Similarity estimation failed; fallback to translation")
            raise RuntimeError('affine_partial_failed')
            
    except Exception as e:
        print(f"   ⚠️ Similarity estimation failed: {e}")
        print(f"   🔄 Fallback to simple translation...")
        
        # 단순 이동으로 fallback
        deltas = left_pts - right_pts
        mean_delta = np.mean(deltas, axis=0)
        tx, ty = float(mean_delta[0]), float(mean_delta[1])
        H = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float64)
        print(f"   ↔️ Fallback translation: (tx,ty)=({tx:.2f},{ty:.2f})")
        visualize_homography_matching(left_img, right_img, left_pts, right_pts, H)
        return H, None

def calculate_optimal_canvas_size(left_img, right_img, H):
    """최소 중첩을 위한 최적 캔버스 크기를 계산합니다."""
    print(f"\n📐 최적 캔버스 크기 계산 (최소 중첩)")
    
    h, w = left_img.shape[:2]
    
    # 오른쪽 이미지를 호모그래피로 변환
    right_transformed = cv2.warpPerspective(right_img, H, (w*2, h))
    
    # 변환된 오른쪽 이미지의 유효 영역 찾기
    right_mask = (right_transformed.sum(axis=2) > 0).astype(np.uint8)
    
    # 왼쪽 이미지 마스크
    left_mask = np.ones((h, w), dtype=np.uint8)
    left_mask_padded = np.zeros((h, w*2), dtype=np.uint8)
    left_mask_padded[:, :w] = left_mask
    
    # 중첩 영역 계산
    overlap = np.logical_and(left_mask_padded, right_mask)
    overlap_area = np.sum(overlap)
    
    print(f"   중첩 영역: {overlap_area} 픽셀")
    print(f"   중첩 비율: {overlap_area/(w*h)*100:.1f}%")
    
    # 최적 캔버스 크기 (중첩 최소화)
    optimal_width = int(w * 1.8)  # 중첩을 줄이기 위해 가로 길이 조정
    optimal_height = h
    
    print(f"   최적 캔버스 크기: {optimal_width} x {optimal_height}")
    
    return (optimal_width, optimal_height), overlap_area

def compute_canvas_with_translation(left_img, right_img, H):
    """H로 오른쪽을 왼쪽 좌표계로 워핑할 때, 전체를 양의 좌표로 옮길 캔버스와 T를 계산.

    반환:
      H_canvas: T @ H (우측 이미지를 캔버스로 워핑하는 최종 행렬)
      canvas_size: (W, H)
      left_offset: (tx, ty) 캔버스에서 왼쪽 이미지를 배치할 좌표
      overlap_area: 대략적인 중첩 픽셀 수(참고용)
    """
    h1, w1 = left_img.shape[:2]
    h2, w2 = right_img.shape[:2]

    # 오른쪽 이미지 네 모서리를 H로 변환
    corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    # 좌표계 통합: 왼쪽 이미지(0..w1, 0..h1)와 변환된 우측 코너의 범위
    all_x = np.concatenate([np.array([0, w1]), warped[:, 0]])
    all_y = np.concatenate([np.array([0, h1]), warped[:, 1]])

    min_x, min_y = np.min(all_x), np.min(all_y)
    max_x, max_y = np.max(all_x), np.max(all_y)

    # 모두 양수로 이동시키는 translation
    tx = -min(0.0, float(min_x))
    ty = -min(0.0, float(min_y))

    T = np.array([[1.0, 0.0, tx],
                  [0.0, 1.0, ty],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    H_canvas = T @ H

    # 캔버스 크기 계산 - 두 이미지가 모두 들어갈 수 있도록
    canvas_w = int(np.ceil(max_x + tx))
    canvas_h = int(np.ceil(max_y + ty))
    
    # 왼쪽 이미지가 들어갈 수 있는 최소 크기 보장
    canvas_w = max(canvas_w, w1 + int(np.ceil(tx)))
    canvas_h = max(canvas_h, h1 + int(np.ceil(ty)))
    
    canvas_size = (canvas_w, canvas_h)

    # 중첩 평가용 대략치
    right_transformed = cv2.warpPerspective(right_img, H_canvas, canvas_size)
    left_mask = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.uint8)
    lx, ly = int(np.floor(tx)), int(np.floor(ty))
    left_mask[ly:ly + h1, lx:lx + w1] = 1
    right_mask = (right_transformed.sum(axis=2) > 0).astype(np.uint8)
    overlap_area = int(np.sum(np.logical_and(left_mask, right_mask)))

    left_offset = (lx, ly)
    
    print(f"   캔버스 크기: {canvas_size}")
    print(f"   왼쪽 오프셋: {left_offset}")
    print(f"   중첩 영역: {overlap_area} 픽셀")
    
    return H_canvas, canvas_size, left_offset, overlap_area

def create_blending_masks(left_img, right_img, H, canvas_size, left_offset=(0,0), seam_width=32):
    """중첩 영역에서 seamless한 블렌딩을 위한 마스크를 생성합니다."""
    print(f"\n🎨 Seamless 블렌딩 마스크 생성")
    
    h, w = left_img.shape[:2]
    
    # 왼쪽 이미지를 캔버스에 배치
    left_canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    ox, oy = int(left_offset[0]), int(left_offset[1])
    left_canvas[oy:oy+h, ox:ox+w] = left_img
    
    # 오른쪽 이미지를 변환하여 캔버스에 배치
    right_transformed = cv2.warpPerspective(right_img, H, canvas_size)
    
    # 중첩 영역 찾기
    left_mask = (left_canvas.sum(axis=2) > 0).astype(np.uint8)
    right_mask = (right_transformed.sum(axis=2) > 0).astype(np.uint8)
    overlap = np.logical_and(left_mask, right_mask)
    
    # 블렌딩 마스크 생성 (중첩 영역에서 그라데이션)
    left_blend_mask = np.zeros_like(left_mask, dtype=np.float32)
    right_blend_mask = np.zeros_like(right_mask, dtype=np.float32)
    
    # 비중첩 영역은 각각 1로 설정
    left_blend_mask[left_mask > 0] = 1.0
    right_blend_mask[right_mask > 0] = 1.0
    
    # 중첩 영역에서 거리 기반 가중치 계산
    overlap_y, overlap_x = np.where(overlap)
    if len(overlap_y) > 0:
        # 중첩 영역의 중심선 찾기 (X 좌표 기준)
        center_x = np.mean(overlap_x)
        
        for y, x in zip(overlap_y, overlap_x):
            # 중심선으로부터의 거리 계산
            dist_to_center = abs(x - center_x)
            max_dist = seam_width / 2
            
            # 거리에 따른 가중치 계산 (중심선에서 멀어질수록 가중치 감소)
            if dist_to_center < max_dist:
                weight = max(0, 1 - dist_to_center / max_dist)
                left_blend_mask[y, x] = weight
                right_blend_mask[y, x] = 1 - weight
            else:
                # 중첩 영역이지만 seam_width 밖이면 각각 0.5씩
                left_blend_mask[y, x] = 0.5
                right_blend_mask[y, x] = 0.5
    
    print(f"   ✅ Seamless 블렌딩 마스크 생성 완료")
    print(f"   마스크 크기: {left_blend_mask.shape}")
    print(f"   중첩 영역 크기: {len(overlap_y)} 픽셀")
    
    return left_blend_mask, right_blend_mask

def perform_stitching(left_img, right_img, H, canvas_size, left_offset, left_blend_mask, right_blend_mask):
    """이미지를 스티칭합니다."""
    print(f"\n🔗 이미지 스티칭 실행")
    
    h, w = left_img.shape[:2]
    
    # 왼쪽 이미지를 캔버스에 배치
    left_canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    ox, oy = int(left_offset[0]), int(left_offset[1])
    left_canvas[oy:oy+h, ox:ox+w] = left_img
    
    # 오른쪽 이미지를 변환하여 캔버스에 배치
    right_transformed = cv2.warpPerspective(right_img, H, canvas_size)
    
    # 블렌딩 마스크를 3채널로 확장
    left_mask_3ch = left_blend_mask[:, :, np.newaxis]
    right_mask_3ch = right_blend_mask[:, :, np.newaxis]
    
    # 가중치 적용하여 블렌딩
    left_weighted = (left_canvas.astype(np.float32) * left_mask_3ch).astype(np.uint8)
    right_weighted = (right_transformed.astype(np.float32) * right_mask_3ch).astype(np.uint8)
    
    # 최종 스티칭 결과 (가중 평균)
    final_result = cv2.add(left_weighted, right_weighted)
    
    # 중첩 영역에서 블렌딩 품질 확인
    overlap_mask = np.logical_and(left_mask_3ch[:,:,0] > 0, right_mask_3ch[:,:,0] > 0)
    overlap_count = np.sum(overlap_mask)
    
    print(f"   ✅ 스티칭 완료")
    print(f"   최종 이미지 크기: {final_result.shape}")
    print(f"   중첩 영역 크기: {overlap_count} 픽셀")
    
    return final_result, left_canvas, right_transformed

def save_stitching_parameters(H, canvas_size, left_offset, left_blend_mask, right_blend_mask, 
                             config_dir='./data/config'):
    """실시간 비디오 스티칭을 위한 파라미터를 저장합니다."""
    print(f"\n💾 스티칭 파라미터 저장")
    
    # 디렉토리 생성
    os.makedirs(config_dir, exist_ok=True)
    
    # 스티칭 파라미터 저장
    stitching_params = {
        'homography': H.tolist(),
        'canvas_size': canvas_size,
        'left_offset': [int(left_offset[0]), int(left_offset[1])],
        'left_blend_mask': left_blend_mask.tolist(),
        'right_blend_mask': right_blend_mask.tolist(),
        'description': 'Left ↔ Right stitching parameters for real-time video',
        'stitching_method': 'Edge feature-based homography with minimal overlap',
        'usage': 'Load these parameters for real-time Left ↔ Right video stitching'
    }
    
    params_file = os.path.join(config_dir, "LR_stitching_parameters.json")
    with open(params_file, 'w') as f:
        json.dump(stitching_params, f, indent=2)
    
    print(f"   ✅ 스티칭 파라미터 저장 완료")
    print(f"   파일 경로: {params_file}")
    
    return params_file

def save_stitching_results(left_canvas, right_transformed, final_result, 
                          left_blend_mask, right_blend_mask, output_dir='./data/stitching_results'):
    """스티칭 결과를 저장합니다."""
    print(f"\n💾 스티칭 결과 저장")
    
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 개별 이미지 저장
    cv2.imwrite(os.path.join(output_dir, "left_canvas.png"), left_canvas)
    cv2.imwrite(os.path.join(output_dir, "right_transformed.png"), right_transformed)
    cv2.imwrite(os.path.join(output_dir, "final_stitched.png"), final_result)
    
    # 블렌딩 마스크 저장
    cv2.imwrite(os.path.join(output_dir, "left_blend_mask.png"), (left_blend_mask * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, "right_blend_mask.png"), (right_blend_mask * 255).astype(np.uint8))
    
    print(f"   ✅ 결과 이미지 저장 완료")
    print(f"   저장 경로: {output_dir}")

def create_stitching_pipeline(config_LC_file, config_CR_file, left_img, right_img, pattern_size=(1, 10)):
    """전체 스티칭 파이프라인을 실행합니다."""
    print(f"\n🎯 Left ↔ Right 스티칭 파이프라인 시작")
    print(f"{'='*60}")
    print(f"🎯 체크보드 패턴: {pattern_size[0]} x {pattern_size[1]}")
    print(f"💡 내부 코너: {pattern_size[0]} x {pattern_size[1]}")
    print(f"💡 method: homography from overlap chessboard + minimal overlap stitching")
    print(f"{'='*60}")
    
    # 1. 캘리브레이션 설정 로드
    print(f"\n📁 1단계: 캘리브레이션 설정 로드")
    config_LC = load_calibration_config(config_LC_file)
    config_CR = load_calibration_config(config_CR_file)
    
    # 2. 이미지 정렬 (캘리브레이션 맵 적용)
    print(f"\n🔄 2단계: 이미지 정렬")
    left_rectified, right_rectified = apply_rectification_maps(left_img, right_img, config_LC, config_CR)
    
    # 1) 시각화: 검출된 특징점
    print(f"\n🎯 1/3: 특징점 검출")
    print(f"   패턴 크기: {pattern_size[0]} x {pattern_size[1]}")
    left_corners, right_corners = detect_overlap_features(left_rectified, right_rectified, pattern_size)
    
    if left_corners is None or right_corners is None:
        print(f"❌ 중첩 영역 체크보드 코너 검출 실패")
        print(f"💡 특징점 검출 실패 시 해결 방법:")
        print(f"   1. 체크보드가 정렬된 이미지에 완전히 보이는지 확인")
        print(f"   2. 체크보드 크기 확인: 현재 {pattern_size[0]} x {pattern_size[1]} (실제 체커보드)")
        print(f"   3. 내부 코너: 1 x 9 = 9개 (OpenCV 4×11 패턴에서 중간 줄 추출)")
        print(f"   4. 이미지 품질 개선: 선명하게, 충분한 조명으로")
        print(f"   5. 가운데 화살표가 있는 중첩 영역이 양쪽 이미지에 보이는지 확인")
        return None
    
    # 2) 시각화: 호모그래피 매칭 과정
    print(f"\n🔗 2/3: 호모그래피 계산 및 매칭 시각화")
    result = calculate_homography_from_overlap_corners(left_corners, right_corners, left_rectified, right_rectified)
    if result is None:
        print(f"❌ 호모그래피 계산 실패")
        return None
    
    H, mask = result
    
    # 3) 시각화: 매칭결과로 스티칭된 이미지
    print(f"\n📐 3/3: 스티칭 결과 생성")
    # 최적 캔버스/오프셋 계산 (좌표계 통합)
    H_canvas, canvas_size, left_offset, overlap_area = compute_canvas_with_translation(left_rectified, right_rectified, H)
    
    # 6. 블렌딩 마스크 생성
    print(f"\n🎨 6단계: 블렌딩 마스크 생성")
    left_blend_mask, right_blend_mask = create_blending_masks(
        left_rectified, right_rectified, H_canvas, canvas_size, left_offset
    )
    
    # 7. 이미지 스티칭
    print(f"\n🔗 7단계: 이미지 스티칭 실행")
    # 캔버스 좌표에 왼쪽 이미지를 오프셋 배치하고, 오른쪽은 H_canvas로 워핑하여 겹쳐 스티칭
    final_result, left_canvas, right_transformed = perform_stitching(
        left_rectified, right_rectified, H_canvas, canvas_size, left_offset,
        left_blend_mask, right_blend_mask
    )
    
    # 8. 파라미터 저장
    print(f"\n💾 8단계: 스티칭 파라미터 저장")
    params_file = save_stitching_parameters(H_canvas, canvas_size, left_offset, left_blend_mask, right_blend_mask)
    
    # 8-1. 통합된 homography_params.json 저장 (참조 코드 구조용)
    print(f"\n💾 8-1단계: 통합된 homography_params.json 저장")
    unified_file = save_unified_homography_params(H_canvas, canvas_size, left_offset, config_LC_file, config_CR_file)
    
    # 8-2. NPY 파일들 생성 (Realtime_Video_3_CPU - 복사본.py용)
    print(f"\n💾 8-2단계: NPY 파일들 생성")
    params_dir = './data/params'
    os.makedirs(params_dir, exist_ok=True)
    
    # LC의 Left 카메라 맵을 왼쪽 카메라용으로 저장
    np.save(os.path.join(params_dir, 'stereo_map_left_x.npy'), config_LC['left_map1_x'])
    
    # CR의 Right 카메라 맵을 오른쪽 카메라용으로 저장
    np.save(os.path.join(params_dir, 'stereo_map_right_x.npy'), config_CR['right_map1_x'])
    
    # 블렌딩 마스크를 NPY로 저장
    np.save(os.path.join(params_dir, 'left_blend_mask.npy'), left_blend_mask)
    np.save(os.path.join(params_dir, 'right_blend_mask.npy'), right_blend_mask)
    
    print(f"   ✅ NPY 파일들 생성 완료:")
    print(f"      - stereo_map_left_x.npy: {config_LC['left_map1_x'].shape}")
    print(f"      - stereo_map_right_x.npy: {config_CR['right_map1_x'].shape}")
    print(f"      - left_blend_mask.npy: {left_blend_mask.shape}")
    print(f"      - right_blend_mask.npy: {right_blend_mask.shape}")
    
    # 9. 결과 저장
    print(f"\n💾 9단계: 결과 이미지 저장")
    save_stitching_results(left_canvas, right_transformed, final_result, 
                          left_blend_mask, right_blend_mask)
    
    # 10. 중첩 영역 시각화
    print(f"\n🔍 10단계: 중첩 영역 시각화")
    visualize_overlap_analysis(left_rectified, right_rectified, H_canvas, canvas_size, left_offset, left_blend_mask, right_blend_mask)
    visualize_stitching_process(left_rectified, right_rectified, H_canvas, canvas_size, left_offset, left_blend_mask, right_blend_mask, final_result)
    visualize_canvas_layout(left_rectified, right_rectified, H_canvas, canvas_size, left_offset)
    
    print(f"\n🎉 Left ↔ Right 스티칭 파이프라인 완료!")
    print(f"📁 파라미터 파일: {params_file}")
    print(f"📊 중첩 영역: {overlap_area} 픽셀")
    print(f"🎯 체크보드 패턴: {pattern_size[0]} x {pattern_size[1]}")
    print(f"💡 내부 코너: {CHESSBOARD_SIZE[0]} x {CHESSBOARD_SIZE[1]} (corners)")
    
    return {
        'params_file': params_file,
        'homography': H,
        'canvas_size': canvas_size,
        'left_offset': left_offset,
        'overlap_area': overlap_area,
        'final_result': final_result
    }

def main():
    """메인 함수: Left ↔ Right 스티칭 파이프라인 실행"""
    print(f"\n🎯 Left ↔ Right stitching engine (overlap chessboard corners)")
    print(f"{'='*60}")
    print(f"💡 method: homography from overlap chessboard + minimal overlap stitching")
    print(f"💡 목적: 실시간 비디오 스티칭을 위한 파라미터 생성")
    print(f"🎯 체크보드 패턴: {CHESSBOARD_SIZE[0]} x {CHESSBOARD_SIZE[1]} (실제 체커보드)")
    print(f"💡 특징점: 중첩 영역의 공통 체크보드 코너로 seamless 연결")
    print(f"💡 내부 코너: 1 x 9 = 9개 (OpenCV 4×11 패턴에서 중간 줄 추출)")
    print(f"{'='*60}")
    
    # 설정 파일 경로
    config_LC_file = "./data/config/LC_calibration_config.json"
    config_CR_file = "./data/config/CR_calibration_config.json"
    
    # 설정 파일 존재 확인
    if not os.path.exists(config_LC_file):
        print(f"❌ LC 캘리브레이션 설정 파일을 찾을 수 없습니다: {config_LC_file}")
        print(f"💡 먼저 Calibration.py를 실행하여 캘리브레이션을 완료해주세요.")
        return
    
    if not os.path.exists(config_CR_file):
        print(f"❌ CR 캘리브레이션 설정 파일을 찾을 수 없습니다: {config_CR_file}")
        print(f"💡 먼저 Calibration.py를 실행하여 캘리브레이션을 완료해주세요.")
        return
    
    # Left-Right 쌍 이미지 로드
    left_img = cv2.imread("./data/images/pair_LR/left/img00.png")
    right_img = cv2.imread("./data/images/pair_LR/right/img00.png")
    
    if left_img is None or right_img is None:
        print(f"❌ Left-Right 쌍 이미지를 찾을 수 없습니다.")
        print(f"💡 먼저 Camera_1.py 모드 2로 Left-Right 쌍 이미지를 촬영해주세요.")
        print(f"   • 모드 2 선택 후 '3'번 키로 Left-Right 쌍 촬영")
        print(f"   • 이미지 경로: ./data/images/pair_LR/left/, ./data/images/pair_LR/right/")
        print(f"   • 체크보드가 양쪽 이미지에 걸쳐 보이도록 촬영")
        print(f"   • 체크보드 크기: {CHESSBOARD_SIZE[0]} x {CHESSBOARD_SIZE[1]} (실제 체커보드)")
        print(f"   • 내부 코너: 1 x 9 = 9개 (OpenCV 4×11 패턴에서 중간 줄 추출)")
        return
    
    print(f"✅ Left-Right 쌍 이미지 로드 완료:")
    print(f"   왼쪽: {left_img.shape}")
    print(f"   오른쪽: {right_img.shape}")
    
    try:
        # 스티칭 파이프라인 실행
        result = create_stitching_pipeline(
            config_LC_file, config_CR_file, 
            left_img, right_img, pattern_size=CHESSBOARD_SIZE
        )
        
        if result:
            print(f"\n{'='*60}")
            print(f"🎉 스티칭 엔진 실행 완료!")
            print(f"{'='*60}")
            print(f"📁 생성된 파일:")
            print(f"   • 스티칭 파라미터: {result['params_file']}")
            print(f"   • 통합 파라미터: ./data/config/homography_params.json (참조 코드 구조용)")
            print(f"   • 결과 이미지: ./data/stitching_results/")
            
            # ROI 선택 기능 추가
            print(f"\n🎯 사용자 지정 ROI 선택 모드")
            print(f"💡 마우스로 드래그하여 실시간 스티칭에서 표시할 영역을 선택할 수 있습니다")
            print(f"💡 이 기능을 사용하면 검은 영역을 제거하고 FPS를 향상시킬 수 있습니다")
            
            user_input = input(f"\n🎯 ROI 선택을 진행하시겠습니까? (y/n): ").strip().lower()
            
            if user_input in ['y', 'yes', 'ㅇ']:
                print(f"\n🎯 ROI 선택 모드 시작...")
                print(f"💡 최종 스티칭 결과 이미지가 표시됩니다")
                print(f"💡 마우스로 드래그하여 원하는 영역을 선택하세요")
                
                # ROI 선택 실행
                roi_result = interactive_roi_selection(result['final_result'])
                
                if roi_result:
                    roi_info, roi_mask = roi_result
                    
                    print(f"\n🎯 ROI 선택 완료! 이제 ROI 기반 블렌딩 파라미터를 생성합니다...")
                    
                    # ROI 기반 블렌딩 파라미터 생성
                    roi_blending_params = create_roi_based_blending_parameters(
                        result['final_result'], roi_info, result['homography'], 
                        result['canvas_size'], result['left_offset']
                    )
                    
                    print(f"\n{'='*60}")
                    print(f"🎉 ROI 최적화 완료!")
                    print(f"{'='*60}")
                    print(f"📁 생성된 파일:")
                    print(f"   • ROI 정보: ./data/params/user_roi_info.json")
                    print(f"   • ROI 마스크: ./data/params/user_roi_mask.npy")
                    print(f"   • ROI 블렌딩 파라미터: ./data/params/roi_blending_params.json")
                    print(f"   • ROI 미리보기: ./data/params/user_roi_preview.png")
                    print(f"\n🚀 다음 단계:")
                    print(f"   • 기존 Realtime_Video_4.py에 ROI가 자동으로 적용됩니다")
                    print(f"   • 지정된 ROI 영역만 표시되어 FPS 향상")
                    print(f"   • 검은 영역 제거로 더 깔끔한 화면")
                    print(f"{'='*60}")
                else:
                    print(f"⚠️ ROI 선택이 취소되었습니다")
                    print(f"💡 나중에 ./data/stitching_results/final_stitched.png 이미지를 사용하여 수동으로 ROI를 지정할 수 있습니다")
            else:
                print(f"ℹ️ ROI 선택을 건너뜁니다")
                print(f"💡 나중에 ./data/stitching_results/final_stitched.png 이미지를 사용하여 수동으로 ROI를 지정할 수 있습니다")
            
            print(f"\n{'='*60}")
            print(f"🎉 스티칭 엔진 실행 완료!")
            print(f"{'='*60}")
            print(f"📁 생성된 파일:")
            print(f"   • 스티칭 파라미터: {result['params_file']}")
            print(f"   • 통합 파라미터: ./data/config/homography_params.json (참조 코드 구조용)")
            print(f"   • 결과 이미지: ./data/stitching_results/")
            print(f"\n🚀 다음 단계:")
            print(f"   • 저장된 파라미터를 사용하여 실시간 비디오 스티칭")
            print(f"   • 중앙 카메라 없이 Left ↔ Right 직접 연결")
            print(f"   • 참조 코드 구조로 성능 최적화 가능")
            print(f"{'='*60}")
        else:
            print(f"\n❌ 스티칭 파이프라인 실행 실패")
            print(f"💡 특징점 검출 실패 시 해결 방법:")
            print(f"   1. 체크보드가 정렬된 이미지에 완전히 보이는지 확인")
            print(f"   2. 체크보드 크기 확인: 현재 {CHESSBOARD_SIZE[0]} x {CHESSBOARD_SIZE[1]}")
            print(f"   3. 내부 코너: {CHESSBOARD_SIZE[0]} x {CHESSBOARD_SIZE[1]}")
            print(f"   4. 이미지 품질 개선: 선명하게, 충분한 조명으로")
            print(f"   5. 체크보드가 양쪽 이미지에 걸쳐 보이도록 촬영")
        
    except Exception as e:
        print(f"\n❌ 스티칭 파이프라인 실행 실패: {str(e)}")
        raise e

def visualize_overlap_analysis(left_img, right_img, H, canvas_size, left_offset, left_blend_mask, right_blend_mask):
    """중첩 영역 분석을 시각화합니다."""
    print(f"\n🔍 중첩 영역 분석 시각화")
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"   ❌ matplotlib이 설치되지 않았습니다. pip install matplotlib")
        return
    
    # 왼쪽 이미지를 캔버스에 배치
    h, w = left_img.shape[:2]
    left_canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    ox, oy = int(left_offset[0]), int(left_offset[1])
    left_canvas[oy:oy+h, ox:ox+w] = left_img
    
    # 오른쪽 이미지를 변환하여 캔버스에 배치
    right_transformed = cv2.warpPerspective(right_img, H, canvas_size)
    
    # 중첩 영역 찾기
    left_mask = (left_canvas.sum(axis=2) > 0).astype(np.uint8)
    right_mask = (right_transformed.sum(axis=2) > 0).astype(np.uint8)
    overlap = np.logical_and(left_mask, right_mask)
    
    # 2x3 서브플롯 생성
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    # 1. 왼쪽 이미지 (캔버스에 배치된 상태)
    left_rgb = cv2.cvtColor(left_canvas, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(left_rgb)
    axes[0, 0].set_title('1. Left Image on Canvas', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. 오른쪽 변환된 이미지
    right_rgb = cv2.cvtColor(right_transformed, cv2.COLOR_BGR2RGB)
    axes[0, 1].imshow(right_rgb)
    axes[0, 1].set_title('2. Right Transformed Image', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. 중첩 영역 마스크
    overlap_vis = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    overlap_vis[overlap] = [255, 255, 0]  # 노란색으로 중첩 영역 표시
    axes[0, 2].imshow(overlap_vis)
    axes[0, 2].set_title(f'3. Overlap Region ({np.sum(overlap)} pixels)', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # 4. 왼쪽 블렌딩 마스크 (히트맵)
    left_blend_vis = axes[1, 0].imshow(left_blend_mask, cmap='Reds', vmin=0, vmax=1)
    axes[1, 0].set_title('4. Left Blending Mask', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(left_blend_vis, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # 5. 오른쪽 블렌딩 마스크 (히트맵)
    right_blend_vis = axes[1, 1].imshow(right_blend_mask, cmap='Blues', vmin=0, vmax=1)
    axes[1, 1].set_title('5. Right Blending Mask', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(right_blend_vis, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # 6. 블렌딩 마스크 합계 (중첩 영역에서 1이 되어야 함)
    total_mask = left_blend_mask + right_blend_mask
    total_vis = axes[1, 2].imshow(total_mask, cmap='viridis', vmin=0, vmax=2)
    axes[1, 2].set_title('6. Total Blending Mask (should be 1 in overlap)', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(total_vis, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # 저장
    output_dir = './data/visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    overlap_plot_file = os.path.join(output_dir, 'overlap_analysis.png')
    plt.savefig(overlap_plot_file, dpi=300, bbox_inches='tight')
    print(f"   💾 중첩 영역 분석 시각화 저장: {overlap_plot_file}")
    
    plt.show()
    
    # 중첩 영역 통계 출력
    print(f"\n📊 중첩 영역 통계:")
    print(f"   캔버스 크기: {canvas_size}")
    print(f"   왼쪽 이미지 오프셋: {left_offset}")
    print(f"   중첩 영역 크기: {np.sum(overlap)} 픽셀")
    print(f"   중첩 영역 비율: {np.sum(overlap)/(canvas_size[0]*canvas_size[1])*100:.2f}%")
    
    # 블렌딩 마스크 품질 확인
    overlap_indices = np.where(overlap)
    if len(overlap_indices[0]) > 0:
        total_weights = left_blend_mask[overlap] + right_blend_mask[overlap]
        weight_error = np.abs(total_weights - 1.0)
        max_error = np.max(weight_error)
        mean_error = np.mean(weight_error)
        
        print(f"   블렌딩 마스크 품질:")
        print(f"     최대 가중치 오차: {max_error:.4f}")
        print(f"     평균 가중치 오차: {mean_error:.4f}")
        print(f"     완벽한 블렌딩: {'✅' if max_error < 0.01 else '❌'}")


def visualize_stitching_process(left_img, right_img, H, canvas_size, left_offset, left_blend_mask, right_blend_mask, final_result):
    """스티칭 과정을 단계별로 시각화합니다."""
    print(f"\n🎬 스티칭 과정 단계별 시각화")
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"   ❌ matplotlib이 설치되지 않았습니다. pip install matplotlib")
        return
    
    # 왼쪽 이미지를 캔버스에 배치
    h, w = left_img.shape[:2]
    left_canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    ox, oy = int(left_offset[0]), int(left_offset[1])
    left_canvas[oy:oy+h, ox:ox+w] = left_img
    
    # 오른쪽 이미지를 변환하여 캔버스에 배치
    right_transformed = cv2.warpPerspective(right_img, H, canvas_size)
    
    # 2x3 서브플롯 생성
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    # 1단계: 원본 왼쪽 이미지
    left_orig_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(left_orig_rgb)
    axes[0, 0].set_title('1. Original Left Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2단계: 원본 오른쪽 이미지
    right_orig_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
    axes[0, 1].imshow(right_orig_rgb)
    axes[0, 1].set_title('2. Original Right Image', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3단계: 캔버스에 배치된 왼쪽 이미지
    left_canvas_rgb = cv2.cvtColor(left_canvas, cv2.COLOR_BGR2RGB)
    axes[0, 2].imshow(left_canvas_rgb)
    axes[0, 2].set_title('3. Left Image on Canvas', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # 4단계: 변환된 오른쪽 이미지
    right_trans_rgb = cv2.cvtColor(right_transformed, cv2.COLOR_BGR2RGB)
    axes[1, 0].imshow(right_trans_rgb)
    axes[1, 0].set_title('4. Transformed Right Image', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 5단계: 블렌딩 마스크 적용된 왼쪽 이미지
    left_weighted = (left_canvas.astype(np.float32) * left_blend_mask[:, :, np.newaxis]).astype(np.uint8)
    left_weighted_rgb = cv2.cvtColor(left_weighted, cv2.COLOR_BGR2RGB)
    axes[1, 1].imshow(left_weighted_rgb)
    axes[1, 1].set_title('5. Left Image with Blending Mask', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # 6단계: 최종 스티칭 결과
    final_rgb = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)
    axes[1, 2].imshow(final_rgb)
    axes[1, 2].set_title('6. Final Stitched Result', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # 저장
    output_dir = './data/visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    process_plot_file = os.path.join(output_dir, 'stitching_process_steps.png')
    plt.savefig(process_plot_file, dpi=300, bbox_inches='tight')
    print(f"   💾 스티칭 과정 단계별 시각화 저장: {process_plot_file}")
    
    plt.show()


def visualize_canvas_layout(left_img, right_img, H, canvas_size, left_offset):
    """캔버스 레이아웃과 이미지 배치를 시각화합니다."""
    print(f"\n📐 캔버스 레이아웃 시각화")
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print(f"   ❌ matplotlib이 설치되지 않았습니다. pip install matplotlib")
        return
    
    # 왼쪽 이미지를 캔버스에 배치
    h, w = left_img.shape[:2]
    left_canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    ox, oy = int(left_offset[0]), int(left_offset[1])
    left_canvas[oy:oy+h, ox:ox+w] = left_img
    
    # 오른쪽 이미지를 변환하여 캔버스에 배치
    right_transformed = cv2.warpPerspective(right_img, H, canvas_size)
    
    # 캔버스 레이아웃 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 1. 이미지 배치 결과
    final_result = cv2.add(left_canvas, right_transformed)
    final_rgb = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)
    ax1.imshow(final_rgb)
    ax1.set_title('1. Canvas Layout with Images', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # 2. 레이아웃 다이어그램
    ax2.set_xlim(0, canvas_size[0])
    ax2.set_ylim(canvas_size[1], 0)  # OpenCV 좌표계 (y축 반전)
    ax2.set_aspect('equal')
    ax2.set_title('2. Canvas Layout Diagram', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 왼쪽 이미지 영역 표시
    left_rect = patches.Rectangle((ox, oy), w, h, linewidth=3, edgecolor='red', facecolor='red', alpha=0.3)
    ax2.add_patch(left_rect)
    ax2.text(ox + w/2, oy + h/2, 'Left Image', ha='center', va='center', fontsize=12, fontweight='bold', color='red')
    
    # 오른쪽 이미지 영역 표시 (변환된 네 모서리)
    corners = np.float32([[0, 0], [right_img.shape[1], 0], [right_img.shape[1], right_img.shape[0]], [0, right_img.shape[0]]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    
    # 변환된 오른쪽 이미지 영역을 다각형으로 표시
    right_poly = patches.Polygon(warped_corners, linewidth=3, edgecolor='blue', facecolor='blue', alpha=0.3)
    ax2.add_patch(right_poly)
    
    # 오른쪽 이미지 중심점 계산
    center_x = np.mean(warped_corners[:, 0])
    center_y = np.mean(warped_corners[:, 1])
    ax2.text(center_x, center_y, 'Right Image\n(Transformed)', ha='center', va='center', fontsize=12, fontweight='bold', color='blue')
    
    # 캔버스 경계 표시
    canvas_rect = patches.Rectangle((0, 0), canvas_size[0], canvas_size[1], linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
    ax2.add_patch(canvas_rect)
    ax2.text(canvas_size[0]/2, -20, f'Canvas: {canvas_size[0]} x {canvas_size[1]}', ha='center', va='top', fontsize=14, fontweight='bold')
    
    # 좌표 정보 표시
    ax2.text(10, 10, f'Left Offset: ({ox}, {oy})', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # 저장
    output_dir = './data/visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    layout_plot_file = os.path.join(output_dir, 'canvas_layout_diagram.png')
    plt.savefig(layout_plot_file, dpi=300, bbox_inches='tight')
    print(f"   💾 캔버스 레이아웃 다이어그램 저장: {layout_plot_file}")
    
    plt.show()

def save_unified_homography_params(H, canvas_size, left_offset, config_LC_file, config_CR_file, config_dir='./data/config'):
    """참조 코드 구조에 맞는 통합된 homography_params.json을 생성합니다."""
    print(f"\n🔗 통합된 homography_params.json 생성")
    
    # 기존 캘리브레이션 설정에서 카메라 해상도 추출
    config_LC = load_calibration_config(config_LC_file)
    config_CR = load_calibration_config(config_CR_file)
    
    # 카메라 해상도 추출 (LC 또는 CR에서 가져오기)
    camera_resolution = [1920, 1080]  # 기본값
    if 'camera_resolution' in config_LC:
        camera_resolution = config_LC['camera_resolution']
    elif 'camera_resolution' in config_CR:
        camera_resolution = config_CR['camera_resolution']
    
    # 렌즈 왜곡 보정 맵 파일 경로 설정
    params_dir = os.path.join(os.path.dirname(config_dir), "params")
    
    # 통합된 파라미터 구조 (참조 코드와 동일)
    unified_params = {
        'homography_matrix': H.tolist(),
        'final_size': [int(canvas_size[0]), int(canvas_size[1])],
        'camera_resolution': camera_resolution,
        'left_image_offset': [int(left_offset[0]), int(left_offset[1])],
        'rectification_maps': {
            'map_left_x': 'stereo_map_left_x.npy',
            'map_right_x': 'stereo_map_right_x.npy'
        },
        'blending_optimization': {
            'left_mask_file': 'left_blend_mask.npy',
            'right_mask_file': 'right_blend_mask.npy'
        },
        'description': 'Unified homography parameters for real-time video stitching',
        'stitching_method': 'Edge feature-based homography with minimal overlap',
        'usage': 'Load these parameters for real-time Left ↔ Right video stitching',
        'file_structure': 'Compatible with PreCalibratedVideoStitcher class',
        'created_by': 'Stitching_Engine.py (Unified Version)'
    }
    
    # 통합된 파일 저장
    unified_file = os.path.join(config_dir, "homography_params.json")
    with open(unified_file, 'w') as f:
        json.dump(unified_params, f, indent=2)
    
    print(f"   ✅ 통합된 homography_params.json 저장 완료")
    print(f"   파일 경로: {unified_file}")
    print(f"   📊 파라미터 정보:")
    print(f"      - 호모그래피 행렬: {H.shape}")
    print(f"      - 최종 캔버스 크기: {canvas_size}")
    print(f"      - 카메라 해상도: {camera_resolution}")
    print(f"      - 왼쪽 이미지 오프셋: {left_offset}")
    print(f"      - 렌즈 왜곡 보정 맵: {params_dir}")
    
    return unified_file


# 거울모드 시각화 함수 제거됨

def fallback_chessboard_detection(gray, img_name):
    """표준 방법이 실패했을 때 사용하는 대체 검출 방법"""
    print(f"         🆘 대체 검출 방법 시도 중...")
    
    # 1. 엣지 기반 검출
    edges = cv2.Canny(gray, 50, 150)
    
    # 2. 윤곽선 검출
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. 사각형 모양의 윤곽선 찾기
    potential_chessboards = []
    for contour in contours:
        # 윤곽선을 근사화
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 사각형인지 확인
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            if area > 1000:  # 최소 면적
                potential_chessboards.append(approx)
    
    if potential_chessboards:
        print(f"           ✅ {len(potential_chessboards)}개의 잠재적 체커보드 영역 발견")
        # 가장 큰 영역 선택
        largest_contour = max(potential_chessboards, key=cv2.contourArea)
        
        # 해당 영역에서 체커보드 검출 재시도
        x, y, w, h = cv2.boundingRect(largest_contour)
        roi = gray[y:y+h, x:x+w]
        
        # ROI에서 다양한 패턴 크기로 검출 시도
        for pattern in [(3, 3), (4, 4), (5, 5), (3, 5), (3, 7)]:
            ret, corners = cv2.findChessboardCorners(roi, pattern, 
                                                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                # 원본 좌표로 변환
                corners[:, :, 0] += x
                corners[:, :, 1] += y
                
                # 코너 검증
                if validate_chessboard_corners(corners, pattern, gray):
                    print(f"           ✅ 대체 방법으로 {pattern} 패턴 검출 성공!")
                    return True, corners, pattern
        
    print(f"           ❌ 대체 검출 방법도 실패")
    return False, None, None

def interactive_roi_selection(final_result, output_dir='./data/params'):
    """
    사용자가 마우스로 드래그하여 ROI를 직접 지정할 수 있는 인터랙티브 함수
    """
    print(f"\n🎯 인터랙티브 ROI 선택 모드")
    print(f"💡 마우스로 드래그하여 표시할 영역을 선택하세요")
    print(f"💡 선택 완료 후 'Enter' 키를 누르세요")
    print(f"💡 취소하려면 'ESC' 키를 누르세요")
    
    # ROI 선택을 위한 전역 변수
    roi_coords = {'start': None, 'end': None}
    drawing = False
    
    # DISPLAY_SCALE 적용하여 표시용 이미지 생성
    if DISPLAY_SCALE != 1.0:
        display_width = int(final_result.shape[1] * DISPLAY_SCALE)
        display_height = int(final_result.shape[0] * DISPLAY_SCALE)
        display_img = cv2.resize(final_result, (display_width, display_height))
        print(f"📱 디스플레이 스케일 적용: {DISPLAY_SCALE:.1f}x ({display_width}x{display_height})")
    else:
        display_img = final_result.copy()
        display_width = final_result.shape[1]
        display_height = final_result.shape[0]
    
    temp_img = display_img.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal roi_coords, drawing, temp_img
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # 마우스 왼쪽 버튼 클릭 시작
            roi_coords['start'] = (x, y)
            drawing = True
            temp_img = display_img.copy()
            
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            # 마우스 드래그 중 - 실시간 미리보기
            temp_img = display_img.copy()
            cv2.rectangle(temp_img, roi_coords['start'], (x, y), (0, 255, 0), 2)
            
        elif event == cv2.EVENT_LBUTTONUP:
            # 마우스 왼쪽 버튼 놓음
            roi_coords['end'] = (x, y)
            drawing = False
            
            # 최종 선택 영역 표시
            temp_img = display_img.copy()
            x1, y1 = roi_coords['start']
            x2, y2 = roi_coords['end']
            
            # 좌표 정규화 (시작점이 항상 왼쪽 위)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(temp_img, f"ROI: {x2-x1}x{y2-y1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 선택 영역 정보 표시 (스케일된 좌표를 원본 좌표로 변환하여 표시)
            scale_factor = 1.0 / DISPLAY_SCALE if DISPLAY_SCALE != 1.0 else 1.0
            orig_x1 = int(x1 * scale_factor)
            orig_y1 = int(y1 * scale_factor)
            orig_x2 = int(x2 * scale_factor)
            orig_y2 = int(y2 * scale_factor)
            
            area = (orig_x2-orig_x1) * (orig_y2-orig_y1)
            total_area = final_result.shape[0] * final_result.shape[1]
            ratio = (area / total_area) * 100
            
            cv2.putText(temp_img, f"Area: {area:,} pixels ({ratio:.1f}%)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(temp_img, f"Display Coords: ({x1}, {y1}) to ({x2}, {y2})", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(temp_img, f"Original Coords: ({orig_x1}, {orig_y1}) to ({orig_x2}, {orig_y2})", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 윈도우 생성 및 마우스 콜백 설정
    window_name = "Interactive ROI Selection - Drag to select area, Press Enter to confirm"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 윈도우 속성 설정 (matplotlib과 유사한 표시를 위해)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    
    # 윈도우 크기를 스케일된 이미지 해상도로 설정
    cv2.resizeWindow(window_name, display_width, display_height)
    
    # 윈도우 위치를 화면 중앙에 설정 (matplotlib과 유사하게)
    screen_width = 1920  # 일반적인 화면 해상도
    screen_height = 1080
    window_x = max(0, (screen_width - display_width) // 2)
    window_y = max(0, (screen_height - display_height) // 2)
    cv2.moveWindow(window_name, window_x, window_y)
    
    # 이미지가 올바르게 표시되는지 확인 (BGR to RGB 변환 없이)
    # OpenCV는 BGR 형식이므로 그대로 사용
    
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # 초기 이미지 표시
    cv2.imshow(window_name, display_img)
    
    print(f"🎯 마우스로 드래그하여 영역을 선택하세요...")
    print(f"💡 디스플레이 정보: 원본 이미지 크기 {final_result.shape[1]}x{final_result.shape[0]}, 스케일된 크기 {display_width}x{display_height}, 윈도우 위치 ({window_x}, {window_y})")
    
    while True:
        cv2.imshow(window_name, temp_img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter 키
            if roi_coords['start'] and roi_coords['end']:
                break
            else:
                print("⚠️ 먼저 영역을 선택해주세요")
        elif key == 27:  # ESC 키
            print("❌ ROI 선택 취소됨")
            cv2.destroyAllWindows()
            return None
        elif key == ord('r'):  # R 키로 리셋
            roi_coords = {'start': None, 'end': None}
            temp_img = display_img.copy()
            print("🔄 ROI 선택 리셋됨")
    
    # 최종 ROI 좌표 계산 (스케일된 좌표를 원본 좌표로 변환)
    x1, y1 = roi_coords['start']
    x2, y2 = roi_coords['end']
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # 스케일된 좌표를 원본 이미지 좌표로 변환
    scale_factor = 1.0 / DISPLAY_SCALE if DISPLAY_SCALE != 1.0 else 1.0
    orig_x1 = int(x1 * scale_factor)
    orig_y1 = int(y1 * scale_factor)
    orig_x2 = int(x2 * scale_factor)
    orig_y2 = int(y2 * scale_factor)
    
    # ROI 마스크 생성 (원본 이미지 크기로)
    roi_mask = np.zeros(final_result.shape[:2], dtype=np.uint8)
    roi_mask[orig_y1:orig_y2, orig_x1:orig_x2] = 255
    
    # ROI 정보 (원본 이미지 좌표 기준)
    roi_info = {
        'x1': int(orig_x1), 'y1': int(orig_y1), 'x2': int(orig_x2), 'y2': int(orig_y2),
        'width': int(orig_x2 - orig_x1), 'height': int(orig_y2 - orig_y1),
        'area': int((orig_x2 - orig_x1) * (orig_y2 - orig_y1)),
        'total_area': int(final_result.shape[0] * final_result.shape[1]),
        'area_ratio': float(((orig_x2 - orig_x1) * (orig_y2 - orig_y1)) / (final_result.shape[0] * final_result.shape[1]) * 100)
    }
    
    print(f"✅ ROI 선택 완료!")
    print(f"   선택 영역: {roi_info['width']}x{roi_info['height']}")
    print(f"   영역 비율: {roi_info['area_ratio']:.1f}%")
    print(f"   좌표: ({x1}, {y1}) ~ ({x2}, {y2})")
    
    # ROI 마스크 저장
    os.makedirs(output_dir, exist_ok=True)
    mask_file = os.path.join(output_dir, 'user_roi_mask.npy')
    np.save(mask_file, roi_mask)
    
    # ROI 정보 JSON 저장
    roi_file = os.path.join(output_dir, 'user_roi_info.json')
    with open(roi_file, 'w') as f:
        json.dump(roi_info, f, indent=2)
    
    # ROI 미리보기 이미지 저장 (원본 이미지에 원본 좌표로 표시)
    preview_img = final_result.copy()
    cv2.rectangle(preview_img, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 0), 3)
    cv2.putText(preview_img, f"USER SELECTED ROI: {orig_x2-orig_x1}x{orig_y2-orig_y1}", (orig_x1, orig_y1-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    preview_file = os.path.join(output_dir, 'user_roi_preview.png')
    cv2.imwrite(preview_file, preview_img)
    
    print(f"💾 ROI 정보 저장 완료:")
    print(f"   마스크: {mask_file}")
    print(f"   정보: {roi_file}")
    print(f"   미리보기: {preview_file}")
    
    cv2.destroyAllWindows()
    
    return roi_info, roi_mask

def create_roi_based_blending_parameters(final_result, roi_info, homography, canvas_size, left_offset, output_dir='./data/params'):
    """
    ROI 영역에 맞춘 블렌딩 파라미터를 생성합니다.
    ROI 크기로 블렌딩 마스크를 생성하여 실시간 스티칭에서 FPS 향상을 기대할 수 있습니다.
    """
    print(f"\n🎨 ROI 기반 블렌딩 파라미터 생성")
    print(f"   ROI 크기: {roi_info['width']} x {roi_info['height']}")
    print(f"   원본 캔버스: {canvas_size[0]} x {canvas_size[1]}")
    
    # ROI 좌표 추출
    roi_x1, roi_y1 = roi_info['x1'], roi_info['y1']
    roi_x2, roi_y2 = roi_info['x2'], roi_info['y2']
    roi_width, roi_height = roi_info['width'], roi_info['height']
    
    # ROI 영역에서의 왼쪽 이미지 위치 계산
    # ROI가 전체 캔버스에서 어느 부분에 있는지에 따라 왼쪽 이미지의 ROI 내 위치가 결정됨
    roi_left_offset_x = max(0, left_offset[0] - roi_x1)
    roi_left_offset_y = max(0, left_offset[1] - roi_y1)
    
    # ROI 내에서의 왼쪽 이미지 크기 (올바른 계산)
    # 왼쪽 이미지가 ROI와 겹치는 영역의 크기를 계산
    left_img_start_x = max(roi_x1, left_offset[0])
    left_img_end_x = min(roi_x2, left_offset[0] + 1920)
    left_img_start_y = max(roi_y1, left_offset[1])
    left_img_end_y = min(roi_y2, left_offset[1] + 1080)
    
    roi_left_width = max(0, left_img_end_x - left_img_start_x)
    roi_left_height = max(0, left_img_end_y - left_img_start_y)
    
    # ROI 내에서의 왼쪽 이미지 상대적 위치 재계산
    roi_left_offset_x = left_img_start_x - roi_x1
    roi_left_offset_y = left_img_start_y - roi_y1
    
    # ROI 크기로 블렌딩 마스크 생성
    roi_left_blend_mask = np.zeros((roi_height, roi_width), dtype=np.float32)
    roi_right_blend_mask = np.zeros((roi_height, roi_width), dtype=np.float32)
    
    # ROI 내에서의 왼쪽 이미지 영역 설정
    if roi_left_width > 0 and roi_left_height > 0:
        roi_left_blend_mask[roi_left_offset_y:roi_left_offset_y + roi_left_height, 
                           roi_left_offset_x:roi_left_offset_x + roi_left_width] = 1.0
    
    # ROI 내에서의 오른쪽 이미지 영역 설정 (호모그래피 변환 고려)
    # 오른쪽 이미지가 ROI 내에서 어느 부분에 위치하는지 계산
    roi_right_corners = np.float32([[0, 0], [1920, 0], [1920, 1080], [0, 1080]]).reshape(-1, 1, 2)
    roi_right_transformed = cv2.perspectiveTransform(roi_right_corners, homography).reshape(-1, 2)
    
    # 변환된 오른쪽 이미지가 ROI와 겹치는 영역 찾기
    roi_right_blend_mask = np.zeros((roi_height, roi_width), dtype=np.float32)
    
    # ROI 내에서의 오른쪽 이미지 가중치 설정
    for y in range(roi_height):
        for x in range(roi_width):
            # ROI 좌표를 전체 캔버스 좌표로 변환
            canvas_x = roi_x1 + x
            canvas_y = roi_y1 + y
            
            # 왼쪽 이미지 가중치 (이미 설정됨)
            left_weight = roi_left_blend_mask[y, x]
            
            # 오른쪽 이미지 가중치 (중첩 영역에서 1 - left_weight)
            if left_weight > 0:
                roi_right_blend_mask[y, x] = 1.0 - left_weight
            else:
                roi_right_blend_mask[y, x] = 1.0
    
    # 블렌딩 파라미터 저장
    roi_blending_params = {
        'roi_info': roi_info,
        'roi_left_offset': [int(roi_left_offset_x), int(roi_left_offset_y)],
        'roi_left_size': [int(roi_left_width), int(roi_left_height)],
        'roi_canvas_size': [int(roi_width), int(roi_height)],
        'roi_left_blend_mask': roi_left_blend_mask.tolist(),
        'roi_right_blend_mask': roi_right_blend_mask.tolist(),
        'description': 'ROI-based blending parameters for optimized real-time stitching',
        'usage': 'Load these parameters for ROI-optimized real-time video stitching'
    }
    
    # 파일 저장
    os.makedirs(output_dir, exist_ok=True)
    params_file = os.path.join(output_dir, 'roi_blending_params.json')
    
    with open(params_file, 'w') as f:
        json.dump(roi_blending_params, f, indent=2)
    
    # ROI 블렌딩 마스크를 NPY로도 저장
    np.save(os.path.join(output_dir, 'roi_left_blend_mask.npy'), roi_left_blend_mask)
    np.save(os.path.join(output_dir, 'roi_right_blend_mask.npy'), roi_right_blend_mask)
    
    print(f"   ✅ ROI 블렌딩 파라미터 생성 완료")
    print(f"   파일 경로: {params_file}")
    print(f"   ROI 왼쪽 오프셋: ({roi_left_offset_x}, {roi_left_offset_y})")
    print(f"   ROI 왼쪽 크기: {roi_left_width} x {roi_left_height}")
    
    return roi_blending_params




if __name__ == "__main__":
    main()
