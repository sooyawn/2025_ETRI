import cv2
import numpy as np
import os
import json


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

def load_calibration_config(config_file):
    """캘리브레이션 설정 파일을 로드합니다."""
    print(f"\n📁 캘리브레이션 설정 로드: {config_file}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # 카메라 내부 파라미터
    left_mtx = np.array(config['left_camera']['intrinsic_matrix'])
    left_dist = np.array(config['left_camera']['distortion_coefficients'])
    right_mtx = np.array(config['right_camera']['intrinsic_matrix'])
    right_dist = np.array(config['right_camera']['distortion_coefficients'])
    
    # 스테레오 관계
    R = np.array(config['stereo_calibration']['rotation_matrix'])
    T = np.array(config['stereo_calibration']['translation_vector'])
    
    # 정렬 맵 로드
    config_dir = os.path.dirname(config_file)
    pair_name = config['pair_name']
    maps_file = os.path.join(config_dir, f"{pair_name}_rectification_maps.npz")
    
    if os.path.exists(maps_file):
        maps_data = np.load(maps_file)
        left_map1_x = np.array(maps_data['left_map1_x'], dtype=np.float32)
        right_map1_x = np.array(maps_data['right_map1_x'], dtype=np.float32)
        print(f"   ✅ 정렬 맵 로드 성공")
    else:
        raise FileNotFoundError(f"정렬 맵 파일을 찾을 수 없습니다: {maps_file}")
    
    return {
        'left_mtx': left_mtx, 'left_dist': left_dist,
        'right_mtx': right_mtx, 'right_dist': right_dist,
        'R': R, 'T': T,
        'left_map1_x': left_map1_x, 'right_map1_x': right_map1_x
    }

def apply_rectification_maps(left_img, right_img, config_LC, config_CR):
    """캘리브레이션 결과를 사용하여 이미지를 정렬합니다."""
    print(f"\n🔄 이미지 정렬 (캘리브레이션 맵 적용)")
    
    # LC의 Left 카메라 정렬 맵 사용
    left_rectified = cv2.remap(left_img, config_LC['left_map1_x'], None, 
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # CR의 Right 카메라 정렬 맵 사용
    right_rectified = cv2.remap(right_img, config_CR['right_map1_x'], None, 
                                cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    print(f"   ✅ 이미지 정렬 완료")
    return left_rectified, right_rectified

# Helper functions from Stitching_Engine_3.py
def preprocess_image_for_chessboard(gray):
    """체커보드 검출을 위한 이미지 전처리 (Calibration_2.py와 동일한 강화된 방식)"""
    # 가우시안 블러로 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 히스토그램 평활화로 대비 향상
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    
    # 추가 필터링 옵션들
    gaussian_3x3 = cv2.GaussianBlur(gray, (3, 3), 0)
    median_5 = cv2.medianBlur(gray, 5)
    
    # 이진화 (적응적 임계값)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    return enhanced, binary, gaussian_3x3, median_5

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
    """여러 스케일에서 체커보드 검출 시도 (Calibration_2.py와 동일한 강화된 방식)"""
    # Calibration_2.py와 동일한 스케일 시도
    scales = [0.5, 0.75, 1.5, 2.0]
    
    for scale in scales:
        if scale != 1.0:
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
                print(f"           ✅ 크기 조정 {scale}x 성공: {len(corners)}개")
                return True, corners
    
    print(f"           ❌ 다중 스케일 검출 실패")
    return False, None

def save_debug_images(gray, enhanced, binary, gaussian_3x3, median_5, img_name, pattern_size):
    """디버깅을 위한 이미지 저장 (Calibration_2.py와 동일한 강화된 방식)"""
    debug_dir = './data/debug_images'
    os.makedirs(debug_dir, exist_ok=True)
    
    # 원본 이미지 저장
    cv2.imwrite(os.path.join(debug_dir, f'{img_name}_original.png'), gray)
    
    # 전처리된 이미지들 저장
    cv2.imwrite(os.path.join(debug_dir, f'{img_name}_enhanced.png'), enhanced)
    cv2.imwrite(os.path.join(debug_dir, f'{img_name}_binary.png'), binary)
    cv2.imwrite(os.path.join(debug_dir, f'{img_name}_gaussian_3x3.png'), gaussian_3x3)
    cv2.imwrite(os.path.join(debug_dir, f'{img_name}_median_5.png'), median_5)
    
    # 체커보드 패턴 정보 저장
    with open(os.path.join(debug_dir, f'{img_name}_pattern_info.txt'), 'w') as f:
        f.write(f"Pattern Size: {pattern_size}\n")
        f.write(f"Image Shape: {gray.shape}\n")
        f.write(f"Image Type: {gray.dtype}\n")
        f.write(f"Enhanced: blurred + CLAHE\n")
        f.write(f"Gaussian 3x3: 노이즈 제거\n")
        f.write(f"Median 5: 노이즈 제거\n")
    
    print(f"         💾 디버그 이미지 저장: {debug_dir}")

def detect_chessboard_corners_simple(gray, img_name, pattern_size):
    """Stitching_Engine_3.py와 완전히 동일한 체크보드 검출 (fallback 제외)"""
    base_cols, base_rows = int(pattern_size[0]), int(pattern_size[1])
    print(f"       🔍 OpenCV 체크보드 코너 검출 시작... (요청: {base_cols}x{base_rows})")

    # 이미지 전처리 (Calibration_2.py와 동일한 강화된 방식)
    print(f"         🎨 이미지 전처리 중...")
    enhanced, binary, gaussian_3x3, median_5 = preprocess_image_for_chessboard(gray)
    
    # Calibration_2.py와 동일한 강화된 검출 플래그 조합
    flags_combinations = [
        cv2.CALIB_CB_ADAPTIVE_THRESH,
        cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_FAST_CHECK,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK,
        cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ]

    # 0) 오버랩 가장자리 ROI 우선 시도 (체커보드가 작아졌을 때 보이는 가장자리만 확대 탐색)
    h, w = gray.shape
    side = 'left' if '왼쪽' in str(img_name) else 'right'
    roi_width = max(100, int(w * 0.18))
    if side == 'left':
        roi = gray[:, :roi_width]
        roi_x0 = 0
    else:
        roi = gray[:, w - roi_width:]
        roi_x0 = w - roi_width

    # ROI 업스케일 탐색 (작은 보드 대응)
    upscale_scales = [1.5, 2.0, 2.5, 3.0, 3.5]
    # ROI 전처리(블러+CLAHE)
    roi_blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    roi_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(roi_blurred)
    for scale in upscale_scales:
        roi_src = roi_clahe
        roi_resized = cv2.resize(roi_src, (int(roi_src.shape[1] * scale), int(roi_src.shape[0] * scale)))

        # 0) SB 먼저 시도 (강검출)
        sb_flags_list = [
            0,
            cv2.CALIB_CB_NORMALIZE_IMAGE,
            cv2.CALIB_CB_EXHAUSTIVE if hasattr(cv2, 'CALIB_CB_EXHAUSTIVE') else 0,
            (cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE) if hasattr(cv2, 'CALIB_CB_EXHAUSTIVE') else cv2.CALIB_CB_NORMALIZE_IMAGE,
            cv2.CALIB_CB_ACCURACY if hasattr(cv2, 'CALIB_CB_ACCURACY') else 0
        ]
        found = False
        corners_roi = None
        for sbf in sb_flags_list:
            if sbf == 0 and sb_flags_list.count(0) > 1:
                pass
            try:
                ret_roi, c = cv2.findChessboardCornersSB(roi_resized, (base_cols, base_rows), flags=sbf)
            except Exception:
                ret_roi, c = False, None
            if ret_roi:
                corners_roi = c
                found = True
                break

        # 1) 표준으로도 시도
        if not found:
            for flg in flags_combinations:
                ret_roi, c = cv2.findChessboardCorners(roi_resized, (base_cols, base_rows), flg)
                if ret_roi:
                    corners_roi = c
                    found = True
                    break

        if found and corners_roi is not None:
            # 원본 좌표로 복원
            corners_roi = corners_roi / scale
            corners_roi[:, 0, 0] += roi_x0
            # 서브픽셀 정밀도 향상
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
            corners_roi = cv2.cornerSubPix(gray, corners_roi, (15, 15), (-1, -1), criteria)

            if validate_chessboard_corners(corners_roi, (base_cols, base_rows), gray):
                corners_xy = corners_roi.reshape(-1, 2)
                corners_grid = corners_xy.reshape(base_rows, base_cols, 2)
                col_means_x = [float(np.mean(corners_grid[:, j, 0])) for j in range(base_cols)]
                if '왼쪽' in str(img_name):
                    col_index = int(np.argmax(col_means_x))
                elif '오른쪽' in str(img_name):
                    col_index = int(np.argmin(col_means_x))
                else:
                    col_index = int(np.argmin(col_means_x))
                indices = [r * base_cols + col_index for r in range(base_rows)]
                selected_corners = corners_xy[indices]
                order = np.argsort(selected_corners[:, 1])
                selected_corners = selected_corners[order]
                print(f"           ✅ ROI 업스케일({scale:.1f}x) 검출 성공 - {img_name}")
                return selected_corners

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
            # 서브픽셀 정밀도 향상
            try:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
                multiscale_corners = cv2.cornerSubPix(gray, multiscale_corners, (15, 15), (-1, -1), criteria)
            except Exception:
                pass
            if validate_chessboard_corners(multiscale_corners, (cols, rows), gray):
                corners_xy = multiscale_corners.reshape(-1, 2)
                # 좌표 기반으로 실제 왼쪽/오른쪽 열 선택 (3.py와 동일 로직)
                corners_grid = corners_xy.reshape(rows, cols, 2)
                col_means_x = [float(np.mean(corners_grid[:, j, 0])) for j in range(cols)]
                if '왼쪽' in str(img_name):
                    col_index = int(np.argmax(col_means_x))  # 왼쪽: 실제 가장 오른쪽 열
                elif '오른쪽' in str(img_name):
                    col_index = int(np.argmin(col_means_x))  # 오른쪽: 실제 가장 왼쪽 열
                else:
                    col_index = int(np.argmin(col_means_x))

                indices = [r * cols + col_index for r in range(rows)]
                selected_corners = corners_xy[indices]
                # 행 정렬 표준화: y 오름차순(위→아래)
                order = np.argsort(selected_corners[:, 1])
                selected_corners = selected_corners[order]
                
                # 🔍 디버깅: 선택된 코너들의 X 좌표 출력
                print(f"           ✅ {img_name}: 열 {col_index} 선택, {len(selected_corners)}개 코너 추출 (다중 스케일)")
                print(f"           📍 선택된 코너 X좌표: {[f'{corner[0]:.1f}' for corner in selected_corners]}")
                print(f"           📍 전체 코너 X좌표 범위: {corners_xy[:, 0].min():.1f} ~ {corners_xy[:, 0].max():.1f}")
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
                
                # Calibration_2.py와 동일한 전처리된 이미지들로 검출 시도
                detection_methods = [
                    ("기본", gray),
                    ("블러", enhanced),  # enhanced는 이미 blurred + CLAHE
                    ("대비향상", enhanced),
                    ("가우시안", gaussian_3x3),
                    ("중간값필터", median_5)
                ]
                
                for method_name, processed_img in detection_methods:
                    ret, corners = cv2.findChessboardCornersSB(processed_img, (cols, rows), flags=sb_flags)
                    if ret:
                        print(f"           ✅ SB 성공 ({cols}x{rows}) - {method_name} 이미지 사용")
                        
                        # 코너 서브픽셀 정밀도 향상 (더 정밀한 파라미터)
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
                        corners = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criteria)
                        
                        # 코너 검증
                        if validate_chessboard_corners(corners, (cols, rows), gray):
                            corners_xy = corners.reshape(-1, 2)
                            total_corners = corners_xy.shape[0]
                            expected_corners = cols * rows
                            
                            if total_corners >= expected_corners:
                                # 좌표 기반 열 선택 (3.py와 동일)
                                corners_grid = corners_xy.reshape(rows, cols, 2)
                                col_means_x = [float(np.mean(corners_grid[:, j, 0])) for j in range(cols)]
                                if '왼쪽' in str(img_name):
                                    col_index = int(np.argmax(col_means_x))
                                elif '오른쪽' in str(img_name):
                                    col_index = int(np.argmin(col_means_x))
                                else:
                                    col_index = int(np.argmin(col_means_x))

                                indices = [r * cols + col_index for r in range(rows)]
                                selected_corners = corners_xy[indices]
                                # 행 정렬 표준화
                                order = np.argsort(selected_corners[:, 1])
                                selected_corners = selected_corners[order]
                                
                                # 🔍 디버깅: 선택된 코너들의 X 좌표 출력
                                print(f"           ✅ {img_name}: 열 {col_index} 선택, {len(selected_corners)}개 코너 추출")
                                print(f"           📍 선택된 코너 X좌표: {[f'{corner[0]:.1f}' for corner in selected_corners]}")
                                print(f"           📍 전체 코너 X좌표 범위: {corners_xy[:, 0].min():.1f} ~ {corners_xy[:, 0].max():.1f}")
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

            # Calibration_2.py와 동일한 전처리된 이미지들로 검출 시도
            detection_methods = [
                ("기본", gray),
                ("블러", enhanced),  # enhanced는 이미 blurred + CLAHE
                ("대비향상", enhanced),
                ("가우시안", gaussian_3x3),
                ("중간값필터", median_5)
            ]
            
            for method_name, processed_img in detection_methods:
                try:
                    ret, corners = cv2.findChessboardCorners(processed_img, (cols, rows), flags=flags)

                    if ret:
                        print(f"           ✅ {img_name}: 체크보드 검출 성공 ({cols}x{rows}) - {method_name} 이미지 사용!")

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
                                # 좌표 기반 열 선택 (3.py와 동일)
                                corners_grid = corners_xy.reshape(rows, cols, 2)
                                col_means_x = [float(np.mean(corners_grid[:, j, 0])) for j in range(cols)]
                                if '왼쪽' in str(img_name):
                                    col_index = int(np.argmax(col_means_x))
                                elif '오른쪽' in str(img_name):
                                    col_index = int(np.argmin(col_means_x))
                                else:
                                    col_index = int(np.argmin(col_means_x))

                                indices = [r * cols + col_index for r in range(rows)]
                                selected_corners = corners_xy[indices]
                                # 행 정렬 표준화
                                order = np.argsort(selected_corners[:, 1])
                                selected_corners = selected_corners[order]
                                
                                # 🔍 디버깅: 선택된 코너들의 X 좌표 출력
                                print(f"           ✅ {img_name}: 열 {col_index} 선택, {len(selected_corners)}개 코너 추출")
                                print(f"           📍 선택된 코너 X좌표: {[f'{corner[0]:.1f}' for corner in selected_corners]}")
                                print(f"           📍 전체 코너 X좌표 범위: {corners_xy[:, 0].min():.1f} ~ {corners_xy[:, 0].max():.1f}")
                                return selected_corners
                            else:
                                print(f"           ⚠️ {img_name}: 코너 개수 부족 ({total_corners}/{expected_corners})")
                                # 이 경우도 다음 cols로 재시도
                        else:
                            print(f"           ⚠️ {img_name}: 코너 검증 실패")
                            continue
                    else:
                        print(f"           ❌ {img_name}: 방법 {i+1} 실패 ({method_name} 이미지)")

                except Exception as e:
                    print(f"           ❌ {img_name}: 방법 {i+1} 오류 - {e}")
                    continue

    print(f"       ❌ {img_name}: 모든 cols 감소 재시도 실패 (최종 {base_cols}→3)")
    return None

def detect_overlap_features(rectified_left, rectified_right, pattern_size=(1, 9)):
    """정렬된 이미지에서 중첩 영역의 체크보드 코너를 검출합니다."""
    print(f"\n🎯 중첩 영역 체크보드 코너 검출")
    print(f"   💡 왼쪽 이미지: 4×9 패턴, 오른쪽 이미지: 4×9 패턴 (동일한 패턴 크기)")
    print(f"   왼쪽 이미지 크기: {rectified_left.shape}")
    print(f"   오른쪽 이미지 크기: {rectified_right.shape}")
    print(f"   💡 중첩 영역: chessboard in overlap area")
    
    def extract_overlap_corners(img, img_name, specific_pattern_size):
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
        
        # 체크보드 OpenCV 표준 검출 (각 이미지별 패턴 사용)
        print(f"     🔍 Chessboard detection start... ({specific_pattern_size[0]}x{specific_pattern_size[1]})")
        
        # OpenCV 체크보드 검출 사용
        corners = detect_chessboard_corners_simple(gray, img_name, specific_pattern_size)
        if corners is not None:
            return corners
        
        print(f"     ❌ {img_name}: 체크보드 검출 실패")
        return None
    
    # 왼쪽 이미지에서 중첩 영역 코너 추출 (4×9 패턴)
    left_pattern_size = (4, 9)
    left_corners = extract_overlap_corners(rectified_left, "왼쪽", left_pattern_size)
    
    # 오른쪽 이미지에서 중첩 영역 코너 추출 (3×9 패턴)
    right_pattern_size = (3, 9)
    right_corners = extract_overlap_corners(rectified_right, "오른쪽", right_pattern_size)
    
    # 결과 확인
    if left_corners is None or right_corners is None:
        print(f"   ❌ 중첩 영역 코너 추출 실패")
        print(f"   💡 해결 방법:")
        print(f"     1. 체크보드가 정렬된 이미지에 완전히 보이는지 확인")
        print(f"     2. 체크보드 크기 확인: 왼쪽 4×9, 오른쪽 4×9")
        print(f"     3. 이미지 품질 개선: 선명하게, 충분한 조명으로")
        print(f"     4. 가운데 화살표가 있는 중첩 영역이 양쪽 이미지에 보이는지 확인")
        return None, None
    
    print(f"   ✅ 양쪽 이미지 모두 중첩 영역 코너 추출 성공!")
    print(f"   왼쪽 (4×9): {len(left_corners)}개 점")
    print(f"   오른쪽 (4×9): {len(right_corners)}개 점")
    print(f"   💡 중첩 영역의 공통 특징점으로 호모그래피 계산 가능!")
    
    return left_corners, right_corners

def calculate_homography_from_overlap_corners(left_corners, right_corners, left_img=None, right_img=None):
    """중첩 영역 코너로부터 호모그래피를 계산합니다."""
    print(f"\n🔗 호모그래피 계산")
    
    left_pts = np.float32(left_corners.reshape(-1, 2))
    right_pts = np.float32(right_corners.reshape(-1, 2))
    
    if len(left_pts) < 4 or len(right_pts) < 4:
        print(f"   ❌ 특징점이 부족합니다 (최소 4개 필요)")
        return None
    
    try:
        # 유사변환 추정
        A, inliers = cv2.estimateAffinePartial2D(
            right_pts, left_pts, 
            method=cv2.RANSAC, 
            ransacReprojThreshold=2.0,
            maxIters=2000,
            confidence=0.99
        )
        
        if A is not None:
            # 수평 오프셋 보정: 예측 좌표와 실제 좌표의 x-잔차의 중앙값을 tx에 반영
            try:
                pred = (right_pts @ A[:, :2].T) + A[:, 2]
                residuals = left_pts - pred
                median_dx = float(np.median(residuals[:, 0]))
                # tx 보정만 수행 (수직 정합은 이미 양호)
                A[0, 2] += median_dx
                print(f"   🔧 수평 바이어스 보정: Δtx={median_dx:.2f} px")
            except Exception:
                pass

            H = np.vstack([A, [0, 0, 1]]).astype(np.float64)
            print(f"   ✅ 호모그래피 계산 성공")
            return H, inliers
        else:
            # 단순 이동으로 fallback
            deltas = left_pts - right_pts
            mean_delta = np.mean(deltas, axis=0)
            tx, ty = float(mean_delta[0]), float(mean_delta[1])
            H = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float64)
            print(f"   ↔️ Fallback 이동 변환: (tx,ty)=({tx:.2f},{ty:.2f})")
            return H, None
            
    except Exception as e:
        print(f"   ❌ 호모그래피 계산 실패: {e}")
        return None

def compute_canvas_with_translation(left_img, right_img, H):
    """캔버스 크기와 오프셋을 계산합니다."""
    h1, w1 = left_img.shape[:2]
    h2, w2 = right_img.shape[:2]

    # 오른쪽 이미지 네 모서리를 H로 변환
    corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    # 좌표계 통합
    all_x = np.concatenate([np.array([0, w1]), warped[:, 0]])
    all_y = np.concatenate([np.array([0, h1]), warped[:, 1]])

    min_x, min_y = np.min(all_x), np.min(all_y)
    max_x, max_y = np.max(all_x), np.max(all_y)

    # 모두 양수로 이동시키는 translation
    tx = -min(0.0, float(min_x))
    ty = -min(0.0, float(min_y))

    T = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float64)
    H_canvas = T @ H

    # 캔버스 크기 계산
    canvas_w = max(int(np.ceil(max_x + tx)), w1 + int(np.ceil(tx)))
    canvas_h = max(int(np.ceil(max_y + ty)), h1 + int(np.ceil(ty)))
    canvas_size = (canvas_w, canvas_h)

    left_offset = (int(np.floor(tx)), int(np.floor(ty)))
    
    print(f"   캔버스 크기: {canvas_size}, 왼쪽 오프셋: {left_offset}")
    return H_canvas, canvas_size, left_offset, 0

def create_blending_masks(left_img, right_img, H, canvas_size, left_offset=(0,0)):
    """블렌딩 마스크를 생성합니다."""
    h, w = left_img.shape[:2]
    
    # 왼쪽 이미지를 캔버스에 배치
    left_canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    ox, oy = int(left_offset[0]), int(left_offset[1])
    left_canvas[oy:oy+h, ox:ox+w] = left_img
    
    # 오른쪽 이미지를 변환하여 캔버스에 배치
    right_transformed = cv2.warpPerspective(right_img, H, canvas_size)
    
    # 블렌딩 마스크 생성
    left_mask = (left_canvas.sum(axis=2) > 0).astype(np.float32)
    right_mask = (right_transformed.sum(axis=2) > 0).astype(np.float32)
    
    # 중첩 영역에서 가중치 조정
    overlap = np.logical_and(left_mask > 0, right_mask > 0)
    left_blend_mask = left_mask.copy()
    right_blend_mask = right_mask.copy()
    
    # 중첩 영역에서는 0.5씩 할당
    left_blend_mask[overlap] = 0.5
    right_blend_mask[overlap] = 0.5
    
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
    
    # 블렌딩
    left_mask_3ch = left_blend_mask[:, :, np.newaxis]
    right_mask_3ch = right_blend_mask[:, :, np.newaxis]
    
    left_weighted = (left_canvas.astype(np.float32) * left_mask_3ch).astype(np.uint8)
    right_weighted = (right_transformed.astype(np.float32) * right_mask_3ch).astype(np.uint8)
    
    final_result = cv2.add(left_weighted, right_weighted)
    
    print(f"   ✅ 스티칭 완료")
    return final_result, left_canvas, right_transformed

def save_stitching_parameters(H, canvas_size, left_offset, left_blend_mask, right_blend_mask, config_dir='./data/config'):
    """🚫 불필요한 LR_stitching_parameters.json 생성 제거 (실시간 코드에서 사용되지 않음)"""
    print(f"\n💾 스티칭 파라미터 저장 (homography_params.json만 생성)")
    
    # LR_stitching_parameters.json은 실시간 코드에서 사용되지 않으므로 생성하지 않음
    print(f"   ℹ️ LR_stitching_parameters.json 생성 건너뜀 (실시간 코드에서 미사용)")
    return None

def save_unified_homography_params(H, canvas_size, left_offset, config_LC_file, config_CR_file, config_dir='./data/config'):
    """실시간 코드에서 사용하는 homography_params.json만 생성합니다."""
    try:
        with open(config_LC_file, 'r') as f:
            config_LC = json.load(f)
        camera_resolution = config_LC.get('camera_resolution', [1920, 1080])
    except:
        camera_resolution = [1920, 1080]
    
    # 🎯 실시간 코드에서 실제 사용하는 파라미터만 저장
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
        }
    }
    
    unified_file = os.path.join(config_dir, "homography_params.json")
    with open(unified_file, 'w') as f:
        json.dump(unified_params, f, indent=2)
    
    print(f"   ✅ 실시간 코드용 파라미터 저장: {unified_file}")
    return unified_file

def save_stitching_results(left_canvas, right_transformed, final_result, left_blend_mask, right_blend_mask, output_dir='./data/stitching_results'):
    """스티칭 결과를 저장합니다."""
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(output_dir, "left_canvas.png"), left_canvas)
    cv2.imwrite(os.path.join(output_dir, "right_transformed.png"), right_transformed)
    cv2.imwrite(os.path.join(output_dir, "final_stitched.png"), final_result)
    cv2.imwrite(os.path.join(output_dir, "left_blend_mask.png"), (left_blend_mask * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, "right_blend_mask.png"), (right_blend_mask * 255).astype(np.uint8))
    
    print(f"   ✅ 결과 이미지 저장 완료: {output_dir}")

# ========================================
# 시각화 함수들 (필수 4개)
# ========================================

def create_pipeline_visualization_1(left_original, right_original, left_rectified, right_rectified):
    """1단계: 원본 이미지 → 정렬된 이미지 시각화"""
    print(f"\n🎨 1단계 시각화: 원본 → 정렬")
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"   ❌ matplotlib이 설치되지 않았습니다. pip install matplotlib")
        return
    
    # BGR을 RGB로 변환
    if len(left_original.shape) == 3:
        left_orig_rgb = cv2.cvtColor(left_original, cv2.COLOR_BGR2RGB)
        right_orig_rgb = cv2.cvtColor(right_original, cv2.COLOR_BGR2RGB)
        left_rect_rgb = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2RGB)
        right_rect_rgb = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2RGB)
    else:
        left_orig_rgb = left_original
        right_orig_rgb = right_original
        left_rect_rgb = left_rectified
        right_rect_rgb = right_rectified
    
    # 2x2 서브플롯 생성
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    axes[0, 0].imshow(left_orig_rgb)
    axes[0, 0].set_title('1. Original Left Image', fontsize=16, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(right_orig_rgb)
    axes[0, 1].set_title('2. Original Right Image', fontsize=16, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(left_rect_rgb)
    axes[1, 0].set_title('3. Rectified Left Image', fontsize=16, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(right_rect_rgb)
    axes[1, 1].set_title('4. Rectified Right Image', fontsize=16, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    output_dir = './data/visualization'
    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, '1_calibration_rectification.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"   💾 1단계 시각화 저장: {plot_file}")
    plt.show()

def create_pipeline_visualization_2(left_img, right_img, left_corners, right_corners):
    """2단계: 특징점 검출 결과 시각화"""
    print(f"\n🎨 2단계 시각화: 특징점 검출")
    
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
        cv2.circle(left_with_corners, (x, y), 3, (255, 0, 0), -1)
        cv2.circle(left_with_corners, (x, y), 5, (255, 255, 255), 2)
        cv2.putText(left_with_corners, str(i), (x+15, y+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 오른쪽 이미지에 코너점 표시
    for i, corner in enumerate(right_corners):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(right_with_corners, (x, y), 3, (0, 255, 0), -1)
        cv2.circle(right_with_corners, (x, y), 5, (255, 255, 255), 2)
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
    
    output_dir = './data/visualization'
    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, '2_feature_detection.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"   💾 2단계 시각화 저장: {plot_file}")
    plt.show()

def create_pipeline_visualization_3(left_img, right_img, left_corners, right_corners, H):
    """3단계: 호모그래피 매칭 결과 시각화"""
    print(f"\n🎨 3단계 시각화: 호모그래피 매칭")
    
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
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.imshow(concat)
    ax.set_title('Homography Matching - Left-Right Correspondence', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # 왼쪽 이미지와 오른쪽 이미지 간 매칭선 그리기
    for i, (lp, rp) in enumerate(zip(left_corners, right_corners)):
        lx, ly = float(lp[0]), float(lp[1])  # 왼쪽 이미지의 코너
        rx, ry = float(rp[0]) + w1, float(rp[1])  # 오른쪽 이미지의 코너 (x좌표에 w1 오프셋)
        
        # 왼쪽과 오른쪽 이미지 간 매칭선
        ax.plot([lx, rx], [ly, ry], '-', color='yellow', linewidth=2, alpha=0.8)
        
        # 코너점 표시
        ax.plot(lx, ly, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
        ax.plot(rx, ry, 'bo', markersize=8, markeredgecolor='white', markeredgewidth=2)
        
        # 코너점 번호 표시
        ax.text(lx+10, ly-10, str(i), color='white', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
        ax.text(rx+10, ry-10, str(i), color='white', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7))
    
    plt.tight_layout()
    
    output_dir = './data/visualization'
    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, '3_homography_matching.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"   💾 3단계 시각화 저장: {plot_file}")
    plt.show()

def create_pipeline_visualization_4(left_img, right_img, final_result):
    """4단계: 최종 스티칭 결과 시각화"""
    print(f"\n🎨 4단계 시각화: 최종 스티칭 결과")
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"   ❌ matplotlib이 설치되지 않았습니다. pip install matplotlib")
        return
    
    # BGR을 RGB로 변환
    final_rgb = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB) if len(final_result.shape) == 3 else final_result
    
    # 1x1 서브플롯 생성 (최종 결과만)
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    ax.imshow(final_rgb)
    ax.set_title('Final Stitched Result', fontsize=18, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    output_dir = './data/visualization'
    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, '4_final_stitching.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"   💾 4단계 시각화 저장: {plot_file}")
    plt.show()

# ========================================
# ROI 선택 기능 (필수)
# ========================================

def interactive_roi_selection(final_result, output_dir='./data/params'):
    """사용자가 마우스로 드래그하여 ROI를 직접 지정할 수 있는 인터랙티브 함수"""
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
    else:
        display_img = final_result.copy()
        display_width = final_result.shape[1]
        display_height = final_result.shape[0]
    
    temp_img = display_img.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal roi_coords, drawing, temp_img
        
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_coords['start'] = (x, y)
            drawing = True
            temp_img = display_img.copy()
            
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            temp_img = display_img.copy()
            cv2.rectangle(temp_img, roi_coords['start'], (x, y), (0, 255, 0), 2)
            
        elif event == cv2.EVENT_LBUTTONUP:
            roi_coords['end'] = (x, y)
            drawing = False
            
            temp_img = display_img.copy()
            x1, y1 = roi_coords['start']
            x2, y2 = roi_coords['end']
            
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(temp_img, f"ROI: {x2-x1}x{y2-y1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # 윈도우 생성 및 마우스 콜백 설정
    window_name = "Interactive ROI Selection - Drag to select area, Press Enter to confirm"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, display_img)
    
    print(f"🎯 마우스로 드래그하여 영역을 선택하세요...")
    
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
    
    # ROI 마스크 저장
    os.makedirs(output_dir, exist_ok=True)
    mask_file = os.path.join(output_dir, 'user_roi_mask.npy')
    np.save(mask_file, roi_mask)
    
    # ROI 정보 JSON 저장
    roi_file = os.path.join(output_dir, 'user_roi_info.json')
    with open(roi_file, 'w') as f:
        json.dump(roi_info, f, indent=2)
    
    # ROI 미리보기 이미지 저장
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
    """ROI 영역에 맞춘 블렌딩 파라미터를 생성합니다."""
    print(f"\n🎨 ROI 기반 블렌딩 파라미터 생성")
    
    roi_x1, roi_y1 = roi_info['x1'], roi_info['y1']
    roi_x2, roi_y2 = roi_info['x2'], roi_info['y2']
    roi_width, roi_height = roi_info['width'], roi_info['height']
    
    # ROI 크기로 블렌딩 마스크 생성
    roi_left_blend_mask = np.zeros((roi_height, roi_width), dtype=np.float32)
    roi_right_blend_mask = np.zeros((roi_height, roi_width), dtype=np.float32)
    
    # 왼쪽 이미지가 ROI와 겹치는 영역 계산
    left_img_start_x = max(roi_x1, left_offset[0])
    left_img_end_x = min(roi_x2, left_offset[0] + 1920)
    left_img_start_y = max(roi_y1, left_offset[1])
    left_img_end_y = min(roi_y2, left_offset[1] + 1080)
    
    roi_left_width = max(0, left_img_end_x - left_img_start_x)
    roi_left_height = max(0, left_img_end_y - left_img_start_y)
    roi_left_offset_x = left_img_start_x - roi_x1
    roi_left_offset_y = left_img_start_y - roi_y1
    
    # ROI 내에서의 왼쪽 이미지 영역 설정
    if roi_left_width > 0 and roi_left_height > 0:
        roi_left_blend_mask[roi_left_offset_y:roi_left_offset_y + roi_left_height, 
                           roi_left_offset_x:roi_left_offset_x + roi_left_width] = 1.0
    
    # 오른쪽 이미지 가중치 설정
    for y in range(roi_height):
        for x in range(roi_width):
            left_weight = roi_left_blend_mask[y, x]
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
        'roi_right_blend_mask': roi_right_blend_mask.tolist()
    }
    
    os.makedirs(output_dir, exist_ok=True)
    params_file = os.path.join(output_dir, 'roi_blending_params.json')
    
    with open(params_file, 'w') as f:
        json.dump(roi_blending_params, f, indent=2)
    
    # ROI 블렌딩 마스크를 NPY로도 저장
    np.save(os.path.join(output_dir, 'roi_left_blend_mask.npy'), roi_left_blend_mask)
    np.save(os.path.join(output_dir, 'roi_right_blend_mask.npy'), roi_right_blend_mask)
    
    print(f"   ✅ ROI 블렌딩 파라미터 생성 완료: {params_file}")
    return roi_blending_params

# ========================================
# 파이프라인 함수들
# ========================================

def print_pipeline_progress(step, total_steps, title, description=""):
    """파이프라인 진행 상황을 표시합니다."""
    progress_bar = '█' * step + '░' * (total_steps - step)
    percentage = (step / total_steps) * 100
    print(f"\n{'='*60}")
    print(f"🚀 단계 {step}/{total_steps}: {title}")
    print(f"📋 진행도: [{progress_bar}] {percentage:.1f}%")
    if description:
        print(f"💡 {description}")
    print(f"{'='*60}")

def create_stitching_pipeline(config_LC_file, config_CR_file, left_img, right_img, pattern_size=(1, 10)):
    """전체 스티칭 파이프라인을 실행합니다."""
    print(f"\n🎯 Left ↔ Right 스티칭 파이프라인 시작")
    print(f"{'='*60}")
    print(f"🎯 체크보드 패턴: {pattern_size[0]} x {pattern_size[1]}")
    print(f"💡 내부 코너: {pattern_size[0]} x {pattern_size[1]}")
    print(f"💡 method: homography from overlap chessboard + minimal overlap stitching")
    print(f"{'='*60}")
    
    total_steps = 4
    
    # 1단계: 캘리브레이션 설정 로드 및 이미지 정렬
    print_pipeline_progress(1, total_steps, "캘리브레이션 설정 로드 및 이미지 정렬", 
                           "캘리브레이션 데이터를 로드하고 이미지를 정렬합니다")
    config_LC = load_calibration_config(config_LC_file)
    config_CR = load_calibration_config(config_CR_file)
    left_rectified, right_rectified = apply_rectification_maps(left_img, right_img, config_LC, config_CR)
    
    create_pipeline_visualization_1(left_img, right_img, left_rectified, right_rectified)
    
    # 2단계: 특징점 검출
    print_pipeline_progress(2, total_steps, "특징점 검출", 
                           "중첩 영역에서 체크보드 코너를 검출합니다")
    left_corners, right_corners = detect_overlap_features(left_rectified, right_rectified, pattern_size)
    
    if left_corners is None or right_corners is None:
        print(f"❌ 중첩 영역 체크보드 코너 검출 실패")
        return None
    
    create_pipeline_visualization_2(left_rectified, right_rectified, left_corners, right_corners)
    
    # 3단계: 호모그래피 계산 및 매칭
    print_pipeline_progress(3, total_steps, "호모그래피 계산 및 매칭", 
                           "특징점을 사용하여 호모그래피 행렬을 계산합니다")
    result = calculate_homography_from_overlap_corners(left_corners, right_corners, left_rectified, right_rectified)
    if result is None:
        print(f"❌ 호모그래피 계산 실패")
        return None
    
    H, mask = result
    create_pipeline_visualization_3(left_rectified, right_rectified, left_corners, right_corners, H)
    
    # 4단계: 스티칭 결과 생성
    print_pipeline_progress(4, total_steps, "스티칭 결과 생성", 
                           "호모그래피를 사용하여 이미지를 스티칭합니다")
    H_canvas, canvas_size, left_offset, overlap_area = compute_canvas_with_translation(left_rectified, right_rectified, H)
    left_blend_mask, right_blend_mask = create_blending_masks(
        left_rectified, right_rectified, H_canvas, canvas_size, left_offset
    )
    final_result, left_canvas, right_transformed = perform_stitching(
        left_rectified, right_rectified, H_canvas, canvas_size, left_offset,
        left_blend_mask, right_blend_mask
    )
    
    create_pipeline_visualization_4(left_rectified, right_rectified, final_result)
    
    # 파라미터 저장 (실시간 코드에서 사용하는 것만)
    print(f"\n💾 실시간 코드용 파라미터 저장")
    save_stitching_parameters(H_canvas, canvas_size, left_offset, left_blend_mask, right_blend_mask)  # 불필요한 파일 생성 안함
    unified_file = save_unified_homography_params(H_canvas, canvas_size, left_offset, config_LC_file, config_CR_file)
    
    # NPY 파일들 생성
    params_dir = './data/params'
    os.makedirs(params_dir, exist_ok=True)
    np.save(os.path.join(params_dir, 'stereo_map_left_x.npy'), config_LC['left_map1_x'])
    np.save(os.path.join(params_dir, 'stereo_map_right_x.npy'), config_CR['right_map1_x'])
    np.save(os.path.join(params_dir, 'left_blend_mask.npy'), left_blend_mask)
    np.save(os.path.join(params_dir, 'right_blend_mask.npy'), right_blend_mask)
    
    # 결과 저장
    save_stitching_results(left_canvas, right_transformed, final_result, 
                          left_blend_mask, right_blend_mask)
    
    print(f"\n🎉 Left ↔ Right 스티칭 파이프라인 완료!")
    print(f"📁 실시간 코드용 파라미터: {unified_file}")
    print(f"📊 중첩 영역: {overlap_area} 픽셀")
    
    return {
        'params_file': unified_file,
        'homography': H,
        'canvas_size': canvas_size,
        'left_offset': left_offset,
        'overlap_area': overlap_area,
        'final_result': final_result
    }

def main():
    """메인 함수: Left ↔ Right 스티칭 파이프라인 실행"""
    print(f"\n🎯 Left ↔ Right stitching engine (간소화 버전)")
    print(f"{'='*60}")
    print(f"💡 method: homography from overlap chessboard + minimal overlap stitching")
    print(f"💡 목적: 실시간 비디오 스티칭을 위한 파라미터 생성")
    print(f"🎯 체크보드 패턴: {CHESSBOARD_SIZE[0]} x {CHESSBOARD_SIZE[1]} (실제 체커보드)")
    print(f"💡 특징점: 중첩 영역의 공통 체크보드 코너로 seamless 연결")
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
            print(f"   • 실시간 코드용 파라미터: {result['params_file']}")
            print(f"   • 결과 이미지: ./data/stitching_results/")
            print(f"   • NPY 파일들: ./data/params/")
            
            # ROI 선택 기능
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
                else:
                    print(f"⚠️ ROI 선택이 취소되었습니다")
            else:
                print(f"ℹ️ ROI 선택을 건너뜁니다")
            
            print(f"\n{'='*60}")
            print(f"🎉 스티칭 엔진 실행 완료!")
            print(f"{'='*60}")
        else:
            print(f"\n❌ 스티칭 파이프라인 실행 실패")
        
    except Exception as e:
        print(f"\n❌ 스티칭 파이프라인 실행 실패: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
