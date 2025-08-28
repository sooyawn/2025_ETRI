import cv2
import numpy as np
import os
import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime


# ========================================
# 설정 (사용자가 쉽게 수정 가능)
# ========================================
CHESSBOARD_SIZE = (9, 6)      # 체크보드 크기 (가로, 세로) - 내부 코너 개수
SQUARE_SIZE = 0.05          # 체크보드 정사각형 한 변의 크기 (미터)

# 카메라 인덱스 설정
LEFT_CAMERA_INDEX = 2         # 왼쪽 카메라
CENTER_CAMERA_INDEX = 1       # 중앙 카메라 (캘리브레이션용)
RIGHT_CAMERA_INDEX = 0        # 오른쪽 카메라

# 이미지 품질 설정
MIN_IMAGES_FOR_CALIBRATION = 5  # 최소 캘리브레이션 이미지 수
CORNER_ACCURACY = 0.001        # 코너 검출 정확도


def load_image_pairs(left_dir, right_dir, pair_name):
    """좌우 카메라 이미지 쌍을 로드합니다."""
    print(f"\n📁 [{pair_name}] 이미지 로딩 시작")
    print(f"   왼쪽 카메라: {left_dir}")
    print(f"   오른쪽 카메라: {right_dir}")
    
    # PNG와 JPG 모두 지원
    left_images = sorted(glob.glob(os.path.join(left_dir, "*.png")) + glob.glob(os.path.join(left_dir, "*.jpg")))
    right_images = sorted(glob.glob(os.path.join(right_dir, "*.png")) + glob.glob(os.path.join(right_dir, "*.jpg")))
    
    if len(left_images) != len(right_images):
        raise ValueError(f"이미지 개수가 다릅니다: 왼쪽 {len(left_images)}, 오른쪽 {len(right_images)}")
    
    if len(left_images) < MIN_IMAGES_FOR_CALIBRATION:
        raise ValueError(f"캘리브레이션을 위한 이미지가 부족합니다: {len(left_images)}개 (최소 {MIN_IMAGES_FOR_CALIBRATION}개 필요)")
    
    print(f"   ✅ 총 {len(left_images)}쌍의 이미지 로드 완료")
    return left_images, right_images


def detect_chessboard_corners(images, pattern_size, camera_name):
    """체크보드 코너를 찾습니다. (강화된 검출 알고리즘)"""
    print(f"\n🔍 [{camera_name}] 체크보드 코너 검출 시작 (강화된 검출 알고리즘):")
    print(f"   패턴 크기: {pattern_size[0]} x {pattern_size[1]}")
    print(f"   정사각형 크기: {SQUARE_SIZE}m")
    
    objpoints = []
    imgpoints = []
    
    # corners 폴더 생성
    corners_dir = f"./data/corners/{camera_name}"
    os.makedirs(corners_dir, exist_ok=True)
    print(f"   📁 코너 검출 이미지 저장 경로: {corners_dir}")
    
    for i, img_path in enumerate(images):
        print(f"   이미지 {i+1}/{len(images)}: {os.path.basename(img_path)}")
        
        # 이미지 로드
        img = cv2.imread(img_path)
        if img is None:
            print(f"     ❌ 이미지 로드 실패: {img_path}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"     이미지 크기: {img.shape[1]} x {img.shape[0]}")
        
        # 강화된 코너 검출 시도
        corners = None
        ret = False
        
        # 1. 기본 검출 시도
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            print(f"     ✅ 기본 검출 성공: {len(corners)}개")
        else:
            print(f"     ⚠️ 기본 검출 실패, 고급 검출 시도 중...")
            
            # 2. 이미지 전처리 후 재시도
            # 가우시안 블러로 노이즈 제거
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 히스토그램 평활화로 대비 향상
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(blurred)
            
            # 여러 검출 옵션 시도
            detection_methods = [
                ("기본", gray),
                ("블러", blurred),
                ("대비향상", enhanced),
                ("가우시안", cv2.GaussianBlur(gray, (3, 3), 0)),
                ("중간값필터", cv2.medianBlur(gray, 5))
            ]
            
            for method_name, processed_img in detection_methods:
                # 다양한 검출 플래그 시도
                flags_list = [
                    cv2.CALIB_CB_ADAPTIVE_THRESH,
                    cv2.CALIB_CB_NORMALIZE_IMAGE,
                    cv2.CALIB_CB_FAST_CHECK,
                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK,
                    cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
                ]
                
                for flags in flags_list:
                    ret, corners = cv2.findChessboardCorners(processed_img, pattern_size, flags)
                    if ret:
                        print(f"     ✅ {method_name} + 플래그 {flags} 성공: {len(corners)}개")
                        break
                
                if ret:
                    break
            
            # 3. 여전히 실패하면 이미지 크기 조정 시도
            if not ret:
                print(f"     ⚠️ 고급 검출도 실패, 이미지 크기 조정 시도 중...")
                
                # scale 변수 초기화
                scale = 1.0
                
                # 이미지를 0.5배, 0.75배, 1.5배, 2배로 조정하여 시도
                scales = [0.5, 0.75, 1.5, 2.0]
                for scale in scales:
                    if scale != 1.0:
                        h, w = gray.shape
                        new_h, new_w = int(h * scale), int(w * scale)
                        resized = cv2.resize(gray, (new_w, new_h))
                        
                        ret, corners = cv2.findChessboardCorners(resized, pattern_size, 
                                                               cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
                        if ret:
                            # 원본 크기로 좌표 변환
                            corners = corners / scale
                            print(f"     ✅ 크기 조정 {scale}x 성공: {len(corners)}개")
                            break
        
        if ret and corners is not None:
            # 서브픽셀 정확도 향상
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # 3D 점 좌표 생성 (올바른 형식)
            objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
            objp = objp * SQUARE_SIZE
            
            # 데이터 형식 강제 변환 (OpenCV 호환성)
            objp = np.asarray(objp, dtype=np.float32)
            corners = np.asarray(corners, dtype=np.float32)
            
            objpoints.append(objp)
            imgpoints.append(corners)
            print(f"     ✅ 최종 코너 검출 성공: {len(corners)}개")
            
            # 코너가 검출된 이미지 저장
            img_with_corners = img.copy()
            cv2.drawChessboardCorners(img_with_corners, pattern_size, corners, True)
            
            # 파일명 생성
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            corner_filename = f"{base_name}_corners.png"
            corner_path = os.path.join(corners_dir, corner_filename)
            
            # 이미지 저장
            cv2.imwrite(corner_path, img_with_corners)
            print(f"     💾 코너 검출 이미지 저장: {corner_filename}")
            
        else:
            print(f"     ❌ 모든 검출 방법 실패")
    
    print(f"   📊 [{camera_name}] 전체 결과: {len(objpoints)}/{len(images)} 이미지에서 코너 검출 성공")
    
    if len(objpoints) == 0:
        print(f"   🚨 [{camera_name}] 모든 이미지에서 코너 검출 실패!")
        print(f"   💡 체크보드 크기를 확인하고 이미지를 다시 촬영해주세요.")
        print(f"   💡 현재 설정: {pattern_size[0]} x {pattern_size[1]} (내부 코너 개수)")
        print(f"   🔍 디버깅을 위해 첫 번째 이미지를 저장합니다...")
        
        # 첫 번째 이미지 저장하여 디버깅
        if len(images) > 0:
            debug_img = cv2.imread(images[0])
            debug_path = f"./debug_{camera_name}_first_image.png"
            cv2.imwrite(debug_path, debug_img)
            print(f"   💾 디버그 이미지 저장: {debug_path}")
    
    return objpoints, imgpoints


def calibrate_single_camera(objpoints, imgpoints, image_size, camera_name):
    """단일 카메라 캘리브레이션을 수행합니다."""
    print(f"\n📷 [{camera_name}] 단일 카메라 캘리브레이션 시작")
    
    # 카메라 캘리브레이션
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    
    if not ret:
        raise RuntimeError(f"{camera_name} 카메라 캘리브레이션 실패")
    
    # 재투영 오차 계산
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    mean_error /= len(objpoints)
    
    print(f"   ✅ 캘리브레이션 완료")
    print(f"   📐 내부 파라미터 행렬 크기: {mtx.shape}")
    print(f"   📏 재투영 오차: {mean_error:.6f} 픽셀")
    
    return mtx, dist, mean_error


def calibrate_stereo_camera(left_objpoints, left_imgpoints, right_objpoints, right_imgpoints,
                             left_mtx, left_dist, right_mtx, right_dist, image_size, pair_name):
    """스테레오 카메라 캘리브레이션을 수행합니다."""
    print(f"\n🔗 [{pair_name}] 스테레오 카메라 캘리브레이션 시작")
    
    # 공통 이미지 찾기
    common_images = min(len(left_objpoints), len(right_objpoints))
    if common_images < 1:
        raise RuntimeError("공통 이미지가 부족합니다")
    
    print(f"   공통 이미지 개수: {common_images}")
    print(f"   이미지 크기: {image_size}")
    
    # 데이터 형식 검증 및 디버깅
    print(f"   🔍 데이터 형식 검증:")
    print(f"      - left_objpoints[0] 형태: {left_objpoints[0].shape}, 타입: {left_objpoints[0].dtype}")
    print(f"      - left_imgpoints[0] 형태: {left_imgpoints[0].shape}, 타입: {left_imgpoints[0].dtype}")
    print(f"      - right_objpoints[0] 형태: {right_objpoints[0].shape}, 타입: {right_objpoints[0].dtype}")
    print(f"      - right_imgpoints[0] 형태: {right_imgpoints[0].shape}, 타입: {right_imgpoints[0].dtype}")
    
    # 데이터 복사본 생성 (메모리 문제 방지)
    left_objpoints_copy = [objp.copy() for objp in left_objpoints[:common_images]]
    left_imgpoints_copy = [imgp.copy() for imgp in left_imgpoints[:common_images]]
    right_objpoints_copy = [objp.copy() for objp in right_objpoints[:common_images]]
    right_imgpoints_copy = [imgp.copy() for imgp in right_imgpoints[:common_images]]
    
    print(f"   ✅ 데이터 복사 완료")
    
    # 스테레오 캘리브레이션
    ret, left_mtx, left_dist, right_mtx, right_dist, R, T, E, F = cv2.stereoCalibrate(
        left_objpoints_copy, 
        left_imgpoints_copy, 
        right_imgpoints_copy,  # ✅ right_objpoints_copy → right_imgpoints_copy로 수정
        left_mtx, left_dist, right_mtx, right_dist,
        image_size
    )
    
    if not ret:
        raise RuntimeError(f"{pair_name} 스테레오 캘리브레이션 실패")
    
    print(f"   ✅ 스테레오 캘리브레이션 완료")
    print(f"   📐 회전 행렬 R: {R.shape}")
    print(f"   📏 변환 벡터 T: {T.shape}")
    print(f"   📊 본질 행렬 E: {E.shape}")
    print(f"   📊 기본 행렬 F: {F.shape}")
    
    return R, T, E, F


def rectify_images(left_mtx, left_dist, right_mtx, right_dist, R, T, 
                                left_img, right_img, pair_name):
    """기존 Calibration.py와 동일한 방식으로 이미지를 정렬합니다."""
    print(f"\n🔄 [{pair_name}] 기존 방식 정렬 시작")
    
    h, w = left_img.shape[:2]
    image_size = (w, h)
    
    print(f"   📐 원본 이미지 크기: {w} x {h}")
    
    # 기존 코드와 동일한 스테레오 정렬 방식
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        left_mtx, left_dist, right_mtx, right_dist,
        image_size, R, T,
        alpha=1.0,              # alpha=1.0 (모든 픽셀 보존, 과도한 확대 방지)
        newImageSize=(0, 0),    # 원본 크기 유지
    )
    
    print(f"   📊 ROI 정보:")
    print(f"      - ROI1: {roi1}")
    print(f"      - ROI2: {roi2}")
    
    # 기존 코드와 동일한 정렬 맵 생성 방식
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        left_mtx, left_dist, R1, P1, image_size, cv2.CV_16SC2
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        right_mtx, right_dist, R2, P2, image_size, cv2.CV_16SC2
    )
    
    print(f"   ✅ 정렬 맵 생성 완료:")
    print(f"   - 왼쪽 맵: {left_map1.shape}")
    print(f"   - 오른쪽 맵: {right_map1.shape}")
    
    # 이미지 정렬 적용
    left_rect = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    right_rect = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    
    print(f"   ✅ 정렬 완료")
    print(f"   📐 정렬 후 이미지 크기: 왼쪽 {left_rect.shape}, 오른쪽 {right_rect.shape}")
    
    return (left_rect, right_rect, left_map1, left_map2, right_map1, right_map2, roi1, roi2)


def save_calibration_data(left_mtx, left_dist, right_mtx, right_dist, R, T, E, F,
                          left_map1, left_map2, right_map1, right_map2, roi1, roi2,
                          pair_name, output_dir):
    """캘리브레이션 데이터를 저장합니다."""
    print(f"\n💾 [{pair_name}] 캘리브레이션 데이터 저장 시작")
    
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON 설정 파일 저장
    config_data = {
        'pair_name': pair_name,
        'camera_resolution': {
            'width': 1920,
            'height': 1080
        },
        'chessboard_size': CHESSBOARD_SIZE,
        'square_size': SQUARE_SIZE,
        'calibration_date': datetime.now().isoformat(),
        'left_camera': {
            'intrinsic_matrix': left_mtx.tolist(),
            'distortion_coefficients': left_dist.tolist()
        },
        'right_camera': {
            'intrinsic_matrix': right_mtx.tolist(),
            'distortion_coefficients': right_dist.tolist()
        },
        'stereo_calibration': {
            'rotation_matrix': R.tolist(),
            'translation_vector': T.tolist(),
            'essential_matrix': E.tolist(),
            'fundamental_matrix': F.tolist()
        },
        'rectification': {
            'roi1': roi1.tolist() if hasattr(roi1, 'tolist') else roi1,
            'roi2': roi2.tolist() if hasattr(roi2, 'tolist') else roi2
        }
    }
    
    config_file = os.path.join(output_dir, f"{pair_name}_calibration_config.json")
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"   💾 설정 파일 저장: {config_file}")
    
    # NPY 파일로 정렬 맵 저장
    maps_file = os.path.join(output_dir, f"{pair_name}_rectification_maps.npz")
    np.savez_compressed(maps_file,
                        left_map1_x=left_map1,
                        left_map1_y=left_map2,
                        right_map1_x=right_map1,
                        right_map1_y=right_map2)
    
    print(f"   💾 정렬 맵 저장: {maps_file}")
    
    return config_file, maps_file
 
 
def visualize_final_results(config_LC, config_CR):
     """LC, CR 쌍의 최종 결과를 시각화합니다."""
     print(f"\n🎨 최종 결과 시각화 시작")
     
     # LC 쌍의 정렬 맵 로드
     LC_maps = np.load(config_LC['maps_file'])
     left_map1 = LC_maps['left_map1_x']
     left_map2 = LC_maps['left_map1_y']
     
     # CR 쌍의 정렬 맵 로드
     CR_maps = np.load(config_CR['maps_file'])
     right_map1 = CR_maps['right_map1_x']
     right_map2 = CR_maps['right_map1_y']
     
     # 테스트 이미지 로드 (첫 번째 이미지 사용)
     left_img = cv2.imread('./data/images/pair_LC/left/img00.png')
     right_img = cv2.imread('./data/images/pair_CR/right/img00.png')
     
     if left_img is None or right_img is None:
         print("   ❌ test images not found")
         return
     
     print(f"   📐 원본 이미지 크기: 왼쪽 {left_img.shape}, 오른쪽 {right_img.shape}")
     
     # 이미지 정렬 적용
     left_rect = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
     right_rect = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
     
     print(f"   ✅ 정렬 완료")
     print(f"   📐 정렬 후 이미지 크기: 왼쪽 {left_rect.shape}, 오른쪽 {right_rect.shape}")
     
     # 화면 표시용 크기 조정 (원본은 1920x1080 유지)
     scale_factor = 0.4
     display_width = int(1920 * scale_factor)
     display_height = int(1080 * scale_factor)
     
     # 화면 표시용 리사이즈
     left_orig_display = cv2.resize(left_img, (display_width, display_height))
     right_orig_display = cv2.resize(right_img, (display_width, display_height))
     left_rect_display = cv2.resize(left_rect, (display_width, display_height))
     right_rect_display = cv2.resize(right_rect, (display_width, display_height))
     
     # 제목 추가 (영문 표기)
     font_scale = 0.6
     thickness = 2
     
     cv2.putText(left_orig_display, "Original Left", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
     cv2.putText(right_orig_display, "Original Right", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
     cv2.putText(left_rect_display, "Rectified Left (LC)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
     cv2.putText(right_rect_display, "Rectified Right (CR)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
     
     # 이미지 결합 (2x2 그리드)
     top_row = np.hstack([left_orig_display, right_orig_display])
     bottom_row = np.hstack([left_rect_display, right_rect_display])
     combined = np.vstack([top_row, bottom_row])
     
     # 결과 저장
     output_dir = "./data/outputs"
     os.makedirs(output_dir, exist_ok=True)
     
     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
     output_path = os.path.join(output_dir, f"Final_LC_CR_Results_{timestamp}.png")
     
     cv2.imwrite(output_path, combined)
     print(f"   💾 최종 시각화 결과 저장: {output_path}")
     print(f"   📐 화면 표시 크기: {display_width} x {display_height}")
     
     # 화면에 표시 (표시 실패는 무시)
     try:
         cv2.imshow("Final LC-CR Calibration Results", combined)
         cv2.waitKey(0)
         cv2.destroyAllWindows()
     except Exception as e:
         print(f"   ⚠️ 화면 표시 실패: {e}")


def visualize_final_results(config_LC, config_CR):
    """LC, CR 쌍의 최종 결과를 시각화합니다."""
    print(f"\n🎨 최종 결과 시각화 시작")
    
    # LC 쌍의 정렬 맵 로드
    LC_maps = np.load(config_LC['maps_file'])
    left_map1_LC = LC_maps['left_map1_x']
    left_map2_LC = LC_maps['left_map1_y']
    
    # CR 쌍의 정렬 맵 로드  
    CR_maps = np.load(config_CR['maps_file'])
    right_map1_CR = CR_maps['right_map1_x']
    right_map2_CR = CR_maps['right_map1_y']
    
    # 테스트 이미지 로드 (첫 번째 이미지 사용)
    left_img = cv2.imread('./data/images/pair_LC/left/img00.png')
    right_img = cv2.imread('./data/images/pair_CR/right/img00.png')
    
    if left_img is None or right_img is None:
        print("   ❌ 테스트 이미지 로드 실패")
        return
    
    print(f"   📐 원본 이미지 크기: 왼쪽 {left_img.shape}, 오른쪽 {right_img.shape}")
    
    # 이미지 정렬 적용
    left_rect = cv2.remap(left_img, left_map1_LC, left_map2_LC, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    right_rect = cv2.remap(right_img, right_map1_CR, right_map2_CR, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
    
    print(f"   ✅ 정렬 완료")
    print(f"   📐 정렬 후 이미지 크기: 왼쪽 {left_rect.shape}, 오른쪽 {right_rect.shape}")
    
    # 화면 표시용 크기 조정 (원본은 1920x1080 유지)
    scale_factor = 0.4
    display_width = int(1920 * scale_factor)
    display_height = int(1080 * scale_factor)
    
    # 화면 표시용 리사이즈
    left_orig_display = cv2.resize(left_img, (display_width, display_height))
    right_orig_display = cv2.resize(right_img, (display_width, display_height))
    left_rect_display = cv2.resize(left_rect, (display_width, display_height))
    right_rect_display = cv2.resize(right_rect, (display_width, display_height))
    
    # 제목 추가
    font_scale = 0.6
    thickness = 2
    
    cv2.putText(left_orig_display, "Original Left", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    cv2.putText(right_orig_display, "Original Right", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    cv2.putText(left_rect_display, "Rectified Left (LC)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
    cv2.putText(right_rect_display, "Rectified Right (CR)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
    
    # 이미지 결합 (2x2 그리드)
    top_row = np.hstack([left_orig_display, right_orig_display])
    bottom_row = np.hstack([left_rect_display, right_rect_display])
    combined = np.vstack([top_row, bottom_row])
    
    # 결과 저장
    output_dir = "./data/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"Final_LC_CR_Results_{timestamp}.png")
    
    cv2.imwrite(output_path, combined)
    print(f"   💾 최종 시각화 결과 저장: {output_path}")
    print(f"   📐 원본 크기 유지: 1920 x 1080")
    print(f"   📐 화면 표시 크기: {display_width} x {display_height}")
    
    # 화면에 표시
    try:
        cv2.imshow("Final LC-CR Calibration Results", combined)
        print(f"   👁️ 최종 결과를 화면에 표시합니다.")
        print(f"   💡 아무 키나 누르면 창이 닫힙니다.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"   ⚠️ 화면 표시 실패: {e}")
    
    print(f"   🎯 최종 결과 요약:")
    print(f"      • LC 쌍: Left ↔ Center 정렬 맵 생성 완료")
    print(f"      • CR 쌍: Center ↔ Right 정렬 맵 생성 완료")
    print(f"      • 중앙 카메라: 캘리브레이션 용도로만 사용")
    print(f"      • 다음 단계: Stitching_Engine.py에서 Left ↔ Right 직접 연결")


def calibrate_camera_pair(left_dir, right_dir, pair_name):
    """한 쌍의 카메라에 대해 캘리브레이션을 수행합니다."""
    print(f"\n{'='*60}")
    print(f"🎯 {pair_name} 쌍 캘리브레이션 시작")
    print(f"{'='*60}")
    
    # 1. 이미지 로드
    print(f"\n📸 1단계: 이미지 로드")
    left_images, right_images = load_image_pairs(left_dir, right_dir, pair_name)
    
    # 2. 코너 검출
    print(f"\n🔍 2단계: 코너 검출")
    left_objpoints, left_imgpoints = detect_chessboard_corners(left_images, CHESSBOARD_SIZE, f"{pair_name}_Left")
    right_objpoints, right_imgpoints = detect_chessboard_corners(right_images, CHESSBOARD_SIZE, f"{pair_name}_Right")
    
    # 3. 단일 카메라 캘리브레이션
    print(f"\n📷 3단계: 단일 카메라 캘리브레이션")
    left_img = cv2.imread(left_images[0])
    image_size = (left_img.shape[1], left_img.shape[0])
    
    left_mtx, left_dist, left_error = calibrate_single_camera(
        left_objpoints, left_imgpoints, image_size, f"{pair_name}_Left"
    )
    right_mtx, right_dist, right_error = calibrate_single_camera(
        right_objpoints, right_imgpoints, image_size, f"{pair_name}_Right"
    )
    
    # 4. 스테레오 캘리브레이션
    print(f"\n🔗 4단계: 스테레오 캘리브레이션")
    R, T, E, F = calibrate_stereo_camera(
        left_objpoints, left_imgpoints, right_objpoints, right_imgpoints,
        left_mtx, left_dist, right_mtx, right_dist, image_size, pair_name
    )
    
    # 5. 이미지 정렬
    print(f"\n🔄 5단계: 이미지 정렬")
    right_img = cv2.imread(right_images[0])
    
    left_rect, right_rect, left_map1, left_map2, right_map1, right_map2, roi1, roi2 = \
        rectify_images(left_mtx, left_dist, right_mtx, right_dist, R, T, 
                                   left_img, right_img, pair_name)
    
    # 6. 데이터 저장
    print(f"\n💾 6단계: 데이터 저장")
    output_dir = "./data/config"
    config_file, maps_file = save_calibration_data(
        left_mtx, left_dist, right_mtx, right_dist, R, T, E, F,
        left_map1, left_map2, right_map1, right_map2, roi1, roi2,
        pair_name, output_dir
    )
    
    print(f"\n✅ {pair_name} 쌍 캘리브레이션 완료!")
    print(f"   📁 설정 파일: {config_file}")
    print(f"   📁 정렬 맵: {maps_file}")
    
    return {
        'pair_name': pair_name,
        'config_file': config_file,
        'maps_file': maps_file,
        'left_mtx': left_mtx,
        'left_dist': left_dist,
        'right_mtx': right_mtx,
        'right_dist': right_dist,
        'R': R,
        'T': T,
        'E': E,
        'F': F,
        'left_map1': left_map1,
        'left_map2': left_map2,
        'right_map1': right_map1,
        'right_map2': right_map2,
        'roi1': roi1,
        'roi2': roi2
    }


def main():
    """메인 함수: LC, CR 쌍 독립적 캘리브레이션"""
    print(f"\n🎯 3카메라 캘리브레이션 시작")
    print(f"   체크보드 크기: {CHESSBOARD_SIZE[0]} x {CHESSBOARD_SIZE[1]}")
    print(f"   정사각형 크기: {SQUARE_SIZE}m")
    print(f"   �� 목표: LC, CR 쌍 독립적 캘리브레이션으로 정렬 맵 생성")
    print(f"   🎯 최종 목표: Stitching_Engine.py에서 Left ↔ Right 직접 연결")
    
    try:
        # 1. LC 쌍 캘리브레이션 (Left ↔ Center)
        print(f"\n�� 1단계: Left-Center 쌍 캘리브레이션")
        config_LC = calibrate_camera_pair(
            left_dir='./data/images/pair_LC/left',
            right_dir='./data/images/pair_LC/center',
            pair_name='LC'
        )
        
        # 2. CR 쌍 캘리브레이션 (Center ↔ Right)
        print(f"\n📷 2단계: Center-Right 쌍 캘리브레이션")
        config_CR = calibrate_camera_pair(
            left_dir='./data/images/pair_CR/center',
            right_dir='./data/images/pair_CR/right',
            pair_name='CR'
        )
        
        # 3. 최종 결과 시각화 (Left ↔ Right)
        print(f"\n🎨 3단계: 최종 결과 시각화 (Left ↔ Right)")
        visualize_final_results(config_LC, config_CR)
        
        print(f"\n{'='*60}")
        print(f"🎉 전체 캘리브레이션 완료!")
        print(f"{'='*60}")
        print(f"�� 생성된 파일들:")
        print(f"   • LC 쌍: {config_LC['config_file']}")
        print(f"   • LC 맵: {config_LC['maps_file']}")
        print(f"   • CR 쌍: {config_CR['config_file']}")
        print(f"   • CR 맵: {config_CR['maps_file']}")
        
        print(f"\n🚀 다음 단계:")
        print(f"   • Stitching_Engine.py로 Left ↔ Right 직접 연결")
        print(f"   • 중앙 카메라 제거 및 최소 중첩 영역 스티칭")
        print(f"   �� 각 쌍 독립적 캘리브레이션으로 안정적인 정렬 맵 생성!")
        print(f"{'='*60}")
        
        return config_LC, config_CR
        
    except Exception as e:
        print(f"\n❌ 캘리브레이션 실패: {e}")
        print(f"💡 해결 방법:")
        print(f"   1. 이미지 품질 확인: 선명하게, 충분한 조명으로")
        print(f"   2. 체크보드가 이미지에 완전히 보이도록 촬영")
        print(f"   3. 체크보드 크기 확인: 현재 {CHESSBOARD_SIZE[0]} x {CHESSBOARD_SIZE[1]}")
        raise e


if __name__ == '__main__':
    # 사용 예시
    try:
        config_LC, config_CR = main()
        print("✅ 모든 과정 완료")
        print(f"📁 LC 설정: {config_LC['config_file']}")
        print(f"📁 CR 설정: {config_CR['config_file']}")
    except Exception as e:
        print(f"❌ 최종 오류: {e}")