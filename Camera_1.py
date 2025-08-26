import cv2
import os
import numpy as np

# ========================================
# 체크보드 설정 
# ========================================
CHESSBOARD_SIZE_LC_CR = (6, 9)    # LC, CR 모드용 체크보드 크기 (가로, 세로) - 내부 코너 개수
CHESSBOARD_SIZE_LR = (4, 9)       # LR 모드용 체크보드 크기 (가로, 세로) - 내부 코너 개수

# ========================================
# 카메라 인덱스 설정 
# ========================================
LEFT_CAMERA_INDEX = 2         # 왼쪽 카메라
CENTER_CAMERA_INDEX = 1      # 중앙 카메라
RIGHT_CAMERA_INDEX = 0        # 오른쪽 카메라

# ========================================
# 기본 해상도 설정 
# ========================================
DEFAULT_WIDTH = 1920          # 기본 해상도 너비
DEFAULT_HEIGHT = 1080         # 기본 해상도 높이
DEFAULT_FPS = 60              # 기본 FPS
# ========================================

def get_chessboard_size(mode):
    """모드에 따른 체크보드 크기 반환"""
    if mode == 1:  # LC, CR 모드
        return CHESSBOARD_SIZE_LC_CR
    elif mode == 2:  # LR 모드
        return CHESSBOARD_SIZE_LR
    else:
        return CHESSBOARD_SIZE_LC_CR  # 기본값

def get_available_resolutions(cap):
    """사용 가능한 해상도 목록 반환"""
    resolutions = [
        (3840, 2160),  # 4K
        (1920, 1080),  # Full HD
        (1280, 720),   # HD
        (640, 480)     # VGA
    ]
    
    available = []
    for width, height in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_w == width and actual_h == height:
            available.append((width, height))
    
    return available

def open_cam(index, backend, selected_res=None, selected_fov=None):
    
    # 선택된 백엔드로 카메라 연결
    try:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            print(f"카메라 {index}: 백엔드 {backend} 연결 실패")
            return None
        
        backend_names = {cv2.CAP_MSMF: "Media Foundation", cv2.CAP_DSHOW: "DirectShow"}
        backend_name = backend_names.get(backend, f"Backend {backend}")
        print(f"카메라 {index}: {backend_name} 백엔드로 연결 성공")
        
    except Exception as e:
        print(f"카메라 {index}: 백엔드 {backend} 연결 오류: {e}")
        return None
    
    # 선택된 해상도가 있으면 적용
    if selected_res:
        width, height = selected_res
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        print(f"카메라 {index} 해상도 설정: {width}x{height}")
    else:
        # 자동 해상도 감지
        available_res = get_available_resolutions(cap)
        if available_res:
            best_width, best_height = available_res[0]  # 가장 높은 해상도
            print(f"카메라 {index} 지원 해상도: {available_res}")
            print(f"카메라 {index} 최고 해상도 설정: {best_width}x{best_height}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, best_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best_height)
        else:
            print(f"⚠️ 카메라 {index}: 지원 해상도를 찾을 수 없습니다. 수동으로 해상도를 선택해주세요.")
    
    # 🚀 시야각(FoV) 설정
    if selected_fov is not None:
        try:
            # 시야각 설정 (카메라가 지원하는 경우)
            # 일반적으로 CAP_PROP_ZOOM 또는 카메라별 속성으로 설정
            if hasattr(cap, 'set') and hasattr(cap, 'get'):
                # 시야각별 줌 값 설정 (카메라마다 다를 수 있음)
                if selected_fov == 65:
                    # 65° 시야각 (가장 좁은 시야)
                    cap.set(cv2.CAP_PROP_ZOOM, 1.5)  # 줌 인
                    print(f"📐 카메라 {index} 시야각: 65° (좁은 시야)")
                elif selected_fov == 78:
                    # 78° 시야각 (중간 시야)
                    cap.set(cv2.CAP_PROP_ZOOM, 1.2)  # 약간 줌 인
                    print(f"📐 카메라 {index} 시야각: 78° (중간 시야)")
                elif selected_fov == 90:
                    # 90° 시야각 (가장 넓은 시야)
                    cap.set(cv2.CAP_PROP_ZOOM, 1.0)  # 줌 아웃
                    print(f"📐 카메라 {index} 시야각: 90° (넓은 시야)")
                
                # 시야각 설정 확인
                actual_zoom = cap.get(cv2.CAP_PROP_ZOOM)
                print(f"   실제 줌 값: {actual_zoom:.2f}")
                
        except Exception as e:
            print(f"⚠️ 카메라 {index} 시야각 설정 실패: {e}")
            print("   기본 시야각으로 설정됩니다.")
    
    # 🚀 백엔드별 최적화 설정
    if backend == cv2.CAP_MSMF:
        # Media Foundation 최적화 설정
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 최소화
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        print("🚀 Media Foundation 최적화 설정 적용")
    else:
        # DirectShow 및 기타 백엔드 기본 설정
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        print("🔧 기본 백엔드 설정 적용")
    
    # ========================================
    # 3개 카메라 동일한 설정 적용
    # ========================================
    
    # 노출 설정 (수동 모드)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 수동 노출
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)         # 노출값 (동일하게)
    
    # 밝기 및 대비 설정 (동일하게)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)      # 밝기 중간값
    cap.set(cv2.CAP_PROP_CONTRAST, 128)        # 대비 중간값
    
    # 게인 설정 (동일하게)
    cap.set(cv2.CAP_PROP_GAIN, 0)              # 게인 0 (자연스러운 이미지)
    
    # 추가 카메라 속성 설정 (동일하게)
    cap.set(cv2.CAP_PROP_SATURATION, 128)      # 채도 중간값
    cap.set(cv2.CAP_PROP_HUE, 0)               # 색조 0 (중립)
    cap.set(cv2.CAP_PROP_GAMMA, 128)           # 감마 중간값
    cap.set(cv2.CAP_PROP_SHARPNESS, 128)       # 선명도 중간값
    cap.set(cv2.CAP_PROP_BACKLIGHT, 0)         # 백라이트 보정 0
    
    
    return cap

def read_actual(cap):
    return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS))

def adjust_brightness_to_target(frame, target):
    diff = target - np.mean(frame)
    return cv2.add(frame, np.clip(diff, -50, 50))  # 과조정 방지

def equalize_triplet(a, b, c):
    # 세 프레임 평균 밝기를 동일 목표로 정렬
    target = (np.mean(a)+np.mean(b)+np.mean(c))/3.0
    return (adjust_brightness_to_target(a, target),
            adjust_brightness_to_target(b, target),
            adjust_brightness_to_target(c, target))

def ensure_dirs(base='./data/images'):

    # 폴더 정의
    folder_structure = {
        'pair_LC_left': os.path.join(base, 'pair_LC', 'left'),
        'pair_LC_center': os.path.join(base, 'pair_LC', 'center'),
        'pair_CR_center': os.path.join(base, 'pair_CR', 'center'),
        'pair_CR_right': os.path.join(base, 'pair_CR', 'right'),
        'pair_LR_left': os.path.join(base, 'pair_LR', 'left'),
        'pair_LR_right': os.path.join(base, 'pair_LR', 'right'),
    }
    
    # 필요한 폴더 생성
    for folder_path in folder_structure.values():
        os.makedirs(folder_path, exist_ok=True)
        print(f"폴더 생성: {folder_path}")
    
    return folder_structure


def main():
    print("카메라 초기화 (왼:0, 중:1, 오:2)")
    print("거울모드 자동 활성화")
    
    # 백엔드 선택 메뉴
    print("\n=== 백엔드 선택 ===")
    print("1: Media Foundation (성능)")
    print("2: DirectShow (안정성)")
    
    while True:
        backend_choice = input("백엔드를 선택하세요 (1 또는 2): ").strip()
        if backend_choice in ['1', '2']:
            break
        print("잘못된 선택입니다. 1 또는 2를 입력해주세요.")
    
    # 선택된 백엔드 설정
    if backend_choice == '1':
        selected_backend = cv2.CAP_MSMF
        backend_name = "Media Foundation"
    else:  # backend_choice == '2'
        selected_backend = cv2.CAP_DSHOW
        backend_name = "DirectShow"
    
    print(f"선택된 백엔드: {backend_name}")
    
    # 해상도 선택 메뉴
    print("\n=== 해상도 선택 ===")
    print("1: 640x480 (VGA)")
    print("2: 1280x720 (HD)")
    print("3: 1920x1080 (Full HD)")
    print("4: 3840x2160 (4K)")
    
    while True:
        choice = input("해상도를 선택하세요 (1-4): ").strip()
        if choice in ['1', '2', '3', '4']:
            break
        print("잘못된 선택입니다. 1-4 중에서 선택해주세요.")
    
    # 선택된 해상도 설정
    if choice == '1':
        selected_res = (640, 480)
        res_name = "VGA"
    elif choice == '2':
        selected_res = (1280, 720)
        res_name = "HD"
    elif choice == '3':
        selected_res = (1920, 1080)
        res_name = "Full HD"
    else:  # choice == '4'
        selected_res = (3840, 2160)
        res_name = "4K"
    
    print(f"선택된 해상도: {res_name} ({selected_res[0]}x{selected_res[1]})")

    # 시야각(FoV) 선택 메뉴
    print("\n=== 시야각(FoV) 선택 ===")
    print("1: 65° (좁은 시야 - 가장 좁게 보임)")
    print("2: 78° (중간 시야 - 기본값)")
    print("3: 90° (넓은 시야 - 가장 넓게 보임)")
    
    while True:
        fov_choice = input("시야각을 선택하세요 (1, 2, 3): ").strip()
        if fov_choice in ['1', '2', '3']:
            break
        print("잘못된 선택입니다. 1, 2, 3 중에서 선택해주세요.")
    
    # 선택된 시야각 설정
    if fov_choice == '1':
        selected_fov = 65
        fov_name = "65° (좁은 시야)"
    elif fov_choice == '2':
        selected_fov = 78
        fov_name = "78° (중간 시야)"
    else:  # fov_choice == '3'
        selected_fov = 90
        fov_name = "90° (넓은 시야)"
    
    print(f"선택된 시야각: {fov_name}")

    # 거울모드 자동 활성화
    mirror_mode = True

    cap_L = open_cam(LEFT_CAMERA_INDEX, selected_backend, selected_res, selected_fov)  # 왼쪽 카메라
    cap_C = open_cam(CENTER_CAMERA_INDEX, selected_backend, selected_res, selected_fov)  # 중앙 카메라
    cap_R = open_cam(RIGHT_CAMERA_INDEX, selected_backend, selected_res, selected_fov)  # 오른쪽 카메라

    if any([c is None for c in (cap_L, cap_C, cap_R)]):
        print("3대 모두 열리지 않았습니다. 인덱스/연결 상태 확인")
        for c in (cap_L, cap_C, cap_R):
            if c: c.release()
        return

    wL,hL,fpsL = read_actual(cap_L)
    wC,hC,fpsC = read_actual(cap_C)
    wR,hR,fpsR = read_actual(cap_R)
    
    print("\n=== 카메라 설정 정보 ===")
    print(f"선택된 백엔드: {backend_name}")
    print(f"선택된 해상도: {res_name}")
    print(f"선택된 시야각: {fov_name}")
    print(f"왼쪽 카메라:  {wL}x{hL} @ {fpsL:.1f}fps")
    print(f"중앙 카메라:  {wC}x{hC} @ {fpsC:.1f}fps")
    print(f"오른쪽 카메라:{wR}x{hR} @ {fpsR:.1f}fps")
    
    # 해상도 일치 여부 확인
    if wL == selected_res[0] and hL == selected_res[1]:
        print("요청한 해상도와 일치합니다!")
    else:
        print(f"요청한 해상도({selected_res[0]}x{selected_res[1]})와 실제 해상도({wL}x{hL})가 다릅니다")
    
    # 카메라별 해상도 일치 여부
    if wL == wC == wR and hL == hC == hR:
        print("모든 카메라 해상도가 일치합니다.")
    else:
        print("카메라별 해상도가 다릅니다. 동일하게 설정하세요.")
        print(f"  - 왼쪽: {wL}x{hL}")
        print(f"  - 중앙: {wC}x{hC}")
        print(f"  - 오른쪽: {wR}x{hR}")

    paths = ensure_dirs('./data/images')
    
    # Left-Center와 Center-Right에 대해 별도 인덱스 관리
    def get_next_index_pair(folder_path):
        """특정 폴더의 다음 인덱스 반환"""
        if not os.path.isdir(folder_path): return 0
        nums = []
        for f in os.listdir(folder_path):
            if f.lower().endswith(('.png','.jpg','.jpeg')):
                try:
                    n = int(os.path.splitext(f)[0].replace('img',''))
                    nums.append(n)
                except:
                    pass
        return max(nums) + 1 if nums else 0
    
    # 초기 인덱스 설정 - 카메라 새로 켤 때마다 0번부터 시작
    idx_LC = 0    # Left-Center 인덱스 (0번부터 시작)
    idx_CR = 0    # Center-Right 인덱스 (0번부터 시작)
    idx_LR = 0    # Left-Right 인덱스 (0번부터 시작)
    
    print(f"이미지 인덱스 초기화: LC={idx_LC}, CR={idx_CR}, LR={idx_LR}")
    print("이전 이미지가 있다면 덮어쓰기됩니다!")
    
    # 모드 선택 (프로그램 시작 시)
    print(f"\n=== 촬영 모드 선택 ===")
    print(f"1: 모드 1 - LC, CR 동시 표시 + 촬영 (기존 기능)")
    print(f"2: 모드 2 - Left-Right 연결용 이미지 촬영")
    
    while True:
        mode_choice = input("모드를 선택하세요 (1 또는 2): ").strip()
        if mode_choice in ['1', '2']:
            break
        print("잘못된 선택입니다. 1 또는 2를 입력해주세요.")
    
    selected_mode = int(mode_choice)
    if selected_mode == 1:
        print(f"모드 1 선택: LC, CR 동시 표시 + 촬영")
        print(f"체크보드 크기: {CHESSBOARD_SIZE_LC_CR[0]}x{CHESSBOARD_SIZE_LC_CR[1]} (내부 코너)")
    else:
        print(f"모드 2 선택: Left-Right 연결용 이미지 촬영")
        print(f"체크보드 크기: {CHESSBOARD_SIZE_LR[0]}x{CHESSBOARD_SIZE_LR[1]} (내부 코너)")
    
    # 선택된 모드에 따른 체크보드 크기 출력
    current_chessboard_size = get_chessboard_size(selected_mode)
    print(f"현재 모드 체크보드 크기: {current_chessboard_size[0]}x{current_chessboard_size[1]}")

    while True:
        okL, frameL = cap_L.read()
        okC, frameC = cap_C.read()
        okR, frameR = cap_R.read()
        if not (okL and okC and okR):
            print("프레임 수신 실패")
            break

        frameL, frameC, frameR = equalize_triplet(frameL, frameC, frameR)

        # 거울모드 적용 (좌우 반전)
        if mirror_mode:
            frameL = cv2.flip(frameL, 1)  # 1 = 좌우 반전
            frameC = cv2.flip(frameC, 1)
            frameR = cv2.flip(frameR, 1)

        # 미리보기 리사이즈 (시스템 해상도 기반)
        width, height = wL, hL  # 첫 번째 카메라 해상도 사용
        disp_w = min(640, width//2)
        disp_h = int(disp_w * height / max(1,width))
        dispL = cv2.resize(frameL, (disp_w, disp_h))
        dispC = cv2.resize(frameC, (disp_w, disp_h))
        dispR = cv2.resize(frameR, (disp_w, disp_h))

        # 라벨 (거울모드 표시)
        mirror_text = " (MIRROR)" if mirror_mode else ""
        cv2.putText(dispL, f"LEFT {wL}x{hL}{mirror_text}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,0),2)
        cv2.putText(dispC, f"CENTER {wC}x{hC}{mirror_text}",(10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,255),2)
        cv2.putText(dispR, f"RIGHT {wR}x{hR}{mirror_text}",(10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,0),2)

        # 모드별 창 표시
        if selected_mode == 1:
            # 모드 1: LC, CR 동시 표시 
            # 1) Left-Center 창
            left_center = cv2.hconcat([dispL, dispC])
            cv2.putText(left_center, f"Left-Center | {wL}x{hL} | {wC}x{hC} (MIRROR)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(left_center, f"Press '1' to capture Left-Center pair (idx: {idx_LC:02d})", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.imshow("Left-Center Pair", left_center)

            # 2) Center-Right 창
            center_right = cv2.hconcat([dispC, dispR])
            cv2.putText(center_right, f"Center-Right | {wC}x{hC} | {wR}x{hR} (MIRROR)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(center_right, f"Press '2' to capture Center-Right pair (idx: {idx_CR:02d})", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.imshow("Center-Right Pair", center_right)

        else:
            # 모드 2: Left-Right 창만 표시 (중간 카메라 제외)
            # 거울모드 상태에 따라 카메라 순서 조정
            if mirror_mode:
                # 거울모드 ON: Left-Right 순서 
                left_right = cv2.hconcat([dispL, dispR])
                mirror_text = " (MIRROR)"
            else:
                # 거울모드 OFF: Right-Left 순서 
                left_right = cv2.hconcat([dispR, dispL])
                mirror_text = ""
            
            cv2.putText(left_right, f"Left-Right | {wL}x{hL} | {wR}x{hR}{mirror_text}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(left_right, f"Press '3' to capture Left-Right pair (idx: {idx_LR:02d})", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.imshow("Left-Right Pair (Center Excluded)", left_right)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('1'):
            try:
                # Left-Center 페어만 저장
                pLC_L = os.path.join(paths['pair_LC_left'],   f'img{idx_LC:02d}.png')
                pLC_C = os.path.join(paths['pair_LC_center'], f'img{idx_LC:02d}.png')
                cv2.imwrite(pLC_L, frameL)
                cv2.imwrite(pLC_C, frameC)
                print(f'Left-Center 페어 저장 완료 idx={idx_LC:02d}')
                print(f'  Pair LC: {pLC_L}, {pLC_C}')
                idx_LC += 1  # Left-Center 인덱스만 증가
            except Exception as e:
                print(f"저장 중 오류 발생: {e}")
                print("프로그램을 계속 실행합니다...")

        elif key == ord('2'):
            try:
                # Center-Right 페어만 저장
                pCR_C = os.path.join(paths['pair_CR_center'], f'img{idx_CR:02d}.png')
                pCR_R = os.path.join(paths['pair_CR_right'],  f'img{idx_CR:02d}.png')
                cv2.imwrite(pCR_C, frameC)
                cv2.imwrite(pCR_R, frameR)
                print(f'Center-Right 페어 저장 완료 idx={idx_CR:02d}')
                print(f'  Pair CR: {pCR_C}, {pCR_R}')
                idx_CR += 1  # Center-Right 인덱스만 증가
            except Exception as e:
                print(f"저장 중 오류 발생: {e}")
                print("프로그램을 계속 실행합니다...")

        elif key == ord('3'):
            try:
                # Left-Right 페어만 저장 (중간 카메라 제외)
                pLR_L = os.path.join(paths['pair_LR_left'],  f'img{idx_LR:02d}.png')
                pLR_R = os.path.join(paths['pair_LR_right'], f'img{idx_LR:02d}.png')
                cv2.imwrite(pLR_L, frameL)
                cv2.imwrite(pLR_R, frameR)
                print(f'Left-Right 페어 저장 완료 idx={idx_LR:02d}')
                print(f'  Pair LR: {pLR_L}, {pLR_R}')
                print(f'  중간 카메라 제외: Left ↔ Right 직접 연결용')
                idx_LR += 1  # Left-Right 인덱스만 증가
            except Exception as e:
                print(f"저장 중 오류 발생: {e}")
                print("프로그램을 계속 실행합니다...")

        elif key == ord('m'):
            # 거울모드 토글
            mirror_mode = not mirror_mode
            if mirror_mode:
                print("거울모드 활성화")
            else:
                print("거울모드 비활성화")

        elif key == ord('q'):
            print("종료 요청 감지...")
            break

    print("카메라 해제 중...")
    cap_L.release(); cap_C.release(); cap_R.release()
    print("윈도우 정리 중...")
    cv2.destroyAllWindows()
    
    # 강제 종료를 위한 추가 처리
    for i in range(5):
        cv2.waitKey(1)
        cv2.destroyAllWindows()
    
    print("종료 완료.")
    print(f"Left-Center 마지막 인덱스: {idx_LC-1:02d}")
    print(f"Center-Right 마지막 인덱스: {idx_CR-1:02d}")
    print(f"Left-Right 마지막 인덱스: {idx_LR-1:02d}")
    return  # main 함수 정상 종료

if __name__ == "__main__":
    main()
