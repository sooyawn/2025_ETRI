import numpy as np

# 카메라 해상도
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# 프레임 속도 (FPS)
FPS = 30

# 카메라 내부 행렬 계산 (focal length는 화면 너비의 1.5배)
focal_length = 1.5 * FRAME_WIDTH
center = (FRAME_WIDTH / 2, FRAME_HEIGHT / 2)
CAMERA_MATRIX = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype=np.float64)

# 왜곡 계수 (초기엔 0, 실제 캘리브레이션 후 값 교체)
DIST_COEFFS = np.zeros((4,1))

# 카메라 설치 각도 (라디안 단위) - 수정됨
# 실제 카메라 설치 각도에 맞게 조정하세요
CAMERA_ANGLES = {
    0: 0.0,       # 카메라0 설치 각도
    1: 0.0        # 카메라1 설치 각도 (0도로 수정)
}

# 카메라 위치 (필요시)
CAMERA_POSITIONS = {
    0: np.array([0, 0, 0]),          # 기준점
    1: np.array([100, 0, 0])         # 예: 오른쪽으로 100mm 떨어진 위치
}
