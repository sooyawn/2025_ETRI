# Eyetracking Dual‑Camera Pipeline (1→2→3→4)

이 프로젝트는 **두 대의 카메라**로 데이터를 수집하고(1), **캘리브레이션**(2), **스티칭 파라미터 산출**(3), 마지막으로 **실시간 스티칭 + 아이트래킹**(4)을 수행합니다.  
파일 이름 끝의 숫자(1, 2, 3, 4) 순서대로 실행하면 됩니다.

---

## 0) 환경 준비

### 가상환경 & 설치
```bash
# 프로젝트 루트(이 README가 있는 위치)에서
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip



pip install -r requirements.txt
```

> **Windows에서 MediaPipe 오류**가 나면 `protobuf==3.20.*`를 추가로 설치하세요.
>
> ```bash
> pip install "protobuf==3.20.*"
> ```

---

## 1) 프로젝트 구조 & 필요한 파일

실행 전 기본 폴더 구조:

```text
project_root/
├─ Camera_1.py                # (1) 데이터 수집
├─ Calibration_2.py           # (2) 캘리브레이션 (LC, CR 쌍)
├─ Stitching_3.py             # (3) 스티칭 파라미터/ROI 생성
├─ Eyetracking_4.py           # (4) 실시간 스티칭 + 아이트래킹
├─ models/
│  └─ yolov8n-face.pt         # YOLO 얼굴 모델 (저장되어 있어야 함)
└─ data/
   ├─ images/
   │  ├─ pair_LC/
   │  │  ├─ left/             # LC 쌍의 Left 이미지들
   │  │  └─ center/           # LC 쌍의 Center 이미지들
   │  ├─ pair_CR/
   │  │  ├─ center/           # CR 쌍의 Center 이미지들
   │  │  └─ right/            # CR 쌍의 Right 이미지들
   │  └─ pair_LR/
   │     ├─ left/
   │     │   └─ img00.png     # LR 스티칭에 사용할 샘플 (최소 1장 필요)
   │     └─ right/
   │         └─ img00.png
   ├─ config/                 # 스크립트가 생성
   ├─ params/                 # 스크립트가 생성
   └─ stitching_results/      # 스크립트가 생성
```

각 스크립트는 상단에 **카메라 인덱스**(LEFT/RIGHT/CENTER)가 하드코딩되어 있습니다.
- 기본값(예): `LEFT=2, CENTER=1, RIGHT=0`

---

## 2) 실행 순서

### (1) Camera_1.py — 데이터 수집
- **목적**: 캘리브레이션용 이미지(LC, CR)와 LR 샘플 이미지 촬영/저장
- **실행**
  ```bash
  python Camera_1.py
  ```
- **결과**: `./data/images/` 아래에 다음 폴더들이 채워집니다.
  - `pair_LC/left`, `pair_LC/center` — **최소 5장 이상**의 체크보드 이미지
  - `pair_CR/center`, `pair_CR/right` — **최소 5장 이상**
  - `pair_LR/left/img00.png`, `pair_LR/right/img00.png` — LR 샘플 1쌍

---

### (2) Calibration_2.py — 캘리브레이션
- **목적**: LC(Left–Center), CR(Center–Right) **각 쌍 독립 캘리브레이션** 및 정렬 맵 생성
- **입력**: (1)에서 저장한 `pair_LC`, `pair_CR` 폴더 이미지
- **실행**
  ```bash
  python Calibration_2.py
  ```
- **출력**
  - `./data/config/LC_calibration_config.json`
  - `./data/config/CR_calibration_config.json`
  - `./data/config/LC_rectification_maps.npz`
  - `./data/config/CR_rectification_maps.npz`
  - 시각화 이미지: `./data/outputs/Final_LC_CR_Results_*.png`
- 참고: 내부에서 OpenCV의 `findChessboardCorners(SB)` 등 **강화된 검출**을 사용합니다.
  (코너 검출 실패 시 전처리/스케일 재시도 로직 포함)

---

### (3) Stitching_3.py — 스티칭 파라미터/ROI 생성
- **목적**: (2)에서 만든 정렬 맵으로 LR 샘플(`pair_LR/img00.png`)을 정렬 후,
  **중첩 체크보드 코너**로 **호모그래피**를 계산하여 실시간용 파라미터를 생성
- **실행**
  ```bash
  python Stitching_3.py
  ```
- **출력**
  - `./data/config/homography_params.json`  ← (실시간 코드가 읽음)
  - `./data/params/stereo_map_left_x.npy`
  - `./data/params/stereo_map_right_x.npy`
  - `./data/params/left_blend_mask.npy`
  - `./data/params/right_blend_mask.npy`
  - 결과/시각화: `./data/stitching_results/*.png`, `./data/visualization/*.png`
- **ROI 선택(선택사항)**: 실행 중 `ROI를 선택할지` 물으면 `y` 입력 → 마우스로 영역 드래그 → **Enter** 확정  
  ROI 관련 산출물:
  - `./data/params/user_roi_info.json`
  - `./data/params/user_roi_mask.npy`
  - `./data/params/roi_blending_params.json`
  - `./data/params/user_roi_preview.png`

---

### (4) Eyetracking_4.py — 실시간 스티칭 + 아이트래킹
- **목적**: (3)에서 생성한 파라미터로 **실시간 스티칭(그레이스케일 최적화)** 수행 + **시선 추적**
- **실행**
  ```bash
  python Eyetracking_4.py
  ```
- **키보드**
  - `q` : 종료
  - `m` : 거울 모드 on/off 
  ### 기본 값이 거울 모드 입니다. 

- **필수 파일**
  - `./data/config/homography_params.json`
  - `./data/params/stereo_map_left_x.npy`, `stereo_map_right_x.npy`
  - `./data/params/left_blend_mask.npy`, `right_blend_mask.npy`
  - `./models/yolov8n-face.pt` (이미 저장되어 있어야 함)
- **추가 메모**
  - YOLO(ultralytics) + MediaPipe FaceMesh 사용
  - CUDA가 있으면 자동으로 GPU 사용(`torch.cuda.is_available()` 기준)

---

## 현재 오류

### Stitching_3.py 코드에서 특징점 선택이 정확하게 이루어지지 않고 있습니다. 코드 수정이 필요할 것 같습니다. 
### Eyetracking_4.py 코드에서 호모그래피 파라미터, 정렬맵 파라미터를 불러오는데 심한 병목 현상이 발생중입니다. 