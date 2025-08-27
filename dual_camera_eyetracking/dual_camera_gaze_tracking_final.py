import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
import mediapipe as mp
from collections import deque
import threading
from queue import Queue
import copy
from system_params import FRAME_WIDTH, FRAME_HEIGHT, FPS, CAMERA_MATRIX, DIST_COEFFS, CAMERA_ANGLES

# GPU 가속 설정
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print(f"Using GPU: {torch.cuda.get_device_name()}")
else:
    device = torch.device('cpu')
    print("Using CPU")

# YOLO 모델 로드 (GPU 가속)
model = YOLO("yolov8n-face.pt")  # 얼굴 특화 모델 사용
model.to(device)

# MediaPipe FaceMesh 설정 (성능 한계 테스트: 5명까지 랜드마크 추출)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=5,  # 성능 테스트를 위해 5명으로 증가
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 3D 얼굴 모델 포인트
face_3d_model_points = np.array([
    [0.0, 0.0, 0.0], [0.0, -330.0, -65.0],
    [-225.0, 170.0, -135.0], [225.0, 170.0, -135.0],
    [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]
], dtype=np.float32)

landmark_ids = [1, 152, 33, 263, 61, 291]

# 🔥 눈 주위 랜드마크 ID들 (더 정확한 동공 추정을 위해)
left_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
right_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# 홍채 중심점 ID
iris_ids = [468, 473]  # 왼쪽, 오른쪽 홍채 중심

# 🔥 안정화를 위한 히스토리 버퍼 (각 카메라별, 각 사람별)
class GazeStabilizer:
    def __init__(self, buffer_size=5, outlier_threshold=0.1):
        self.buffer_size = buffer_size
        self.outlier_threshold = outlier_threshold
        self.gaze_history = deque(maxlen=buffer_size)
        self.iris_history = deque(maxlen=buffer_size)
        
    def add_sample(self, gaze, iris_coords):
        """새로운 샘플 추가"""
        self.gaze_history.append(gaze)
        self.iris_history.append(iris_coords)
    
    def get_stabilized_gaze(self):
        """안정화된 시선 좌표 반환"""
        if len(self.gaze_history) < 2:
            return self.gaze_history[-1] if self.gaze_history else (0, 0)
        
        # 아웃라이어 제거
        valid_gazes = []
        for gaze in self.gaze_history:
            if self._is_valid_gaze(gaze):
                valid_gazes.append(gaze)
        
        if not valid_gazes:
            return self.gaze_history[-1]
        
        # 가중 평균 (최근 값에 더 높은 가중치)
        weights = np.linspace(0.5, 1.0, len(valid_gazes))
        weights = weights / np.sum(weights)
        
        avg_x = np.average([g[0] for g in valid_gazes], weights=weights)
        avg_y = np.average([g[1] for g in valid_gazes], weights=weights)
        
        return (avg_x, avg_y)
    
    def _is_valid_gaze(self, gaze):
        """아웃라이어 검출"""
        if len(self.gaze_history) < 2:
            return True
        
        recent_gazes = list(self.gaze_history)[-3:]  # 최근 3개 샘플
        avg_x = np.mean([g[0] for g in recent_gazes])
        avg_y = np.mean([g[1] for g in recent_gazes])
        
        distance = np.sqrt((gaze[0] - avg_x)**2 + (gaze[1] - avg_y)**2)
        return distance < self.outlier_threshold

# 🔥 멀티스레딩 처리 클래스
class ThreadedProcessor:
    def __init__(self):
        self.yolo_queue = Queue(maxsize=4)  # 큐 크기 증가
        self.mediapipe_queue = Queue(maxsize=4)  # 큐 크기 증가  
        self.result_queue = Queue(maxsize=4)  # 큐 크기 증가
        self.running = True
        
        # YOLO 처리 스레드
        self.yolo_thread = threading.Thread(target=self._yolo_worker, daemon=True)
        self.yolo_thread.start()
        
        # MediaPipe 처리 스레드
        self.mediapipe_thread = threading.Thread(target=self._mediapipe_worker, daemon=True)
        self.mediapipe_thread.start()
    
    def _yolo_worker(self):
        """YOLO 얼굴 검출 전용 스레드"""
        while self.running:
            try:
                frame, camera_id = self.yolo_queue.get(timeout=0.1)
                yolo_results = model.predict(frame, verbose=False)[0]
                face_boxes = [
                    box for box in yolo_results.boxes.data.cpu().numpy()
                    if int(box[5]) == 0
                ]
                self.mediapipe_queue.put((frame, face_boxes, camera_id))
                self.yolo_queue.task_done()
            except:
                continue
    
    def _mediapipe_worker(self):
        """MediaPipe 랜드마크 처리 전용 스레드"""
        while self.running:
            try:
                frame, face_boxes, camera_id = self.mediapipe_queue.get(timeout=0.1)
                frame_processed, results = self._process_mediapipe(frame, face_boxes, camera_id)
                self.result_queue.put((frame_processed, results, camera_id))
                self.mediapipe_queue.task_done()
            except:
                continue
    
    def _process_mediapipe(self, frame, face_boxes, camera_id):
        """MediaPipe 랜드마크 처리 (기존 process_faces_yolo_mediapipe 로직 사용)"""
        results = []
        h, w = frame.shape[:2]
        
        for face_idx, box in enumerate(face_boxes):
            if face_idx >= 5:  # 성능 테스트: 최대 5명까지 처리
                break
                
            x1, y1, x2, y2, conf, cls = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_roi = frame[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            mesh_result = face_mesh.process(face_rgb)

            if mesh_result.multi_face_landmarks:
                for face_landmarks in mesh_result.multi_face_landmarks:
                    # 기본 랜드마크 포인트들
                    image_points = []
                    for idx in landmark_ids:
                        lm = face_landmarks.landmark[idx]
                        x_lm = int(lm.x * (x2 - x1)) + x1
                        y_lm = int(lm.y * (y2 - y1)) + y1
                        image_points.append((x_lm, y_lm))
                        cv2.circle(frame, (x_lm, y_lm), 2, (0, 0, 255), -1)

                    # 🔥 개선된 동공 추정: 홍채 중심 + 눈 주위 랜드마크 평균
                    iris_coords = []
                    eye_centers = []
                    
                    # 홍채 중심점 추출
                    for idx in iris_ids:
                        if idx < len(face_landmarks.landmark):
                            lm = face_landmarks.landmark[idx]
                            x_lm = int(lm.x * (x2 - x1)) + x1
                            y_lm = int(lm.y * (y2 - y1)) + y1
                            iris_coords.append((x_lm, y_lm))
                            cv2.circle(frame, (x_lm, y_lm), 3, (0, 255, 0), -1)
                    
                    # 눈 주위 랜드마크 기반 눈 중심 계산
                    roi_info = (x1, y1, x2-x1, y2-y1)
                    
                    left_eye_center = get_eye_center_from_landmarks(face_landmarks, left_eye_landmarks, roi_info)
                    right_eye_center = get_eye_center_from_landmarks(face_landmarks, right_eye_landmarks, roi_info)
                    
                    if left_eye_center:
                        eye_centers.append(left_eye_center)
                    if right_eye_center:
                        eye_centers.append(right_eye_center)

                    # 3D 포즈 추정
                    rot = (0, 0, 0)
                    pos = (0, 0, 0)
                    if len(image_points) == 6:
                        image_points_np = np.array(image_points, dtype=np.float32)
                        success, rotation_vector, translation_vector = cv2.solvePnP(
                            face_3d_model_points, image_points_np, CAMERA_MATRIX, DIST_COEFFS)
                        if success:
                            rot = rotation_vector.flatten()
                            pos = translation_vector.flatten()

                    # 🔥 개선된 시선 추정: 다중 방법 결합
                    gaze = (0, 0)
                    
                    # 방법 1: 홍채 중심 기반
                    if len(iris_coords) == 2:
                        lx, ly = iris_coords[0]
                        rx, ry = iris_coords[1]
                        iris_gaze = ((rx + lx) / 2 / w - 0.5, (ry + ly) / 2 / h - 0.5)
                    else:
                        iris_gaze = None
                    
                    # 방법 2: 눈 주위 랜드마크 중심 기반
                    if len(eye_centers) == 2:
                        lx, ly = eye_centers[0]
                        rx, ry = eye_centers[1]
                        landmark_gaze = ((rx + lx) / 2 / w - 0.5, (ry + ly) / 2 / h - 0.5)
                    else:
                        landmark_gaze = None
                    
                    # 🔥 두 방법의 가중 평균 (홍채가 더 정확하므로 높은 가중치)
                    if iris_gaze and landmark_gaze:
                        gaze = (
                            0.7 * iris_gaze[0] + 0.3 * landmark_gaze[0],
                            0.7 * iris_gaze[1] + 0.3 * landmark_gaze[1]
                        )
                    elif iris_gaze:
                        gaze = iris_gaze
                    elif landmark_gaze:
                        gaze = landmark_gaze

                    # 🔥 시간적 안정화 적용
                    if camera_id in stabilizers and face_idx < len(stabilizers[camera_id]):
                        stabilizer = stabilizers[camera_id][face_idx]
                        stabilizer.add_sample(gaze, iris_coords)
                        stable_gaze = stabilizer.get_stabilized_gaze()
                        gaze = stable_gaze

                    results.append({"pos": pos, "rot": rot, "gaze": gaze})

        return frame, results
    
    def process_frame(self, frame, camera_id):
        """프레임 처리 요청"""
        try:
            self.yolo_queue.put_nowait((frame.copy(), camera_id))
        except:
            # Queue가 가득찬 경우 이전 항목 제거 후 새로운 것 추가
            try:
                self.yolo_queue.get_nowait()
                self.yolo_queue.task_done()
                self.yolo_queue.put_nowait((frame.copy(), camera_id))
            except:
                pass
    
    def get_result(self):
        """처리 결과 가져오기"""
        try:
            return self.result_queue.get_nowait()
        except:
            return None
    
    def stop(self):
        """스레드 정리"""
        self.running = False

# 각 카메라별 안정화 객체 (성능 테스트: 최대 5명까지)
stabilizers = {
    0: [GazeStabilizer() for _ in range(5)],  # 카메라 0: 5명
    1: [GazeStabilizer() for _ in range(5)]   # 카메라 1: 5명
}

# 멀티스레딩 프로세서 초기화
processor = ThreadedProcessor()

# 카메라 초기화 (DirectShow 백엔드로 최적화)
cap0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# 카메라 1이 실패하면 카메라 0으로 대체
if not cap1.isOpened():
    cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)

for cap in [cap0, cap1]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)  
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

prev_frame_time = time.time()

def rotate_frame(frame, angle_rad):
    if angle_rad == 0:
        return frame
    angle_deg = np.rad2deg(angle_rad)
    h, w = frame.shape[:2]
    center = (w//2, h//2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(frame, rot_mat, (w, h))

def get_eye_center_from_landmarks(face_landmarks, eye_landmark_ids, roi_offset):
    """눈 주위 랜드마크들의 평균으로 눈 중심 계산"""
    eye_points = []
    x1, y1, roi_width, roi_height = roi_offset
    
    for idx in eye_landmark_ids:
        if idx < len(face_landmarks.landmark):
            lm = face_landmarks.landmark[idx]
            x_lm = int(lm.x * roi_width) + x1
            y_lm = int(lm.y * roi_height) + y1
            eye_points.append((x_lm, y_lm))
    
    if eye_points:
        # 눈 주위 랜드마크들의 중심점 계산
        center_x = int(np.mean([p[0] for p in eye_points]))
        center_y = int(np.mean([p[1] for p in eye_points]))
        return (center_x, center_y)
    return None

# 기존 process_faces_yolo_mediapipe 함수는 ThreadedProcessor 클래스 내부로 이동됨

def draw_info(frame, data, fps, camera_id):
    # 카메라 상태 표시
    status_color = (0, 255, 0) if len(data) > 0 else (0, 0, 255)  # 초록색: 추적 중, 빨간색: 추적 안됨
    cv2.putText(frame, f"Tracking: {'ON' if len(data) > 0 else 'OFF'}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
    
    for i, d in enumerate(data):
        y_offset = 40 + i * 120  # 위치 조정
        cv2.putText(frame, f"Person {i+1}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(frame, f"Pos: {np.round(d['pos'],1)}", (10, y_offset+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, f"Rot: {np.round(d['rot'],2)}", (10, y_offset+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, f"Gaze: ({d['gaze'][0]:.3f}, {d['gaze'][1]:.3f})", (10, y_offset+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(frame, f"Multi-threaded", (10, y_offset+80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

# 이전 프레임 데이터 저장 (멀티스레딩 대기용)
last_frame_data = {0: [], 1: []}


try:
    while True: #메인 루프 시작점
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        
        if not ret0:
            break
        if not ret1:
            frame1 = frame0.copy()

        frame0 = rotate_frame(frame0, CAMERA_ANGLES[0])
        frame1 = rotate_frame(frame1, CAMERA_ANGLES[1])

        # 🔥 멀티스레딩 처리: YOLO와 MediaPipe 병렬 실행
        processor.process_frame(frame0, 0)
        processor.process_frame(frame1, 1)

        # 결과 수집
        data0 = last_frame_data[0]
        data1 = last_frame_data[1]
        
        # 처리 완료된 결과 가져오기 (두 카메라 모두 처리될 때까지 대기)
        processed_cameras = set()
        max_attempts = 20  # 최대 20회 시도 -> 최대 20ms 대기
        
        for attempt in range(max_attempts):
            result = processor.get_result()
            if result is None:
                time.sleep(0.001)  # 1ms 대기
                continue
                
            frame_processed, results, camera_id = result
            processed_cameras.add(camera_id)
            
            last_frame_data[camera_id] = results
            if camera_id == 0:
                frame0 = frame_processed
                data0 = results
            else:
                frame1 = frame_processed
                data1 = results
            
            # 두 카메라 모두 처리되면 종료
            if len(processed_cameras) >= 2:
                break

        # FPS 계산
        now = time.time()
        fps = 1.0 / (now - prev_frame_time)
        prev_frame_time = now

        # 정보 표시
        draw_info(frame0, data0, fps, 0)
        draw_info(frame1, data1, fps, 1)

        cv2.putText(frame0, "Camera 0 (Left)", (10, frame0.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame1, "Camera 1 (Right)", (10, frame1.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        combined = np.hstack((frame0, frame1))
        cv2.imshow("🔥 Multi-threaded Gaze Tracking (GPU + Parallel Processing)", combined)

        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    pass

finally:
    processor.stop() 
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()
