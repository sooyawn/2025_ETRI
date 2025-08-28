import os
import json
import cv2
import numpy as np
from datetime import datetime
import time
from typing import Tuple, Optional
import torch
from ultralytics import YOLO
import mediapipe as mp
from collections import deque
import threading
from queue import Queue

# ========================================
# 해상도 설정 (Camera_1.py와 동일)
# ========================================
DEFAULT_WIDTH = 1920          # 기본 해상도 너비
DEFAULT_HEIGHT = 1080         # 기본 해상도 높이
DEFAULT_FPS = 60              # 기본 FPS

# ========================================
# 카메라 인덱스 설정 (Camera_1.py와 동일)
# ========================================
LEFT_CAMERA_INDEX = 2         # 왼쪽 카메라 (Camera_1.py: cap_L = open_cam(2, selected_res))
RIGHT_CAMERA_INDEX = 0        # 오른쪽 카메라 (Camera_1.py: cap_R = open_cam(0, selected_res))
# ========================================

# ========================================
# 디스플레이 스케일 설정
# ========================================
DISPLAY_SCALE = 0.5          # 화면 표시용 스케일 (0.5 = 50% 크기, 1.0 = 100% 크기)

# ========================================
# 아이트래킹 설정
# ========================================
GAZE_TRACKING_CONFIG = {
    'max_faces': 3,  # 최대 추적할 얼굴 수
    'detection_confidence': 0.7,
    'tracking_confidence': 0.7,
    'stabilizer_buffer_size': 5,
    'outlier_threshold': 0.1
}

# ========================================
# 동공 추적 실험 설정
# ========================================
GAZE_EXPERIMENT_CONFIG = {
    'experiment_mode': False,  # 실험 모드 활성화 여부
    'target_a_mode': False,    # 타겟 A 응시 모드
    'target_b_mode': False,    # 타겟 B 응시 모드
    'recording_duration': 10,  # 기록 시간 (초)
    'min_recording_duration': 1,  # 최소 기록 시간 (초)
    'frame_rate': 30,          # 예상 프레임 레이트
    'expected_frames': 300,    # 예상 프레임 수 (10초 * 30fps)
    'target_distance_cm': 150, # 사용자와 화면 간 거리 (cm)
    'target_angle_deg': 0.5,   # 목표 시선 추적 정확도 (도)
    'target_distance_pixels': None,  # 계산된 목표 픽셀 거리
    'min_detection_threshold': 0.1   # 최소 감지 임계값 (픽셀)
}

# ========================================

class GazeStabilizer:
    """시선 추적 안정화 클래스"""
    
    def __init__(self, buffer_size=None, outlier_threshold=None):
        self.buffer_size = buffer_size or GAZE_TRACKING_CONFIG['stabilizer_buffer_size']
        self.outlier_threshold = outlier_threshold or GAZE_TRACKING_CONFIG['outlier_threshold']
        self.gaze_history = deque(maxlen=self.buffer_size)
        self.iris_history = deque(maxlen=self.buffer_size)
        
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
        
        recent_gazes = list(self.gaze_history)[-3:]
        avg_x = np.mean([g[0] for g in recent_gazes])
        avg_y = np.mean([g[1] for g in recent_gazes])
        
        distance = np.sqrt((gaze[0] - avg_x)**2 + (gaze[1] - avg_y)**2)
        return distance < self.outlier_threshold


class GazeExperimentRecorder:
    """동공 추적 실험 데이터 기록 클래스"""
    
    def __init__(self):
        self.reset()
        self._calculate_target_distance()
    
    def reset(self):
        """실험 데이터 초기화"""
        self.gaze_coordinates = []  # 시선 좌표 기록
        self.timestamps = []        # 타임스탬프 기록
        self.recording_start_time = None
        self.recording_end_time = None
        self.is_recording = False
        self.current_target = None  # 'A' 또는 'B'
        self.frame_count = 0

    
    def _calculate_target_distance(self):
        """목표 픽셀 거리 계산 (0.5도 각도 기준)"""
        # 150cm * tan(0.5도) ≈ 1.3cm
        target_distance_cm = GAZE_EXPERIMENT_CONFIG['target_distance_cm']
        target_angle_rad = np.radians(GAZE_EXPERIMENT_CONFIG['target_angle_deg'])
        target_distance_cm = target_distance_cm * np.tan(target_angle_rad)
        
        # 화면 해상도에 따른 픽셀 거리 계산
        # 일반적인 24인치 모니터 기준 (1920x1080, 약 53cm 너비)
        screen_width_cm = 53.0  # 24인치 모니터 너비
        screen_width_pixels = 1920
        
        # 1cm당 픽셀 수 계산
        pixels_per_cm = screen_width_pixels / screen_width_cm
        
        # 목표 픽셀 거리 계산
        target_pixels = target_distance_cm * pixels_per_cm
        
        GAZE_EXPERIMENT_CONFIG['target_distance_pixels'] = target_pixels
        
        print(f"🔬 동공 추적 실험 설정:")
        print(f"   - 사용자-화면 거리: {target_distance_cm}cm")
        print(f"   - 목표 각도: {GAZE_EXPERIMENT_CONFIG['target_angle_deg']}도")
        print(f"   - 계산된 거리: {target_distance_cm:.2f}cm")
        print(f"   - 목표 픽셀 거리: {target_pixels:.1f} 픽셀")
        print(f"   - 최소 감지 임계값: {GAZE_EXPERIMENT_CONFIG['min_detection_threshold']} 픽셀")
    
    def start_recording(self, target):
        """실험 기록 시작"""
        if self.is_recording:
            print(f"⚠️ 이미 기록 중입니다. 먼저 중지해주세요.")
            return False
        
        self.reset()
        self.current_target = target
        self.recording_start_time = time.time()
        self.is_recording = True
        
        print(f"🔬 타겟 {target} 응시 기록 시작")
        print(f"   - 목표 기록 시간: {GAZE_EXPERIMENT_CONFIG['recording_duration']}초")
        print(f"   - 유효 기록 시간: {GAZE_EXPERIMENT_CONFIG['min_recording_duration']}초 이상 (제한 없음)")
        print(f"   - 예상 프레임 수: {GAZE_EXPERIMENT_CONFIG['expected_frames']}")
        print(f"   - 목표 픽셀 거리: {GAZE_EXPERIMENT_CONFIG['target_distance_pixels']:.1f} 픽셀")
        print(f"   - 최소 감지 임계값: {GAZE_EXPERIMENT_CONFIG['min_detection_threshold']} 픽셀")
        return True
    
    def stop_recording(self):
        """실험 기록 중지"""
        if not self.is_recording:
            print(f"⚠️ 기록 중이 아닙니다.")
            return False
        
        self.recording_end_time = time.time()
        self.is_recording = False
        
        actual_duration = self.recording_end_time - self.recording_start_time
        actual_frames = len(self.gaze_coordinates)
        
        print(f"\n🔬 타겟 {self.current_target} 응시 기록 완료!")
        print(f"   - 실제 기록 시간: {actual_duration:.2f}초")
        print(f"   - 실제 프레임 수: {actual_frames}")
        print(f"   - 평균 FPS: {actual_frames/actual_duration:.1f}")
        
        # 유효한 기록 시간인지 확인 (최소 1초만 확인)
        if actual_duration < GAZE_EXPERIMENT_CONFIG['min_recording_duration']:
            print(f"⚠️ 기록 시간이 너무 짧습니다. 최소 {GAZE_EXPERIMENT_CONFIG['min_recording_duration']}초 필요")
            return False
        
        # 9초 제한 제거 - 모든 데이터 사용!
        print(f"✅ 기록이 성공적으로 완료되었습니다!")
        print(f"   - 모든 {actual_frames}개 프레임 데이터 사용")
        return True
    
    def add_gaze_sample(self, gaze_x, gaze_y):
        """시선 좌표 샘플 추가"""
        if not self.is_recording:
            return
        
        current_time = time.time()
        elapsed_time = current_time - self.recording_start_time
        
        # 목표 시간에 도달하면 자동 중지
        if elapsed_time >= GAZE_EXPERIMENT_CONFIG['recording_duration']:
            # 자동 중지 시 즉시 통계 저장
            self.stop_recording()
            
            # 부모 클래스에 통계 저장 요청
            if hasattr(self, '_parent_stitcher'):
                self._parent_stitcher._save_experiment_stats_auto(self.current_target)
            
            return
        
        # 좌표 기록
        self.gaze_coordinates.append((gaze_x, gaze_y))
        self.timestamps.append(current_time)
        self.frame_count += 1
    
    def get_recording_progress(self):
        """현재 기록 진행 상황 반환"""
        if not self.is_recording:
            return 0.0, 0.0
        
        current_time = time.time()
        elapsed_time = current_time - self.recording_start_time
        progress_percent = (elapsed_time / GAZE_EXPERIMENT_CONFIG['recording_duration']) * 100
        
        return elapsed_time, progress_percent
    
    def get_statistics(self):
        """기록된 데이터 통계 계산"""
        if not self.gaze_coordinates:
            return None
        
        # 좌표 분리
        x_coords = [coord[0] for coord in self.gaze_coordinates]
        y_coords = [coord[1] for coord in self.gaze_coordinates]
        
        # 기본 통계
        stats = {
            'target': self.current_target,
            'frame_count': len(self.gaze_coordinates),
            'duration': self.recording_end_time - self.recording_start_time if self.recording_end_time else 0,
            'mean_x': np.mean(x_coords),
            'mean_y': np.mean(y_coords),
            'std_x': np.std(x_coords),
            'std_y': np.std(y_coords),
            'min_x': np.min(x_coords),
            'max_x': np.max(x_coords),
            'min_y': np.min(y_coords),
            'max_y': np.max(y_coords),
            'range_x': np.max(x_coords) - np.min(x_coords),
            'range_y': np.max(y_coords) - np.min(y_coords)
        }
        
        # 시선 안정성 지표 (표준편차의 평균)
        stats['stability_score'] = np.mean([stats['std_x'], stats['std_y']])
        
        return stats
    
    def print_statistics(self):
        """통계 결과 출력"""
        stats = self.get_statistics()
        if not stats:
            print("⚠️ 통계 데이터가 없습니다.")
            return
        
        print(f"\n🔬 타겟 {stats['target']} 응시 통계 결과:")
        print(f"{'='*50}")
        print(f"   📊 기본 정보:")
        print(f"      - 프레임 수: {stats['frame_count']}")
        print(f"      - 기록 시간: {stats['duration']:.2f}초")
        print(f"      - 평균 FPS: {stats['frame_count']/stats['duration']:.1f}")
        
        print(f"   📍 시선 좌표 통계:")
        print(f"      - 평균 위치: ({stats['mean_x']:.2f}, {stats['mean_y']:.2f})")
        print(f"      - X축 표준편차: {stats['std_x']:.2f} 픽셀")
        print(f"      - Y축 표준편차: {stats['std_y']:.2f} 픽셀")
        print(f"      - X축 범위: {stats['min_x']:.1f} ~ {stats['max_x']:.1f} ({stats['range_x']:.1f} 픽셀)")
        print(f"      - Y축 범위: {stats['min_y']:.1f} ~ {stats['max_y']:.1f} ({stats['range_y']:.1f} 픽셀)")
        
        print(f"   🎯 안정성 지표:")
        print(f"      - 시선 안정성 점수: {stats['stability_score']:.2f} 픽셀")
        print(f"      - 목표 감지 거리: {GAZE_EXPERIMENT_CONFIG['target_distance_pixels']:.1f} 픽셀")
        
        # 0.5도 정확도 달성 여부 판단
        if stats['stability_score'] <= GAZE_EXPERIMENT_CONFIG['target_distance_pixels']:
            print(f"   ✅ 0.5도 시선 추적 정확도 달성!")
            print(f"      - 현재 안정성: {stats['stability_score']:.2f} 픽셀")
            print(f"      - 목표 안정성: {GAZE_EXPERIMENT_CONFIG['target_distance_pixels']:.1f} 픽셀 이하")
        else:
            print(f"   ❌ 0.5도 시선 추적 정확도 미달성")
            print(f"      - 필요 안정성: {GAZE_EXPERIMENT_CONFIG['target_distance_pixels']:.1f} 픽셀 이하")
            print(f"      - 현재 안정성: {stats['stability_score']:.2f} 픽셀")
        
        print(f"{'='*50}")
        print(f"💡 다음 단계:")
        if stats['target'] == 'A':
            print(f"   - 'b' 키를 눌러 타겟 B 응시 기록을 시작하세요")
        else:
            print(f"   - 'c' 키를 눌러 두 타겟 결과를 비교하세요")
            print(f"   - 'f' 키를 눌러 결과를 파일로 저장하세요")
    
    def compare_targets(self, target_a_stats, target_b_stats):
        """타겟 A와 B의 결과 비교"""
        if not target_a_stats or not target_b_stats:
            print("⚠️ 두 타겟의 통계 데이터가 필요합니다.")
            return
        
        print(f"\n🔬 타겟 A vs 타겟 B 비교 분석:")
        print(f"{'='*60}")
        
        # 평균 위치 차이 계산
        mean_diff_x = abs(target_b_stats['mean_x'] - target_a_stats['mean_x'])
        mean_diff_y = abs(target_b_stats['mean_y'] - target_a_stats['mean_y'])
        mean_distance = np.sqrt(mean_diff_x**2 + mean_diff_y**2)
        
        print(f"   📍 평균 위치 변화:")
        print(f"      - 타겟 A: ({target_a_stats['mean_x']:.2f}, {target_a_stats['mean_y']:.2f})")
        print(f"      - 타겟 B: ({target_b_stats['mean_x']:.2f}, {target_b_stats['mean_y']:.2f})")
        print(f"      - X축 변화: {mean_diff_x:.2f} 픽셀")
        print(f"      - Y축 변화: {mean_diff_y:.2f} 픽셀")
        print(f"      - 총 변화 거리: {mean_distance:.2f} 픽셀")
        
        # 안정성 비교
        stability_diff = abs(target_b_stats['stability_score'] - target_a_stats['stability_score'])
        print(f"   🎯 안정성 비교:")
        print(f"      - 타겟 A 안정성: {target_a_stats['stability_score']:.2f} 픽셀")
        print(f"      - 타겟 B 안정성: {target_b_stats['stability_score']:.2f} 픽셀")
        print(f"      - 안정성 차이: {stability_diff:.2f} 픽셀")
        
        # 0.5도 정확도 달성 여부
        target_distance = GAZE_EXPERIMENT_CONFIG['target_distance_pixels']
        print(f"   🎯 0.5도 정확도 분석:")
        print(f"      - 목표 감지 거리: {target_distance:.1f} 픽셀")
        print(f"      - 실제 변화 거리: {mean_distance:.2f} 픽셀")
        
        if mean_distance >= target_distance:
            print(f"      ✅ 변화 거리가 목표 거리 이상으로 감지됨!")
            print(f"         - 시스템이 0.5도 시선 변화를 감지할 수 있음")
        else:
            print(f"      ❌ 변화 거리가 목표 거리 미만")
            print(f"         - 시스템이 0.5도 시선 변화를 감지하기 어려움")
        
        print(f"{'='*60}")
    
    def save_results_to_file(self, filename=None):
        """실험 결과를 파일로 저장"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gaze_experiment_results_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("동공 추적 실험 결과\n")
                f.write("="*50 + "\n\n")
                
                # 실험 설정
                f.write("실험 설정:\n")
                f.write(f"- 목표 각도: {GAZE_EXPERIMENT_CONFIG['target_angle_deg']}도\n")
                f.write(f"- 사용자-화면 거리: {GAZE_EXPERIMENT_CONFIG['target_distance_cm']}cm\n")
                f.write(f"- 목표 픽셀 거리: {GAZE_EXPERIMENT_CONFIG['target_distance_pixels']:.1f} 픽셀\n")
                f.write(f"- 기록 시간: {GAZE_EXPERIMENT_CONFIG['recording_duration']}초\n\n")
                
                # 기록된 데이터
                if self.gaze_coordinates:
                    f.write("기록된 시선 좌표:\n")
                    f.write("프레임\tX좌표\tY좌표\t타임스탬프\n")
                    for i, (coord, ts) in enumerate(zip(self.gaze_coordinates, self.timestamps)):
                        f.write(f"{i+1}\t{coord[0]:.2f}\t{coord[1]:.2f}\t{ts:.3f}\n")
                    
                    f.write("\n")
                    
                    # 통계 결과
                    stats = self.get_statistics()
                    if stats:
                        f.write("통계 결과:\n")
                        f.write(f"- 평균 위치: ({stats['mean_x']:.2f}, {stats['mean_y']:.2f})\n")
                        f.write(f"- X축 표준편차: {stats['std_x']:.2f}\n")
                        f.write(f"- Y축 표준편차: {stats['std_y']:.2f}\n")
                        f.write(f"- 시선 안정성 점수: {stats['stability_score']:.2f}\n")
                
            print(f"✅ 실험 결과가 {filename}에 저장되었습니다.")
            return filename
            
        except Exception as e:
            print(f"❌ 파일 저장 실패: {e}")
            return None


class UltraFastVideoStitcherGrayscale:
    """초고속 최적화된 실시간 비디오 스티처 (FPS 향상 버전)"""
    
    def __init__(self, calibration_config_path: str):
        self.config_path = calibration_config_path
        
        # FPS 측정용 변수들
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # 🚀 최적화: 한 번만 모든 캘리브레이션 데이터 로드
        self._load_all_calibration_data_once()
        
        # 🚀 최적화: 모든 사전 계산 적용
        self._apply_ultra_optimizations()
        
        # 🚀 아이트래킹 초기화
        self._initialize_gaze_tracking()
        
        # 🚀 멀티스레딩 아이트래킹 초기화
        self._initialize_threaded_gaze_tracking()
        
        # 🚀 동공 추적 실험 초기화
        self._initialize_gaze_experiment()
        
        # 카메라 객체 초기화
        self.cap_left = None
        self.cap_right = None
        
        # 프레임 카운터
        self.frame_count = 0
        
        # 🚀 성능 측정용 변수들
        self.step_times = {
            'frame_read': [],
            'mirror_flip': [],
            'lens_rectification': [],
            'input_grayscale': [],  # 🆕 입력 그레이스케일 변환 시간
            'canvas_copy': [],
            'homography_warp': [],
            'left_placement': [],
            'blending': [],
            'roi_crop': [],
            'output_grayscale': [],  # 🆕 출력 그레이스케일 변환 시간
            'display_resize': [],
            'total_stitching': []
        }
        self.performance_measurement = False  # 성능 측정 비활성화 (콘솔 출력 줄이기)
        print(f"✅ 성능 측정 초기화 완료: {len(self.step_times)} 단계")
        print(f"   - 성능 측정: {'활성화' if self.performance_measurement else '비활성화'}")
        print(f"   - FPS 로그: 10초마다 출력")
        print(f"   - 성능 통계: 30초마다 출력")

    def _load_all_calibration_data_once(self):
        """🚀 최적화: JSON 파일을 한 번만 로딩하여 모든 데이터 처리"""
        print("🔧 한 번에 모든 캘리브레이션 데이터 로딩 중...")
        
        config_dir = os.path.dirname(self.config_path)
        homography_file = os.path.join(config_dir, "homography_params.json")
        
        try:
            with open(homography_file, 'r') as f:
                self.homo_data = json.load(f)  # 한 번만 로딩!
            
            print("✅ homography_params.json 로드 완료")
            
            # 모든 데이터를 한 번에 처리
            self._process_homography_data()
            self._process_roi_data()  # 🆕 ROI 데이터 처리 추가
            self._process_blend_masks()  
            self._process_rectification_maps()
            
        except FileNotFoundError as e:
            print(f"homography_params.json 파일을 찾을 수 없습니다: {e}")
            print("먼저 Stitching_Engine.py를 실행하여 캘리브레이션을 완료해주세요")
            raise
        except Exception as e:
            print(f"JSON 로드 오류: {e}")
            raise

    def _process_homography_data(self):
        """호모그래피 데이터 처리"""
        data = self.homo_data
        self.homography_matrix = np.array(data['homography_matrix'])
        self.canvas_size = tuple(data['final_size'])
        self.left_offset = tuple(data['left_image_offset'])
        self.camera_resolution = data.get('camera_resolution', [1920, 1080])
        print(f"✅ 호모그래피 데이터 처리 완료: {self.camera_resolution}")

    def _process_roi_data(self):
        """🆕 사용자가 선택한 ROI 데이터 처리"""
        print("🔧 ROI 데이터 처리 중...")
        
        # ROI 정보 파일 경로 - 올바른 경로 계산
        config_dir = os.path.dirname(self.config_path)  # "./data/config"
        data_dir = os.path.dirname(config_dir)          # "./data"
        params_dir = os.path.join(data_dir, "params")   # "./data/params"
        roi_info_path = os.path.join(params_dir, "user_roi_info.json")
        roi_mask_path = os.path.join(params_dir, "user_roi_mask.npy")
        
        print(f"🔍 ROI 파일 경로 확인:")
        print(f"   config_path: {self.config_path}")
        print(f"   config_dir: {config_dir}")
        print(f"   data_dir: {data_dir}")
        print(f"   params_dir: {params_dir}")
        print(f"   roi_info_path: {roi_info_path}")
        print(f"   roi_mask_path: {roi_mask_path}")
        
        # ROI 데이터가 있는지 확인
        if os.path.exists(roi_info_path) and os.path.exists(roi_mask_path):
            try:
                # ROI 정보 로드
                with open(roi_info_path, 'r') as f:
                    self.roi_info = json.load(f)
                
                # ROI 마스크 로드
                self.roi_mask = np.load(roi_mask_path).astype(np.uint8)
                
                # ROI 좌표 추출 (원본 해상도 기준)
                self.roi_x1 = self.roi_info['x1']
                self.roi_y1 = self.roi_info['y1']
                self.roi_x2 = self.roi_info['x2']
                self.roi_y2 = self.roi_info['y2']
                
                # 🚨 ROI 좌표는 이미 전체 캔버스 기준이므로 스케일링 불필요!
                # Stitching_Engine에서 생성된 ROI 좌표를 그대로 사용
                # (사용자가 0.5 스케일로 선택했지만, 좌표는 전체 캔버스 기준)
                
                self.roi_width = self.roi_x2 - self.roi_x1
                self.roi_height = self.roi_y2 - self.roi_y1
                
                print(f"🔧 ROI 좌표 스케일 조정:")
                print(f"   - 원본 ROI: ({self.roi_info['x1']}, {self.roi_info['y1']}) -> ({self.roi_info['x2']}, {self.roi_info['y2']})")
                print(f"   - 스케일 조정 후: ({self.roi_x1}, {self.roi_y1}) -> ({self.roi_x2}, {self.roi_y2})")
                print(f"   - ROI 크기: {self.roi_width} x {self.roi_height}")
                
                # 🚨 ROI 좌표 범위 검증 (원본 해상도 내에 있는지 확인)
                if hasattr(self, 'canvas_size'):
                    canvas_w, canvas_h = self.canvas_size[0], self.canvas_size[1]
                    if (self.roi_x1 < 0 or self.roi_y1 < 0 or 
                        self.roi_x2 > canvas_w or self.roi_y2 > canvas_h):
                        print(f"⚠️ ROI 좌표가 원본 해상도({canvas_w}x{canvas_h})를 벗어남!")
                        print(f"   ROI 범위: ({self.roi_x1}, {self.roi_y1}) -> ({self.roi_x2}, {self.roi_y2})")
                        # 좌표를 원본 해상도 내로 클램프
                        self.roi_x1 = max(0, min(self.roi_x1, canvas_w-1))
                        self.roi_y1 = max(0, min(self.roi_y1, canvas_h-1))
                        self.roi_x2 = max(self.roi_x1+1, min(self.roi_x2, canvas_w))
                        self.roi_y2 = max(self.roi_y1+1, min(self.roi_y2, canvas_h))
                        self.roi_width = self.roi_x2 - self.roi_x1
                        self.roi_height = self.roi_y2 - self.roi_y1
                        print(f"   클램프 후 ROI: ({self.roi_x1}, {self.roi_y1}) -> ({self.roi_x2}, {self.roi_y2})")
                        print(f"   클램프 후 크기: {self.roi_width} x {self.roi_height}")
                
                self.use_roi = True
                
                # 🆕 ROI가 있을 때 원본 캔버스 크기 보존 (전체 스티칭 후 크롭)
                if hasattr(self, 'canvas_size'):
                    self.original_canvas_size = self.canvas_size  # 원본 크기 보존
                    print(f"✅ 원본 캔버스 크기 보존: {self.canvas_size[0]}x{self.canvas_size[1]}")
                    print(f"   ROI 크기: {self.roi_width}x{self.roi_height}")
                
                print(f"✅ 사용자 정의 ROI 로드 완료: ({self.roi_x1}, {self.roi_y1}) -> ({self.roi_x2}, {self.roi_y2})")
                print(f"   ROI 크기: {self.roi_width} x {self.roi_height}")
                
                # 🆕 ROI 블렌딩 파라미터 로드
                roi_blending_file = os.path.join(params_dir, 'roi_blending_params.json')
                if os.path.exists(roi_blending_file):
                    with open(roi_blending_file, 'r') as f:
                        self.roi_blending_params = json.load(f)
                    
                    # ROI 블렌딩 마스크 로드
                    roi_left_mask_file = os.path.join(params_dir, 'roi_left_blend_mask.npy')
                    roi_right_mask_file = os.path.join(params_dir, 'roi_right_blend_mask.npy')
                    
                    if os.path.exists(roi_left_mask_file) and os.path.exists(roi_right_mask_file):
                        self.roi_left_blend_mask = np.load(roi_left_mask_file)
                        self.roi_right_blend_mask = np.load(roi_right_mask_file)
                        print(f"✅ ROI 블렌딩 파라미터 로드 완료:")
                        print(f"   - 블렌딩 마스크: {self.roi_left_blend_mask.shape}")
                        print(f"   - ROI 왼쪽 오프셋: {self.roi_blending_params['roi_left_offset']}")
                        print(f"   - ROI 왼쪽 크기: {self.roi_blending_params['roi_left_size']}")
                    else:
                        print(f"⚠️ ROI 블렌딩 마스크 파일을 찾을 수 없습니다")
                        print(f"   찾는 파일: {roi_left_mask_file}, {roi_right_mask_file}")
                else:
                    print(f"ℹ️ ROI 블렌딩 파라미터 파일이 없습니다: {roi_blending_file}")
                    print(f"   Stitching_Engine_3.py에서 ROI를 선택하여 블렌딩 파라미터를 생성해주세요")
                
            except Exception as e:
                print(f"⚠️ ROI 데이터 로드 실패: {e}")
                self.use_roi = False
        else:
            print("ℹ️ 사용자 정의 ROI가 없습니다. 전체 화면을 사용합니다.")
            print(f"   파일 존재 여부:")
            print(f"     roi_info_path: {os.path.exists(roi_info_path)}")
            print(f"     roi_mask_path: {os.path.exists(roi_mask_path)}")
            self.use_roi = False

    def _process_blend_masks(self):
        """블렌딩 마스크 처리"""
        self.use_precomputed_blending = False
        
        blend_info = self.homo_data.get("blending_optimization", None)
        if blend_info is not None:
            data_dir = os.path.dirname(os.path.dirname(self.config_path))
            params_dir = os.path.join(data_dir, "params")
            left_mask_path = os.path.join(params_dir, blend_info["left_mask_file"])
            right_mask_path = os.path.join(params_dir, blend_info["right_mask_file"])
            
            if os.path.exists(left_mask_path) and os.path.exists(right_mask_path):
                # 원본 블렌딩 마스크 로드 (ROI 조정 전)
                self.left_blend_mask_original = np.load(left_mask_path).astype(np.float32)
                self.right_blend_mask_original = np.load(right_mask_path).astype(np.float32)
                self.use_precomputed_blending = True
                print(f"✅ 원본 블렌딩 마스크 로드 완료: {self.left_blend_mask_original.shape}")
                
                # 항상 원본 마스크를 그대로 사용 (ROI는 표시 시에만 적용)
                self.left_blend_mask = self.left_blend_mask_original
                self.right_blend_mask = self.right_blend_mask_original
        else:
            print("ℹ️ 블렌딩 마스크 정보가 없습니다.")

    def _process_rectification_maps(self):
        """렌즈 왜곡 보정 맵 처리"""
        rect_maps = self.homo_data.get("rectification_maps", None)
        if rect_maps is None:
            raise ValueError("homography_params.json에 'rectification_maps' 정보가 없습니다.")
        
        data_dir = os.path.dirname(os.path.dirname(self.config_path))
        params_dir = os.path.join(data_dir, "params")
        
        map_left_x_path = os.path.join(params_dir, rect_maps["map_left_x"])
        map_right_x_path = os.path.join(params_dir, rect_maps["map_right_x"])
        
        if not os.path.exists(map_left_x_path) or not os.path.exists(map_right_x_path):
            raise FileNotFoundError("렌즈 왜곡 보정 맵 파일을 찾을 수 없습니다.")
        
        self.map_left_x = np.load(map_left_x_path).astype(np.float32)
        self.map_right_x = np.load(map_right_x_path).astype(np.float32)
        
        # 2채널 맵 처리
        if len(self.map_left_x.shape) == 3 and self.map_left_x.shape[2] == 2:
            # 2채널 맵을 x, y로 분리
            self.map_left_y = self.map_left_x[:, :, 1].astype(np.float32)
            self.map_left_x = self.map_left_x[:, :, 0].astype(np.float32)
            self.map_right_y = self.map_right_x[:, :, 1].astype(np.float32)
            self.map_right_x = self.map_right_x[:, :, 0].astype(np.float32)
            print("✅ 2채널 렌즈 왜곡 보정 맵을 x, y로 분리하여 로드 완료")
        else:
            # 1채널 맵 처리
            map_left_y_path = os.path.join(params_dir, rect_maps["map_left_y"])
            map_right_y_path = os.path.join(params_dir, rect_maps["map_right_y"])
            self.map_left_y = np.load(map_left_y_path).astype(np.float32)
            self.map_right_y = np.load(map_right_y_path).astype(np.float32)
            print("✅ 1채널 렌즈 왜곡 보정 맵 로드 완료")
        
        print(f"   왼쪽 맵: {self.map_left_x.shape}")
        print(f"   오른쪽 맵: {self.map_right_x.shape}")
        print(f"   왼쪽 map2: {self.map_left_y.shape}")
        print(f"   오른쪽 map2: {self.map_right_y.shape}")

    def _apply_ultra_optimizations(self):
        """🚀 모든 초고속 최적화를 한 번에 적용"""
        print("🚀 초고속 최적화 적용 중...")
        
        # 🚀 1. 캔버스 템플릿 미리 생성 (항상 전체 크기!)
        # 🚨 항상 전체 크기로 생성 (ROI는 나중에 크롭!)
        # 🆕 그레이스케일용 1채널 캔버스 생성
        self.canvas_template = np.zeros((self.canvas_size[1], self.canvas_size[0]), dtype=np.uint8)
        print(f"✅ 전체 크기 그레이스케일 캔버스 템플릿 생성: {self.canvas_template.shape}")
        
        # 🚨 항상 카메라 해상도 기준으로 슬라이스 인덱스 설정
        self.left_slice_y = slice(self.left_offset[1], self.left_offset[1] + self.camera_resolution[1])
        self.left_slice_x = slice(self.left_offset[0], self.left_offset[0] + self.camera_resolution[0])
        print(f"✅ 카메라 해상도 기반 슬라이스 인덱스 계산: Y{self.left_slice_y}, X{self.left_slice_x}")
        
        # 🆕 ROI가 있는 경우 ROI 최적화 계산
        if hasattr(self, 'use_roi') and self.use_roi:
            self._calculate_roi_optimizations()
        
        # 🚀 3. 렌즈 보정 함수 사전 설정 (조건문 제거) - 더 하드코딩
        # lambda 대신 직접 함수 정의로 성능 향상
        def rectify_left_hardcoded(img):
            return cv2.remap(img, self.map_left_x, self.map_left_y,
                           cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
        
        def rectify_right_hardcoded(img):
            return cv2.remap(img, self.map_right_x, self.map_right_y,
                           cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
        
        self.rectify_left = rectify_left_hardcoded
        self.rectify_right = rectify_right_hardcoded
        
        # 🚀 4. 블렌딩 마스크 최종 크기로 미리 리사이즈 (항상 전체 크기!)
        if hasattr(self, 'use_precomputed_blending') and self.use_precomputed_blending:
            # 🚨 항상 전체 크기로 리사이즈 (ROI는 나중에 크롭!)
            self.left_mask_final = cv2.resize(
                self.left_blend_mask, 
                self.canvas_size, 
                interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)
            
            self.right_mask_final = cv2.resize(
                self.right_blend_mask, 
                self.canvas_size, 
                interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)
            
            print(f"✅ 전체 크기 블렌딩 마스크 생성: {self.left_mask_final.shape}")
            
            # 🆕 그레이스케일용 1채널 마스크 생성 (3채널 대신)
            self.left_mask_1ch = self.left_mask_final  # 1채널 그대로 사용
            self.right_mask_1ch = self.right_mask_final  # 1채널 그대로 사용
        else:
            # 🚀 블렌딩 마스크가 없으면 기본값 생성
            self.left_mask_1ch = None
            self.right_mask_1ch = None
        
        # 🚀 5. 호모그래피 행렬 최적화 (더 하드코딩)
        self.homography_matrix_opt = self.homography_matrix.astype(np.float32)  # float32로 고정
        
        # 🚀 6. 캔버스 크기 하드코딩 (항상 전체 크기!)
        # 🚨 항상 전체 크기 사용 (ROI는 나중에 크롭!)
        self.canvas_width = int(self.canvas_size[0])
        self.canvas_height = int(self.canvas_size[1])
        print(f"✅ 전체 캔버스 크기 설정: {self.canvas_width}x{self.canvas_height}")
        
        # 🚀 7. 호모그래피 변환 크기 미리 계산 (캐시!)
        self.homography_size = (self.canvas_width, self.canvas_height)
        print(f"✅ 호모그래피 변환 크기 캐시: {self.homography_size}")
        
        # 🚀 8. ROI 기반 사전 계산 좌표 맵 생성 (극한 최적화!)
        if hasattr(self, 'use_roi') and self.use_roi:
            self._precompute_roi_homography_coordinates()
        
        # 🚀 9. 왼쪽 오프셋 하드코딩
        self.left_offset_x = int(self.left_offset[0])
        self.left_offset_y = int(self.left_offset[1])
        
        # 🚀 8. 카메라 해상도 하드코딩
        self.camera_width = int(self.camera_resolution[0])
        self.camera_height = int(self.camera_resolution[1])

        # 🚫 렌즈 보정 LUT 생성 제거 (성능 저하로 인해)
        # self._create_lens_correction_lut()
        
        # 🚫 호모그래피 좌표 맵 사전 계산 제거 (성능 저하로 인해)
        # self._precompute_homography_coordinates()

        print("🚀 초고속 최적화 완료! 예상 FPS 향상: 50-100%")
        
        # 🚀 캐시 상태 확인
        print(f"   📊 캐시 상태:")
        print(f"      - 캔버스 템플릿: {self.canvas_template.shape}")
        print(f"      - 슬라이스 인덱스: Y{self.left_slice_y}, X{self.left_slice_x}")
        print(f"      - 렌즈 보정 함수: {self.rectify_left.__name__}, {self.rectify_right.__name__}")
        print(f"      - 블렌딩 마스크: {'활성화' if hasattr(self, 'left_mask_1ch') and self.left_mask_1ch is not None else '비활성화'}")
        print(f"      - 호모그래피 행렬: {self.homography_matrix_opt.shape}")
        print(f"      - 하드코딩 크기: {self.canvas_width}x{self.canvas_height}")
        if hasattr(self, 'use_roi') and self.use_roi:
            print(f"      - ROI 모드: 활성화 ({self.roi_width}x{self.roi_height})")
        else:
            print(f"      - ROI 모드: 비활성화 (전체 화면)")
        print(f"      - 🚫 렌즈 보정 LUT: 비활성화 (성능 저하로 인해)")
        print(f"      - 🚫 호모그래피 좌표 맵: 비활성화 (성능 저하로 인해)")

    def _calculate_roi_optimizations(self):
        """🆕 ROI 영역을 미리 계산하여 하드코딩 (한 번만!)"""
        print("🔧 ROI 최적화 계산 중...")
        
        # ROI 영역을 미리 계산하여 하드코딩 (전체 스티칭 후 크롭용)
        self.roi_extract_y = slice(self.roi_y1, self.roi_y2)
        self.roi_extract_x = slice(self.roi_x1, self.roi_x2)
        
        print(f"✅ ROI 최적화 계산 완료:")
        print(f"   - ROI 추출 영역: Y{self.roi_extract_y}, X{self.roi_extract_x}")
        print(f"   - 하드코딩 완료: 매 프레임마다 계산하지 않음!")

    def _create_lens_correction_lut(self):
        """🚀 렌즈 보정 LUT 생성으로 매 프레임 계산 제거"""
        print("🔧 렌즈 보정 LUT 생성 중...")
        
        h, w = self.camera_resolution[1], self.camera_resolution[0]
        
        # 좌표 그리드 생성
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # 렌즈 보정 맵 적용하여 LUT 생성
        self.left_lut_x = cv2.remap(x_coords, self.map_left_x, self.map_left_y, 
                                    cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
        self.left_lut_y = cv2.remap(y_coords, self.map_left_x, self.map_left_y, 
                                    cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
        
        self.right_lut_x = cv2.remap(x_coords, self.map_right_x, self.map_right_y, 
                                     cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
        self.right_lut_y = cv2.remap(y_coords, self.map_right_x, self.map_right_y, 
                                     cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
        
        print(f"✅ 렌즈 보정 LUT 생성 완료: {self.left_lut_x.shape}")
        print(f"   예상 성능 향상: 26-38ms → 2-5ms (5-10배 향상!)")

    def _precompute_homography_coordinates(self):
        """🚀 호모그래피 좌표 맵 사전 계산으로 매 프레임 변환 제거"""
        print("🔧 호모그래피 좌표 맵 사전 계산 중...")
        
        # 🚨 중요: 전체 캔버스 크기로 좌표 맵 생성 (카메라 해상도가 아님!)
        canvas_h, canvas_w = self.canvas_size[1], self.canvas_size[0]
        
        # 전체 캔버스 기준 좌표 그리드
        y_coords, x_coords = np.mgrid[0:canvas_h, 0:canvas_w].astype(np.float32)
        
        # 호모그래피 행렬로 변환된 좌표 계산 (한 번만!)
        coords = np.stack([x_coords, y_coords, np.ones_like(x_coords)], axis=-1)
        transformed_coords = (self.homography_matrix_opt @ coords.reshape(-1, 3).T).T
        
        # 2D 좌표로 변환
        transformed_coords = transformed_coords[:, :2] / transformed_coords[:, 2:]
        self.homography_coords = transformed_coords.reshape(canvas_h, canvas_w, 2)
        
        print(f"✅ 호모그래피 좌표 맵 생성 완료: {self.homography_coords.shape}")
        print(f"   - 캔버스 크기: {canvas_w}x{canvas_h}")
        print(f"   - 카메라 해상도: {self.camera_resolution[0]}x{self.camera_resolution[1]}")
        print(f"   예상 성능 향상: 35-58ms → 3-8ms (5-10배 향상!)")

    def _precompute_roi_homography_coordinates(self):
        """🚀 ROI 영역만을 위한 호모그래피 좌표 맵 사전 계산 (원래 방식!)"""
        print("🔧 ROI 기반 호모그래피 좌표 맵 사전 계산 중...")
        
        # 🎯 원래대로! ROI 크기로만 처리!
        roi_h, roi_w = self.roi_height, self.roi_width  # 618, 3081
        
        # ROI 크기로 좌표 그리드 생성
        y_coords, x_coords = np.mgrid[0:roi_h, 0:roi_w].astype(np.float32)
        
        # ROI 좌표를 전체 캔버스 좌표로 변환
        x_coords += self.roi_x1  # ROI 시작점 더하기
        y_coords += self.roi_y1
        
        # 호모그래피 행렬로 변환된 좌표 계산 (한 번만!)
        coords = np.stack([x_coords, y_coords, np.ones_like(x_coords)], axis=-1)
        transformed_coords = (self.homography_matrix_opt @ coords.reshape(-1, 3).T).T
        
        # 2D 좌표로 변환
        transformed_coords = transformed_coords[:, :2] / transformed_coords[:, 2:]
        self.roi_homography_coords = transformed_coords.reshape(roi_h, roi_w, 2)
        
        # 🚀 ROI 영역만을 위한 마스크도 생성 (차원 일치 보장!)
        self.roi_mask = np.ones((roi_h, roi_w), dtype=np.uint8)
        
        print(f"✅ ROI 호모그래피 좌표 맵 생성 완료:")
        print(f"   - ROI 크기: {roi_w}x{roi_h}")
        print(f"   - 좌표 맵 크기: {self.roi_homography_coords.shape}")
        print(f"   - ROI 마스크 크기: {self.roi_mask.shape}")
        print(f"   - 예상 성능 향상: 36ms → 5-10ms (3-7배 향상!)")

    def _apply_roi_homography_fast(self, frame):
        """🚀 사전 계산된 ROI 좌표 맵을 사용하여 극한 빠른 호모그래피 변환"""
        if not hasattr(self, 'roi_homography_coords'):
            # 좌표 맵이 없으면 기존 방식 사용
            return cv2.warpPerspective(frame, self.homography_matrix_opt, 
                                     self.homography_size, flags=cv2.INTER_LINEAR)
        
        # 🚀 ROI 크기로 결과 이미지 초기화 (차원 일치 보장!)
        roi_h, roi_w = self.roi_height, self.roi_width
        warped = np.zeros((roi_h, roi_w, 3), dtype=frame.dtype)
        
        # 사전 계산된 좌표 맵 사용
        coords = self.roi_homography_coords
        
        # 좌표를 정수로 변환하여 샘플링
        coords_int = coords.astype(np.int32)
        coords_int[:, :, 0] = np.clip(coords_int[:, :, 0], 0, frame.shape[1]-1)
        coords_int[:, :, 1] = np.clip(coords_int[:, :, 1], 0, frame.shape[0]-1)
        
        # 벡터화된 샘플링 (훨씬 빠름!)
        warped = frame[coords_int[:, :, 1], coords_int[:, :, 0]]
        
        return warped

    def _initialize_gaze_tracking(self):
        """🚀 아이트래킹 초기화"""
        print("🚀 아이트래킹 초기화 중...")
        
        # GPU 설정 (YOLO 시선추적용만)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            print(f"🚀 YOLO GPU 사용: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            print("⚠️ YOLO CPU 사용")
        
        # YOLO 모델 로드
        try:
            self.yolo_model = YOLO("models/yolov8n-face.pt")
            self.yolo_model.to(self.device)
            print("✅ YOLO 얼굴 검출 모델 로드 완료")
        except Exception as e:
            print(f"⚠️ YOLO 모델 로드 실패: {e}")
            self.yolo_model = None
        
        # MediaPipe FaceMesh 설정
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False, 
                max_num_faces=GAZE_TRACKING_CONFIG['max_faces'],
                refine_landmarks=True,
                min_detection_confidence=GAZE_TRACKING_CONFIG['detection_confidence'],
                min_tracking_confidence=GAZE_TRACKING_CONFIG['tracking_confidence']
            )
            print("✅ MediaPipe FaceMesh 초기화 완료")
        except Exception as e:
            print(f"⚠️ MediaPipe 초기화 실패: {e}")
            self.face_mesh = None
        
        # 시선 안정화 객체들
        self.gaze_stabilizers = [GazeStabilizer() for _ in range(GAZE_TRACKING_CONFIG['max_faces'])]
        
        # 3D 얼굴 모델 포인트
        self.face_3d_model_points = np.array([
            [0.0, 0.0, 0.0], [0.0, -330.0, -65.0],
            [-225.0, 170.0, -135.0], [225.0, 170.0, -135.0],
            [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]
        ], dtype=np.float64)
        
        # 랜드마크 ID들
        self.landmark_ids = [1, 152, 33, 263, 61, 291]
        self.left_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.iris_ids = [468, 473]  # 왼쪽, 오른쪽 홍채 중심
        
        print("✅ 아이트래킹 초기화 완료")

    def _initialize_threaded_gaze_tracking(self):
        """🚀 멀티스레딩 아이트래킹 초기화"""
        print("🚀 멀티스레딩 아이트래킹 초기화 중...")
        
        # 스레드 안전한 큐들
        self.gaze_input_queue = Queue(maxsize=2)  # 입력 큐 (최신 프레임만)
        self.gaze_output_queue = Queue(maxsize=2)  # 출력 큐
        
        # 아이트래킹 스레드 시작
        self.gaze_thread_running = True
        self.gaze_thread = threading.Thread(target=self._gaze_worker_thread, daemon=True)
        self.gaze_thread.start()
        
        print("✅ 멀티스레딩 아이트래킹 초기화 완료")

    def _initialize_gaze_experiment(self):
        """🚀 동공 추적 실험 초기화"""
        print("🚀 동공 추적 실험 초기화 중...")
        
        # 실험 기록기 초기화
        self.gaze_experiment = GazeExperimentRecorder()
        
        # 부모 참조 설정 (자동 통계 저장용)
        self.gaze_experiment._parent_stitcher = self
        
        # 실험 상태 변수들
        self.experiment_mode = False
        self.target_a_stats = None
        self.target_b_stats = None
        
        # 실험 UI 상태
        self.show_experiment_info = True
        self.experiment_instruction = "Press 'e' to start experiment mode"
        
        print("✅ 동공 추적 실험 초기화 완료")
        print(f"   - 목표 픽셀 거리: {GAZE_EXPERIMENT_CONFIG['target_distance_pixels']:.1f} 픽셀")
        print(f"   - 기록 시간: {GAZE_EXPERIMENT_CONFIG['recording_duration']}초")
        print(f"   - 예상 프레임 수: {GAZE_EXPERIMENT_CONFIG['expected_frames']}")
    
    def _save_experiment_stats_auto(self, target):
        """🚀 자동 기록 완료 시 통계 저장 (내부 메서드)"""
        print(f"🔧 자동 통계 저장 시작: 타겟 {target}")
        
        # 통계 계산 및 저장
        stats = self.gaze_experiment.get_statistics()
        if stats:
            if target == 'A':
                self.target_a_stats = stats
                print(f"✅ 타겟 A 통계 자동 저장 완료!")
                print(f"   - 평균 위치: ({stats['mean_x']:.1f}, {stats['mean_y']:.1f})")
                print(f"   - 안정성 점수: {stats['stability_score']:.2f} 픽셀")
                print(f"   - 프레임 수: {stats['frame_count']}")
            else:
                self.target_b_stats = stats
                print(f"✅ 타겟 B 통계 자동 저장 완료!")
                print(f"   - 평균 위치: ({stats['mean_x']:.1f}, {stats['mean_y']:.1f})")
                print(f"   - 안정성 점수: {stats['stability_score']:.2f} 픽셀")
                print(f"   - 프레임 수: {stats['frame_count']}")
            
            # 통계 출력
            self.gaze_experiment.print_statistics()
            
            # 실험 지시사항 업데이트
            self.experiment_instruction = "Experiment Mode: Press 'a' or 'b' to start recording"
        else:
            print(f"❌ 자동 통계 저장 실패: 타겟 {target}")
            print(f"   - gaze_coordinates 개수: {len(self.gaze_experiment.gaze_coordinates)}")
            print(f"   - timestamps 개수: {len(self.gaze_experiment.timestamps)}")

    def _gaze_worker_thread(self):
        """🚀 아이트래킹 전용 워커 스레드"""
        while self.gaze_thread_running:
            try:
                # 입력 큐에서 프레임 가져오기 (타임아웃으로 블로킹 방지)
                frame = self.gaze_input_queue.get(timeout=0.1)
                
                # 아이트래킹 처리
                processed_frame, gaze_data = self.process_gaze_tracking(frame)
                
                # 출력 큐에 결과 전달 (이전 결과 덮어쓰기)
                try:
                    if not self.gaze_output_queue.empty():
                        self.gaze_output_queue.get_nowait()  # 이전 결과 제거
                    self.gaze_output_queue.put_nowait((processed_frame, gaze_data))
                except:
                    pass
                    
            except:
                continue  # 큐가 비어있으면 계속 진행

    def process_gaze_tracking_async(self, frame):
        """🚀 비동기 아이트래킹 요청"""
        try:
            # 입력 큐에 프레임 전달 (이전 프레임 덮어쓰기)
            if not self.gaze_input_queue.empty():
                self.gaze_input_queue.get_nowait()  # 이전 프레임 제거
            self.gaze_input_queue.put_nowait(frame)
        except:
            pass

    def get_gaze_result_async(self):
        """🚀 비동기 아이트래킹 결과 가져오기"""
        try:
            return self.gaze_output_queue.get_nowait()
        except:
            return None, []

    def switch_to_mirror_mode(self, enable_mirror: bool):
        """거울모드 전환 함수"""
        if enable_mirror:
            # 거울모드: 이미지를 좌우 반전
            print(f"🔄 거울모드 활성화")
        else:
            # 일반모드: 원본 이미지
            print(f"🔄 일반모드 활성화")

    def toggle_experiment_mode(self):
        """실험 모드 토글"""
        self.experiment_mode = not self.experiment_mode
        
        if self.experiment_mode:
            print(f"\n🔬 실험 모드 활성화")
            print(f"   - 'a': 타겟 A 응시 기록 시작")
            print(f"   - 'b': 타겟 B 응시 기록 시작")
            print(f"   - 's': 현재 통계 출력")
            print(f"   - 'c': 타겟 A vs B 비교")
            print(f"   - 'f': 결과를 파일로 저장")
            print(f"   - 'r': 실험 데이터 초기화")
            print(f"   - 'x': 수동 기록 중지")
            print(f"   - 'e': 실험 모드 비활성화")
            self.experiment_instruction = "Experiment Mode: Press 'a' or 'b' to start recording"
            print(f"✅ 실험 모드가 활성화되었습니다. 'a' 또는 'b' 키를 눌러 기록을 시작하세요.")
        else:
            print(f"\n🔬 실험 모드 비활성화")
            self.experiment_instruction = "Press 'e' to start experiment mode"
            print(f"✅ 실험 모드가 비활성화되었습니다.")
    
    def start_target_recording(self, target):
        """타겟 응시 기록 시작"""
        if not self.experiment_mode:
            print(f"⚠️ 실험 모드를 먼저 활성화해주세요 ('e' 키)")
            return False
        
        if self.gaze_experiment.is_recording:
            print(f"⚠️ 이미 기록 중입니다. 먼저 중지해주세요.")
            return False
        
        success = self.gaze_experiment.start_recording(target)
        if success:
            if target == 'A':
                self.experiment_instruction = f"Recording Target A... ({GAZE_EXPERIMENT_CONFIG['recording_duration']}s)"
                print(f"\n🎯 타겟 A 응시 시작!")
                print(f"   - 10초간 응시해주세요")
                print(f"   - 타이머 바가 화면 상단에 표시됩니다")
                print(f"   - 시선 위치가 빨간색 점으로 표시됩니다")
                print(f"   - 자동으로 10초 후 기록이 완료됩니다")
                print(f"   - 10초를 초과해도 모든 데이터가 기록됩니다")
            else:
                self.experiment_instruction = f"Recording Target B... ({GAZE_EXPERIMENT_CONFIG['recording_duration']}s)"
                print(f"\n🎯 타겟 B 응시 시작!")
                print(f"   - 10초간 응시해주세요 (타겟 A에서 1.3cm 옆 지점)")
                print(f"   - 타이머 바가 화면 상단에 표시됩니다")
                print(f"   - 시선 위치가 빨간색 점으로 표시됩니다")
                print(f"   - 자동으로 10초 후 기록이 완료됩니다")
                print(f"   - 10초를 초과해도 모든 데이터가 기록됩니다")
        
        return success
    
    def stop_target_recording(self):
        """타겟 응시 기록 중지"""
        if not self.gaze_experiment.is_recording:
            print(f"⚠️ 기록 중이 아닙니다.")
            return False
        
        success = self.gaze_experiment.stop_recording()
        if success:
            # 통계 계산 및 저장
            stats = self.gaze_experiment.get_statistics()
            if stats:
                if stats['target'] == 'A':
                    self.target_a_stats = stats
                    print(f"✅ 타겟 A 통계 저장 완료")
                    print(f"   - 평균 위치: ({stats['mean_x']:.1f}, {stats['mean_y']:.1f})")
                    print(f"   - 안정성 점수: {stats['stability_score']:.2f} 픽셀")
                else:
                    self.target_b_stats = stats
                    print(f"✅ 타겟 B 통계 저장 완료")
                    print(f"   - 평균 위치: ({stats['mean_x']:.1f}, {stats['mean_y']:.1f})")
                    print(f"   - 안정성 점수: {stats['stability_score']:.2f} 픽셀")
                
                # 통계 출력
                self.gaze_experiment.print_statistics()
            
            self.experiment_instruction = "Experiment Mode: Press 'a' or 'b' to start recording"
        else:
            print(f"❌ 기록 중지 실패. 기록 시간이 너무 짧거나 깁니다.")
        
        return success
    
    def get_current_gaze_coordinates(self):
        """현재 시선 좌표 반환 (실험 모드에서만)"""
        if not self.experiment_mode or not hasattr(self, 'gaze_experiment'):
            return None, None
        
        if self.gaze_experiment.is_recording and self.gaze_experiment.gaze_coordinates:
            # 가장 최근 좌표 반환
            latest_coord = self.gaze_experiment.gaze_coordinates[-1]
            return latest_coord[0], latest_coord[1]
        
        return None, None
    
    def show_experiment_statistics(self):
        """실험 통계 출력"""
        if not self.experiment_mode:
            print(f"⚠️ 실험 모드를 먼저 활성화해주세요 ('e' 키)")
            return
        
        print(f"\n🔬 현재 실험 상태:")
        print(f"{'='*40}")
        
        if self.target_a_stats:
            print(f"✅ 타겟 A: 완료")
            print(f"   - 평균 위치: ({self.target_a_stats['mean_x']:.2f}, {self.target_a_stats['mean_y']:.2f})")
            print(f"   - 안정성 점수: {self.target_a_stats['stability_score']:.2f} 픽셀")
        else:
            print(f"❌ 타겟 A: 미완료")
        
        if self.target_b_stats:
            print(f"✅ 타겟 B: 완료")
            print(f"   - 평균 위치: ({self.target_b_stats['mean_x']:.2f}, {self.target_b_stats['mean_y']:.2f})")
            print(f"   - 안정성 점수: {self.target_b_stats['stability_score']:.2f} 픽셀")
        else:
            print(f"❌ 타겟 B: 미완료")
        
        if self.target_a_stats and self.target_b_stats:
            print(f"\n🎯 두 타겟 모두 완료! 'c' 키로 비교 분석을 실행할 수 있습니다.")
        
        print(f"{'='*40}")
    
    def compare_experiment_targets(self):
        """타겟 A와 B 비교 분석"""
        if not self.experiment_mode:
            print(f"⚠️ 실험 모드를 먼저 활성화해주세요 ('e' 키)")
            return
        
        if not self.target_a_stats or not self.target_b_stats:
            print(f"⚠️ 두 타겟의 데이터가 모두 필요합니다.")
            print(f"   - 타겟 A: {'완료' if self.target_a_stats else '미완료'}")
            print(f"   - 타겟 B: {'완료' if self.target_b_stats else '미완료'}")
            return
        
        # 비교 분석 실행
        self.gaze_experiment.compare_targets(self.target_a_stats, self.target_b_stats)
    
    def save_experiment_results(self):
        """실험 결과를 파일로 저장"""
        if not self.experiment_mode:
            print(f"⚠️ 실험 모드를 먼저 활성화해주세요 ('e' 키)")
            return
        
        if not self.target_a_stats and not self.target_b_stats:
            print(f"⚠️ 저장할 실험 데이터가 없습니다.")
            return
        
        filename = self.gaze_experiment.save_results_to_file()
        if filename:
            print(f"✅ 실험 결과가 {filename}에 저장되었습니다.")
    
    def reset_experiment_data(self):
        """실험 데이터 초기화"""
        if not self.experiment_mode:
            print(f"⚠️ 실험 모드를 먼저 활성화해주세요 ('e' 키)")
            return
        
        self.gaze_experiment.reset()
        self.target_a_stats = None
        self.target_b_stats = None
        self.experiment_instruction = "Experiment Mode: Press 'a' or 'b' to start recording"
        print(f"✅ 실험 데이터가 초기화되었습니다.")

    def setup_cameras(self, left_id: int = LEFT_CAMERA_INDEX, right_id: int = RIGHT_CAMERA_INDEX) -> bool:
        """카메라 초기화 및 설정"""
        print(f"카메라 초기화: 왼쪽={left_id}, 오른쪽={right_id}")
        
        try:
            # 사용 가능한 백엔드 목록 (우선순위 순)
            backends = [
                (cv2.CAP_DSHOW, "DirectShow"),
                (cv2.CAP_MSMF, "Media Foundation"),
                (cv2.CAP_ANY, "Auto")
            ]
            
            # 각 백엔드로 카메라 초기화 시도
            for backend_id, backend_name in backends:
                try:
                    print(f"백엔드 {backend_name} ({backend_id}) 시도 중...")
                    self.cap_left = cv2.VideoCapture(left_id, backend_id)
                    self.cap_right = cv2.VideoCapture(right_id, backend_id)
                    
                    if self.cap_left.isOpened() and self.cap_right.isOpened():
                        print(f"백엔드 {backend_name} ({backend_id}) 사용")
                        break
                    else:
                        print(f"백엔드 {backend_name} ({backend_id}) 실패")
                        self.cap_left.release()
                        self.cap_right.release()
                except Exception as e:
                    print(f"백엔드 {backend_name} ({backend_id}) 오류: {e}")
                    continue
            else:
                print("모든 백엔드에서 카메라 초기화 실패")
                return False
            
            # 고성능 카메라 설정 (하드코딩)
            target_width = self.camera_width
            target_height = self.camera_height
            
            for i, cap in enumerate([self.cap_left, self.cap_right]):
                camera_name = "왼쪽" if i == 0 else "오른쪽"
                
                # 해상도 설정 (하드코딩)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
                
                # 🚀 FPS 최적화: 60fps로 설정 (하드코딩)
                cap.set(cv2.CAP_PROP_FPS, 60)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 최소화
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"   {camera_name} 카메라: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
                
                # 해상도 불일치 경고 (하드코딩된 임계값)
                if abs(actual_width - target_width) > 100 or abs(actual_height - target_height) > 100:
                    print(f"⚠️ {camera_name} 카메라 해상도 불일치: 목표 {target_width}x{target_height}, 실제 {actual_width}x{actual_height}")
            
            return True
            
        except Exception as e:
            print(f"❌ 카메라 설정 오류: {e}")
            return False

    def _blend_images_ultra_fast(self, left_translated: np.ndarray, right_warped: np.ndarray) -> np.ndarray:
        """🚀 극한 속도 블렌딩 (진짜 빠른 알고리즘!)"""
        
        # 🚀 방법 1: 제자리 연산 (가장 빠름 - 메모리 복사 없음!)
        # left_translated에 직접 덮어쓰기 (원본 수정됨 주의!)
        mask = right_warped > 10
        left_translated[mask] = right_warped[mask]
        return left_translated
        
        # 🚫 기존 방법들 (느림)
        # result = left_translated.copy()  # 14.6MB 복사!
        # mask = np.any(right_warped > 10, axis=2)  # 4.9MB 마스크!
        # result[mask] = right_warped[mask]
        # return result

    def update_fps(self):
        """🚀 하드코딩된 FPS 업데이트 (최대 성능)"""
        self.fps_counter += 1
        current_time = time.time()
        
        # 1초마다 FPS 계산 (하드코딩된 시간)
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
            
            # FPS 로그 출력 제한 (10초마다만 출력, 성능 측정이 활성화된 경우에만)
            if self.performance_measurement:
                if hasattr(self, '_last_fps_log_time'):
                    if current_time - self._last_fps_log_time >= 10.0:
                        print(f"📊 현재 FPS: {self.current_fps:.1f}")
                        self._last_fps_log_time = current_time
                else:
                    self._last_fps_log_time = current_time
                    print(f"📊 현재 FPS: {self.current_fps:.1f}")
                    
                # 🚀 성능 측정 결과 출력 제한 (30초마다만 출력)
                if self.frame_count > 0:
                    if hasattr(self, '_last_performance_log_time'):
                        if current_time - self._last_performance_log_time >= 30.0:
                            print(f"🔍 성능 측정 호출: frame_count={self.frame_count}, step_times={len(self.step_times['total_stitching'])}")
                            self._print_performance_stats()
                            self._last_performance_log_time = current_time
                    else:
                        self._last_performance_log_time = current_time
                        print(f"🔍 성능 측정 호출: frame_count={self.frame_count}, step_times={len(self.step_times['total_stitching'])}")
                        self._print_performance_stats()

    def _print_performance_stats(self):
        """🚀 각 단계별 성능 통계 출력"""
        print(f"\n🔍 성능 분석 (프레임 {self.frame_count}):")
        print(f"{'='*50}")
        
        for step_name, times in self.step_times.items():
            if times:
                avg_time = np.mean(times) * 1000  # ms로 변환
                min_time = np.min(times) * 1000
                max_time = np.max(times) * 1000
                print(f"   {step_name:15s}: 평균 {avg_time:6.2f}ms (최소 {min_time:6.2f}ms, 최대 {max_time:6.2f}ms)")
        
        # 총 스티칭 시간 분석
        if self.step_times['total_stitching']:
            total_avg = np.mean(self.step_times['total_stitching']) * 1000
            theoretical_fps = 1000 / total_avg if total_avg > 0 else 0
            print(f"{'='*50}")
            print(f"   🎯 총 스티칭 시간: {total_avg:.2f}ms")
            print(f"   🚀 이론적 최대 FPS: {theoretical_fps:.1f}")
        
        # 메모리 사용량 표시
        print(f"   💾 캔버스 크기: {self.canvas_width}x{self.canvas_height}")
        if hasattr(self, 'use_roi') and self.use_roi:
            print(f"   🎯 ROI 크기: {self.roi_width}x{self.roi_height}")
        
        print(f"{'='*50}")
        
        # 측정 데이터 초기화 (메모리 절약)
        for step_name in self.step_times:
            self.step_times[step_name] = []

    def show_stitched_video_grayscale(self):
        """🚀 단색(그레이스케일) 초고속 실시간 스티칭 비디오 표시"""
        print(f"\n🚀 단색(그레이스케일) 초고속 실시간 스티칭 시작 (중간 성능 버전)")
        print(f"📸 카메라 해상도: {self.camera_resolution[0]}x{self.camera_resolution[1]}")
        print(f"🎯 목표 FPS: 40+ (단색 처리로 인한 성능 향상)")
        print(f"🔄 거울모드: {'활성화' if True else '비활성화'}") # 거울모드 기본 활성화
        print(f"📱 디스플레이 스케일: {DISPLAY_SCALE:.1f}x ({int(DISPLAY_SCALE*100)}%)")
        print(f"📐 최종 캔버스 크기: {self.canvas_size[0]}x{self.canvas_size[1]}")
        print(f"📍 왼쪽 이미지 오프셋: ({self.left_offset[0]}, {self.left_offset[1]})")
        print(f"🎨 색상 모드: 그레이스케일 (1채널) - 입력부터 1채널 처리로 극한 메모리 절약")
        print(f"   - 입력 변환: BGR → Gray (메모리 1/3)")
        print(f"   - 스티칭 처리: 1채널로 처리 (처리 속도 2-3배 향상)")
        print(f"   - 아이트래킹: 1채널 → 3채널 복제 (YOLO 호환성)")
        
        # 🆕 ROI 정보 표시
        if hasattr(self, 'use_roi') and self.use_roi:
            print(f"🎯 ROI 모드: 활성화")
            print(f"   ROI 좌표: ({self.roi_x1}, {self.roi_y1}) -> ({self.roi_x2}, {self.roi_y2})")
            print(f"   ROI 크기: {self.roi_width} x {self.roi_height}")
            print(f"   예상 FPS 향상: 단색 처리 + ROI로 인한 추가 100-150% 향상")
        else:
            print(f"🎯 ROI 모드: 비활성화 (전체 화면)")
        
        print(f"🚀 적용된 최적화: 단색 처리 + JSON 단일로딩 + 캔버스캐시 + 조건문제거")
        print(f"👁️ 아이트래킹: YOLO + MediaPipe FaceMesh + 시선 안정화 (그레이스케일에서도 정상 작동)")
        print(f"🚀 멀티스레딩: 아이트래킹 별도 스레드로 분리하여 메인 루프 FPS 향상")
        print(f"{'='*70}")
        
        # 🚀 성능 최적화: 시작 시 한 번만 실행!
        print("🔧 성능 최적화 시작...")
        
        print("✅ 성능 최적화 완료!")
        
        print(f"🎬 단색 실시간 스티칭 시작...")
        print(f"💡 종료: 'q' 키")
        print(f"💡 거울모드 토글: 'm' 키")
        print(f"🎨 그레이스케일 모드로 아이트래킹과 함께 작동합니다")
        print(f"{'='*60}")
        
        # 시작 시 거울모드 활성화
        print("🔄 거울모드가 기본값으로 활성화되었습니다.")
        
        frame_count = 0
        mirror_mode = True  # 거울모드를 기본값으로 설정
        
        try:
            while True:
                # 🚀 전체 스티칭 시간 측정 시작
                total_start_time = time.time()
                
                # 1️⃣ 프레임 읽기 시간 측정
                frame_read_start = time.time()
                retL, frameL = self.cap_left.read()
                retR, frameR = self.cap_right.read()
                frame_read_time = time.time() - frame_read_start
                self.step_times['frame_read'].append(frame_read_time)
                
                if not retL or not retR:
                    print("❌ 프레임 읽기 실패")
                    break

                # 2️⃣ 거울모드 적용 시간 측정
                mirror_start = time.time()
                if mirror_mode:
                    frameL = cv2.flip(frameL, 1)
                    frameR = cv2.flip(frameR, 1)
                mirror_time = time.time() - mirror_start
                self.step_times['mirror_flip'].append(mirror_time)

                # 3️⃣ 🚀 초고속 스티칭 처리 (모든 최적화 적용)
                # 🆕 방법 2: 입력부터 1채널로 변환하여 스티칭 처리
                gray_start = time.time()
                frameL_gray = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
                frameR_gray = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
                gray_time = time.time() - gray_start
                self.step_times['input_grayscale'].append(gray_time)
                
                # 🚀 1채널로 스티칭 처리 (극한 메모리 절약!)
                stitched = self.stitch_frame_pair_ultra_optimized(frameL_gray, frameR_gray)
                
                # 🚀 전체 스티칭 시간 측정 완료
                total_stitching_time = time.time() - total_start_time
                self.step_times['total_stitching'].append(total_stitching_time)
                
                # 🚀 🆕 방법 2: 스티칭부터 1채널로 처리 (극한 메모리 절약!)
                grayscale_start = time.time()
                
                # 🆕 스티칭 결과가 이미 1채널이므로 변환 불필요
                stitched_gray = stitched  # 이미 1채널 그레이스케일
                
                # 🆕 1채널을 3채널로 복제 (YOLO 모델 호환성)
                # 그레이스케일을 3채널로 복제: [H,W] → [H,W,3] (같은 값으로 3번 복제)
                stitched_gray_3ch = np.stack([stitched_gray] * 3, axis=-1)
                
                grayscale_time = time.time() - grayscale_start
                self.step_times['output_grayscale'].append(grayscale_time)
                
                # 🚀 비동기 아이트래킹 요청 (블로킹 없음) - 1채널 기반 3채널로 요청
                self.process_gaze_tracking_async(stitched_gray_3ch)  # 1채널 기반 3채널 사용
                
                # 🚀 비동기 아이트래킹 결과 가져오기
                stitched_with_gaze, gaze_data = self.get_gaze_result_async()
                if stitched_with_gaze is None:
                    stitched_with_gaze = stitched_gray  # 아이트래킹 결과가 없으면 그레이스케일 사용
                    gaze_data = []
                else:
                    # 🆕 아이트래킹 결과를 그레이스케일로 변환 (이미 1채널 기반이므로 빠름)
                    stitched_with_gaze = cv2.cvtColor(stitched_with_gaze, cv2.COLOR_BGR2GRAY)
                
                # 4️⃣ 디스플레이 리사이즈 시간 측정
                resize_start = time.time()
                if DISPLAY_SCALE != 1.0:
                    # 첫 프레임에서만 크기 계산 (캐싱) - 하드코딩
                    if not hasattr(self, '_display_width') or not hasattr(self, '_display_height'):
                        self._display_width = int(stitched_with_gaze.shape[1] * DISPLAY_SCALE)
                        self._display_height = int(stitched_with_gaze.shape[0] * DISPLAY_SCALE)
                        print(f"🔧 디스플레이 크기 캐시: {self._display_width}x{self._display_height}")
                    
                    # 하드코딩된 크기 사용
                    stitched_display = cv2.resize(stitched_with_gaze, (self._display_width, self._display_height), 
                                                interpolation=cv2.INTER_LINEAR)
                else:
                    stitched_display = stitched_with_gaze
                
                resize_time = time.time() - resize_start
                self.step_times['display_resize'].append(resize_time)
            
                # 🚀 최적화된 FPS 계산
                self.update_fps()
                
                # 🚀 실험 모드에서 시선 추적 정보는 정보 패널에만 표시 (화면에는 점 없음)
                
                # 🚀 정보 패널 생성 (화면 위에 새로운 공간)
                info_panel_height = 200  # 정보 패널 높이
                total_height = stitched_display.shape[0] + info_panel_height
                total_width = stitched_display.shape[1]
                
                # 전체 캔버스 생성 (정보 패널 + 비디오 화면)
                combined_display = np.zeros((total_height, total_width, 3), dtype=np.uint8)
                
                # 비디오 화면을 하단에 배치
                combined_display[info_panel_height:, :] = cv2.cvtColor(stitched_display, cv2.COLOR_GRAY2BGR)
                
                # 🚀 실험 모드 정보 패널 표시
                if hasattr(self, 'show_experiment_info') and self.show_experiment_info:
                    # 실험 모드 상태 표시
                    if hasattr(self, 'experiment_mode') and self.experiment_mode:
                        # 실험 진행 상황 표시
                        if hasattr(self, 'gaze_experiment') and self.gaze_experiment.is_recording:
                            # 🚀 실시간 타이머 표시 (정보 패널에)
                            elapsed_time, progress_percent = self.gaze_experiment.get_recording_progress()
                            remaining_time = max(0, GAZE_EXPERIMENT_CONFIG['recording_duration'] - elapsed_time)
                            
                            # 타이머 바 표시 (정보 패널에)
                            timer_bar_width = 300
                            timer_bar_height = 25
                            timer_x, timer_y = 20, 20
                            
                            # 타이머 배경
                            cv2.rectangle(combined_display, (timer_x, timer_y), 
                                        (timer_x + timer_bar_width, timer_y + timer_bar_height), (64, 64, 64), -1)
                            
                            # 타이머 진행률
                            progress_width = int(timer_bar_width * (progress_percent / 100))
                            cv2.rectangle(combined_display, (timer_x, timer_y), 
                                        (timer_x + progress_width, timer_y + timer_bar_height), (0, 255, 0), -1)
                            
                            # 타이머 텍스트
                            timer_text = f"Recording Target {self.gaze_experiment.current_target}: {remaining_time:.1f}s / {GAZE_EXPERIMENT_CONFIG['recording_duration']}s"
                            cv2.putText(combined_display, timer_text, (timer_x, timer_y + 18), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            # 진행률 퍼센트
                            progress_text = f"Progress: {progress_percent:.1f}%"
                            cv2.putText(combined_display, progress_text, (timer_x + timer_bar_width + 20, timer_y + 18), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            # 🚀 실시간 좌표 표시 (정보 패널에만)
                            current_x, current_y = self.get_current_gaze_coordinates()
                            if current_x is not None and current_y is not None:
                                coord_text = f"Current Gaze: ({current_x:.1f}, {current_y:.1f})"
                                cv2.putText(combined_display, coord_text, (timer_x, timer_y + 50), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            # 실험 지시사항
                            instruction_text = "Keep your gaze steady on the target for 10 seconds"
                            cv2.putText(combined_display, instruction_text, (timer_x, timer_y + 80), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                            
                            # 키보드 단축키 안내
                            key_info = "Press 'x' to stop recording manually"
                            cv2.putText(combined_display, key_info, (timer_x, timer_y + 100), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                        
                        # 실험 결과 미리보기 제거 (타겟은 사용자가 직접 만듦)
                        
                        # 실험 모드 상태 표시
                        status_text = "EXPERIMENT MODE: ACTIVE"
                        cv2.putText(combined_display, status_text, (20, 180), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # 실험 진행 단계 표시
                        if hasattr(self, 'target_a_stats') and self.target_a_stats:
                            if hasattr(self, 'target_b_stats') and self.target_b_stats:
                                stage_text = "Stage: COMPLETED - Press 'c' to compare results"
                                cv2.putText(combined_display, stage_text, (400, 180), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            else:
                                stage_text = "Stage: Target A completed - Press 'b' for Target B"
                                cv2.putText(combined_display, stage_text, (400, 180), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        else:
                            stage_text = "Stage: Ready - Press 'a' to start Target A"
                            cv2.putText(combined_display, stage_text, (400, 180), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        # 실험 모드 비활성화 상태 (정보 패널에)
                        status_text = "EXPERIMENT MODE: INACTIVE"
                        cv2.putText(combined_display, status_text, (20, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                        
                        # 실험 모드 활성화 안내
                        instruction_text = "Press 'e' to activate experiment mode"
                        cv2.putText(combined_display, instruction_text, (20, 80), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        
                        # 키보드 단축키 안내
                        key_info = "Available keys: 'e' (experiment), 'm' (mirror), 'q' (quit)"
                        cv2.putText(combined_display, key_info, (20, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                
                # 중첩 영역 품질 확인 (첫 프레임에서만)
                if frame_count == 0:
                    if hasattr(self, 'left_blend_mask') and hasattr(self, 'right_blend_mask'):
                        overlap_quality = np.sum((self.left_blend_mask > 0) & (self.right_blend_mask > 0))
                        print(f"   📊 중첩 영역 품질: {overlap_quality} 픽셀")
                        print(f"   🎯 블렌딩 마스크 적용 완료")
                        print(f"   🎨 그레이스케일 변환 완료")
                        print(f"   📊 채널 정보: 입력 3채널 → 처리 1채널 → 출력 1채널")
                
                # 결과 표시 (정보 패널 + 비디오 화면)
                cv2.imshow('Grayscale Real-time Stitching + Experiment Panel', combined_display)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("💡 'q' 키 입력으로 종료")
                    break
                elif key == ord('m'):
                    mirror_mode = not mirror_mode
                    print(f"🔄 거울모드: {'활성화' if mirror_mode else '비활성화'}")
                    
                    # 거울모드 전환 (이미지 좌우 반전만)
                    self.switch_to_mirror_mode(mirror_mode)
                
                # 🚀 실험 모드 키 입력 처리
                elif key == ord('e'):
                    self.toggle_experiment_mode()
                elif key == ord('a') and self.experiment_mode:
                    self.start_target_recording('A')
                elif key == ord('b') and self.experiment_mode:
                    self.start_target_recording('B')
                elif key == ord('s') and self.experiment_mode:
                    self.show_experiment_statistics()
                elif key == ord('c') and self.experiment_mode:
                    self.compare_experiment_targets()
                elif key == ord('f') and self.experiment_mode:
                    self.save_experiment_results()
                elif key == ord('r') and self.experiment_mode:
                    self.reset_experiment_data()
                elif key == ord('x') and self.experiment_mode:
                    # 🚀 수동으로 기록 중지
                    if hasattr(self, 'gaze_experiment') and self.gaze_experiment.is_recording:
                        print(f"🛑 수동으로 기록을 중지합니다.")
                        self.stop_target_recording()
                    else:
                        print(f"⚠️ 현재 기록 중이 아닙니다.")
                elif key == ord('p'):
                    # 🚀 성능 측정 토글
                    self.performance_measurement = not self.performance_measurement
                    if self.performance_measurement:
                        print(f"📊 성능 측정 활성화 - FPS 로그와 성능 통계가 출력됩니다")
                    else:
                        print(f"📊 성능 측정 비활성화 - 콘솔 출력이 최소화됩니다")
                
                # 🚀 프레임 카운터 증가 (성능 측정용)
                frame_count += 1
                self.frame_count = frame_count  # 클래스 변수에 할당
                

                
        except KeyboardInterrupt:
            print("\n💡 Ctrl+C로 종료")
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
        finally:
            # 정리
            if self.cap_left:
                self.cap_left.release()
            if self.cap_right:
                self.cap_right.release()
            
            # 🚀 아이트래킹 스레드 정리
            self.gaze_thread_running = False
            if hasattr(self, 'gaze_thread') and self.gaze_thread.is_alive():
                self.gaze_thread.join(timeout=1.0)
            
            cv2.destroyAllWindows()
            print("✅ 정리 완료")

    def release(self):
        """리소스 해제"""
        if hasattr(self, 'cap_left') and self.cap_left:
            self.cap_left.release()
        if hasattr(self, 'cap_right') and self.cap_right:
            self.cap_right.release()
        
        # 🚀 아이트래킹 스레드 정리
        if hasattr(self, 'gaze_thread_running'):
            self.gaze_thread_running = False
        if hasattr(self, 'gaze_thread') and self.gaze_thread.is_alive():
            self.gaze_thread.join(timeout=1.0)
        
        cv2.destroyAllWindows()
        print("✅ 모든 리소스 해제 완료")

    def process_gaze_tracking(self, frame: np.ndarray) -> Tuple[np.ndarray, list]:
        """🚀 아이트래킹 처리 (그레이스케일 호환)"""
        if self.yolo_model is None or self.face_mesh is None:
            return frame, []
        
        try:
            # YOLO 얼굴 검출 (그레이스케일에서도 작동)
            yolo_results = self.yolo_model.predict(frame, verbose=False)[0]
            face_boxes = [
                box for box in yolo_results.boxes.data.cpu().numpy()
                if int(box[5]) == 0  # 클래스 0: 얼굴
            ]
            
            results = []
            h, w = frame.shape[:2]  # 그레이스케일: shape[0], shape[1]
            
            for face_idx, box in enumerate(face_boxes):
                if face_idx >= GAZE_TRACKING_CONFIG['max_faces']:
                    break
                    
                x1, y1, x2, y2, conf, cls = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face_roi = frame[y1:y2, x1:x2]
                # 🆕 그레이스케일을 RGB로 변환 (MediaPipe는 RGB 필요)
                if len(face_roi.shape) == 2:  # 그레이스케일인 경우
                    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
                else:  # 이미 RGB인 경우
                    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                mesh_result = self.face_mesh.process(face_rgb)

                if mesh_result.multi_face_landmarks:
                    for face_landmarks in mesh_result.multi_face_landmarks:
                        # 기본 랜드마크 포인트들
                        image_points = []
                        for idx in self.landmark_ids:
                            lm = face_landmarks.landmark[idx]
                            x_lm = int(lm.x * (x2 - x1)) + x1
                            y_lm = int(lm.y * (y2 - y1)) + y1
                            image_points.append((x_lm, y_lm))
                            # 🆕 그레이스케일에서는 흰색으로 표시
                            cv2.circle(frame, (x_lm, y_lm), 2, 255, -1)

                        # 홍채 중심점 추출
                        iris_coords = []
                        for idx in self.iris_ids:
                            if idx < len(face_landmarks.landmark):
                                lm = face_landmarks.landmark[idx]
                                x_lm = int(lm.x * (x2 - x1)) + x1
                                y_lm = int(lm.y * (y2 - y1)) + y1
                                iris_coords.append((x_lm, y_lm))
                                # 🆕 그레이스케일에서는 흰색으로 표시
                                cv2.circle(frame, (x_lm, y_lm), 3, 255, -1)
                        
                        # 눈 중심 계산
                        roi_info = (x1, y1, x2-x1, y2-y1)
                        left_eye_center = self._get_eye_center_from_landmarks(face_landmarks, self.left_eye_landmarks, roi_info)
                        right_eye_center = self._get_eye_center_from_landmarks(face_landmarks, self.right_eye_landmarks, roi_info)
                        
                        eye_centers = []
                        if left_eye_center:
                            eye_centers.append(left_eye_center)
                        if right_eye_center:
                            eye_centers.append(right_eye_center)

                        # 시선 추정
                        gaze = (0, 0)
                        
                        # 홍채 중심 기반
                        if len(iris_coords) == 2:
                            lx, ly = iris_coords[0]
                            rx, ry = iris_coords[1]
                            iris_gaze = ((rx + lx) / 2 / w - 0.5, (ry + ly) / 2 / h - 0.5)
                        else:
                            iris_gaze = None
                        
                        # 눈 주위 랜드마크 중심 기반
                        if len(eye_centers) == 2:
                            lx, ly = eye_centers[0]
                            rx, ry = eye_centers[1]
                            landmark_gaze = ((rx + lx) / 2 / w - 0.5, (ry + ly) / 2 / h - 0.5)
                        else:
                            landmark_gaze = None
                        
                        # 두 방법의 가중 평균
                        if iris_gaze and landmark_gaze:
                            gaze = (
                                0.7 * iris_gaze[0] + 0.3 * landmark_gaze[0],
                                0.7 * iris_gaze[1] + 0.3 * landmark_gaze[1]
                            )
                        elif iris_gaze:
                            gaze = iris_gaze
                        elif landmark_gaze:
                            gaze = landmark_gaze

                        # 시간적 안정화 적용
                        if face_idx < len(self.gaze_stabilizers):
                            stabilizer = self.gaze_stabilizers[face_idx]
                            stabilizer.add_sample(gaze, iris_coords)
                            stable_gaze = stabilizer.get_stabilized_gaze()
                            gaze = stable_gaze

                        # 🚀 실험 데이터 기록 (첫 번째 얼굴만)
                        if face_idx == 0 and hasattr(self, 'gaze_experiment'):
                            # 시선 좌표를 픽셀 좌표로 변환
                            gaze_x = (gaze[0] + 0.5) * w  # -0.5~0.5 → 0~w
                            gaze_y = (gaze[1] + 0.5) * h  # -0.5~0.5 → 0~h
                            
                            # 실험 기록기에 시선 좌표 추가
                            self.gaze_experiment.add_gaze_sample(gaze_x, gaze_y)

                        results.append({"gaze": gaze, "iris_coords": iris_coords})

            return frame, results
            
        except Exception as e:
            print(f"⚠️ 아이트래킹 처리 오류: {e}")
            return frame, []

    def _get_eye_center_from_landmarks(self, face_landmarks, eye_landmark_ids, roi_offset):
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
            center_x = int(np.mean([p[0] for p in eye_points]))
            center_y = int(np.mean([p[1] for p in eye_points]))
            return (center_x, center_y)
        return None

    def draw_gaze_info(self, frame: np.ndarray, gaze_data: list, fps: float):
        """🚀 시선 추적 정보 표시 (그레이스케일 호환)"""
        # FPS 표시 (그레이스케일에서는 흰색)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        
        # 추적 상태 표시 (그레이스케일에서는 흰색/검은색)
        status_color = 255 if len(gaze_data) > 0 else 0
        cv2.putText(frame, f"Faces: {len(gaze_data)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # 각 얼굴의 시선 정보 표시
        for i, data in enumerate(gaze_data):
            y_offset = 90 + i * 120
            cv2.putText(frame, f"Person {i+1}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
            cv2.putText(frame, f"Gaze: ({data['gaze'][0]:.3f}, {data['gaze'][1]:.3f})", (10, y_offset+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
            
            # 시선 방향 표시 (화면 중앙 기준)
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            gaze_x = int(center_x + data['gaze'][0] * 200)  # 스케일 조정
            gaze_y = int(center_y + data['gaze'][1] * 200)
            cv2.arrowedLine(frame, (center_x, center_y), (gaze_x, gaze_y), 255, 3)
        
        # 아이트래킹 활성화 표시 (그레이스케일에서는 흰색)
        cv2.putText(frame, "EYE TRACKING: ON", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)

    def stitch_frame_pair_ultra_optimized(self, left_frame: np.ndarray, right_frame: np.ndarray) -> np.ndarray:
        """🚀 초고속 프레임 스티칭 (ROI 기반 극한 최적화!)"""
        
        # 🚀 ROI 모드와 전체 모드 분기 처리
        if hasattr(self, 'use_roi') and self.use_roi and hasattr(self, 'roi_homography_coords'):
            return self._stitch_frame_pair_roi_optimized(left_frame, right_frame)
        else:
            return self._stitch_frame_pair_full_optimized(left_frame, right_frame)
    
    def _stitch_frame_pair_roi_optimized(self, left_frame: np.ndarray, right_frame: np.ndarray) -> np.ndarray:
        """🚀 ROI 기반 극한 최적화 스티칭 (전체 스티칭 후 ROI 크롭!)"""
        # 🎯 사용자가 원한 방식: 전체 스티칭 후 ROI 크롭!
        
        # 🚀 1단계: 전체 스티칭 (기존 방식 사용!)
        full_stitched = self._stitch_frame_pair_full_optimized(left_frame, right_frame)
        
        # 🚀 2단계: ROI 크롭 (사용자가 선택한 ROI 영역만!)
        roi_start = time.time()
        
        # 🚨 ROI 좌표는 전체 캔버스 기준이므로 그대로 사용
        # (Stitching_Engine에서 0.5 스케일로 선택했지만, 좌표는 전체 캔버스 기준)
        roi_image = full_stitched[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2]
        
        roi_time = time.time() - roi_start
        self.step_times['roi_crop'].append(roi_time)
        
        return roi_image
    
    def _stitch_frame_pair_full_optimized(self, left_frame: np.ndarray, right_frame: np.ndarray) -> np.ndarray:
        """🚀 전체 크기 기존 최적화 스티칭 (기존 로직 유지)"""
        # 🚀 1단계: 렌즈 보정 시간 측정 (원래 방식으로 복원!)
        lens_start = time.time()
        left_rectified = self.rectify_left(left_frame)
        right_rectified = self.rectify_right(right_frame)
        lens_time = time.time() - lens_start
        self.step_times['lens_rectification'].append(lens_time)
        
        # 🚀 2단계: 캔버스 복사 시간 측정
        canvas_start = time.time()
        
        # 🚨 항상 전체 크기 캔버스 사용 (ROI는 나중에 크롭!)
        # 🆕 그레이스케일용 1채널 캔버스 생성
        canvas = np.zeros((self.canvas_size[1], self.canvas_size[0]), dtype=np.uint8)
        canvas[self.left_slice_y, self.left_slice_x] = left_rectified
        
        canvas_time = time.time() - canvas_start
        self.step_times['canvas_copy'].append(canvas_time)
        
        # 🚀 3단계: 호모그래피 변환 시간 측정 (원래 방식으로 복원!)
        homography_start = time.time()
        
        # 🚨 원래 방식: cv2.warpPerspective 사용 (ROI는 나중에 크롭!)
        warped_right = cv2.warpPerspective(right_rectified, self.homography_matrix_opt, 
                                       self.homography_size, flags=cv2.INTER_LINEAR)
        
        homography_time = time.time() - homography_start
        self.step_times['homography_warp'].append(homography_time)
        
        # 🚀 4단계: 블렌딩 시간 측정
        blending_start = time.time()
        
        # 🚀 빠른 블렌딩 사용 (마스크 계산 안함!)
        final_image = self._blend_images_ultra_fast(canvas, warped_right)
        
        blending_time = time.time() - blending_start
        self.step_times['blending'].append(blending_time)
        
        return final_image

    def _blend_images_roi_fast(self, left_roi: np.ndarray, right_roi: np.ndarray) -> np.ndarray:
        """🚀 ROI 크기 이미지 초고속 블렌딩 (ROI 전용!)"""
        # ROI 크기로 간단한 블렌딩 (가장 빠른 방식)
        # 왼쪽 이미지를 기본으로 하고, 오른쪽 이미지가 있는 부분만 덮어쓰기
        result = left_roi.copy()
        mask = right_roi > 10  # 임계값으로 마스크 생성
        result[mask] = right_roi[mask]
        return result

    def _blend_images_ultra_fast_full(self, left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
        """🚀 전체 크기 이미지 초고속 블렌딩 (ROI 처리용)"""
        if hasattr(self, 'use_precomputed_blending') and self.use_precomputed_blending:
            # 원본 블렌딩 마스크 사용 (전체 크기)
            full_left_mask = self.left_blend_mask_original
            full_right_mask = self.right_blend_mask_original
            
            # 3채널 마스크 생성
            full_left_mask_3ch = np.stack([full_left_mask] * 3, axis=-1)
            full_right_mask_3ch = np.stack([full_right_mask] * 3, axis=-1)
            
            # 가중 평균 블렌딩
            blended = (left_image * full_left_mask_3ch + right_image * full_right_mask_3ch).astype(np.uint8)
        else:
            # 마스크가 없으면 단순 덧셈
            blended = np.clip(left_image + right_image, 0, 255).astype(np.uint8)
        
        return blended

    def _apply_lens_correction_fast(self, frame, is_left=True):
        """🚀 LUT 기반 빠른 렌즈 보정"""
        if is_left:
            return cv2.remap(frame, self.left_lut_x, self.left_lut_y, 
                            cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)
        else:
            return cv2.remap(frame, self.right_lut_x, self.right_lut_y, 
                            cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)

    def _apply_homography_fast(self, frame):
        """🚀 사전 계산된 좌표 맵으로 빠른 호모그래피 변환"""
        # 🚨 중요: frame은 카메라 해상도(1080x1920), 좌표 맵은 전체 캔버스 크기(1342x3633)
        frame_h, frame_w = frame.shape[:2]
        canvas_h, canvas_w = self.canvas_size[1], self.canvas_size[0]
        
        # 좌표를 정수로 변환하여 샘플링
        coords = self.homography_coords  # 1342x3633x2
        
        # 좌표 범위 검증 및 클리핑
        coords_int = coords.astype(np.int32)
        coords_int[:, :, 0] = np.clip(coords_int[:, :, 0], 0, frame_w-1)  # x 좌표
        coords_int[:, :, 1] = np.clip(coords_int[:, :, 1], 0, frame_h-1)  # y 좌표
        
        # 벡터화된 샘플링 (훨씬 빠름!)
        warped = frame[coords_int[:, :, 1], coords_int[:, :, 0]]
        
        return warped


if __name__ == "__main__":
# 🚀 단색(그레이스케일) 초고속 최적화된 실시간 스티칭 실행
    print("🚀 Ultra-Fast Video Stitcher Grayscale + Eye Tracking v4.0 시작")
    print("예상 성능 향상: 10 FPS → 40+ FPS (300%+ 향상)")
    print("👁️ 아이트래킹: YOLO 얼굴 검출 + MediaPipe FaceMesh + 시선 안정화 (그레이스케일)")
    print("🎯 ROI 최적화: 사용자 정의 ROI 지원으로 추가 FPS 향상")
    print("🚫 렌즈 보정 LUT: 사전 계산 제거 (성능 저하로 인해)")
    print("🚫 호모그래피 좌표 맵: 사전 계산 제거 (성능 저하로 인해)")
    print("🎨 그레이스케일 모드: 1채널 처리로 메모리 1/3, 처리 속도 2-3배 향상")
    
    # 🚀 동공 추적 실험 기능 안내
    print("\n🔬 동공 추적 실험 기능:")
    print("   - 0.5도 시선 추적 정확도 측정")
    print("   - 타겟 A, B 응시 데이터 수집")
    print("   - 실시간 통계 분석 및 비교")
    print("   - 실험 결과 파일 저장")
    print("   - 키보드 단축키:")
    print("     'e': 실험 모드 토글")
    print("     'a': 타겟 A 응시 기록 시작")
    print("     'b': 타겟 B 응시 기록 시작")
    print("     'x': 수동 기록 중지")
    print("     's': 현재 통계 출력")
    print("     'c': 타겟 A vs B 비교")
    print("     'f': 결과를 파일로 저장")
    print("     'r': 실험 데이터 초기화")
    print("     'm': 거울모드 토글")
    print("     'p': 성능 측정 토글 (콘솔 출력 제어)")
    print("     'q': 종료")
    print("   - 실험 진행:")
    print("     1. 'e'로 실험 모드 활성화")
    print("     2. 'a'로 타겟 A 응시 시작 (10초 자동 기록)")
    print("     3. 'b'로 타겟 B 응시 시작 (10초 자동 기록)")
    print("     4. 'c'로 결과 비교 분석")
    print("     5. 'f'로 결과 파일 저장")
    print("   - 화면 표시:")
    print("     - 모든 텍스트 제거됨 (깔끔한 화면)")
    print("     - 상단에 타이머 바만 표시")
    print("     - 현재 시선 위치를 빨간색 점으로 표시")
    print("     - 콘솔에서 실시간 진행 상황 확인")
    print("   - 성능 측정:")
    print("     - 기본값: 비활성화 (콘솔 출력 최소화)")
    print("     - 'p' 키로 필요시에만 활성화 가능")
    print("     - 활성화 시: FPS 로그 10초마다, 성능 통계 30초마다")
    
    stitcher = UltraFastVideoStitcherGrayscale("./data/config/homography_params.json")
    
    # 🆕 ROI 모드 상태 확인
    if hasattr(stitcher, 'use_roi') and stitcher.use_roi:
        print(f"✅ ROI 모드 활성화: {stitcher.roi_width} x {stitcher.roi_height}")
        print("   🚀 ROI 기반 최적화 + 그레이스케일로 극한 FPS 향상 예상")
    else:
        print("ℹ️ ROI 모드 비활성화: 전체 화면 모드")
        print("   💡 ROI를 사용하려면 Stitching_Engine_3.py에서 ROI를 선택하세요")
    
    # 카메라 설정
    if not stitcher.setup_cameras():
        print("❌ 카메라 설정 실패")
        exit(1)
    
    stitcher.show_stitched_video_grayscale()
