# IR 기반 운전자 모니터링 시스템(DMS) — 실시간 성능 평가 보고서

**평가일**: 2026년 3월 27일  
**평가자**: 임베디드 시스템 및 실시간 AI 추론 최적화 전문가  
**프로젝트**: 저조도/차폐 상황 강건한 운전자 모니터링 시스템

---

## 1. 각 케이스별 예상 레이턴시 추정

### 1.1 레이턴시 예상 모델 (GPU RTX 3080 기준)

| 모듈 | 예상 시간 | 비고 |
|------|---------|------|
| IR 전처리 | 5ms | 히스토그램 균등화, 대비 조정 |
| 얼굴 검출 (YOLO) | 8ms | 필수 병목 단계 |
| MediaPipe Face Mesh | 12ms | 468개 3D 랜드마크 |
| MediaPipe Pose | 10ms | 33개 바디 키포인트 |
| MoveNet Lightning | 6ms | 경량 포즈 추정 |
| 눈 영역 CNN (MobileNet) | 8ms | 눈 특징 추출 |
| 머리 포즈 추정 (6DRepNet) | 8ms | 3축 회전 각도 |
| 시선 추정 CNN (ResNet-18) | 12ms | 이미지 기반 |
| 시선 추정 MLP (랜드마크) | 5ms | 경량 |
| Adaptive Fusion | 3ms | confidence score 계산 |
| ST-GCN | 15ms | 그래프 컨볼루션 (~100 노드) |
| Temporal Transformer | 20ms | 시퀀스 모델링 (5-10 프레임) |
| Temporal LSTM | 10ms | 경량 시간축 모델 |
| MLP Classifier | 3ms | 최종 분류기 |

---

### 1.2 케이스별 상세 레이턴시 분석

#### **Case 1 (가장 복잡): 랜드마크 + 포즈 + 눈영역 → ST-GCN → Temporal Transformer**

**파이프라인:**
```
IR 입력 (5ms)
  ↓
얼굴 검출 (8ms)
  ↓
┌─────────────────────────────┐
│ 병렬 처리 (최대 12ms):        │
│ • Face Mesh (12ms)          │
│ • Pose Extraction (10ms)    │
│ • Eye Region CNN (8ms)      │
└─────────────────────────────┘
  ↓
Adaptive Fusion (3ms)
  ↓
ST-GCN (15ms) → Temporal Transformer (20ms)
  ↓
분류기 (3ms)
```

**레이턴시 계산:**
- 순차 부분: 5 + 8 + 3 + 15 + 20 + 3 = 54ms
- 병렬 부분: max(12, 10, 8) = 12ms (최대값)
- **총합: 54 + 12 = 66ms**
- 시퀀스 모델 오버헤드: ~5ms (프레임 버퍼링)
- **최종 추정 레이턴시: ~71ms**

**300ms 여유도**: 229ms ✓ (매우 여유 있음)

**평가:**
- ✓ 0.3초 요구사항 충족 (여유도 76%)
- ⚠ ST-GCN (15ms) + Transformer (20ms)가 총 시간의 50% 점유
- ⚠ 가장 복잡하지만 충분한 여유로 안정성 확보 가능

---

#### **Case 2: 랜드마크 → 시선 추정 MLP + 포즈 → LSTM**

**파이프라인:**
```
IR 입력 (5ms)
  ↓
얼굴 검출 (8ms) → Face Mesh (12ms)
  ↓
┌──────────────────────────┐
│ 병렬 처리 (최대 10ms):   │
│ • Gaze MLP (5ms)         │
│ • Pose (10ms)            │
└──────────────────────────┘
  ↓
Adaptive Fusion (3ms) → LSTM (10ms)
  ↓
분류기 (3ms)
```

**레이턴시 계산:**
- 순차: 5 + 8 + 12 + 3 + 10 + 3 = 41ms
- 병렬: max(5, 10) = 10ms
- **총합: 41 + 10 = 51ms**
- 시퀀스 오버헤드: ~5ms
- **최종 추정 레이턴시: ~56ms**

**300ms 여유도**: 244ms ✓ (여유도 81%)

**평가:**
- ✓ 0.3초 요구사항 충족 (여유도 매우 큼)
- ✓ 시선 추정 중간 단계가 해석성 개선
- ✓ LSTM (10ms)으로 경량화 가능
- ✓ Colab Pro 환경에서 실행 용이

---

#### **Case 3: 이미지 → 시선 CNN + 포즈 → LSTM**

**파이프라인:**
```
IR 입력 (5ms)
  ↓
얼굴 검출 (8ms)
  ↓
┌──────────────────────────┐
│ 병렬 처리 (최대 12ms):   │
│ • Gaze CNN (12ms)        │
│ • Pose (10ms)            │
└──────────────────────────┘
  ↓
Adaptive Fusion (3ms) → LSTM (10ms)
  ↓
분류기 (3ms)
```

**레이턴시 계산:**
- 순차: 5 + 8 + 3 + 10 + 3 = 29ms
- 병렬: max(12, 10) = 12ms
- **총합: 29 + 12 = 41ms**
- 시퀀스 오버헤드: ~5ms
- **최종 추정 레이턴시: ~46ms**

**300ms 여유도**: 254ms ✓ (여유도 85%)

**평가:**
- ✓ 0.3초 요구사항 충족 (최고 여유도)
- ✓ End-to-End 시선 추정으로 단순화
- ✓ 랜드마크 검출 오류 전파 감소
- ✓ 가장 빠른 추론 속도

---

#### **Case 4 (베이스라인): 머리 포즈 + 스켈레톤 → MLP**

**파이프라인:**
```
IR 입력 (5ms)
  ↓
얼굴 검출 (8ms)
  ↓
┌──────────────────────────┐
│ 병렬 처리 (최대 10ms):   │
│ • Head Pose (8ms)        │
│ • Pose (10ms)            │
└──────────────────────────┘
  ↓
MLP 분류기 (3ms)
```

**레이턴시 계산:**
- 순차: 5 + 8 + 3 = 16ms
- 병렬: max(8, 10) = 10ms
- **총합: 16 + 10 = 26ms**
- 시퀀스 오버헤드: ~5ms
- **최종 추정 레이턴시: ~31ms**

**300ms 여유도**: 269ms ✓ (여유도 90%)

**평가:**
- ✓ 0.3초 요구사항 충족 (초과 여유)
- ✓ 가장 단순한 구조로 안정성 우수
- ⚠ 시선 정보 부재 → 정확도 저하 가능성
- ✓ Colab Pro, DGX Spark 모두 무난

---

#### **Case 5 (최소 베이스라인): 스켈레톤만 → MLP**

**파이프라인:**
```
IR 입력 (5ms)
  ↓
얼굴 검출 (8ms)
  ↓
Pose 추출 (10ms)
  ↓
MLP 분류기 (3ms)
```

**레이턴시 계산:**
- 순차: 5 + 8 + 10 + 3 = 26ms
- 병렬: 없음
- **총합: 26ms**
- 시퀀스 오버헤드: ~5ms
- **최종 추정 레이턴시: ~31ms**

**300ms 여유도**: 269ms ✓ (여유도 90%)

**평가:**
- ✓ 0.3초 요구사항 충족
- ✓ 병렬화 불필요 (메모리 효율)
- ✗ 시선 정보 전무 → 정확도 하한선
- ✓ 벤치마크 용도로만 적합

---

### 1.3 레이턴시 비교 요약

| 케이스 | 추정 레이턴시 | 여유도 | 0.3초 충족 | 평가 |
|--------|------------|------|----------|------|
| Case 1 | 71ms | 229ms (76%) | ✓ | 최고 성능, 안정성 우수 |
| Case 2 | 56ms | 244ms (81%) | ✓ | 균형잡힘 |
| Case 3 | 46ms | 254ms (85%) | ✓ | 가장 빠름 |
| Case 4 | 31ms | 269ms (90%) | ✓ | 단순하지만 정확도 의문 |
| Case 5 | 31ms | 269ms (90%) | ✓ | 하한선 (비추천) |

**결론: 모든 케이스가 0.3초 요구사항을 충족합니다.**

---

## 2. 0.3초 요구사항 충족 평가

### 2.1 충족 가능 케이스

**✓ Case 1, 2, 3 (강력히 추천)**
- 여유도 70% 이상으로 안정적 동작 확보
- 예상외 지연(네트워크, OS 스케줄링, 메모리 페이징 등) 대응 가능
- 추가 후처리(신뢰도 시각화, 경고음 재생 등) 추가 여유

**△ Case 4 (조건부 충족)**
- 레이턴시는 매우 낮지만 (31ms)
- 정확도 미검증 → 실제 성능 불확실
- 포즈만으로는 야간/차폐 상황에서 문제 가능성

### 2.2 어려운 케이스

**✗ Case 5 (비권장)**
- 순수 신체 움직임만으로 운전 집중도 판단 어려움
- 벤치마크용도로만 가치 있음

---

## 3. 경량화 방안의 실효성 및 우선순위

### 3.1 경량화 기법 평가

| 기법 | 효과 | 난이도 | 우선순위 | 비고 |
|------|------|--------|---------|------|
| **Knowledge Distillation** | 30-40% 감소 | 중 | 1순위 | Case 1의 Transformer 경량화에 최적 |
| **INT8 Quantization** | 20-30% 감소 | 낮 | 1순위 | 추론 전용, 호환성 우수 |
| **Pruning (50%)** | 40-50% 감소 | 중 | 2순위 | ST-GCN/Transformer 스파시티화 |
| **경량 백본 선택** | 30-40% 감소 | 낮 | 3순위 | MobileNetV3, EfficientNet-B0 |
| **프레임 샘플링** | 25-30% 감소 | 낮 | 2순위 | 2-3 프레임 간격 처리 |
| **Adaptive Computation** | 15-25% 감소 | 중 | 2순위 | Early exit, 신뢰도 threshold |
| **ONNX/TensorRT 최적화** | 15-20% 감소 | 낮 | 3순위 | 배포 단계 적용 |

### 3.2 권장 적용 순서

**1단계 (즉시, 개발 초기)**
- INT8 Quantization: RTX 3080 호환성 100%, 구현 간단
- 프레임 샘플링 정책 결정: 2-3 프레임 간격 추천 (50-75fps 카메라 대응)

**2단계 (중간 발표 전)**
- Knowledge Distillation (Case 1의 Transformer 경량화)
  - Teacher: 현재 모델
  - Student: 50% 파라미터 축소 Transformer
  - 성능 유지율: 95-98% 목표
  
**3단계 (최종 발표)**
- 선택적 Pruning (50% 스파시티)
- TensorRT 최적화
- 모바일/엣지 배포 검토

---

## 4. 병렬 처리 모듈 분석

### 4.1 병렬 처리 가능 모듈 (Case별)

#### **Case 1**
```
얼굴 검출 (8ms) [필수 순차]
        ↓
     [병렬 시작]
     ↙   ↓   ↘
Face  Pose  Eye
Mesh  (10) CNN
(12)       (8)
[병렬 끝]
     ↓
Fusion (3ms) → ST-GCN (15ms) → Transformer (20ms)
```

**병렬화 이득:**
- 순차 실행: 12 + 10 + 8 = 30ms
- 병렬 실행: max(12, 10, 8) = 12ms
- **절감: 18ms (60% 단축)**

**구현 방안:**
```python
# 예시 코드 구조
with ThreadPoolExecutor(max_workers=3) as executor:
    face_future = executor.submit(extract_face_mesh, roi)
    pose_future = executor.submit(extract_pose, roi)
    eye_future = executor.submit(extract_eye_features, eye_roi)
    
    face_result = face_future.result()
    pose_result = pose_future.result()
    eye_result = eye_future.result()
```

#### **Case 2**
```
Face Mesh (12ms) [순차 필수]
     ↓
  [병렬 시작]
  ↙          ↘
Gaze MLP   Pose
(5ms)      (10ms)
  [병렬 끝]
     ↓
Fusion → LSTM
```

**병렬화 이득:**
- 순차: 5 + 10 = 15ms
- 병렬: max(5, 10) = 10ms
- **절감: 5ms (33% 단축)**

#### **Case 3**
```
얼굴 검출 (8ms)
     ↓
  [병렬 시작]
  ↙          ↘
Gaze CNN   Pose
(12ms)     (10ms)
  [병렬 끝]
     ↓
Fusion → LSTM
```

**병렬화 이득:**
- 순차: 12 + 10 = 22ms
- 병렬: max(12, 10) = 12ms
- **절감: 10ms (45% 단축)**

#### **Case 4**
```
얼굴 검출 (8ms)
     ↓
  [병렬 시작]
  ↙          ↘
Head Pose  Pose
(8ms)      (10ms)
  [병렬 끝]
     ↓
MLP
```

**병렬화 이득:**
- 순차: 8 + 10 = 18ms
- 병렬: max(8, 10) = 10ms
- **절감: 8ms (44% 단축)**

### 4.2 순차 처리 필수 모듈

모든 케이스에서 순차 필수:
1. **IR 전처리** (5ms) - 입력 정규화
2. **얼굴 검출** (8ms) - ROI 추출
3. **Fusion 이후** - 시계열 의존성 발생
4. **최종 분류** (3ms)

---

## 5. Colab Pro / DGX Spark 환경 제약사항

### 5.1 Colab Pro 환경 분석

| 항목 | 사양 | 충분성 |
|------|------|--------|
| GPU | T4 or L4 | ✓ (모든 케이스 가능) |
| VRAM | 15-16GB | ✓ (Case 1도 무난) |
| 토큰 속도 | ~60M tokens/min | N/A |
| 배치 크기 | 최대 128 | ✓ |
| 세션 지속시간 | 최대 24시간 | △ (중단 가능성) |
| 네트워크 | 전용 | ✓ |

**예상 추론 속도:**
- **T4 GPU**: Case 1 약 80-100ms (RTX 3080 기준 71ms대비 10-40% 느림)
- **L4 GPU**: Case 1 약 60-70ms (거의 동일)

**권장:**
- L4 GPU 사용 (약간의 추가 비용으로 안정성 확보)
- 학습 초기: T4로 시작, 최종 추론: L4로 최적화

**제약:**
- 세션 중단 시 재연결 필요 (체크포인트 저장 필수)
- 데이터셋 대용량 처리 시 타임아웃 위험
- 실시간 스트림 처리는 어려움

---

### 5.2 DGX Spark 환경 분석

| 항목 | 사양 | 충분성 |
|------|------|--------|
| GPU | A100 (40GB) | ✓✓ (과도) |
| VRAM | 40GB | ✓✓ (과도) |
| 추론 속도 | RTX 3080 대비 3-5배 | ✓✓ |
| 세션 지속시간 | 실제 제한 없음 | ✓✓ |
| 네트워크 지연 | 낮음 | ✓ |
| 예약 가능성 | 제한적 | △ |

**예상 추론 속도:**
- **Case 1**: 15-25ms (매우 빠름)
- **Case 3**: 12-18ms

**권장:**
- 데이터셋 구축, 학습용으로 최적 (단기 사용)
- 최종 성능 검증에 활용
- 중간 발표 전 테스트 완료

**제약:**
- 할당 시간 제한 (교수님 협의 필요)
- 4월 초까지 승인 프로세스 (중간 발표 후)
- 대기 시간 가능성

---

## 6. 확산 모델(GazeD)의 실시간 추론 가능성

### 6.1 GazeD 개요

**논문**: "GazeD: Context-Aware Diffusion for Accurate 3D Gaze Estimation" (2026)

**구조:**
- 2D → 3D 리프팅 (확산 모델 기반)
- Forward Process: 사람의 시선 분포 학습
- Reverse Process: 2D 특징 → 3D 시선 벡터 복원

---

### 6.2 확산 모델 레이턴시 분석

**확산 모델의 근본적 문제:**

```
표준 확산 모델 추론:
1단계 (Noise 초기화): 5ms
2단계 (Reverse Process):
    Step 1: 20ms (1000 스텝 중)
    Step 2: 20ms
    ...
    Step N: 20ms
    = 총 1000 × 20ms = 20,000ms ❌

실시간 불가능!
```

**가능한 최적화 기법:**

| 기법 | 효과 | 추정 레이턴시 |
|------|------|-------------|
| **Reverse Step 축소** | 1000 → 10 스텝 | 200ms |
| **Classifier-Free Guidance 제거** | 50% 속도 향상 | 100ms |
| **Continuous Reverse Process** | 1-스텝 증류 | 50-80ms |
| **확산 모델 대체 (VAE)** | 완전 대체 | 10-15ms |

### 6.3 권장 접근법

#### **방안 A: 확산 모델 배제 (강력히 권장)**

**현실적 이유:**
- 표준 확산 모델: 1000 스텝 → 20초 추론 (실시간 불가)
- 경량화: 100 스텝 축소 → 2초 (여전히 부족)
- 1-스텝 증류: 이론적 가능하나 학습 복잡도 매우 높음

**대체안: 결정론적 3D 시선 추정**
```
방법 1: Regression-Based (권장)
  • 2D 랜드마크 → MLP → 3D 벡터
  • 레이턴시: 5ms
  • 정확도: 매우 좋음 (적절히 학습 시)

방법 2: Graph-Based
  • 2D 좌표 → GCN → 3D 벡터
  • 레이턴시: 8-10ms
  • 정확도: 우수

방법 3: 물리적 제약 (최적)
  • 머리 포즈 + 눈 특징 → 기하학적 계산
  • 레이턴시: 3ms
  • 정확도: 강건함
```

#### **방안 B: 경량화된 확산 모델 (시간 여유 시)**

**필요 조건:**
- 충분한 개발 시간 (4-5주 이상)
- 고급 GPU 리소스 (DGX Spark 장시간 할당)
- 선행 연구 참조 가능성

**목표 사양:**
```
Reverse Step: 10-20 스텝 (학습 데이터 기반 클러스터링)
Guidance 제거: Unconditional 생성
추론 속도: 80-120ms

파이프라인:
1. 2D 특징 추출 (30ms)
2. 경량 확산 역과정 (80ms)
3. 3D 벡터 후처리 (5ms)
---
총합: ~115ms (300ms 내)
```

**실제 구현 난이도: 매우 높음**
- Reverse process step 수 감소 시 품질 저하
- 학습 데이터: IR 환경 특화 필수
- 추가 검증: 생물학적/물리적 타당성 확인 필수

### 6.4 최종 권장사항

**✗ 확산 모델(GazeD) 추천하지 않음**

**이유:**
1. **시간 제약**: 중간 발표(4월 중순) 전 구현 불가능
2. **복잡도**: 팀 역량대비 과도한 난이도
3. **불필요**: Case 1-3만으로도 충분한 성능 확보
4. **위험**: 하이퍼파라미터 튜닝 시간 초과 가능

**대신 추천:**
- **우선순위 1**: Case 1 또는 Case 3 선택 (보조 시선 추정)
- **우선순위 2**: 확산 모델 대신 **결정론적 3D 회귀 모델** 사용
  ```python
  # 예: 2D 랜드마크 → MLP → (yaw, pitch, roll, confidence)
  class GazeRegressor(nn.Module):
      def __init__(self):
          self.fc1 = nn.Linear(136, 256)  # 68 landmarks * 2
          self.fc2 = nn.Linear(256, 128)
          self.gaze_head = nn.Linear(128, 3)  # yaw, pitch, roll
      
      def forward(self, landmarks):
          x = F.relu(self.fc1(landmarks))
          x = F.relu(self.fc2(x))
          return self.gaze_head(x)  # 5ms 추론
  ```

---

## 7. 추가 최적화 제안

### 7.1 하드웨어 레벨 최적화

| 최적화 | 효과 | 구현 난이도 |
|--------|------|-----------|
| **ONNX 변환** | 5-10% 속도 향상 | 낮 |
| **TensorRT** | 20-30% 속도 향상 | 중 |
| **INT8 Quantization** | 20-30% 속도 + 메모리 50% 감소 | 낮 |
| **FP16 Mixed Precision** | 10-15% 속도 향상 | 낮 |
| **배치 처리** (학습) | N/A | N/A |
| **CUDA Graph** | 5% 속도 향상 | 높 |

**권장 적용:**
```python
# 최소 구현 (개발 초기)
model = model.half()  # FP16
model = torch.jit.script(model)  # TorchScript

# 배포 단계 (최종 발표)
import torch2trt
model_trt = torch2trt.torch2trt(model_fp16, inputs=[dummy_input])
```

### 7.2 소프트웨어 레벨 최적화

#### **A. 파이프라인 최적화**

```python
# 1. 프레임 버퍼링 (최대 5 프레임)
frame_buffer = collections.deque(maxlen=5)

# 2. Non-blocking I/O
def infer_async():
    input_queue = queue.Queue(maxsize=2)
    output_queue = queue.Queue()
    
    # 캡처 스레드
    def capture():
        while True:
            frame = camera.read()
            input_queue.put_nowait(frame)
    
    # 추론 스레드
    def infer():
        while True:
            frame = input_queue.get()
            result = model(frame)
            output_queue.put(result)
    
    # 병렬 실행
    Thread(target=capture, daemon=True).start()
    Thread(target=infer, daemon=True).start()

# 3. 결과 캐싱
result_cache = {}
def get_cached_result(frame_id):
    return result_cache.get(frame_id)
```

#### **B. Early Exit 메커니즘**

```python
class AdaptiveGazeEstimator(nn.Module):
    def forward(self, x, confidence_threshold=0.8):
        # 1단계: 빠른 분류기
        quick_result, confidence = self.quick_classifier(x)
        
        # 2단계: 신뢰도 체크
        if confidence > confidence_threshold:
            return quick_result, confidence  # 35ms 절감
        
        # 3단계: 풀 모델
        detailed_result = self.detailed_model(x)
        return detailed_result, 1.0
```

#### **C. 신뢰도 기반 선택적 처리**

```python
def adaptive_fusion(landmarks, pose, eye_features, confidences):
    c_lm = confidences['landmark']  # 0-1
    c_po = confidences['pose']
    c_ey = confidences['eye']
    
    # 동적 가중치 계산
    total_conf = c_lm + c_po + c_ey
    w_lm = c_lm / total_conf
    w_po = c_po / total_conf
    w_ey = c_ey / total_conf
    
    # 특정 모듈 신뢰도 너무 낮으면 생략
    if c_lm < 0.3:
        # 랜드마크 기반 시선 추정 스킵
        result = pose_based_inference(pose, eye_features)
    else:
        result = full_inference(landmarks, pose, eye_features)
    
    return result
```

### 7.3 Colab Pro 환경 최적화

```python
# 1. 메모리 효율화
import torch.cuda as cuda

def optimize_memory():
    cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.9)  # 최대 90% 사용
    
    # 모델 메모리 효율화
    model = torch.jit.trace(model, example_input)
    model.eval()
    return model

# 2. 배치 처리 최적화
def process_video_batch(video_path, batch_size=4):
    frames = []
    results = []
    
    for frame in read_video(video_path):
        frames.append(preprocess(frame))
        
        if len(frames) == batch_size:
            batch_result = model(torch.stack(frames))
            results.extend(batch_result)
            frames = []
    
    return results

# 3. 데이터셋 사전 로딩
from torch.utils.data import DataLoader
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Colab: 최대 4-8
    pin_memory=True,
    prefetch_factor=2
)
```

### 7.4 데이터 전처리 가속화

```python
# 1. NumPy 병렬화 (Numba JIT 컴파일)
from numba import njit

@njit
def fast_preprocess(ir_image):
    # CLAHE 등가 구현
    result = np.zeros_like(ir_image)
    for i in range(ir_image.shape[0]):
        for j in range(ir_image.shape[1]):
            result[i,j] = np.clip(ir_image[i,j] * 1.5 + 30, 0, 255)
    return result.astype(np.uint8)

# 2. GPU 전처리
def gpu_preprocess(ir_image_batch):
    ir_tensor = torch.from_numpy(ir_image_batch).cuda()
    
    # 밝기 조정
    ir_tensor = torch.clamp(ir_tensor * 1.5 + 30, 0, 255)
    
    # 대비 정규화
    ir_tensor = (ir_tensor - ir_tensor.mean()) / (ir_tensor.std() + 1e-6)
    
    return ir_tensor
```

### 7.5 실시간 스트림 처리 아키텍처

```
카메라 입력 (30fps)
    ↓
프레임 큐 (2-3 프레임 버퍼)
    ↓
[병렬 처리]
├─ 얼굴 검출 (CPU/GPU)
├─ 포즈 추정 (GPU)
└─ 시선 추정 (GPU)
    ↓
Fusion & 분류 (GPU)
    ↓
결과 캐시 (최근 5 프레임)
    ↓
경고 출력 (메인 스레드)
```

**추정 처리량:**
- 입력: 30fps (33.3ms 주기)
- 추론: 70ms (Case 1)
- 지연: 2-3 프레임 (약 70-100ms)
- 실제 응답: 100-150ms ✓

---

## 8. 최종 권장 기술 스택

### 8.1 케이스 선택

**권장 순서:**
1. **1순위**: Case 3 (이미지 → 시선CNN + Pose → LSTM)
   - 가장 빠름 (46ms)
   - End-to-End 학습 용이
   - 랜드마크 오류 전파 최소화

2. **2순위**: Case 1 (최복잡)
   - 최고 정확도 (추정)
   - 충분한 여유 (229ms)
   - 강건성 우수

3. **3순위**: Case 2 (균형)
   - 해석성 있는 중간 표현
   - 의료/안전 영역 선호

**비권장**: Case 4, 5 (정확도 미검증)

### 8.2 모델 선택

| 모듈 | 추천 | 이유 |
|------|------|------|
| 시선 추정 | ResNet-18 | 빠르고 충분한 용량 |
| 포즈 추정 | MediaPipe Pose | 효율적, 안정적 |
| 백본 | MobileNetV3 | 경량, IR 친화적 |
| 시퀀스 | LSTM (중) | 경량화 쉬움 |
| 시선 대체 | 회귀 MLP (확산 모델 NO) | 빠르고 단순 |

### 8.3 배포 환경

**개발:**
- Colab Pro (L4 GPU 권장)
- INT8 Quantization 미적용

**테스트:**
- DGX Spark (TensorRT 최적화)
- 최종 성능 검증

**최종:**
- ONNX + TensorRT 변환
- FP16 Mixed Precision
- INT8 Quantization (선택사항)

---

## 9. 프로젝트 일정 권장안

### 9.1 수정된 타임라인

| 주차 | Case 1 | Case 3 | 병렬 최적화 | 비고 |
|------|--------|--------|-----------|------|
| 1-2 | 구상 | 구상 | 분석 | 제안서 완성 |
| 3-4 | 데이터 수집 | 데이터 수집 | - | 벤치마크 데이터 확보 |
| 5-6 | 기본 구현 | 기본 구현 | 스레드 풀 설계 | 병렬화 계획 서면화 |
| 7-8 | ST-GCN | CNN 시선 | 병렬 구현 | **중간 발표 예정 시점** |
| 9-10 | Transformer 학습 | LSTM 학습 | 경량화 | Distillation 시작 |
| 11-12 | Fusion 구현 | Fusion 구현 | INT8 Quantization | 신뢰도 점수 설계 |
| 13-14 | TensorRT | TensorRT | 최적화 | 최종 성능 측정 |
| 15 | 최종 보고 | 최종 보고 | - | **최종 발표** |

---

## 10. 체크리스트 및 주요 위험요소

### 10.1 기술적 위험

| 위험 | 확률 | 영향 | 완화 방안 |
|------|------|------|---------|
| IR 데이터 부족 | 중 | 높음 | 벤치마크 데이터 + 증강 |
| Transformer 학습 불안정 | 중 | 중 | LSTM으로 대체 |
| Colab 토큰 초과 | 중 | 중 | DGX Spark 조기 확보 |
| 차폐 대응 정확도 | 중 | 높음 | Case 1 선택 (다중 특징) |
| 실시간 성능 미달성 | 낮 | 높음 | 경량화 기법 사전 준비 |

### 10.2 권장 마일스톤

**2026년 4월 15일경 (중간 발표 직전)**
- [ ] Case 선택 확정
- [ ] 데이터셋 50% 수집 완료
- [ ] 기본 모델 학습 완료
- [ ] 예상 레이턴시 ≤ 100ms 확인

**2026년 5월 15일경**
- [ ] 최종 데이터셋 수집 완료
- [ ] Fusion 메커니즘 검증
- [ ] 정확도 ≥ 85%

**2026년 6월 1일경 (최종 발표 직전)**
- [ ] TensorRT 최적화 완료
- [ ] 레이턴시 ≤ 70ms
- [ ] 정확도 ≥ 90%

---

## 결론

### 종합 평가

✓ **0.3초 요구사항 충족 확실**
- 모든 케이스에서 여유도 70% 이상
- 추가 최적화로 20-30ms 추가 확보 가능

✓ **Case 3 강력 권장**
- 가장 빠르고 단순함
- 정확도도 Case 1과 크게 다르지 않을 것으로 예상
- 개발 리스크 최소

✓ **병렬화로 20-60% 속도 향상 가능**
- Case 1: 18ms 절감 (25% 향상)
- Case 3: 10ms 절감 (21% 향상)

✗ **확산 모델(GazeD) 부적합**
- 최소 50-80ms 필요 (최적화 후)
- 개발 복잡도 과도
- 결정론적 회귀 모델로 충분

✓ **Colab Pro + DGX Spark 조합 충분**
- Colab Pro: 개발/초기 학습
- DGX Spark: 대규모 실험/최종 검증

---

## 참고 자료

**관련 논문 (제안서에서 제시)**
- Kim et al. (2023): Real-time Driver Monitoring with Facial Landmark
- Choi & Kim (2018): Deep Learning-Based Gaze Detection with NIR Camera
- Sharma & Chakraborty (2024): Driver Gaze Tracking Systems
- GazeD (2026): Context-Aware Diffusion for 3D Gaze

**추천 구현 참고**
- MediaPipe: 얼굴/포즈 검출
- PyTorch Lightning: 모델 학습
- TensorRT: 배포 최적화
- ONNX: 모델 변환

---

**평가 완료**: 2026년 3월 27일
