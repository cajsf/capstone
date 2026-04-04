# ADR-001: IR 기반 강건한 운전자 모니터링 시스템(DMS) — 종합 아키텍처 설계서

**Status:** Proposed
**Date:** 2026-03-27
**Deciders:** 스꾸삐 팀 (유건, 강성찬, 황영인, 이수찬, 강민성), 지도교수 이동훈
**Version:** 1.0

---

## Part 1: End-to-End 모델 아키텍처

---

### 1. Context — 왜 이 설계가 필요한가

기존 DMS는 RGB 기반 단일 모달리티에 의존하여 야간 저조도, 선글라스/마스크 착용, 손에 의한 얼굴 가림 등 실제 주행 환경의 failure case에서 성능이 급격히 저하된다. 본 프로젝트는 IR 카메라 기반으로 **시선(Gaze)**, **머리 포즈(Head Pose)**, **상체 스켈레톤(Skeleton)**, **차폐 감지(Occlusion)** 4개 branch를 독립적으로 운용하고, 각 branch의 **신뢰도(confidence)**를 실시간으로 산출하여 **동적 융합(Dynamic Fusion)**하는 구조를 설계한다.

핵심 설계 원칙:
- **Graceful Degradation**: 일부 branch가 실패해도 나머지 branch로 동작 유지
- **실시간 처리**: 입력~출력 전체 지연 300ms 이내
- **모듈성**: 각 branch를 독립적으로 개발·교체·평가 가능
- **IR-First**: 모든 모듈이 IR 입력을 기본으로 설계

---

### 2. 전체 시스템 구조도

```
┌─────────────────────────────────────────────────────────────────────┐
│                        IR CAMERA INPUT (60fps)                       │
│                         960×540 grayscale                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 0: IR 전처리 모듈 (IR Preprocessing)                          │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  CLAHE → Bilateral Filter → Gamma Correction                  │  │
│  │  + IR Domain Adaptation Layer (학습 기반)                      │  │
│  │  레이턴시: ~8ms | 출력: 정규화된 IR 프레임                      │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 1: 얼굴 검출 및 ROI 추출 (Face Detection & ROI)              │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  YOLO-FaceV2 (Occlusion-Aware)                                │  │
│  │  출력: face_bbox, face_conf, occlusion_flag                   │  │
│  │  레이턴시: ~8ms | 입력: 전처리된 IR 프레임                      │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ROI 추출:                                                           │
│  ├─ face_roi: 얼굴 영역 crop (224×224)                              │
│  ├─ eye_roi_L/R: 좌/우 눈 영역 crop (64×64)                        │
│  └─ upper_body_roi: 상체 영역 crop (256×192)                        │
└──────────────┬───────────┬───────────┬───────────┬──────────────────┘
               ↓           ↓           ↓           ↓
         ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
         │ BRANCH  │ │ BRANCH  │ │ BRANCH  │ │ BRANCH  │
         │   A     │ │   B     │ │   C     │ │   D     │
         │  Gaze   │ │  Head   │ │ Skeleton│ │Occlusion│
         │Estimat. │ │  Pose   │ │  Action │ │Detector │
         └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
              ↓           ↓           ↓           ↓
         ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
         │conf_gaze│ │conf_pose│ │conf_skel│ │conf_occl│
         └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
              └───────────┼───────────┼───────────┘
                          ↓
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 3: 신뢰도 기반 동적 융합 (Confidence-Aware Dynamic Fusion)    │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  w = softmax(τ · [conf_gaze, conf_pose, conf_skel])           │  │
│  │  feat_fused = Σ(w_i · feat_i) + occlusion_context             │  │
│  │  mode = 'normal' if max(conf) > θ_n else 'degraded'          │  │
│  │  레이턴시: ~3ms                                               │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 4: 시계열 행동 분류기 (Temporal Behavior Classifier)          │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  TCN (Temporal Convolutional Network) 또는 Bi-LSTM             │  │
│  │  입력: fused_feature sequence (T=10 frames)                   │  │
│  │  출력: [전방주시, 사이드미러, 백미러, 네비게이션,               │  │
│  │         핸드폰, 졸음, 뒤돌아봄, 기타]                          │  │
│  │  레이턴시: ~10ms                                              │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 5: 위험 판별 및 경고 출력 (Alert Decision Engine)             │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  State Machine: Normal → Caution → Warning → Critical         │  │
│  │  이상행동 지속시간 + 심각도 + 신뢰도 종합 판단                   │  │
│  │  레이턴시: ~2ms                                               │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 3. STAGE 0 — IR 전처리 모듈 상세 설계

#### 3.1 목적
IR 카메라 원본 영상의 조도 불균형, 노이즈, 낮은 대비를 보정하여 후속 모듈(얼굴 검출, 랜드마크 추출)의 입력 품질을 안정화한다.

#### 3.2 전처리 파이프라인

```
IR Raw Frame (960×540, uint8)
    ↓
[Step 1] CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - clipLimit: 2.0
    - tileGridSize: (8, 8)
    - 목적: 국소 대비 향상, 과도한 밝기 증폭 방지
    ↓
[Step 2] Bilateral Filter
    - d: 9, sigmaColor: 75, sigmaSpace: 75
    - 목적: 에지 보존하면서 노이즈 제거
    ↓
[Step 3] Adaptive Gamma Correction
    - γ = 자동 계산 (프레임 평균 밝기 기반)
    - 밝기 < 80: γ = 0.6 (밝게)
    - 밝기 80~170: γ = 1.0 (유지)
    - 밝기 > 170: γ = 1.5 (어둡게)
    - 목적: 터널 진입/진출 등 급격한 조도 변화 대응
    ↓
[Step 4] IR Domain Adaptation Layer (학습 기반, 선택적)
    - 경량 U-Net (3 down + 3 up, 채널 16→64→128)
    - 학습: RGB 공개 데이터 → IR 자체 데이터 스타일 변환
    - 목적: 공개 데이터로 학습된 모델이 IR에서도 안정적으로 작동하도록 feature 분포 정렬
    ↓
출력: Preprocessed IR Frame (960×540, float32, [0,1] 정규화)
```

#### 3.3 IR Domain Adaptation Layer 구조

```python
class IRDomainAdapter(nn.Module):
    """경량 U-Net 기반 IR 도메인 적응 모듈"""
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, 2, stride=2),  # skip connection
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU()
        )
        self.final = nn.Conv2d(32, 1, 1)  # skip connection 후 출력

    def forward(self, x):
        # x: (B, 1, H, W) IR grayscale
        e1 = self.enc1(x)       # (B, 16, H, W)
        e2 = self.enc2(e1)      # (B, 32, H/2, W/2)
        e3 = self.enc3(e2)      # (B, 64, H/4, W/4)
        d3 = self.dec3(e3)      # (B, 32, H/2, W/2)
        d3 = torch.cat([d3, e2], dim=1)  # skip: (B, 64, H/2, W/2)
        d2 = self.dec2(d3)      # (B, 16, H, W)
        d2 = torch.cat([d2, e1], dim=1)  # skip: (B, 32, H, W)
        out = self.final(d2)    # (B, 1, H, W)
        return x + out          # residual: 원본 + 보정값
```

**파라미터 수**: ~45K (매우 경량)
**추론 시간**: ~3ms (GPU), ~8ms (CPU)
**학습 전략**: RGB→IR paired data로 L1 loss + perceptual loss 학습

---

### 4. STAGE 1 — 얼굴 검출 및 ROI 추출

#### 4.1 모델 선택: YOLO-FaceV2

| 항목 | 상세 |
|------|------|
| 모델 | YOLOv8n-face (Ultralytics) 또는 YOLO-FaceV2 |
| 입력 | 960×540 IR 프레임 |
| 출력 | face_bbox (x,y,w,h), confidence, 5-point landmarks |
| 특징 | Occlusion-aware: 부분 가려짐에서도 검출 유지 |
| 추론 시간 | 8ms (GPU) |
| 선택 이유 | 2024 논문에서 occlusion + scale 동시 처리 입증 |

#### 4.2 ROI 추출 로직

```python
def extract_rois(frame, face_bbox, landmarks_5pt):
    """
    face_bbox: [x1, y1, x2, y2]
    landmarks_5pt: [[left_eye], [right_eye], [nose], [mouth_l], [mouth_r]]
    """
    x1, y1, x2, y2 = face_bbox
    face_w, face_h = x2 - x1, y2 - y1

    # 1. Face ROI (224×224 정규화)
    face_roi = frame[y1:y2, x1:x2]
    face_roi = cv2.resize(face_roi, (224, 224))

    # 2. Eye ROI (각 64×64)
    left_eye, right_eye = landmarks_5pt[0], landmarks_5pt[1]
    eye_margin = int(face_w * 0.15)  # 얼굴 너비의 15%

    eye_roi_L = crop_around_point(frame, left_eye, margin=eye_margin, size=64)
    eye_roi_R = crop_around_point(frame, right_eye, margin=eye_margin, size=64)

    # 3. Upper Body ROI (256×192)
    # 얼굴 아래로 확장하여 어깨/목 포함
    body_y1 = max(0, y1 - int(face_h * 0.3))
    body_y2 = min(frame.shape[0], y2 + int(face_h * 1.5))
    body_x1 = max(0, x1 - int(face_w * 0.5))
    body_x2 = min(frame.shape[1], x2 + int(face_w * 0.5))
    upper_body_roi = frame[body_y1:body_y2, body_x1:body_x2]
    upper_body_roi = cv2.resize(upper_body_roi, (192, 256))

    return {
        'face_roi': face_roi,          # (224, 224, 1)
        'eye_roi_L': eye_roi_L,        # (64, 64, 1)
        'eye_roi_R': eye_roi_R,        # (64, 64, 1)
        'upper_body_roi': upper_body_roi,  # (256, 192, 1)
        'face_bbox': face_bbox,
        'landmarks_5pt': landmarks_5pt
    }
```

---

### 5. STAGE 2 — 4개 추정 Branch 상세 설계

---

#### 5.1 Branch A: 시선 추정 (Gaze Estimation)

**목적**: 운전자의 시선 방향(yaw, pitch)을 추정하여 전방 주시 여부 판단

**아키텍처 옵션 2가지 (Case 2 vs Case 3 대응)**:

##### Option A-1: 랜드마크 경유 시선 추정 (Case 2 대응)

```
Face Mesh 468점 추출 (MediaPipe or 자체 모델)
    ↓
눈 영역 68점 추출 (좌 36점 + 우 32점)
    ↓
GazeRegressor MLP
    입력: 136차원 (68 landmarks × 2좌표)
    은닉층: 136→256→128→64
    출력: (yaw, pitch) + confidence
    ↓
시선 방향 벡터
```

```python
class GazeRegressorMLP(nn.Module):
    """랜드마크 기반 시선 추정 (Case 2)"""
    def __init__(self, n_landmarks=68):
        super().__init__()
        input_dim = n_landmarks * 2  # x, y 좌표
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.gaze_head = nn.Linear(64, 2)       # yaw, pitch
        self.conf_head = nn.Linear(64, 1)        # confidence

    def forward(self, landmarks):
        # landmarks: (B, 68, 2) → flatten → (B, 136)
        x = landmarks.reshape(landmarks.shape[0], -1)
        feat = self.backbone(x)
        gaze = self.gaze_head(feat)              # (B, 2)
        conf = torch.sigmoid(self.conf_head(feat))  # (B, 1)
        return gaze, conf, feat  # feat은 fusion에 전달
```

**추론 시간**: ~5ms
**장점**: 경량, 해석 가능 (어떤 랜드마크가 기여하는지 추적 가능)
**단점**: Face Mesh 품질에 의존, IR에서 랜드마크 불안정 시 오류 전파

##### Option A-2: 이미지 직접 시선 추정 (Case 3 대응)

```
Eye ROI (좌+우) + Face ROI
    ↓
EfficientNet-B0 (수정: 1ch IR 입력)
    입력: face_roi (224×224×1)
    Feature: 1280차원
    ↓
Gaze Head (FC)
    출력: (yaw, pitch) + confidence
```

```python
class GazeCNN(nn.Module):
    """이미지 직접 시선 추정 (Case 3)"""
    def __init__(self):
        super().__init__()
        # EfficientNet-B0 백본 (1ch IR 적응)
        self.backbone = efficientnet_b0(pretrained=True)
        # 첫 Conv를 1ch 입력으로 수정
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        # RGB pretrained 가중치의 평균을 1ch에 복사
        with torch.no_grad():
            self.backbone.features[0][0].weight[:] = \
                original_conv.weight.mean(dim=1, keepdim=True)

        self.backbone.classifier = nn.Identity()  # 1280차원 feature

        # Eye-specific branch (보조)
        self.eye_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()  # 32차원
        )

        # Fusion + Head
        self.gaze_head = nn.Sequential(
            nn.Linear(1280 + 32*2, 256),  # face feat + left_eye + right_eye
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # yaw, pitch
        )
        self.conf_head = nn.Sequential(
            nn.Linear(1280 + 32*2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, face_roi, eye_roi_L, eye_roi_R):
        face_feat = self.backbone(face_roi)       # (B, 1280)
        eye_L_feat = self.eye_encoder(eye_roi_L)  # (B, 32)
        eye_R_feat = self.eye_encoder(eye_roi_R)  # (B, 32)

        combined = torch.cat([face_feat, eye_L_feat, eye_R_feat], dim=1)
        gaze = self.gaze_head(combined)           # (B, 2)
        conf = self.conf_head(combined)           # (B, 1)

        return gaze, conf, face_feat  # face_feat은 fusion에 전달
```

**추론 시간**: ~12ms
**파라미터**: ~5.3M (EfficientNet-B0) + ~0.1M (eye encoder + heads)
**장점**: 랜드마크 오류에 강건, end-to-end 학습 가능
**단점**: 해석성 낮음, 학습 데이터 다량 필요

##### **권장 선택: Case 3 (이미지 직접 시선 추정)**

| 비교 항목 | Case 2 (랜드마크 경유) | Case 3 (이미지 직접) |
|-----------|---------------------|-------------------|
| IR 강건성 | △ (랜드마크 품질 의존) | ○ (직접 학습 가능) |
| 레이턴시 | 17ms (Face Mesh 12 + MLP 5) | 12ms |
| 오류 전파 | 높음 (2-stage cascade) | 낮음 (end-to-end) |
| 학습 난이도 | 중 | 중-상 |
| 해석 가능성 | 높음 | 낮음 |
| 최신 SOTA 근거 | - | EfficientNet attention 기반 4.08° MAE |

→ **IR 환경에서는 랜드마크 품질이 보장되지 않으므로 Case 3이 더 안정적**
→ 단, Case 2를 보조 branch로 함께 구현하여 ablation study에 활용 권장

---

#### 5.2 Branch B: 머리 포즈 추정 (Head Pose Estimation)

**목적**: 얼굴 3축 회전(yaw, pitch, roll)을 추정하여 시선의 보조 지표로 활용

**모델 선택: 6DRepNet360**

| 항목 | 상세 |
|------|------|
| 모델 | 6DRepNet360 (IEEE TIP 2024) |
| 입력 | face_roi (224×224×1, IR) |
| 출력 | (yaw, pitch, roll) 3축 각도 + rotation matrix |
| 정확도 | MAE 3~5° (AFLW2000 벤치마크) |
| 특징 | 360° 전방위 회전 대응, 6D continuous rotation representation |
| 추론 시간 | 8ms |

```python
class HeadPoseBranch(nn.Module):
    """6DRepNet360 기반 머리 포즈 추정"""
    def __init__(self):
        super().__init__()
        # 6DRepNet360 백본 (RepVGG-B1g2)
        self.backbone = repvgg_b1g2(pretrained=True)
        # 1ch IR 적응
        self._adapt_first_conv(in_channels=1)

        self.backbone.linear = nn.Identity()  # feature만 추출
        feat_dim = 2048  # RepVGG-B1g2 출력

        # 6D rotation representation (continuous)
        self.rotation_head = nn.Linear(feat_dim, 6)  # 6D repr
        self.conf_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, face_roi):
        feat = self.backbone(face_roi)      # (B, 2048)
        rot_6d = self.rotation_head(feat)   # (B, 6)
        conf = self.conf_head(feat)         # (B, 1)

        # 6D → rotation matrix → euler angles
        rot_matrix = compute_rotation_matrix_from_6d(rot_6d)  # (B, 3, 3)
        yaw, pitch, roll = rotation_matrix_to_euler(rot_matrix)

        euler = torch.stack([yaw, pitch, roll], dim=1)  # (B, 3)
        return euler, conf, feat

    def _adapt_first_conv(self, in_channels=1):
        """RGB pretrained → 1ch IR 적응"""
        old_conv = self.backbone.stage0[0]
        new_conv = nn.Conv2d(in_channels, old_conv.out_channels,
                            kernel_size=old_conv.kernel_size,
                            stride=old_conv.stride,
                            padding=old_conv.padding, bias=False)
        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
        self.backbone.stage0[0] = new_conv
```

**IR 적응 전략**:
- RGB pretrained 가중치의 채널 평균으로 1ch conv 초기화
- IR 자체 데이터로 5 epoch fine-tuning (lr=1e-4)
- 논문 근거: 15-25% 성능 저하를 fine-tuning으로 5% 이내로 복원 가능

---

#### 5.3 Branch C: 상체 스켈레톤 행동 인식 (Skeleton-Based Action Recognition)

**목적**: 상체 관절점 시퀀스로부터 운전자 행동 패턴(핸드폰 사용, 졸음 자세, 뒤돌아봄 등)을 분류

**아키텍처: CTR-GCN + Temporal Transformer Hybrid**

이 branch가 Case 1에 대응하는 핵심 모듈이며, **ST-GCN을 운전자 행동 인식에 적용하는 것은 2024-2025 문헌에서 선례가 없는 새로운 시도**로 학술적 차별성이 있다.

##### 스켈레톤 그래프 정의 (DMS 특화)

```
운전자 상체 스켈레톤 그래프 (13개 관절점):

노드 정의:
  0: 코 (nose)
  1: 좌눈 (left_eye)         ← 차폐 시 생략 가능
  2: 우눈 (right_eye)        ← 차폐 시 생략 가능
  3: 좌귀 (left_ear)
  4: 우귀 (right_ear)
  5: 좌어깨 (left_shoulder)
  6: 우어깨 (right_shoulder)
  7: 좌팔꿈치 (left_elbow)
  8: 우팔꿈치 (right_elbow)
  9: 좌손목 (left_wrist)
  10: 우손목 (right_wrist)
  11: 목 (neck) ← 좌어깨+우어깨 중점으로 계산
  12: 머리중심 (head_center) ← 코+좌귀+우귀 중점

에지 정의 (인접행렬):
  얼굴: (0,1), (0,2), (0,3), (0,4), (0,12)
  몸통: (11,5), (11,6), (0,11)
  좌팔: (5,7), (7,9)
  우팔: (6,8), (8,10)
  크로스: (9,0), (10,0)  ← 손-얼굴 관계 (핸드폰/얼굴 만짐 탐지)
```

##### CTR-GCN + Temporal Transformer

```python
class SkeletonActionBranch(nn.Module):
    """
    CTR-GCN (Channel-wise Topology Refinement GCN) + Temporal Transformer
    Case 1 대응: 랜드마크+스켈레톤 직접 행동 분류
    """
    def __init__(self, num_joints=13, in_channels=2, num_frames=10,
                 num_classes=8, d_model=128):
        super().__init__()
        self.num_joints = num_joints
        self.num_frames = num_frames

        # === Spatial GCN 블록 (CTR-GCN 기반) ===
        self.spatial_gcn = nn.ModuleList([
            CTRGCNBlock(in_channels, 64, num_joints),
            CTRGCNBlock(64, 128, num_joints),
            CTRGCNBlock(128, d_model, num_joints)
        ])

        # === Temporal Transformer ===
        self.temporal_pos_enc = nn.Parameter(
            torch.randn(1, num_frames, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256,
            dropout=0.1, batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=2
        )

        # === Classification Head ===
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        # === Confidence Head ===
        self.conf_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, skeleton_seq):
        """
        skeleton_seq: (B, T, V, C)
            B: batch, T: num_frames, V: num_joints, C: channels (x, y)
        """
        B, T, V, C = skeleton_seq.shape

        # Spatial GCN: 각 프레임 독립 처리
        # (B, T, V, C) → (B*T, C, V)
        x = skeleton_seq.reshape(B*T, V, C).permute(0, 2, 1)

        for gcn_block in self.spatial_gcn:
            x = gcn_block(x)  # (B*T, d_model, V)

        # Global Average Pooling over joints
        x = x.mean(dim=2)  # (B*T, d_model)
        x = x.reshape(B, T, -1)  # (B, T, d_model)

        # Temporal Transformer
        x = x + self.temporal_pos_enc[:, :T, :]
        x = self.temporal_transformer(x)  # (B, T, d_model)

        # Classification
        x_cls = x.permute(0, 2, 1)  # (B, d_model, T)
        logits = self.classifier(x_cls)  # (B, num_classes)
        conf = self.conf_head(x_cls)     # (B, 1)

        # Feature for fusion (마지막 프레임)
        feat = x[:, -1, :]  # (B, d_model)

        return logits, conf, feat


class CTRGCNBlock(nn.Module):
    """Channel-wise Topology Refinement GCN Block"""
    def __init__(self, in_ch, out_ch, num_joints):
        super().__init__()
        self.num_joints = num_joints

        # 학습 가능한 인접행렬 (topology refinement)
        self.A = nn.Parameter(torch.randn(num_joints, num_joints) * 0.01)

        # Channel-wise refinement
        self.conv = nn.Conv1d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

        # Residual
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # x: (B, C_in, V)
        A_norm = F.softmax(self.A, dim=1)  # 행 정규화

        # Graph convolution: x @ A
        x_gc = torch.matmul(x, A_norm)  # (B, C_in, V)

        out = self.conv(x_gc)   # (B, C_out, V)
        out = self.bn(out)
        out = self.relu(out + self.residual(x))

        return out
```

**스켈레톤 추출기 선택**:

| 모델 | 추론 시간 | 정확도 | IR 호환 |
|------|---------|-------|---------|
| MoveNet Lightning | 6ms | 중 | △ (fine-tuning 필요) |
| MediaPipe Pose | 10ms | 상 | △ (RGB 전용) |
| RTMPose-s | 8ms | 상 | ○ (다양한 도메인) |

→ **권장: RTMPose-s** (MMPose 프레임워크, 다양한 도메인에서 검증, 8ms)

---

#### 5.4 Branch D: 차폐 감지 (Occlusion Detection)

**목적**: 현재 프레임에서 어떤 유형의 차폐가 발생하고 있는지 감지하여 다른 branch의 신뢰도 조정에 활용

```python
class OcclusionDetector(nn.Module):
    """경량 차폐 유형 분류기"""
    def __init__(self, num_occlusion_types=5):
        super().__init__()
        # 유형: [none, sunglasses, mask, hand, phone]

        self.backbone = mobilenet_v3_small(pretrained=True)
        # 1ch IR 적응
        self.backbone.features[0][0] = nn.Conv2d(
            1, 16, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.backbone.classifier = nn.Identity()

        self.occlusion_head = nn.Sequential(
            nn.Linear(576, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_occlusion_types)
        )
        # 차폐 심각도 (0~1, 연속값)
        self.severity_head = nn.Sequential(
            nn.Linear(576, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, face_roi):
        feat = self.backbone(face_roi)          # (B, 576)
        occ_type = self.occlusion_head(feat)    # (B, 5)
        severity = self.severity_head(feat)     # (B, 1)

        return {
            'occlusion_logits': occ_type,
            'occlusion_type': torch.argmax(occ_type, dim=1),
            'severity': severity,
            'confidence': 1.0 - severity  # 차폐가 없을수록 높은 신뢰도
        }
```

**추론 시간**: ~5ms (MobileNetV3-Small)
**파라미터**: ~2.5M
**학습 데이터**: DMD 데이터셋 + 자체 촬영 failure 데이터셋

---

### 6. STAGE 3 — 신뢰도 기반 동적 융합 (핵심 기여)

이 단계가 본 프로젝트의 **핵심 차별화 요소**이다.

#### 6.1 신뢰도 산출 공식

각 branch i에 대해:

```
conf_i = α·landmark_score_i × β·visibility_i × γ·temporal_consistency_i × δ·task_confidence_i
```

여기서:
- **landmark_score**: 검출된 특징점의 평균 신뢰도 (모델 내부 출력)
- **visibility**: 차폐 감지 결과 반영 (Branch D 출력)
- **temporal_consistency**: 이전 N 프레임과의 출력 변화량 역수
- **task_confidence**: 각 branch 모델 자체의 softmax 확률 또는 sigmoid 출력
- α, β, γ, δ: 학습 가능한 가중 계수 (초기값 모두 1.0)

#### 6.2 동적 융합 모듈

```python
class ConfidenceAwareFusion(nn.Module):
    """
    신뢰도 기반 동적 융합 모듈

    핵심 아이디어:
    - 각 branch의 feature를 신뢰도에 비례하여 가중 결합
    - 차폐 정보를 context로 추가 주입
    - degraded mode 자동 전환
    """
    def __init__(self, feat_dims={'gaze': 1280, 'pose': 2048, 'skeleton': 128},
                 fused_dim=256, temperature=2.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # 각 branch feature를 동일 차원으로 projection
        self.projections = nn.ModuleDict({
            'gaze': nn.Sequential(
                nn.Linear(feat_dims['gaze'], fused_dim),
                nn.LayerNorm(fused_dim),
                nn.ReLU()
            ),
            'pose': nn.Sequential(
                nn.Linear(feat_dims['pose'], fused_dim),
                nn.LayerNorm(fused_dim),
                nn.ReLU()
            ),
            'skeleton': nn.Sequential(
                nn.Linear(feat_dims['skeleton'], fused_dim),
                nn.LayerNorm(fused_dim),
                nn.ReLU()
            )
        })

        # 차폐 context embedding
        self.occlusion_embed = nn.Sequential(
            nn.Linear(6, 32),  # 5 class logits + severity
            nn.ReLU(),
            nn.Linear(32, fused_dim)
        )

        # 학습 가능한 confidence 가중 계수
        self.conf_weights = nn.Parameter(torch.ones(4))  # α, β, γ, δ

        # Temporal consistency buffer (비학습)
        self.register_buffer('prev_outputs', torch.zeros(3, 3))

    def forward(self, branch_features, branch_confs, occlusion_info):
        """
        branch_features: dict {'gaze': (B, 1280), 'pose': (B, 2048), 'skeleton': (B, 128)}
        branch_confs: dict {'gaze': (B, 1), 'pose': (B, 1), 'skeleton': (B, 1)}
        occlusion_info: dict {'occlusion_logits': (B, 5), 'severity': (B, 1)}
        """
        B = list(branch_features.values())[0].shape[0]

        # Step 1: Feature projection (동일 차원으로)
        projected = {}
        for name, feat in branch_features.items():
            projected[name] = self.projections[name](feat)  # (B, fused_dim)

        # Step 2: 최종 신뢰도 계산
        # visibility는 차폐 심각도의 역수
        visibility = 1.0 - occlusion_info['severity']  # (B, 1)

        final_confs = {}
        for name, raw_conf in branch_confs.items():
            # 시선은 차폐에 민감, 포즈는 덜 민감, 스켈레톤은 무관
            if name == 'gaze':
                vis_factor = visibility  # 차폐 시 시선 신뢰도 크게 하락
            elif name == 'pose':
                vis_factor = 0.5 + 0.5 * visibility  # 차폐 영향 절반
            else:  # skeleton
                vis_factor = torch.ones_like(visibility) * 0.9 + 0.1 * visibility

            final_confs[name] = raw_conf * vis_factor  # (B, 1)

        # Step 3: Softmax 가중치 (temperature scaling)
        conf_tensor = torch.cat(
            [final_confs['gaze'], final_confs['pose'], final_confs['skeleton']],
            dim=1
        )  # (B, 3)
        weights = F.softmax(self.temperature * conf_tensor, dim=1)  # (B, 3)

        # Step 4: 가중 융합
        feat_stack = torch.stack(
            [projected['gaze'], projected['pose'], projected['skeleton']],
            dim=1
        )  # (B, 3, fused_dim)

        weights_expanded = weights.unsqueeze(2)  # (B, 3, 1)
        fused = (feat_stack * weights_expanded).sum(dim=1)  # (B, fused_dim)

        # Step 5: 차폐 context 추가
        occ_context = self.occlusion_embed(
            torch.cat([occlusion_info['occlusion_logits'],
                      occlusion_info['severity']], dim=1)
        )  # (B, fused_dim)
        fused = fused + 0.1 * occ_context  # residual 방식으로 context 주입

        # Step 6: 모드 판정
        max_conf = conf_tensor.max(dim=1)[0]  # (B,)
        mode = torch.where(max_conf > 0.3,
                          torch.ones_like(max_conf),   # normal
                          torch.zeros_like(max_conf))  # degraded

        return {
            'fused_feature': fused,       # (B, fused_dim) → Stage 4로 전달
            'weights': weights,            # (B, 3) — 어떤 branch가 지배적인지
            'confidence_scores': conf_tensor,  # (B, 3)
            'mode': mode,                  # (B,) — 0: degraded, 1: normal
            'overall_confidence': (conf_tensor * weights).sum(dim=1)  # (B,)
        }
```

#### 6.3 Degraded Mode 동작 정의

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEGRADED MODE 전환 로직                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Level 0 (Normal): max(conf) > 0.5                             │
│  → 모든 branch 정상 동작, 융합 가중치 그대로 적용               │
│                                                                 │
│  Level 1 (Partial Degraded): 0.3 < max(conf) ≤ 0.5            │
│  → 가장 신뢰도 높은 branch 위주로 판단                          │
│  → 경고 임계값 상향 (오경보 억제)                                │
│  → UI에 "신뢰도 저하" 표시                                      │
│                                                                 │
│  Level 2 (Heavy Degraded): 0.1 < max(conf) ≤ 0.3              │
│  → skeleton branch만으로 판단 (가장 차폐에 강건)                 │
│  → 졸음/핸드폰 감지만 유지, 세밀한 시선 분류 중단                │
│  → "센서 가림 감지" 경고 출력                                   │
│                                                                 │
│  Level 3 (System Fail): max(conf) ≤ 0.1                        │
│  → 얼굴 미검출 상태 지속                                        │
│  → 30초 후 자동 경고 발동                                       │
│  → "카메라 확인 필요" 경고 출력                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### 7. STAGE 4 — 시계열 행동 분류기

#### 7.1 모델: TCN (Temporal Convolutional Network)

BiLSTM 대비 TCN을 권장하는 이유:
- 병렬 처리 가능 (LSTM은 순차적)
- 고정 수용 범위(receptive field)로 레이턴시 예측 가능
- 학습 안정성 우수

```python
class TemporalBehaviorClassifier(nn.Module):
    """TCN 기반 시계열 행동 분류기"""
    def __init__(self, input_dim=256, num_classes=8,
                 num_frames=10, hidden_dim=128):
        super().__init__()

        # TCN 블록 (dilated causal convolution)
        self.tcn = nn.Sequential(
            TCNBlock(input_dim, hidden_dim, kernel_size=3, dilation=1),
            TCNBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2),
            TCNBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=4),
        )

        # 분류기
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, fused_sequence):
        """
        fused_sequence: (B, T, D) — T 프레임의 fusion 결과 시퀀스
        """
        x = fused_sequence.permute(0, 2, 1)  # (B, D, T)
        x = self.tcn(x)                       # (B, hidden, T)
        logits = self.classifier(x)           # (B, num_classes)
        return logits


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                             padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.padding = padding

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :-self.padding]  # causal: 미래 정보 제거
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out + self.residual(x)
```

#### 7.2 분류 클래스 정의

| ID | 클래스 | 위험도 | 설명 |
|----|-------|-------|------|
| 0 | 전방 주시 | 안전 | 정상 운전 상태 |
| 1 | 사이드미러 확인 | 안전 | 정상 주행 보조 행동 |
| 2 | 백미러 확인 | 안전 | 정상 주행 보조 행동 |
| 3 | 네비게이션 주시 | 주의 | 짧은 시간은 정상, 5초 이상 시 경고 |
| 4 | 핸드폰 사용 | 위험 | 즉시 경고 |
| 5 | 졸음/눈감음 | 위험 | 즉시 경고 |
| 6 | 뒤돌아봄 | 주의 | 지속 시 경고 |
| 7 | 기타 이상행동 | 주의 | 분류 불가한 비정상 패턴 |

---

### 8. STAGE 5 — 위험 판별 및 경고 출력

#### 8.1 State Machine

```python
class AlertStateMachine:
    """
    위험 수준 상태 머신

    상태 전이:
    NORMAL → CAUTION → WARNING → CRITICAL

    각 전이는 [행동 유형 × 지속 시간 × 신뢰도] 조합으로 결정
    """
    NORMAL = 0
    CAUTION = 1
    WARNING = 2
    CRITICAL = 3

    # 경고 전이 규칙
    RULES = {
        # (행동 ID, 최소 지속 프레임, 최소 신뢰도) → 목표 상태
        (4, 3, 0.6):  WARNING,    # 핸드폰 0.5초 → WARNING
        (4, 10, 0.6): CRITICAL,   # 핸드폰 1.7초 → CRITICAL
        (5, 5, 0.5):  WARNING,    # 졸음 0.8초 → WARNING
        (5, 15, 0.5): CRITICAL,   # 졸음 2.5초 → CRITICAL
        (3, 30, 0.7): CAUTION,    # 네비 5초 → CAUTION
        (3, 60, 0.7): WARNING,    # 네비 10초 → WARNING
        (6, 20, 0.6): CAUTION,    # 뒤돌아봄 3.3초 → CAUTION
        (7, 30, 0.5): CAUTION,    # 기타 5초 → CAUTION
    }

    def __init__(self, fps=60):
        self.state = self.NORMAL
        self.behavior_counter = {}  # {class_id: frame_count}
        self.fps = fps

    def update(self, predicted_class, confidence, mode):
        """매 프레임 호출"""

        # Degraded mode에서는 임계값 상향
        conf_threshold_modifier = 0.1 if mode == 'degraded' else 0.0

        # 안전 행동(0,1,2)이면 카운터 리셋
        if predicted_class in [0, 1, 2]:
            self.behavior_counter = {}
            self.state = max(self.NORMAL, self.state - 1)  # 점진적 해제
            return self.state, None

        # 위험/주의 행동 카운터 증가
        self.behavior_counter[predicted_class] = \
            self.behavior_counter.get(predicted_class, 0) + 1

        # 규칙 매칭
        new_state = self.NORMAL
        for (cls, min_frames, min_conf), target_state in self.RULES.items():
            if (predicted_class == cls and
                self.behavior_counter.get(cls, 0) >= min_frames and
                confidence >= min_conf + conf_threshold_modifier):
                new_state = max(new_state, target_state)

        self.state = new_state

        alert_msg = None
        if self.state == self.CAUTION:
            alert_msg = "주의: 전방 주시를 유지하세요"
        elif self.state == self.WARNING:
            alert_msg = "경고: 운전에 집중하세요!"
        elif self.state == self.CRITICAL:
            alert_msg = "긴급: 즉시 전방을 주시하세요!!!"

        return self.state, alert_msg
```

---

### 9. Case 1~5 매핑 및 비교 결정

| 항목 | Case 1 | Case 2 | Case 3 | Case 4 | Case 5 |
|------|--------|--------|--------|--------|--------|
| **구성** | 랜드마크+스켈레톤 → ST-GCN+Transformer | 랜드마크→시선 MLP + 포즈 | 이미지→시선 CNN + 포즈 | 머리포즈+스켈레톤만 | 스켈레톤만 |
| **Branch 사용** | A(MLP)+B+C+D | A(MLP)+B+C+D | A(CNN)+B+C+D | B+C | C |
| **Adaptive Fusion** | ✓ | ✓ | ✓ | ✗ | ✗ |
| **추정 레이턴시** | 71ms | 56ms | 46ms | 31ms | 31ms |
| **예상 정확도** | 최고 (95%+) | 상 (93%+) | 상 (93%+) | 중 (85%+) | 하 (70%+) |
| **IR 강건성** | 중 | 중 | 상 | 상 | 상 |
| **역할** | **메인 모델** | 보조 비교군 | **IR 최적 대안** | 베이스라인 | 하한선 |

#### 최종 결정

```
메인 구현 우선순위:
  1순위: Case 3 (이미지 직접 시선 CNN + 포즈 + 스켈레톤 + Adaptive Fusion)
         → IR 환경에서 가장 안정적, end-to-end 학습 가능

  2순위: Case 1 (ST-GCN + Transformer, 학술적 신규성)
         → 중간 발표 후 시간 여유가 있으면 구현
         → 최종 발표에서 Case 3 대비 ablation study로 활용

  필수 구현: Case 4 (베이스라인) + Case 5 (하한선)
         → 비교군으로서 가치. 간단하므로 1-2일 내 구현 가능
```

---

## Part 2: 전체 어플리케이션 파이프라인

---

### 10. 실시간 추론 파이프라인

```python
class DMSPipeline:
    """전체 DMS 실시간 추론 파이프라인"""

    def __init__(self, config):
        # Stage 0: 전처리
        self.ir_preprocessor = IRPreprocessor(config.preprocess)
        self.ir_adapter = IRDomainAdapter()  # 선택적

        # Stage 1: 얼굴 검출
        self.face_detector = YOLOFaceDetector(config.detector)

        # Stage 2: Branch 모델들
        self.gaze_estimator = GazeCNN()           # Branch A
        self.head_pose = HeadPoseBranch()          # Branch B
        self.skeleton_action = SkeletonActionBranch()  # Branch C
        self.occlusion_detector = OcclusionDetector()  # Branch D
        self.pose_extractor = RTMPose()            # 스켈레톤 추출기

        # Stage 3: 융합
        self.fusion = ConfidenceAwareFusion()

        # Stage 4: 시계열 분류
        self.temporal_classifier = TemporalBehaviorClassifier()

        # Stage 5: 경고
        self.alert_engine = AlertStateMachine(fps=config.fps)

        # 프레임 버퍼 (시계열 입력용)
        self.frame_buffer = deque(maxlen=config.temporal_window)

    @torch.no_grad()
    def process_frame(self, ir_frame):
        """
        단일 프레임 처리 — 전체 파이프라인

        Args:
            ir_frame: np.array (H, W) IR grayscale

        Returns:
            dict: 최종 결과 (행동 분류, 경고 상태, 디버그 정보)
        """
        t_start = time.perf_counter()

        # === Stage 0: 전처리 (~8ms) ===
        preprocessed = self.ir_preprocessor(ir_frame)
        preprocessed = self.ir_adapter(preprocessed)  # 선택적

        # === Stage 1: 얼굴 검출 (~8ms) ===
        detection = self.face_detector(preprocessed)

        if detection is None:
            # 얼굴 미검출 → degraded mode
            return self._handle_no_face()

        rois = extract_rois(preprocessed, detection.bbox, detection.landmarks)

        # === Stage 2: 병렬 Branch 추론 (~12ms, 병렬) ===
        # ThreadPoolExecutor로 GPU 커널 병렬 실행
        with ThreadPoolExecutor(max_workers=4) as pool:
            future_gaze = pool.submit(
                self.gaze_estimator,
                rois['face_roi'], rois['eye_roi_L'], rois['eye_roi_R']
            )
            future_pose = pool.submit(
                self.head_pose, rois['face_roi']
            )
            future_skeleton = pool.submit(
                self._skeleton_pipeline, rois['upper_body_roi']
            )
            future_occlusion = pool.submit(
                self.occlusion_detector, rois['face_roi']
            )

            gaze_out, gaze_conf, gaze_feat = future_gaze.result()
            pose_out, pose_conf, pose_feat = future_pose.result()
            skel_logits, skel_conf, skel_feat = future_skeleton.result()
            occlusion_info = future_occlusion.result()

        # === Stage 3: 동적 융합 (~3ms) ===
        fusion_result = self.fusion(
            branch_features={'gaze': gaze_feat, 'pose': pose_feat, 'skeleton': skel_feat},
            branch_confs={'gaze': gaze_conf, 'pose': pose_conf, 'skeleton': skel_conf},
            occlusion_info=occlusion_info
        )

        # === Stage 4: 시계열 분류 (~10ms) ===
        self.frame_buffer.append(fusion_result['fused_feature'])

        if len(self.frame_buffer) >= 5:  # 최소 5프레임 축적 후 분류
            fused_sequence = torch.stack(list(self.frame_buffer), dim=0)
            fused_sequence = fused_sequence.unsqueeze(0)  # batch dim

            behavior_logits = self.temporal_classifier(fused_sequence)
            predicted_class = torch.argmax(behavior_logits, dim=1).item()
            class_confidence = F.softmax(behavior_logits, dim=1).max().item()
        else:
            predicted_class = 0  # 전방 주시 (기본값)
            class_confidence = 0.5

        # === Stage 5: 경고 판단 (~2ms) ===
        mode = 'normal' if fusion_result['mode'].item() > 0.5 else 'degraded'
        alert_state, alert_msg = self.alert_engine.update(
            predicted_class, class_confidence, mode
        )

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000

        return {
            'predicted_class': predicted_class,
            'class_confidence': class_confidence,
            'alert_state': alert_state,
            'alert_message': alert_msg,
            'fusion_weights': fusion_result['weights'].cpu().numpy(),
            'branch_confidences': fusion_result['confidence_scores'].cpu().numpy(),
            'occlusion_type': occlusion_info['occlusion_type'].item(),
            'mode': mode,
            'latency_ms': latency_ms
        }

    def _skeleton_pipeline(self, upper_body_roi):
        """스켈레톤 추출 → GCN 추론"""
        keypoints = self.pose_extractor(upper_body_roi)
        # 시계열 버퍼에서 최근 N프레임의 스켈레톤 가져오기
        skeleton_seq = self._get_skeleton_sequence(keypoints)
        return self.skeleton_action(skeleton_seq)
```

---

### 11. 레이턴시 예산 (Budget Breakdown)

```
┌──────────────────────────────────────────────────────────────────┐
│           총 레이턴시 예산: 300ms (0.3초)                         │
│                                                                  │
│  Stage 0: IR 전처리          8ms  ██░░░░░░░░░░░░░░░░░░  2.7%   │
│  Stage 1: 얼굴 검출          8ms  ██░░░░░░░░░░░░░░░░░░  2.7%   │
│  Stage 2: 병렬 Branch       12ms  ████░░░░░░░░░░░░░░░░  4.0%   │
│    ├─ Gaze CNN             (12ms)                               │
│    ├─ Head Pose             (8ms)  ← 병렬이므로 max만 계산       │
│    ├─ Skeleton (RTMPose)    (8ms)                               │
│    └─ Occlusion             (5ms)                               │
│  Stage 3: 동적 융합          3ms  █░░░░░░░░░░░░░░░░░░░  1.0%   │
│  Stage 4: TCN 분류          10ms  ███░░░░░░░░░░░░░░░░░  3.3%   │
│  Stage 5: 경고 판정          2ms  █░░░░░░░░░░░░░░░░░░░  0.7%   │
│  오버헤드 (데이터 전송 등)    5ms  ██░░░░░░░░░░░░░░░░░░  1.7%   │
│  ─────────────────────────────────────────────────────────────   │
│  총합:                      48ms  ████████████████░░░░  16.0%   │
│  여유:                     252ms  ░░░░░░░░░░░░░░░░░░░░  84.0%   │
│                                                                  │
│  ✓ 0.3초 요구사항 충족 (여유도 84%)                               │
└──────────────────────────────────────────────────────────────────┘
```

---

### 12. 학습 파이프라인

#### 12.1 데이터셋 전략

```
┌─────────────────────────────────────────────────────────────────┐
│                     데이터셋 전략 (3계층)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 1: 공개 데이터셋 (Pretrain & Baseline)                   │
│  ├─ DMD (Driver Monitoring Dataset): 41h, RGB+IR, 차폐 포함    │
│  ├─ COCO Keypoints: 포즈 추출기 학습                           │
│  ├─ ETH-XGaze: 시선 추정 pretrain                              │
│  └─ 300W / WFLW: 얼굴 랜드마크 pretrain                       │
│                                                                 │
│  Layer 2: 합성 증강 (Data Augmentation)                         │
│  ├─ 검은색 마스킹: 눈/입 영역 랜덤 차폐                        │
│  ├─ 조도 변환: 밝기/대비/감마 랜덤 변경                        │
│  ├─ 해상도 저하: bilinear 다운샘플링                            │
│  ├─ IR 시뮬레이션: RGB → grayscale + 히스토그램 매칭            │
│  └─ Cutout/GridMask: 랜덤 영역 차폐                            │
│                                                                 │
│  Layer 3: 자체 Failure 데이터셋 (핵심 차별화)                   │
│  ├─ 직접 IR 카메라 촬영                                        │
│  ├─ 시나리오: 야간, 선글라스, 마스크, 핸드폰, 졸음 연기         │
│  ├─ 어노테이션: 행동 레이블 + 차폐 유형 + 시선 영역             │
│  └─ 용도: fine-tuning + 최종 평가 세트                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 12.2 학습 스케줄

```
Phase 1: Pretrain (Week 5-7)
  - 공개 데이터로 각 branch 독립 학습
  - Branch A (Gaze): ETH-XGaze → IR fine-tune
  - Branch B (Pose): 6DRepNet pretrained → IR fine-tune
  - Branch C (Skeleton): COCO → RTMPose fine-tune + GCN 학습
  - Branch D (Occlusion): DMD 차폐 라벨로 학습

Phase 2: Joint Training (Week 8-10)
  - Fusion 모듈 포함 전체 파이프라인 end-to-end
  - Multi-task loss: L = L_gaze + L_pose + L_skeleton + L_occlusion + L_behavior
  - 각 loss의 가중치는 uncertainty weighting (Kendall et al., 2018) 적용

Phase 3: Failure Fine-tune (Week 11-12)
  - 자체 failure 데이터셋으로 집중 fine-tuning
  - Hard negative mining: 오분류 사례 중심 재학습
  - Confidence calibration: 신뢰도가 실제 정확도와 일치하도록 조정
```

#### 12.3 Loss 함수 설계

```python
class DMSMultiTaskLoss(nn.Module):
    """
    Multi-task Loss with Uncertainty Weighting
    Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses" (CVPR 2018)
    """
    def __init__(self):
        super().__init__()
        # 학습 가능한 task 불확실성 파라미터
        self.log_sigma_gaze = nn.Parameter(torch.zeros(1))
        self.log_sigma_pose = nn.Parameter(torch.zeros(1))
        self.log_sigma_skeleton = nn.Parameter(torch.zeros(1))
        self.log_sigma_occlusion = nn.Parameter(torch.zeros(1))
        self.log_sigma_behavior = nn.Parameter(torch.zeros(1))

    def forward(self, predictions, targets):
        # 각 task loss
        L_gaze = F.mse_loss(predictions['gaze'], targets['gaze'])
        L_pose = F.mse_loss(predictions['pose'], targets['pose'])
        L_skeleton = F.cross_entropy(predictions['skeleton'], targets['skeleton'])
        L_occlusion = F.cross_entropy(predictions['occlusion'], targets['occlusion'])
        L_behavior = F.cross_entropy(predictions['behavior'], targets['behavior'])

        # Uncertainty weighting
        total = (
            torch.exp(-self.log_sigma_gaze) * L_gaze + self.log_sigma_gaze +
            torch.exp(-self.log_sigma_pose) * L_pose + self.log_sigma_pose +
            torch.exp(-self.log_sigma_skeleton) * L_skeleton + self.log_sigma_skeleton +
            torch.exp(-self.log_sigma_occlusion) * L_occlusion + self.log_sigma_occlusion +
            torch.exp(-self.log_sigma_behavior) * L_behavior + self.log_sigma_behavior
        )

        return total, {
            'gaze': L_gaze.item(),
            'pose': L_pose.item(),
            'skeleton': L_skeleton.item(),
            'occlusion': L_occlusion.item(),
            'behavior': L_behavior.item()
        }
```

---

## Part 3: 실행 계획

---

### 13. 15주 상세 로드맵

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Week 1-2 (3/6 ~ 3/20): 기획 및 분석 ──────────── [완료]
  ✓ 문제 정의 구체화
  ✓ 관련 논문 분석
  ✓ 제안서 작성 및 제출
  ✓ 입력/출력 구조 및 시나리오 확정

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Week 3-4 (3/20 ~ 4/3): 데이터 확보 및 환경 구축 ── [진행 중]

  [시선팀: 황영인, 이수찬]
  ├─ ETH-XGaze / MPIIFaceGaze 데이터셋 다운로드
  ├─ IR 시선 데이터 확보 방안 조사
  └─ GazeCNN 백본(EfficientNet-B0) 초기 세팅

  [포즈팀: 강성찬, 강민성]
  ├─ COCO Keypoints 데이터셋 준비
  ├─ DMD 데이터셋 접근 (기업 요청 진행 중)
  ├─ RTMPose / 6DRepNet pretrained 모델 다운로드
  └─ IR 카메라 구입 (5만원, 재료비 청구 예정)

  [PM: 유건]
  ├─ 전체 코드 레포지토리 구축 (Git)
  ├─ Colab Pro 환경 세팅 + DGX Spark 수요 조사 제출
  ├─ Failure 데이터 촬영 시나리오 설계 + 라벨 기준 정의
  └─ 아키텍처 설계서 검토 및 팀 공유

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Week 5-7 (4/3 ~ 4/24): 전처리 및 모듈 Pretrain

  [시선팀]
  ├─ W5: EfficientNet-B0 → 1ch IR 적응 (conv1 수정, pretrained 로딩)
  ├─ W5: ETH-XGaze로 GazeCNN pretrain
  ├─ W6: IR 시뮬레이션 데이터로 domain adaptation fine-tune
  ├─ W7: Eye ROI 추출 로직 + confidence head 추가
  └─ W7: 시선 MAE 10° 이내 달성 확인

  [포즈팀]
  ├─ W5: RTMPose 세팅 + COCO pretrain 확인
  ├─ W5: 6DRepNet360 → 1ch IR 적응
  ├─ W6: 스켈레톤 그래프 정의 (13 관절점) + CTR-GCN 구현
  ├─ W6: OcclusionDetector(MobileNetV3-Small) 학습 시작
  ├─ W7: IR 데이터에서 포즈/스켈레톤 추출 테스트
  └─ W7: Head Pose MAE 5° 이내 확인

  [PM]
  ├─ W5: IR 전처리 파이프라인(CLAHE→Bilateral→Gamma) 구현
  ├─ W5-6: IR Domain Adaptation U-Net 구현 및 학습
  ├─ W6: YOLO-FaceV2 세팅 + IR fine-tuning
  ├─ W7: Failure 데이터 1차 촬영 (야간, 선글라스)
  └─ W7: 전체 모듈 통합 테스트 (파이프라인 연결)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Week 8-10 (4/24 ~ 5/15): Branch 개발 + 중간 발표 준비 ★

  ** 중간 발표 예상: 5월 초 **

  [시선팀]
  ├─ W8: Case 3 (GazeCNN) 완성 + Case 2 (GazeMLP) 비교 구현
  ├─ W9: 증강 데이터(검은 마스킹, 조도 변환)로 강건성 훈련
  └─ W10: branch 단독 성능 측정 + 중간 발표 자료 기여

  [포즈팀]
  ├─ W8: Branch B (Head Pose) + Branch C (Skeleton GCN) 완성
  ├─ W8: Branch D (Occlusion) 완성
  ├─ W9: Case 4, Case 5 베이스라인 구현 (비교군)
  └─ W10: branch 단독 성능 측정 + 중간 발표 자료 기여

  [PM]
  ├─ W8: ConfidenceAwareFusion 모듈 구현
  ├─ W9: 전체 파이프라인 통합 (Case 3 기준)
  ├─ W9: 레이턴시 측정 + 병렬화 검증
  ├─ W10: 중간 발표 자료 작성 총괄
  └─ W10: 초기 결과 + expected results 포함

  [중간 발표 전략]
  ├─ Case 3 (메인) vs Case 4 (베이스라인) 비교 결과 제시
  ├─ 정상 환경 정확도 + IR 환경 정확도 비교
  ├─ 레이턴시 측정 결과 (0.3초 달성 근거)
  ├─ 자체 Failure 데이터셋 샘플 시연
  └─ 최종 발표 expected results 제시

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Week 11-12 (5/15 ~ 5/29): 신뢰도 설계 및 융합 고도화

  [시선팀]
  ├─ W11: Gaze confidence calibration (실제 정확도와 일치하도록)
  ├─ W11: 차폐 상황별 시선 branch degradation 패턴 분석
  └─ W12: Case 1 (ST-GCN+Transformer) 구현 시도 (시간 여유 시)

  [포즈팀]
  ├─ W11: Skeleton confidence + temporal consistency 로직 구현
  ├─ W11: Failure 데이터 2차 촬영 (마스크, 핸드폰, 졸음 연기)
  └─ W12: 어노테이션 작업 + Hard negative 재학습

  [PM]
  ├─ W11: Fusion temperature 튜닝 + Degraded mode 테스트
  ├─ W11: Multi-task loss uncertainty weighting 적용
  ├─ W12: TCN 시계열 분류기 학습 + 행동 분류 정확도 평가
  └─ W12: Alert State Machine 룰 튜닝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Week 13-15 (5/29 ~ 6/5): 통합 평가 + 최종 발표 ★★

  [전체 팀]
  ├─ W13: Case 3 vs Case 1 vs Case 4 vs Case 5 전면 비교
  │    ├─ 정상 환경 정확도
  │    ├─ 저조도 환경 정확도
  │    ├─ 차폐 환경 정확도 (선글라스, 마스크, 핸드폰)
  │    ├─ FP/FN 비율 분석
  │    └─ 레이턴시 측정
  │
  ├─ W13: Failure 데이터셋 기반 최종 평가
  │    ├─ 시나리오별 정확도 (야간, 부분 가려짐, 행동 가려짐, 정상 혼동)
  │    └─ Degraded mode 동작 검증
  │
  ├─ W14: 경량화 적용 (INT8 Quantization, ONNX 변환)
  ├─ W14: 실시간 데모 영상 촬영
  │
  └─ W15: 최종 발표 자료 + 보고서 작성
       ├─ 비교군 풍부하게 (공개 데이터 결과 + 자체 데이터 결과)
       ├─ ablation study (각 branch 제거 시 성능 변화)
       ├─ 신뢰도 융합의 효과 정량적 입증
       └─ 실시간 데모 시연

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### 14. 팀 역할 분배 상세

```
┌─────────────────────────────────────────────────────────────────┐
│                        팀 구성 및 역할                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [시선팀] 황영인 + 이수찬                                       │
│  ├─ Branch A: 시선 추정 모듈 (GazeCNN / GazeMLP)               │
│  ├─ 시선 관련 데이터셋 관리 (ETH-XGaze, MPIIFaceGaze)          │
│  ├─ Case 2 vs Case 3 비교 실험                                 │
│  └─ 시선 MAE 지표 관리                                         │
│                                                                 │
│  [포즈팀] 강성찬 + 강민성                                       │
│  ├─ Branch B: 머리 포즈 (6DRepNet360)                           │
│  ├─ Branch C: 스켈레톤 행동 인식 (CTR-GCN + Transformer)       │
│  ├─ Branch D: 차폐 감지 (MobileNetV3-Small)                    │
│  ├─ 포즈/스켈레톤 데이터 관리 (COCO, DMD)                      │
│  └─ Case 4, Case 5 베이스라인 구현                              │
│                                                                 │
│  [PM] 유건                                                      │
│  ├─ Stage 0: IR 전처리 + Domain Adaptation                     │
│  ├─ Stage 1: YOLO-FaceV2 얼굴 검출                             │
│  ├─ Stage 3: ConfidenceAwareFusion 설계 및 구현                │
│  ├─ Stage 4: TCN 시계열 분류기                                  │
│  ├─ Stage 5: Alert State Machine                                │
│  ├─ 전체 파이프라인 통합 + 레이턴시 관리                        │
│  ├─ Failure 데이터셋 촬영 기획 + 어노테이션 기준 설계           │
│  └─ 발표 자료 총괄                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### 15. 리스크 관리 및 백업 플랜

| 리스크 | 확률 | 영향 | 대응 |
|--------|------|------|------|
| IR 카메라 구입 지연 | 중 | 상 | RGB→IR 시뮬레이션으로 우선 개발, 카메라 도착 후 실데이터 전환 |
| MediaPipe IR 성능 부족 | 상 | 중 | RTMPose로 대체 (이미 백업 모델로 선정) |
| DGX Spark 할당 지연 | 상 | 중 | Colab Pro L4 GPU로 전체 학습 가능 (검증 완료) |
| 자체 데이터 어노테이션 시간 부족 | 중 | 중 | 최소 시나리오(4종)로 축소 + pseudo label 활용 |
| Case 1 (ST-GCN) 구현 시간 부족 | 중 | 하 | Case 3을 메인으로, Case 1은 최종 발표 시 보너스 |
| Fusion 모듈 학습 불안정 | 하 | 상 | 고정 가중치 fallback (gaze:pose:skeleton = 0.4:0.3:0.3) |
| 0.3초 레이턴시 미달 | 하 | 상 | INT8 양자화 + 프레임 샘플링 (2프레임 간격) |

---

### 16. 성공 지표 및 평가 기준

| 지표 | 목표 | SOTA 참고 | 평가 시점 |
|------|------|----------|---------|
| 통합 정확도 (Accuracy) | ≥ 95% | 99.75% (CNN-BiLSTM-AM) | 최종 발표 |
| 시선 MAE | ≤ 10° | 4.08° (Attention CNN) | 중간 발표 |
| 머리 포즈 MAE | ≤ 5° | 3-5° (6DRepNet360) | 중간 발표 |
| 이상행동 Recall | ≥ 90% | 95%+ (Occlusion-aware DMS) | 최종 발표 |
| FP Rate (오경보율) | ≤ 10% | - | 최종 발표 |
| 레이턴시 | ≤ 300ms | 48ms (예상) | 중간 발표 |
| Degraded Mode 유지율 | ≥ 80% (중도 차폐) | - | 최종 발표 |

---

### 17. 핵심 참고 논문 (우선 읽기 순서)

1. **[필독]** "Occlusion-aware Driver Monitoring System using DMD" (arXiv:2504.20677, Apr 2025) — 본 프로젝트의 직접적 선행 연구
2. **[필독]** "Enhancing DMS Based on Novel Multi-Task Fusion Algorithm" (Sensors 25(21):6799, Nov 2025) — 동적 융합 핵심 참고
3. **[시선팀]** "Appearance-based Gaze Estimation with Improved Attention Branch" (2025) — 4.08° MAE
4. **[포즈팀]** "6DRepNet360" (IEEE TIP 2024) — 360° 머리 포즈 추정
5. **[포즈팀]** "Two-Stream Spatio-Temporal GCN-Transformer" (Scientific Reports, 2025) — ST-GCN SOTA
6. **[PM]** "WTEFNet: Real-Time Low-Light Object Detection for ADAS" (2025) — IR 전처리
7. **[PM]** "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall, CVPR 2018) — Loss 설계

---

### 18. Consequences (이 설계의 결과)

**쉬워지는 것:**
- 각 branch를 독립적으로 개발·테스트 가능 (2인 2인 1인 구조와 정확히 매핑)
- 차폐 상황에서 시스템이 "무너지는" 대신 "성능이 저하되는" 방식으로 동작
- 비교군 풍부 (Case 1~5, 5개 모델 ablation study)
- 신뢰도 수치화로 정량적 분석 가능

**어려워지는 것:**
- Multi-task loss 가중치 튜닝이 복잡 (uncertainty weighting으로 완화)
- Fusion 모듈 학습 시 branch 간 gradient 충돌 가능성 (gradient surgery 기법 검토)
- 자체 데이터 어노테이션 공수

**재검토 필요 사항:**
- Week 7 기준: IR Domain Adaptation Layer의 실효성 평가 → 효과 미미 시 CLAHE만으로 진행
- Week 10 기준: Case 1(ST-GCN) 구현 여부 결정
- Week 12 기준: 경량화 수준 결정 (INT8만 or TensorRT까지)

---

## Action Items

1. [ ] 팀 전체: arXiv:2504.20677 논문 이번 주 내 읽기
2. [ ] 시선팀: ETH-XGaze 데이터셋 다운로드 + EfficientNet-B0 1ch 적응 코드 작성
3. [ ] 포즈팀: RTMPose + 6DRepNet360 pretrained 모델 세팅
4. [ ] PM: Git 레포 구축 + 전처리 파이프라인(CLAHE→Bilateral→Gamma) 구현
5. [ ] PM: IR 카메라 구매 진행 (멘토 재료비 협의)
6. [ ] PM: Failure 데이터 촬영 시나리오 4종 구체화 및 라벨 가이드라인 작성
7. [ ] PM: DGX Spark 수요 조사 멘토에게 리마인드
