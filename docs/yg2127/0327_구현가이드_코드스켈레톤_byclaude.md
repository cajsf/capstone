# DMS 구현 가이드 & 코드 스켈레톤
## 신뢰도 기반 적응형 융합 시스템

**작성일:** 2026년 3월 27일
**대상:** 스꾸삐 팀 개발자

---

## Part 1: 신뢰도 계산 핵심 로직

### 1.1 특징점 신뢰도 계산

```python
import numpy as np
from scipy.ndimage import gaussian_filter1d

class LandmarkConfidence:
    """특징점 신뢰도 계산 모듈"""

    def __init__(self, smoothing_window=5):
        self.smoothing_window = smoothing_window
        self.history = []  # 프레임 히스토리

    def compute_confidence(self, landmarks, detection_scores, frame_idx):
        """
        Args:
            landmarks: np.array (N_keypoints, 2) - (x, y) 좌표
            detection_scores: np.array (N_keypoints,) - 각 특징점의 신뢰도 [0, 1]
            frame_idx: 현재 프레임 인덱스

        Returns:
            dict: {
                'landmark_score': float,
                'visibility': float,
                'temporal_consistency': float,
                'final_confidence': float
            }
        """

        # 1) 특징점 점수: 평균값
        landmark_score = np.mean(detection_scores)

        # 2) 가시성: 신뢰도 높은 특징점 비율
        visible_ratio = np.sum(detection_scores > 0.5) / len(detection_scores)
        visibility = visible_ratio

        # 3) 시간적 일관성
        temporal_consistency = self._compute_temporal_smoothness(
            landmarks,
            frame_idx,
            threshold=20.0  # 픽셀 단위
        )

        # 4) 최종 신뢰도 (곱셈)
        final_conf = landmark_score * visibility * temporal_consistency

        return {
            'landmark_score': float(landmark_score),
            'visibility': float(visibility),
            'temporal_consistency': float(temporal_consistency),
            'final_confidence': float(final_conf)
        }

    def _compute_temporal_smoothness(self, landmarks, frame_idx, threshold=20.0):
        """프레임 간 변화량으로 일관성 계산"""

        if len(self.history) == 0:
            self.history.append(landmarks.copy())
            return 1.0

        # 이전 프레임과의 거리
        prev_landmarks = self.history[-1]
        distance = np.linalg.norm(landmarks - prev_landmarks, axis=1)
        mean_distance = np.mean(distance)

        # 정규화: 임계값 기준
        consistency = 1.0 - np.clip(mean_distance / threshold, 0, 1)

        self.history.append(landmarks.copy())
        if len(self.history) > 30:  # 메모리 관리
            self.history.pop(0)

        return float(consistency)


class OcclusionDetector:
    """가려짐 감지 모듈"""

    def __init__(self, occlusion_types=['sunglasses', 'mask', 'hand', 'phone']):
        self.occlusion_types = occlusion_types
        # 실제 구현: face_recognition 또는 객체 검출 모델

    def compute_occlusion_confidence(self, face_region, landmarks):
        """
        Args:
            face_region: np.array, 얼굴 영역 이미지
            landmarks: np.array, 얼굴 특징점

        Returns:
            dict: {
                'occlusion_type': str or None,
                'severity': float [0, 1],
                'confidence': float [0, 1]  # 1 - severity
            }
        """

        # 간단한 휴리스틱: 눈 영역 밝기 변화
        # 실제 구현에서는 YOLOv8 또는 dedicated 모델 사용

        eye_region = self._extract_eye_region(face_region, landmarks)
        brightness_change = self._compute_brightness_variance(eye_region)

        if brightness_change < 0.1:
            # 선글라스 가능성 높음
            return {
                'occlusion_type': 'sunglasses',
                'severity': 0.9,
                'confidence': 0.1
            }
        elif brightness_change < 0.3:
            return {
                'occlusion_type': 'mask',
                'severity': 0.3,
                'confidence': 0.7
            }
        else:
            return {
                'occlusion_type': None,
                'severity': 0.0,
                'confidence': 1.0
            }

    def _extract_eye_region(self, face_region, landmarks):
        """눈 영역 추출 (간단화)"""
        if len(landmarks) >= 10:
            eye_center = landmarks[0:2].mean(axis=0)  # 근사
            h, w = face_region.shape[:2]
            x, y = int(eye_center[0]), int(eye_center[1])
            eye_patch = face_region[max(0, y-20):y+20, max(0, x-30):x+30]
            return eye_patch
        return face_region

    def _compute_brightness_variance(self, region):
        """밝기 분산 계산"""
        if region.size == 0:
            return 0.5
        gray = np.mean(region, axis=2) if len(region.shape) == 3 else region
        return float(np.var(gray) / 255.0)
```

### 1.2 다중 Branch 신뢰도 통합

```python
class ConfidenceBasedFusion:
    """신뢰도 기반 동적 융합"""

    def __init__(self, n_branches=3, fusion_mode='soft_blending'):
        self.n_branches = n_branches
        self.fusion_mode = fusion_mode
        self.branch_names = ['gaze', 'pose', 'occlusion']

    def fuse_outputs(self, branch_outputs, confidence_scores):
        """
        Args:
            branch_outputs: list of arrays
                - gaze: (2,) [yaw, pitch]
                - pose: (3,) [yaw, pitch, roll]
                - occlusion: scalar or category
            confidence_scores: list of floats [0, 1]

        Returns:
            dict: {
                'output': fused output,
                'weights': normalized weights,
                'overall_confidence': float,
                'mode': 'normal' or 'degraded'
            }
        """

        # Step 1: 신뢰도 정규화
        scores = np.array(confidence_scores)
        scores = np.clip(scores, 0, 1)  # [0, 1] 범위

        # 정규화 (합 = 1)
        if scores.sum() == 0:
            weights = np.ones(self.n_branches) / self.n_branches
        else:
            weights = scores / scores.sum()

        # Step 2: Soft Blending
        if self.fusion_mode == 'soft_blending':
            fused_output = self._soft_blend(branch_outputs, weights)
        elif self.fusion_mode == 'gating':
            fused_output = self._gated_blend(branch_outputs, scores)
        else:
            fused_output = branch_outputs[np.argmax(weights)]

        # Step 3: 전체 신뢰도
        overall_conf = float(np.average(scores, weights=weights))

        # Step 4: 모드 판정
        mode = 'degraded' if overall_conf < 0.3 else 'normal'

        return {
            'output': fused_output,
            'weights': weights,
            'confidence_scores': scores,
            'overall_confidence': overall_conf,
            'mode': mode,
            'weights_dict': {
                self.branch_names[i]: float(weights[i])
                for i in range(len(weights))
            }
        }

    def _soft_blend(self, outputs, weights):
        """가중 평균 (Soft Blending)"""
        # Gaze와 Pose는 숫자, Occlusion은 카테고리
        # 간단화: Gaze와 Pose만 혼합 (Occlusion은 신뢰도로만 반영)

        gaze = np.array(outputs[0])  # (2,)
        pose = np.array(outputs[1])  # (3,)

        # Gaze-Pose 혼합 (가중치 정규화)
        w_gp = np.array([weights[0], weights[1]])
        w_gp = w_gp / (w_gp.sum() + 1e-8)

        # Gaze와 Pose 방향을 직접 혼합하기는 어려움
        # 대신: Confidence 높은 쪽을 더 반영
        if weights[0] > weights[1]:
            return gaze
        else:
            return pose[:2]  # Pitch & Yaw만

    def _gated_blend(self, outputs, scores):
        """Gating 메커니즘 기반 혼합"""
        # sigmoid 가중치
        gate = 1.0 / (1.0 + np.exp(-4 * (scores[0] - 0.5)))  # [0, 1]

        gaze = np.array(outputs[0])
        pose = np.array(outputs[1])[:2]

        # Linear interpolation with gate
        blended = gate * gaze + (1 - gate) * pose
        return blended


class DynamicConfidenceMonitor:
    """동적 신뢰도 모니터링 (Degraded Mode 관리)"""

    def __init__(self, degraded_threshold=0.3, history_window=30):
        self.degraded_threshold = degraded_threshold
        self.history_window = history_window
        self.confidence_history = []
        self.mode_transition_log = []

    def update_and_get_mode(self, overall_confidence, timestamp):
        """
        신뢰도를 업데이트하고 현재 모드 반환

        Returns:
            str: 'normal', 'degraded_level_1', 'degraded_level_2', 'failure'
        """

        self.confidence_history.append(overall_confidence)
        if len(self.confidence_history) > self.history_window:
            self.confidence_history.pop(0)

        # 평균 신뢰도
        avg_conf = np.mean(self.confidence_history)

        # 모드 결정
        if avg_conf > 0.6:
            mode = 'normal'
        elif avg_conf > 0.3:
            mode = 'degraded_level_1'
        elif avg_conf > 0.0:
            mode = 'degraded_level_2'
        else:
            mode = 'failure'

        # 모드 전환 로깅
        if len(self.mode_transition_log) == 0 or \
           self.mode_transition_log[-1][1] != mode:
            self.mode_transition_log.append((timestamp, mode))

        return mode

    def get_decision_thresholds(self, mode):
        """모드별 결정 임계값 반환"""

        thresholds = {
            'normal': {
                'action_confidence_required': 0.80,
                'alert_threshold': 0.75,
                'classification_classes': 6  # 정밀 분류
            },
            'degraded_level_1': {
                'action_confidence_required': 0.70,
                'alert_threshold': 0.60,
                'classification_classes': 3  # 졸음/주의산만/정상
            },
            'degraded_level_2': {
                'action_confidence_required': 0.50,
                'alert_threshold': 0.40,
                'classification_classes': 2  # 위험/안전
            },
            'failure': {
                'action_confidence_required': 0.00,  # 경고만 발생
                'alert_threshold': 0.00,
                'classification_classes': 0  # 분류 중단
            }
        }

        return thresholds.get(mode, thresholds['degraded_level_2'])
```

---

## Part 2: 시간적 모델링 (TCN-Transformer 하이브리드)

```python
import torch
import torch.nn as nn

class TemporalConvBlock(nn.Module):
    """인과 합성곱 (Causal Convolution) 블록"""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()

        # 인과성을 위해 padding 적용: (kernel_size - 1) * dilation
        padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time_steps)

        Returns:
            out: (batch, out_channels, time_steps)
        """
        out = self.conv(x)
        # 인과성 유지: 미래 정보 제거
        out = out[:, :, :-((self.conv.kernel_size[0] - 1) * self.conv.dilation[0])]
        out = self.relu(out)
        out = self.dropout(out)
        return out


class TCNEncoder(nn.Module):
    """Temporal Convolutional Network 인코더"""

    def __init__(self, input_dim=32, hidden_dim=64, num_layers=3, dropout=0.2):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        layers = []
        for i in range(num_layers):
            in_ch = input_dim if i == 0 else hidden_dim
            out_ch = hidden_dim
            dilation = 2 ** i  # 지수적 팽창

            layers.append(
                TemporalConvBlock(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    dilation=dilation,
                    dropout=dropout
                )
            )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) - 골격 시퀀스

        Returns:
            out: (batch, seq_len, hidden_dim)
        """
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.layers(x)
        x = x.transpose(1, 2)  # (batch, seq_len, hidden_dim)
        return x


class TransformerAttentionBlock(nn.Module):
    """Transformer Self-Attention 블록"""

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_dim)

        Returns:
            out: (batch, seq_len, hidden_dim)
        """
        # Self-Attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward
        mlp_out = self.mlp(x)
        x = self.norm2(x + self.dropout(mlp_out))

        return x


class HybridTCNTransformer(nn.Module):
    """하이브리드 TCN-Transformer 모델"""

    def __init__(self,
                 input_dim=32,           # 골격 특징 차원
                 hidden_dim=64,          # 내부 차원
                 num_tcn_layers=3,       # TCN 깊이
                 num_attention_layers=2, # Attention 깊이
                 num_classes=6,          # 행동 클래스
                 dropout=0.2):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Phase 1: TCN (로컬 시간 패턴)
        self.tcn_encoder = TCNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_tcn_layers,
            dropout=dropout
        )

        # Phase 2: Transformer (전역 관계)
        self.attention_layers = nn.ModuleList([
            TransformerAttentionBlock(hidden_dim, num_heads=4, dropout=dropout)
            for _ in range(num_attention_layers)
        ])

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, skeleton_sequence, confidence_weights=None):
        """
        Args:
            skeleton_sequence: (batch, seq_len, input_dim)
                - seq_len: 32 (1초, 30fps)
                - input_dim: 32 (상체 16개 키포인트 × 2D)
            confidence_weights: (batch, seq_len, 1) 선택사항
                - 신뢰도 기반 가중치

        Returns:
            dict: {
                'logits': (batch, num_classes),
                'confidence': (batch,),
                'action': (batch,)
            }
        """

        # Phase 1: TCN
        tcn_out = self.tcn_encoder(skeleton_sequence)  # (batch, seq_len, hidden_dim)

        # 신뢰도 가중치 적용 (선택사항)
        if confidence_weights is not None:
            tcn_out = tcn_out * confidence_weights

        # Phase 2: Transformer
        attn_out = tcn_out
        for attn_layer in self.attention_layers:
            attn_out = attn_layer(attn_out)

        # Phase 3: Classification
        # 시퀀스의 마지막 토큰 사용
        final_output = attn_out[:, -1, :]  # (batch, hidden_dim)
        logits = self.classifier(final_output)  # (batch, num_classes)

        # 결과
        probs = torch.softmax(logits, dim=-1)
        confidence, action = torch.max(probs, dim=-1)

        return {
            'logits': logits,
            'probabilities': probs,
            'confidence': confidence,
            'action': action
        }


# 사용 예시
def example_usage():
    batch_size = 8
    seq_len = 32  # 1초 (30fps)
    input_dim = 32  # 16개 키포인트 × 2D

    # 모델 생성
    model = HybridTCNTransformer(
        input_dim=input_dim,
        hidden_dim=64,
        num_tcn_layers=3,
        num_attention_layers=2,
        num_classes=6,
        dropout=0.2
    )

    # 더미 입력
    skeleton_sequence = torch.randn(batch_size, seq_len, input_dim)
    confidence_weights = torch.ones(batch_size, seq_len, 1) * 0.8

    # 추론
    output = model(skeleton_sequence, confidence_weights)
    print(f"Action logits: {output['logits'].shape}")
    print(f"Confidence: {output['confidence'].mean():.4f}")
```

---

## Part 3: RGB → IR 도메인 적응

```python
import torch.nn as nn

class DomainDiscriminator(nn.Module):
    """도메인 판별자 (adversarial loss)"""

    def __init__(self, input_dim=256):
        super().__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0, 1]: 1 = RGB, 0 = IR
        )

    def forward(self, features):
        """
        Args:
            features: (batch, input_dim) - CNN 중간 특징

        Returns:
            domain_pred: (batch, 1) - [0, 1]
        """
        return self.discriminator(features)


class AdversarialDomainAdaptation(nn.Module):
    """적대적 도메인 적응"""

    def __init__(self, backbone, feature_dim=256):
        super().__init__()
        self.backbone = backbone  # 사전학습된 CNN (EfficientNet 등)
        self.discriminator = DomainDiscriminator(feature_dim)

    def forward_with_domain_loss(self, rgb_batch, ir_batch, lambda_d=0.5):
        """
        Args:
            rgb_batch: (batch, 3, H, W) - RGB 이미지
            ir_batch: (batch, 1, H, W) - IR 이미지 (또는 3채널)
            lambda_d: 도메인 손실 가중치

        Returns:
            dict: {
                'rgb_features': 특징,
                'ir_features': 적응된 특징,
                'domain_loss': 적대적 손실
            }
        """

        # 특징 추출
        rgb_features = self.backbone(rgb_batch)  # (batch, 256, 7, 7)
        ir_features = self.backbone(ir_batch)    # (batch, 256, 7, 7)

        # 공간 평균화
        rgb_pool = torch.nn.functional.adaptive_avg_pool2d(rgb_features, (1, 1))
        ir_pool = torch.nn.functional.adaptive_avg_pool2d(ir_features, (1, 1))

        rgb_pool = rgb_pool.view(rgb_pool.size(0), -1)  # (batch, 256)
        ir_pool = ir_pool.view(ir_pool.size(0), -1)     # (batch, 256)

        # 도메인 판별
        rgb_domain_pred = self.discriminator(rgb_pool)  # (batch, 1)
        ir_domain_pred = self.discriminator(ir_pool)

        # 적대적 손실
        # RGB를 1로, IR을 1로 분류하도록 생성기 속임
        domain_loss = (
            -torch.log(rgb_domain_pred.mean() + 1e-8)  # RGB: log(D) 최대화
            - torch.log(1 - ir_domain_pred.mean() + 1e-8)  # IR: log(1-D) 최대화
        )

        return {
            'rgb_features': rgb_features,
            'ir_features': ir_features,
            'rgb_pool': rgb_pool,
            'ir_pool': ir_pool,
            'domain_loss': domain_loss * lambda_d
        }


# 학습 루프 (간단화)
def train_domain_adaptation_step(model, rgb_batch, ir_batch,
                                  optimizer_backbone, optimizer_discriminator,
                                  lambda_d=0.5):
    """한 스텝의 도메인 적응 학습"""

    # 생성기 (backbone) 업데이트
    optimizer_backbone.zero_grad()
    output = model.forward_with_domain_loss(rgb_batch, ir_batch, lambda_d)
    domain_loss = output['domain_loss']
    domain_loss.backward()
    optimizer_backbone.step()

    # 판별자 업데이트
    optimizer_discriminator.zero_grad()
    output = model.forward_with_domain_loss(rgb_batch, ir_batch, lambda_d)

    # 판별자 손실
    rgb_features = output['rgb_pool'].detach()
    ir_features = output['ir_pool'].detach()

    rgb_domain_pred = model.discriminator(rgb_features)
    ir_domain_pred = model.discriminator(ir_features)

    disc_loss = (
        torch.nn.functional.binary_cross_entropy(
            rgb_domain_pred,
            torch.ones_like(rgb_domain_pred)  # RGB = 1
        ) +
        torch.nn.functional.binary_cross_entropy(
            ir_domain_pred,
            torch.zeros_like(ir_domain_pred)  # IR = 0
        )
    )

    disc_loss.backward()
    optimizer_discriminator.step()

    return {
        'domain_loss': domain_loss.item(),
        'disc_loss': disc_loss.item()
    }
```

---

## Part 4: 통합 DMS 파이프라인

```python
class DriverMonitoringSystem:
    """통합 운전자 모니터링 시스템"""

    def __init__(self, config):
        self.config = config

        # 컴포넌트 초기화
        self.pose_estimator = PoseEstimator(config['pose_model'])  # MediaPipe
        self.occlusion_detector = OcclusionDetector()
        self.skeleton_classifier = HybridTCNTransformer(...)
        self.confidence_fusion = ConfidenceBasedFusion()
        self.confidence_monitor = DynamicConfidenceMonitor()

    def process_frame(self, frame, timestamp):
        """
        한 프레임 처리

        Args:
            frame: np.array (H, W, 3) RGB 또는 IR
            timestamp: float 타임스탬프

        Returns:
            dict: {
                'action': str (전방주시, 미러, 휴대폰 등),
                'confidence': float,
                'mode': str (normal, degraded, failure),
                'alerts': list,
                'debug_info': dict
            }
        """

        # 1) 자세 추정
        pose_result = self.pose_estimator.estimate(frame)
        landmarks = pose_result['landmarks']
        landmark_scores = pose_result['scores']

        # 2) 신뢰도 계산 (특징점)
        landmark_conf = LandmarkConfidence().compute_confidence(
            landmarks,
            landmark_scores,
            frame_idx=timestamp
        )

        # 3) 가려짐 감지
        occlusion_conf = self.occlusion_detector.compute_occlusion_confidence(
            frame,
            landmarks
        )

        # 4) 시선 추정 (간단화)
        gaze_output = self._estimate_gaze(landmarks)  # (yaw, pitch)

        # 5) 스켈레톤 시퀀스 구축
        skeleton_seq = self._build_skeleton_sequence()  # (1, 32, 32)

        # 6) 행동 분류
        action_output = self.skeleton_classifier(skeleton_seq)

        # 7) 신뢰도 기반 융합
        branch_outputs = [gaze_output, pose_output, occlusion_output]
        branch_confidences = [
            landmark_conf['final_confidence'],
            pose_conf,  # 별도 계산
            occlusion_conf['confidence']
        ]

        fusion_result = self.confidence_fusion.fuse_outputs(
            branch_outputs,
            branch_confidences
        )

        # 8) 동적 모드 판정
        mode = self.confidence_monitor.update_and_get_mode(
            fusion_result['overall_confidence'],
            timestamp
        )

        # 9) 모드별 의사결정
        decision_thresholds = self.confidence_monitor.get_decision_thresholds(mode)

        alerts = []
        if action_output['confidence'] > decision_thresholds['alert_threshold']:
            action = action_output['action']
            if self._is_risky_action(action):
                alerts.append({
                    'level': 'warning' if mode == 'normal' else 'critical',
                    'message': f'Driver distraction: {action}',
                    'confidence': float(action_output['confidence'])
                })

        return {
            'action': self._decode_action(action_output['action']),
            'confidence': float(fusion_result['overall_confidence']),
            'mode': mode,
            'alerts': alerts,
            'debug_info': {
                'landmark_conf': landmark_conf,
                'occlusion': occlusion_conf,
                'weights': fusion_result['weights_dict'],
                'mode': mode
            }
        }

    def _estimate_gaze(self, landmarks):
        """랜드마크에서 시선 추정 (간단화)"""
        # 실제 구현: 3D 포즈 추정 + 시선 백터
        # 여기서는 더미 값
        return np.array([10.0, 5.0])  # (yaw, pitch)

    def _is_risky_action(self, action_id):
        """위험 행동 판정"""
        risky_actions = [4, 5, 6, 7]  # 휴대폰, 졸음, 뒤돌아봄, 고개 숙임
        return action_id in risky_actions

    def _decode_action(self, action_id):
        actions = [
            'forward_looking',
            'side_mirror',
            'rear_mirror',
            'navigation',
            'phone_use',
            'turning_back'
        ]
        return actions[action_id] if action_id < len(actions) else 'unknown'


# 메인 루프
def main():
    config = {
        'pose_model': 'mediapipe',
        'backbone': 'efficientnet_b0',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    dms = DriverMonitoringSystem(config)

    # 비디오 루프 (카메라 또는 파일)
    cap = cv2.VideoCapture(0)  # 또는 video_file.mp4

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # DMS 처리
        result = dms.process_frame(frame, timestamp)

        # 시각화
        display_frame = frame.copy()

        # 모드 표시
        mode_color = (0, 255, 0) if result['mode'] == 'normal' else (0, 165, 255)
        cv2.putText(
            display_frame,
            f"Mode: {result['mode']} ({result['confidence']:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            mode_color,
            2
        )

        # 행동 표시
        cv2.putText(
            display_frame,
            f"Action: {result['action']}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        # 경고 표시
        for i, alert in enumerate(result['alerts']):
            alert_color = (0, 0, 255) if alert['level'] == 'critical' else (0, 165, 255)
            cv2.putText(
                display_frame,
                alert['message'],
                (10, 110 + i * 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                alert_color,
                2
            )

        cv2.imshow('Driver Monitoring System', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
```

---

## Part 5: 성능 평가 메트릭

```python
class DMS_Evaluator:
    """DMS 성능 평가"""

    def __init__(self):
        self.predictions = []
        self.ground_truths = []

    def evaluate_accuracy(self):
        """정확도 계산"""
        correct = sum(p == g for p, g in zip(self.predictions, self.ground_truths))
        return correct / len(self.predictions) if self.predictions else 0

    def evaluate_latency(self, frame_processing_times):
        """지연시간 평가"""
        avg_latency = np.mean(frame_processing_times)
        p95_latency = np.percentile(frame_processing_times, 95)
        p99_latency = np.percentile(frame_processing_times, 99)

        return {
            'average_ms': avg_latency * 1000,
            'p95_ms': p95_latency * 1000,
            'p99_ms': p99_latency * 1000,
            'meets_requirement': avg_latency < 0.3  # 300ms 이내
        }

    def evaluate_per_class_metrics(self, y_true, y_pred, class_names):
        """클래스별 메트릭"""
        from sklearn.metrics import precision_recall_fscore_support

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )

        metrics_by_class = {}
        for i, class_name in enumerate(class_names):
            metrics_by_class[class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }

        return metrics_by_class


# 예시
evaluator = DMS_Evaluator()

# 테스트 반복
latencies = []
for frame in test_frames:
    start = time.time()
    result = dms.process_frame(frame, 0)
    end = time.time()
    latencies.append(end - start)

    evaluator.predictions.append(result['action'])
    evaluator.ground_truths.append(ground_truth[frame])

# 평가
print(f"Accuracy: {evaluator.evaluate_accuracy():.2%}")
print(f"Latency: {evaluator.evaluate_latency(latencies)}")
```

---

## 핵심 체크리스트

- [ ] **신뢰도 계산:** landmark, visibility, temporal, classifier
- [ ] **동적 융합:** soft blending + gating 메커니즘
- [ ] **TCN-Transformer:** 시간적 모델링
- [ ] **도메인 적응:** 3단계 (글로벌, 소도메인, fine-tuning)
- [ ] **Degraded Mode:** 신뢰도별 자동 모드 전환
- [ ] **성능 평가:** 정확도, 지연시간, 클래스별 메트릭

---

**최종 주의사항:**
- 실제 차량 환경에서 추가 최적화 필수
- GPU 메모리 및 배터리 소비 모니터링
- 정기적 모델 재학습 (새로운 실패 사례)

---

**작성자:** Claude AI
**날짜:** 2026년 3월 27일
