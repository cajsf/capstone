# DMS Research 2024-2026: Quick Reference Summary

## Critical Findings by Topic

### 1. IR-Based Gaze Estimation (Best Methods)

| Model | Publication | Year | Key Metric | IR Support | Notes |
|-------|-------------|------|-----------|-----------|-------|
| **Appearance-based CNN + Attention** | MDPI 2025 | 2025 | 4.08° error | No (RGB) | Competitive with IR, calibration-free |
| **Model-Based 3D (TOF)** | MDPI 2024 | 2024 | 3D-robust | YES (IR) | TOF/IR fusion, best for depth |
| **Polarized NIR Illumination** | ETRA 2020 | 2020+ | Robust tracking | YES (NIR) | Minimal cross-talk, 24/7 suitable |
| **IR Domain Adaptation** | MDPI 2024 | 2024 | 15-25% gain | YES | Day-to-night style transfer critical |

**Recommendation:** Attention-based CNN (2025) for robustness; domain adaptation layer mandatory for IR

---

### 2. Occlusion-Robust Face Analysis (Top Papers)

| Technique | Paper | Year | Occlusion Type | Key Result |
|-----------|-------|------|----------------|-----------|
| **Multi-stream + Gating** | MDPI 2025 | 2025 | Sunglasses, masks, hands | Handles >50% occlusion |
| **Occlusion-aware DMS** | arXiv:2504.20677 | 2025 | All types + low-light | RGB+IR pipeline, confidence scores |
| **Vision Transformer** | SciOpen 2025 | 2025 | General occlusion | Superior robustness vs. CNNs |
| **YOLO-FaceV2** | TPAMI 2024 | 2024 | Scale + occlusion | Simultaneous handling |

**Recommendation:** Use arXiv:2504.20677 (Occlusion-aware DMS) as reference implementation

---

### 3. Confidence-Based Fusion (Core Algorithm)

| Paper | Year | Method | Key Innovation |
|-------|------|--------|-----------------|
| **Multi-Task Fusion** | Sensors 2025 | BiLSTM + dynamic weighting | Learned confidence weights per frame |
| **Driver-Net** | IEEE IV 2025 | Triple-camera + voting | Multi-view confidence scoring |
| **Robust Multiview** | arXiv:2304 | Masked attention transformer | Learn which modalities to trust |

**Implementation:**
```
confidence[branch] = (detection_prob × visibility × temporal_consistency × task_confidence)
final_output = weighted_mean(branches, weights=softmax(confidences))
```

---

### 4. Action Recognition Architectures

| Method | Year | Performance | Modality | Real-time |
|--------|------|-------------|----------|-----------|
| **CNN-BiLSTM-AM** | 2025 | 99.75% accuracy | RGB video | <100ms |
| **Two-Stream GCN-Transformer** | 2025 | SOTA skeleton | Pose landmarks | Yes |
| **ST-GCN + Part-Joint Attention** | 2025 | Adaptive joints | Skeleton (IR-adapted) | Yes |
| **SlowFast 3D CNN** | 2025 | 92.4% accuracy | Video | 0.07s/frame |

**For Driver Behavior:** CNN-BiLSTM-AM recommended (proven 99.75% on State Farm)

---

### 5. Lightweight Models for Edge

| Model | Energy (mJ) | Latency | Accuracy Drop | Framework |
|-------|------------|---------|---------------|-----------|
| **MobileNet** | 25 | 50-80ms | Low | Standard |
| **EfficientNet** | 20 | 40-60ms | Lower | Preferred |
| **Knowledge Distill** | 15-18 | 30-50ms | Minimal | Automotive standard |
| **MobGazeNet** | ~15 | <50ms | 1-2% vs teacher | Gaze-specific |

**Recommendation:** EfficientNet as baseline; apply knowledge distillation for automotive SoC

---

### 6. Key Datasets Comparison

| Dataset | Year | Modality | IR | Scale | Occlusion | Status |
|---------|------|----------|-----|-------|-----------|--------|
| **DMD** | 2020 | RGB+D+IR | YES | 41h | Limited | Open source, primary |
| **AVDMD** | 2024 | RGB | NO | Real-world | Various | IEEE Dataport |
| **DAOS** | 2026 | Multi-modal | Likely YES | Comprehensive | Extensive | Newest, best-annotated |
| **ETH-XGaze** | 2020+ | RGB | NO | Moderate | Extreme pose | Benchmark |
| **MPIIGaze** | 2019+ | RGB | NO | 213k frames | No | Standard benchmark |

**For Project:** Use DMD (IR support); supplement with DAOS (2026) for latest annotations

---

### 7. MediaPipe Face Mesh + Pose on IR

| Component | Native IR Support | Status | Adaptation Needed |
|-----------|------------------|--------|------------------|
| Face Mesh (468 landmarks) | NO | RGB-trained | Fine-tune on IR data |
| Pose (33 keypoints) | NO | RGB-trained | Domain adaptation layer |
| Recommendation | - | - | Use as RGB path; custom IR network |

**Solution:** Dual-path system
- RGB: Standard MediaPipe
- IR: Custom lightweight pose network with domain adaptation

---

### 8. 6DRepNet Head Pose

| Metric | Value | Status | Notes |
|--------|-------|--------|-------|
| **Accuracy (AFLW2000)** | ~3-5° MAE | SOTA | 20% better than competitors |
| **Range** | 360° (v360) | Full | Handles extreme rotations |
| **Latency** | ~30-50ms | Typical ResNet50 | Fits 0.3s budget |
| **IR Performance** | Unknown | Gap | Needs fine-tuning |

**Action:** Evaluate pre-trained 6DRepNet360 on IR data; domain adapt if >5% drop

---

### 9. Attention Mechanisms for Gaze (2025 SOTA)

| Architecture | Paper | Key Strength | Best For |
|-------------|-------|-------------|----------|
| **GazeSymCAT** | JCDE 2025 | Extreme head poses | Robust to large rotations |
| **Multi-Head Attention** | Sensors 2025 | Cross-eye fusion | Bilateral consistency |
| **Vision Transformer** | Signal 2025 | Long-range context | Global feature capture |
| **MobGazeNet** | MVA 2025 | Mobile efficiency | Edge deployment |

**Recommendation:** GazeSymCAT for accuracy; MobGazeNet for latency

---

### 10. Regulatory Context & Timeline

| Event | Deadline | Requirement | Impact |
|-------|----------|-------------|--------|
| **EU General Safety Reg (GSR)** | July 2024 | DMS mandatory | Market demand spike |
| **Euro NCAP 2026** | 2026 | Driver engagement scoring | Certification requirement |
| **Additional GSR Phase** | July 2026 | Enhanced monitoring | Stricter specifications |

**Implication:** 2026 timeline aligns with regulatory push for robust DMS

---

## Implementation Roadmap Summary

### Phase 1: Foundation (Weeks 5-7)
- [ ] IR preprocessing with domain adaptation (WTEFNet or custom)
- [ ] YOLO-FaceV2 face detection
- [ ] Eye region extraction

### Phase 2: Estimation (Weeks 8-10)
- [ ] Gaze: GazeSymCAT transformer or attention CNN
- [ ] Pose: 6DRepNet360 (evaluate + fine-tune on IR)
- [ ] Action: CNN-BiLSTM for skeleton recognition
- [ ] Occlusion: Binary classifier for detection

### Phase 3: Fusion (Weeks 11-12)
- [ ] Confidence scoring (4 components per branch)
- [ ] Temporal filtering (Kalman)
- [ ] Dynamic weighting mechanism
- [ ] Degraded mode logic

### Phase 4: Integration & Testing (Weeks 13-15)
- [ ] Behavior classifier (attention, drowsy, distracted)
- [ ] Alert generation with confidence metadata
- [ ] Failure dataset evaluation
- [ ] Latency optimization

---

## Critical Paper References (2024-2026)

### Occlusion & Robust Detection
1. **Occlusion-aware DMS** - arXiv:2504.20677 (April 2025) - **Most relevant**
2. **Multi-Task Fusion** - Sensors 25(21):6799 (Nov 2025)
3. **Driver-Net** - IEEE IV 2025, pp.1841-1848

### Gaze & Head Pose
4. **GazeSymCAT** - JCDE 12(3):115 (2025)
5. **6DRepNet360** - IEEE TIP (2024)
6. **Appearance-based Gaze** - Sensors 2025

### Action Recognition
7. **CNN-BiLSTM-AM Distraction** - Complex & Intelligent Systems (2025)
8. **Two-Stream GCN-Transformer** - Scientific Reports (2025)

### Lightweight & Edge
9. **Tiny Deep Learning DMS** - Journal of Real-Time Image Processing (2025)
10. **Knowledge Distillation Survey** - Springer (2024-2025)

### Vision-Language & New Approaches
11. **Vision-Language Distracted Behavior** - arXiv (2024)
12. **Real-Time In-Cabin Behavior** - arXiv:2512.22298

---

## Key Metrics Target (Project Spec)

| Requirement | Target | Research Finding |
|-------------|--------|------------------|
| Accuracy | 95%+ | CNN-BiLSTM achieves 99.75% on State Farm |
| MAE (Gaze) | <10° | Achieved: 4-5° with modern methods |
| Latency | <0.3s | Budget: 140-200ms typical (achievable) |
| Recall (Detection) | >90% | Occlusion-aware methods support this |
| FP/NP Ratio | Minimize | Confidence-based fusion addresses this |

**Conclusion:** All metrics achievable with recommended techniques

---

## Critical Insights for Project Success

1. **Confidence is King:** Dynamic weighting beats fixed thresholds for occlusion robustness
2. **IR Domain Adaptation:** Not optional - RGB models degrade on IR without adaptation
3. **Multi-branch Diversity:** Each branch fails differently (gaze on glasses, pose on mask)
4. **Temporal Stability:** Kalman filtering + attention mechanisms essential for real-time
5. **Lightweight Critical:** <50ms per component for automotive SoC constraints
6. **Occlusion Awareness:** Explicit detection prevents dangerous misclassification
7. **Test on Failure Cases:** DMD+custom failure dataset more realistic than public benchmarks

