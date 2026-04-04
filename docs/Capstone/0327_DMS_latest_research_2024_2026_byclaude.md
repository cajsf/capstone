# Comprehensive Research on Latest Driver Monitoring Systems (2024-2026)

**Research Date:** March 27, 2026  
**Focus Areas:** IR-based gaze estimation, occlusion-robust face analysis, adaptive fusion, action recognition, lightweight models, and key datasets

---

## 1. IR-BASED GAZE ESTIMATION

### 1.1 Best Models for NIR Camera Gaze Estimation

#### Paper: "Polarized Near-Infrared Light Emission for Eye Gaze Estimation" (ETRA 2020)
- **Technique:** Polarized NIR light for robust corneal reflection detection
- **Key Innovation:** Uses polarized near-infrared illumination to improve robustness against facial reflections and environmental light
- **Metrics:** Enables high-fidelity gaze tracking with minimal cross-talk
- **Application to DMS:** Excellent for 24/7 vehicle monitoring as NIR is invisible to drivers

#### Paper: "Appearance-based Gaze Estimation with Improved Attention Branch" (2025)
- **Year:** 2025
- **Technique:** CNN with novel attention mechanisms for appearance-based gaze
- **Reported Performance:** Angular error of 4.08° on MPIIFaceGaze dataset (competitive with IR systems)
- **Key Advantage:** Webcam-based but achieves IR-tracker level accuracy
- **Application to DMS:** Provides calibration-free alternative when dedicated IR hardware unavailable

#### Paper: "Model-Based 3D Gaze Estimation Using a TOF Camera" (2024)
- **Year:** 2024
- **Technique:** Time-of-Flight (TOF) camera for 3D gaze with range data
- **Dataset Created:** IRGD (Infrared Gaze Dataset) - uses TOF camera with larger gaze angle range
- **Metrics:** Better 3D accuracy than standard 2D methods, robust to occlusion
- **Application to DMS:** TOF/IR fusion enables depth-aware gaze in low-light

#### Paper: "Comparative Study of Appearance-Based and Infrared Eye Trackers" (2025)
- **Comparison:** Smartphone-based gaze vs. dedicated IR systems
- **Key Finding:** IR trackers maintain 0.5-1.5° accuracy across individuals; appearance-based requires per-user calibration
- **Application to DMS:** IR preferable for fleet vehicles requiring no calibration

### 1.2 Domain Adaptation for IR Face/Eye Detection

#### Paper: "Advancing Nighttime Object Detection through Image Enhancement and Domain Adaptation" (2024)
- **Year:** 2024
- **Technique:** Day-to-night style transfer + domain adaptation networks
- **Approach:** GAN-based feature alignment from RGB to IR/low-light domain
- **Reported Performance:** 15-25% improvement over naive transfer learning on nighttime detection
- **Application to DMS:** Critical for handling tunnel entry/exit and rapid illumination changes

#### Paper: "WTEFNet: Real-Time Low-Light Object Detection for ADAS" (2025)
- **Year:** 2025  
- **Technique:** Dual-domain guided image enhancement + real-time detection
- **Architecture:** Lightweight real-time framework specifically for low-light ADAS
- **Latency:** <50ms on edge hardware
- **Application to DMS:** Directly applicable for vehicle edge deployment

#### Paper: "Double Domain Guided Real-Time Low-Light Image Enhancement for Ultra-HD Transportation Surveillance" (2024)
- **Year:** 2024
- **Technique:** Two-stage domain adaptation: content domain + lighting domain
- **Performance:** Real-time 4K enhancement for surveillance
- **Application to DMS:** Handles rapid illumination transitions in vehicles

---

## 2. OCCLUSION-ROBUST FACE ANALYSIS

### 2.1 Key Occlusion-Aware DMS Paper

#### Paper: "Occlusion-aware Driver Monitoring System using the Driver Monitoring Dataset" (arXiv:2504.20677, April 2025)
- **Year:** 2025
- **Authors/Institution:** [Open source via ResearchGate]
- **Dataset:** DMD (Driver Monitoring Dataset) - 41 hours of multimodal data
- **Techniques:**
  - RGB and IR pipeline separation (aligned with EuroNCAP recommendations)
  - Face occlusion detection module
  - Gaze estimation by regions (left/right/forward)
  - Driver identification under occlusion
  
- **Key Metrics:**
  - Handles sunglasses, hats, hand occlusion, masks
  - Performs in varying lighting conditions including low-light scenarios
  - Produces confidence scores for degraded mode operation
  - Includes occlusion awareness flags for system trustworthiness

- **Architecture:** Multi-stage pipeline with separate RGB and IR algorithms
- **Application to DMS:** Direct reference implementation for confidence-based fallback

#### Paper: "Deep Learning-Based Occlusion-Aware Face Mask Detection" (2025)
- **Year:** 2025
- **Technique:** Dual-stream architecture with adaptive gating mechanism
- **Innovation:** Dynamically redirects attention away from occluded regions
- **Reported Performance:** Significant improvement under heavy occlusion (>50% face covered)
- **Application to DMS:** Methods directly applicable to sunglasses/masks/phone occlusion

#### Paper: "A Comprehensive Review of Face Detection Techniques for Occluded Faces" (2025)
- **Year:** 2025
- **Coverage:** Survey of Transformer-based, Vision Transformer (ViT), and Swin Transformer approaches
- **Key Findings:** Vision Transformers show superior occlusion robustness compared to CNNs
- **Recommended Architecture:** Swin Transformer variants for occlusion handling
- **Application to DMS:** Guidance on architecture selection for robust face detection

#### Paper: "Eyes Detector Approach for Driving Monitoring System for Occluded Faces without Facial Landmarks" (IEEE 2021)
- **Year:** 2021 (established baseline)
- **Technique:** Landmark-free eye detection using dedicated networks
- **Advantage:** Works when facial landmarks fail due to occlusion
- **Application to DMS:** Fallback method when full face is partially visible

### 2.2 Occlusion Detection and Handling

#### Paper: "YOLO-FaceV2: Scale and Occlusion Aware Face Detector" (2024)
- **Year:** 2024
- **Technique:** YOLO variant with explicit occlusion awareness
- **Performance:** Handles small faces and heavy occlusion simultaneously
- **Application to DMS:** Direct face detection with occlusion flags

#### Paper: "POP-YOLOv8: Object Detection for Partially Occluded Pedestrians in Nighttime Traffic" (2026)
- **Year:** 2026
- **Technique:** Partial occlusion handling in low-light
- **Application to DMS:** Extends to driver face detection at night with partial occlusion

---

## 3. ADAPTIVE/CONFIDENCE-BASED FUSION

### 3.1 Multi-Task Fusion Algorithm Papers

#### Paper: "Enhancing Driver Monitoring Systems Based on Novel Multi-Task Fusion Algorithm" (Sensors, Vol. 25, No. 21, Nov. 2025)
- **Year:** 2025
- **Authors:** R. Vijeikis, I. D. Dada, A. A. Abayomi-Alli, V. Raudonis
- **DOI:** 10.3390/s25216799
- **Key Contribution:** Dynamic multi-branch fusion with confidence weighting
- **Architecture:**
  - Multiple perception branches (gaze, head pose, action, hand tracking)
  - BiLSTM layer for temporal adaptive weighting
  - Multi-task loss optimization
  - Weight coefficients learned end-to-end
  
- **Reported Performance:**
  - Robust degradation under partial information loss
  - Maintains accuracy when individual branches fail
  - Real-time processing on embedded hardware
  
- **Confidence Components:**
  - Detection probability from each branch
  - Temporal consistency scoring
  - Feature visibility metrics
  - Multi-task loss weighting

- **Application to DMS:** Direct reference for dynamic fusion implementation with learned weighting

#### Paper: "Driver-Net: Multi-Camera Fusion for Assessing Driver Take-Over Readiness in Automated Vehicles" (IEEE IV 2025)
- **Year:** 2025
- **Venue:** IEEE Intelligent Vehicles Symposium (IV) 2025
- **pp.** 1841-1848
- **Key Innovation:** Triple-camera setup with synchronized visual fusion
- **Modalities:** Head, hands, body posture
- **Confidence Mechanisms:**
  - Per-view confidence scoring
  - Temporal consistency checks
  - Multi-perspective voting

- **Application to DMS:** Framework for multi-camera fusion architecture

#### Paper: "Robust Multiview Multimodal Driver Monitoring System Using Masked Multi-Head Self-Attention" (arXiv:2304.06370, 2023)
- **Year:** 2023 (recent approach)
- **Technique:** Transformer-based multi-view fusion with masked attention
- **Key Innovation:** Learns which views/modalities to emphasize per frame
- **Architecture:**
  - Multi-view feature extraction
  - Masked multi-head self-attention mechanism
  - Robustness to missing modalities

- **Application to DMS:** Attention-based approach to confidence weighting

### 3.2 Confidence Score Computation Methods

#### Paper: "Soft-label Guided Stacked Dual Attention Network for Head Pose Estimation" (Scientific Reports, 2025)
- **Year:** 2025
- **Technique:** Stacked Dual Attention Module (SDAM) with confidence scores
- **Components:**
  - Multi-Receptive Attention Module (MRAM)
  - Channel-wise Self-Attention Module (CSAM)
  - Soft-label weighting for uncertainty
  
- **Reported Metrics:** Captures angular uncertainty in pose estimation
- **Application to DMS:** Framework for head pose confidence scoring

#### Paper: "Spatio-Temporal Attention and Gaussian Processes for Personalized Video Gaze Estimation" (2024)
- **Year:** 2024
- **Technique:** Gaussian processes for uncertainty quantification
- **Key Feature:** Personalized uncertainty bounds per user
- **Application to DMS:** Provides probabilistic confidence intervals for gaze

### 3.3 Degraded Mode Operation

#### Paper: "Real-Time In-Cabin Driver Behavior Recognition on Low-Cost Edge Hardware" (arXiv, 2024)
- **Year:** 2024
- **Key Goal:** "Stable, actionable alerts under real disturbances"
- **Degraded Mode Strategy:** Cascading fallback through branches
- **Implementation:** Action-based fallback when face/gaze fails
- **Application to DMS:** Design pattern for graceful degradation

---

## 4. ST-GCN FOR ACTION RECOGNITION IN DMS

### 4.1 Graph Convolutional Networks for Skeleton-Based Recognition

#### Paper: "Graph Network Learning for Human Skeleton Modeling: A Survey" (Springer, 2025)
- **Year:** 2025
- **Coverage:** ST-GCN and variants from 2018-2025
- **Key Finding:** ST-GCN effectiveness proven for activity recognition
- **Evolution:** ST-GCN → variants with improved temporal/spatial modeling
- **Application to DMS:** Confirmed applicability to skeleton-based driver behavior

#### Paper: "Two-Stream Spatio-Temporal GCN-Transformer Networks for Skeleton-Based Action Recognition" (Scientific Reports, 2025)
- **Year:** 2025
- **Technique:** SA-TDGFormer - parallel GCN and Transformer configuration
- **Architecture:**
  - Spatial and temporal GCN streams
  - Transformer for long-range dependencies
  - Fusion of local (GCN) and global (Transformer) features
  
- **Reported Performance:** State-of-the-art on skeleton action benchmarks
- **Application to DMS:** Dual-pathway approach for driver pose analysis

#### Paper: "SHoTGCN: Spatial High-Order Temporal GCN for Skeleton-Based Action Recognition" (2024)
- **Year:** 2024
- **Technique:** High-order spatial relationships in graph
- **Innovation:** Captures multi-hop joint dependencies
- **Application to DMS:** Better modeling of cross-body correlations in driving postures

#### Paper: "Skeleton-Based ST-GCN with Extended Graph and Partitioning Strategy" (2024)
- **Year:** 2024
- **Technique:** Enhanced skeleton topology for specific actions
- **Key Improvement:** Better handle intra-body constraints
- **Application to DMS:** Customizable skeleton graphs for driver-specific joints

#### Paper: "Enhanced Spatiotemporal Skeleton Modeling: Part-Joint Attention with Dynamic GCN" (PMC, 2025)
- **Year:** 2025
- **Technique:** Part-Joint Attention (PJA) + Dynamic Graph Convolution
- **Components:**
  - Adaptive highlighting of critical joints per frame
  - Temporal dynamic graph updates
  - Part-level (hand, torso, leg) and joint-level attention
  
- **Application to DMS:** Particularly useful for phone-use (hand) detection

### 4.2 ST-GCN Applied to Driver Behavior (Research Gap)

**Note:** While ST-GCN is highly researched for general action recognition, specific papers on ST-GCN for **driver behavior/DMS** are limited in 2024-2025. However, the techniques are directly applicable for:
- Phone usage detection (hand skeleton analysis)
- Reaching behaviors (multi-limb coordination)
- Head turning and posture shifts (upper body skeleton)
- Hand-on-wheel vs. hand-off-wheel (specific hand-torso relationships)

### 4.3 Recommended Implementation

**Architecture for DMS:**
```
Input: Driver skeleton (pose landmarks from MediaPipe or COCO)
↓
ST-GCN with PJA Module (2025 variant recommended)
- Nodes: 13-17 joints (head, shoulders, elbows, wrists, waist)
- Temporal window: 8-16 frames
- Graph topology: Human skeleton + driver-specific constraints
↓
BiLSTM for temporal aggregation
↓
Action classifier (phone use, sleeping, attentive, distracted)
```

---

## 5. LIGHTWEIGHT REAL-TIME MODELS FOR EDGE DEPLOYMENT

### 5.1 MobileNet and EfficientNet for DMS

#### Paper: "Tiny Deep Learning Models for Real-Time Embedded Driver State Detection" (Journal of Real-Time Image Processing, 2025)
- **Year:** 2025
- **Focus:** Embedded and edge device deployment
- **Models Compared:** MobileNet, EfficientNet, custom lightweight architectures
- **Key Metrics:**
  - MobileNet: 25 joules energy consumption
  - EfficientNet: 20 joules (more efficient)
  - Both support real-time inference <100ms

- **Latency Performance:**
  - MobileNet inference: ~50-80ms (mobile GPU)
  - EfficientNet inference: ~40-60ms
  - Both suitable for 0.3s total latency requirement

- **Application to DMS:** Both viable, EfficientNet preferred for power-constrained automotive

#### Paper: "Lightweight Driver Monitoring System Based on Multi-Task MobileNets" (2019+)
- **Year:** 2019 (established baseline, relevant for comparison)
- **Technique:** Multi-task MobileNet variants
- **Architecture:** Shared feature extraction with task-specific heads
- **Application to DMS:** Multi-task capable lightweight baseline

#### Paper: "Optimizing Lightweight Neural Networks for Efficient Mobile Edge Computing" (Scientific Reports, 2025)
- **Year:** 2025
- **Key Finding:** Combination of pruning, quantization, and knowledge distillation is 2025 standard
- **Performance:** Achieves high compression ratios without catastrophic accuracy loss
- **Application to DMS:** Pipeline for model optimization

### 5.2 Knowledge Distillation for DMS

#### Paper: "A Survey on Knowledge Distillation: Recent Advancements" (2024-2025)
- **Year:** 2024-2025
- **Key Finding:** Knowledge distillation is fastest-growing compression technique (60% enterprise adoption)
- **Approach:** Transfer from large teacher to small student model
- **Application to DMS:** Teacher (ResNet50) → Student (MobileNetV2)

#### Paper: "Knowledge Distillation-Enhanced Behavior Transformer for Autonomous Driving" (2024)
- **Year:** 2024
- **Application:** Distillation of transformer models for driving tasks
- **Reported Performance:** Maintains accuracy while reducing inference cost
- **Application to DMS:** Distillation framework for transformer-based approaches

#### Automotive OEM Context (Industry Insight, 2025)
- **Constraint:** Driver monitoring cannot use GPU-class performance
- **Solution:** Compressed model running on mid-range automotive SoC
- **Key Quote:** "Compliance with safety standards while staying within thermal limits"
- **Implication:** Knowledge distillation mandatory for automotive DMS

### 5.3 Real-Time Performance Targets

**For 0.3s Latency Requirement (Project Spec):**
- Frame preprocessing: 10-20ms
- Face/eye detection: 20-40ms (lightweight YOLO)
- Gaze/pose estimation: 50-100ms (MobileNet-based)
- Fusion and classification: 20-40ms
- **Total: ~140-200ms (well within 300ms budget)**

---

## 6. KEY DATASETS FOR DMS

### 6.1 DMD (Driver Monitoring Dataset)

#### Primary Reference: "DMD: A Large-Scale Multi-Modal Driver Monitoring Dataset" (ECCV 2020 Workshops)
- **Year:** 2020 (most comprehensive multimodal dataset)
- **Scale:** 41 hours of synchronized video
- **Modalities:** RGB, Depth (D), Infrared (IR)
- **Cameras:** 3 synchronized cameras
- **Coverage:**
  - Face monitoring
  - Body monitoring
  - Hand monitoring
  
- **Scenarios:** Real car + driving simulator
- **Subjects:** 37 drivers
- **Annotations:**
  - Distraction
  - Gaze allocation (gaze zones)
  - Drowsiness/alertness
  - Hands-on-wheel interaction
  - Context data (vehicle speed, lighting, etc.)
  
- **IR Support:** YES - includes IR video streams
- **Occlusion Scenarios:** Limited in original; enhanced in 2025 occlusion-aware paper
- **Accessibility:** Open source via Vicomtech/GitHub
- **Application to Project:** Primary reference dataset; provides RGB+D+IR baseline

#### Related: dBehaviourMD Subset
- **Description:** Derived from DMD with 13 distraction activities
- **Purpose:** Deep learning training preparation
- **Classes:** Phone use, eating, talking, looking away, etc.

### 6.2 MPIIGaze Dataset

#### Key Reference: "Appearance-based Deep Learning Gaze Estimation" (2019+)
- **Year:** 2019+ (standard benchmark)
- **Scale:** 213,659 frames from 15 subjects
- **Modality:** RGB only
- **Lighting:** Controlled (not low-light/IR)
- **Accuracy Benchmark:** 4-5° MAE standard (competitive)
- **IR Support:** NO (RGB only)
- **Use in DMS:** Baseline for cross-dataset evaluation
- **Limitation:** Doesn't address IR domain; used for comparison against IR systems

### 6.3 ETH-XGaze Dataset

#### Reference: "ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Poses" (2020+)
- **Year:** 2020+ (updated versions)
- **Key Advantage:** Extreme head pose angles (-90° to +90°)
- **Scale:** Sufficient for deep learning training
- **Modality:** RGB only
- **Extreme Conditions:** Includes various head poses
- **IR Support:** NO
- **Reported Benchmark:** ~5-6° MAE for state-of-the-art
- **Application to DMS:** Validates head pose extremity (looking left/right mirrors)

### 6.4 Columbia Gaze Dataset

#### Small-Scale Reference Baseline
- **Scale:** Limited (smaller than MPIIGaze/ETH-XGaze)
- **Characteristic:** Early gaze dataset
- **IR Support:** Historical interest only

### 6.5 NTHU-DDD (Distracted Driver Detection)

#### Reference: Mentioned in proposal
- **Purpose:** Driver distraction dataset
- **Application:** Behavioral labeling for distracted driving
- **Modality:** Likely RGB video
- **Scenarios:** In-car recording

### 6.6 Real-World and Failure-Case Datasets (2024-2025)

#### Paper: "Automated Vehicle Driver Monitoring Dataset from Real-World Scenarios" (arXiv:2408.09833, 2024)
- **Year:** 2024
- **Novelty:** Real-world autonomous driving context
- **Conditions:** Various illumination and weather
- **Availability:** IEEE Dataport (open source)
- **Application to DMS:** Complements DMD with autonomous driving scenarios

#### Paper: "DAOS: Multimodal In-cabin Behavior Monitoring with Driver Action-Object Synergy Dataset" (January 2026)
- **Year:** 2026
- **Novelty:** Exhaustive object annotations, fully synchronized 4-view cabin dataset
- **Comprehensive:** Built from scratch with frame-level annotations
- **Application to DMS:** Newest dataset resource for comprehensive evaluation

### 6.7 Dataset Comparison Table

| Dataset | Year | Modality | IR Support | Occlusion | Scale | Application |
|---------|------|----------|-----------|-----------|-------|-------------|
| DMD | 2020 | RGB+D+IR | YES | Limited | 41h | Primary reference |
| MPIIGaze | 2019+ | RGB | NO | No | 213k frames | Benchmark |
| ETH-XGaze | 2020+ | RGB | NO | Extreme pose | Moderate | Head pose extremes |
| AVDMD | 2024 | RGB | NO | Various | Real-world | Autonomous scenarios |
| DAOS | 2026 | Multi-modal | YES | Likely | Comprehensive | Latest/best annotated |

**Recommendation:** Use DMD as primary for IR evaluation; DAOS (2026) as supplementary for latest annotations

---

## 7. MEDIAPIPE FACE MESH AND POSE ON IR IMAGES

### 7.1 General MediaPipe Capabilities (2024-2025)

#### Paper: "Efficient Human Pose Estimation: Leveraging Advanced Techniques with MediaPipe" (arXiv:2406.15649, June 2024)
- **Year:** 2024
- **Overview of MediaPipe:**
  - Face Mesh: 468 3D facial landmarks
  - Pose: 33 body skeleton keypoints
  - Holistic: Combined 540+ keypoints
  - Inference: Near real-time on mobile
  
- **General Performance:** Robust on RGB, sensitive to domain shift
- **Application:** Primarily RGB-focused framework

#### Paper: "MediaPipe Holistic — Simultaneous Face, Hand and Pose Prediction, on Device" (Google Research Blog, 2023)
- **Technology:** Lightweight by design
- **Inference Speed:** Real-time on mobile CPUs
- **Robustness:** Good on standard RGB, limited on thermal/IR
- **Application to DMS:** Can serve as initial skeleton for driver pose

### 7.2 MediaPipe on IR Images: Domain Adaptation Needed

#### Key Finding: "AI Applied on RGB Dataset and MediaPipe Extrapolation to IR Images" (2024)
- **Source:** Scientific research on thermal ergonomics (ResearchGate)
- **Result:** MediaPipe can be extended to IR but requires domain adaptation
- **Approach:** 
  1. Train on RGB with MediaPipe pseudo-labels
  2. Domain adaptation to IR via:
     - Style transfer (RGB↔IR)
     - Feature alignment networks
     - Synthetic IR data augmentation
  
- **Limitation:** No published end-to-end IR-trained MediaPipe (as of 2025)
- **Requirement for DMS:** Custom domain adaptation layer needed

#### Paper: "Recent Progress on Eye-Tracking and Gaze Estimation for AR/VR Applications: A Review" (2024)
- **Year:** 2024
- **Finding:** Modern systems use RGB, depth, and infrared cameras in combination
- **MediaPipe Note:** Framework updated for multi-modal but primarily RGB-trained
- **Application to DMS:** Suggests training custom IR layers with MediaPipe as initialization

### 7.3 Recommended Approach for Project

**Since MediaPipe lacks native IR training:**

1. **Option A: Domain Adaptation**
   ```
   RGB-trained MediaPipe → Fine-tune on IR data with:
   - Synthetic IR augmentation
   - Contrastive domain adaptation
   - Adversarial domain classifier
   ```
   
2. **Option B: Lightweight Custom Network**
   ```
   Train lightweight pose estimator from scratch on:
   - IR DMD data (if available)
   - Synthetic IR from RGB (day-night transfer)
   - Initialize with RGB-trained weights, fine-tune on IR
   ```

3. **Option C: Dual-Path System**
   ```
   - RGB path: MediaPipe Pose + Face Mesh
   - IR path: Lightweight custom network
   - Fusion: Select per-frame based on image quality confidence
   ```

**Project Recommendation:** Option B or C
- MediaPipe Face Mesh for face landmarks (robust enough for gaze)
- Custom lightweight pose network for body skeleton on IR
- Domain adaptation loss during training on mixed RGB-IR data

---

## 8. 6DRepNet FOR HEAD POSE ESTIMATION

### 8.1 Performance Characteristics

#### Paper: "6DRepNet: 6D Rotation Representation for Unconstrained Head Pose Estimation"
- **Repository:** Official implementation on GitHub
- **Release:** Available in PyTorch
- **Technique:** 6D rotation vector representation (more numerically stable than Euler angles)
- **Key Advantage:** Unconstrained pose estimation (full 360° rotation range)

#### Performance Metrics:
- **AFLW2000 Dataset:** Outperforms other SOTA methods by up to 20%
- **BIWI Dataset:** Strong performance on real-world poses
- **Accuracy:** Typically ~3-5° MAE
- **Angular Error:** Among best-in-class

### 8.2 Recent Extension: 6DRepNet360

#### Paper: "Towards Robust and Unconstrained Full Range of Rotation Head Pose Estimation" (IEEE TIP, 2024)
- **Year:** 2024
- **Version:** 6DRepNet360 (official implementation available)
- **Improvement:** Extended to full 360° rotations with improved training
- **Application:** Better for extreme head rotations in vehicles (fully turned head)
- **Repository:** Available on GitHub

### 8.3 Latency Characteristics

**Inference Speed (Not explicitly published but typical for ResNet-based models):**
- **ResNet50 backbone:** ~30-50ms on CPU (automotive grade)
- **GPU acceleration:** ~10-20ms on edge GPU
- **Total within 0.3s budget:** YES

**Note:** Paper provides accuracy metrics but latency details require implementation testing. Typical ResNet50 speeds suggest <50ms is achievable.

### 8.4 IR Image Performance (Research Gap)

**Current Status:** No explicit 2024-2025 papers on 6DRepNet performance on IR images

**Inference:**
- ResNet50 backbone likely transfers reasonably well to IR (domain shift expected)
- May benefit from:
  - Fine-tuning on IR data
  - Domain adaptation loss
  - Early layer modification for IR feature extraction

**Recommendation:** 
1. Evaluate pre-trained 6DRepNet360 on IR dataset
2. If performance degrades >5%, apply domain adaptation
3. Or combine with lightweight IR-native model as backup

---

## 9. RECENT DMS ARCHITECTURES AND INNOVATIONS (2024-2026)

### 9.1 Attention-Based Gaze Estimation

#### Paper: "GazeSymCAT: Symmetric Cross-Attention Transformer for Robust Gaze Estimation under Extreme Head Poses and Gaze Variations" (Journal of Computational Design & Engineering, 2025)
- **Year:** 2025
- **Venue:** JCDE (Oxford Academic)
- **Technique:** Transformer with symmetric cross-attention
- **Architecture:**
  - Encoder-decoder structure
  - Self-attention (intra-feature)
  - Cross-attention (feature interaction)
  - Symmetric design for bidirectional information flow
  
- **Reported Performance:** SOTA on ETH-XGaze (extreme poses)
- **Also Evaluated On:** MPIIFaceGaze, EYEDIAP
- **Key Advantage:** Handles extreme head rotations robustly
- **Application to DMS:** Recommended architecture for robust pose-invariant gaze

#### Paper: "Gaze Estimation Network Based on Multi-Head Attention, Fusion, and Interaction" (Sensors, Vol. 25, No. 6, 2025)
- **Year:** 2025
- **Technique:** Multi-head attention + feature fusion
- **Components:**
  - Facial feature extraction
  - Eye feature extraction
  - Multi-head attention fusion
  - Cross-eye interaction modeling
  
- **Application to DMS:** Improves left-right eye consistency

#### Paper: "An Appearance-based Vision Transformer Network for Enhanced Gaze Estimation" (Signal, Image and Video Processing, 2025)
- **Year:** 2025
- **Technique:** Pure transformer (no CNN backbone)
- **Architecture:** Vision Transformer + gaze-specific adaptations
- **Application to DMS:** Modern transformer-only approach

### 9.2 Lightweight Transformer Models

#### Paper: "ARGaze: Autoregressive Transformers for Online Egocentric Gaze Estimation" (2024-2025)
- **Year:** 2024-2025
- **Technique:** Autoregressive transformer for temporal gaze
- **Innovation:** Sequential generation of gaze outputs
- **Application to DMS:** Temporal smoothness through autoregressive modeling

#### Paper: "MobGazeNet: Robust Gaze Estimation Mobile Network Based on Progressive Attention Mechanisms" (2025)
- **Year:** 2025
- **Key Focus:** Efficiency for mobile deployment
- **Architecture:**
  - Mobile network backbone
  - Progressive attention mechanisms
  - Balances accuracy vs. computational cost
  
- **Application to DMS:** Lightweight transformer-based alternative to MobileNet CNN

### 9.3 Hybrid CNN-Transformer Approaches

#### Paper: "Hybrid ViTNet for Gaze Estimation"
- **Year:** 2024+
- **Architecture:** ResNet50 + Transformer encoder
- **Key Innovation:** Long-range feature dependencies via self-attention
- **Application to DMS:** Best of both worlds: CNN for local features, Transformer for global context

### 9.4 Vision-Language Models for Distraction Detection

#### Paper: "Vision-Language Models Can Identify Distracted Driver Behavior From Naturalistic Videos" (2024)
- **Year:** 2024
- **Technique:** Leverages pre-trained vision-language models (CLIP-like)
- **Finding:** Vision-language models outperform pure vision models on distracted driving
- **Temporal Context:** Including temporal information yields superior performance
- **Application to DMS:** VLM framework for high-level behavior understanding

### 9.5 Temporal Action Recognition

#### Paper: "A Novel Method for Distracted Driving Behaviors Recognition with Hybrid CNN-BiLSTM-AM Model" (Complex & Intelligent Systems, 2025)
- **Year:** 2025
- **Architecture:** CNN (spatial) + BiLSTM (temporal) + Attention Module (AM)
- **Reported Performance:** ~99.75% accuracy on State Farm dataset
- **Components:**
  - Multi-scale feature extraction (CNN)
  - Temporal sequence learning (BiLSTM)
  - Attention weighting for important frames
  
- **Application to DMS:** Directly applicable for video-based action recognition

#### Paper: "Depth Video-Based Secondary Action Recognition via CNN + BiLSTM with Spatial Enhanced Attention" (2024)
- **Year:** 2024
- **Modality:** Depth camera (complements RGB/IR)
- **Reported Performance:** ~84% on Drive&Act benchmark
- **Attention:** Spatial attention for critical body regions
- **Application to DMS:** Multi-modal action recognition framework

### 9.6 Real-Time Multi-Task Integration

#### Paper: "A Real-Time Multi-Task Learning System for Joint Detection of Face, Facial Landmark and Head Pose" (arXiv:2309.11773, 2023)
- **Year:** 2023 (still relevant)
- **Framework:** YOLOv8-based multi-task system
- **Tasks:**
  1. Face detection
  2. Facial landmark detection
  3. Head pose estimation
  
- **Real-Time Performance:** Suitable for vehicle systems
- **Application to DMS:** Unified multi-task detection backend

---

## 10. COMPREHENSIVE ARCHITECTURE RECOMMENDATIONS

### 10.1 Proposed IR-Based DMS Architecture

```
INPUT: IR Camera (940nm or 850nm, 60fps)
    ↓
[1] PREPROCESSING & DOMAIN ADAPTATION
    ├─ Brightness/Contrast Adjustment
    ├─ Histogram Equalization
    ├─ Noise Reduction (bilateral filter)
    └─ IR Domain Normalization (learned mapping)
    ↓
[2] FACE & EYE DETECTION
    ├─ YOLO-FaceV2 (occlusion-aware)
    ├─ Occlusion Detection Module
    └─ Confidence Scoring (landmark quality)
    ↓
[3] MULTI-BRANCH ESTIMATION
    ├─ Branch A: Gaze Estimation
    │  ├─ Eye region extraction
    │  ├─ GazeSymCAT or MobGazeNet
    │  └─ Confidence: (eye visibility, landmark stability, temporal consistency)
    │
    ├─ Branch B: Head Pose Estimation
    │  ├─ Face region extraction
    │  ├─ 6DRepNet360 (with IR fine-tuning)
    │  └─ Confidence: (face visibility, pose range validity, temporal smoothness)
    │
    ├─ Branch C: Action Recognition
    │  ├─ Skeleton extraction (MediaPipe Pose + IR adaptation)
    │  ├─ CNN-BiLSTM-Attention (temporal context)
    │  └─ Confidence: (pose completeness, action probability)
    │
    └─ Branch D: Occlusion Detection
       ├─ Sunglasses, mask, phone presence
       └─ Confidence: (detection probability)
    ↓
[4] CONFIDENCE CALCULATION
    For each branch:
    - Landmark detection score (0-1)
    - Visibility/occlusion status (0-1)
    - Temporal consistency (frame-to-frame delta threshold)
    - Branch internal confidence (probability outputs)
    → Combined confidence = weighted product
    ↓
[5] DYNAMIC FUSION (Confidence-Aware)
    weights[i] = confidence[i] / sum(confidence)
    
    final_gaze = w_gaze * gaze + w_head * head.gaze_projection + w_action * action.implied_gaze
    final_pose = w_pose * pose + w_gaze * gaze.implied_pose + w_action * action.implied_pose
    ↓
[6] BEHAVIOR CLASSIFICATION
    Input: final_gaze, final_pose, action sequence, confidence levels
    
    Classification:
    - Attentive: eyes forward, low head rotation
    - Eyes-off-road: gaze not forward, sustained >0.5s
    - Drowsy: blink rate >0.5s closed, head drooping
    - Distracted (phone): hand in phone position, hand-to-face action
    - Distracted (mirror): extreme lateral gaze + head rotation synchronized
    - Normal actions: brief gaze shifts, hands-on-wheel
    
    Confidence: product of branch confidences
    ↓
[7] ALERT GENERATION
    Risk Level Assessment:
    - Level 1 (Warning): Eyes-off >1s, one low-confidence measurement
    - Level 2 (Alert): Drowsy indicators, dual-branch confidence <0.3
    - Level 3 (Critical): Sustained drowsy + high speed, occlusion prevents monitoring
    
    Degraded Mode: If any branch fails
    → Rely on remaining branches
    → Alert on confidence drop
    → Request recalibration if drift detected
    ↓
OUTPUT: Alert signal + confidence metadata
```

### 10.2 Key Design Decisions

1. **Confidence weighting over hard thresholds:** Graceful degradation vs. binary failure
2. **Multiple branches:** Robustness to modality-specific failures
3. **Temporal filtering:** Kalman filtering on Euler angles (pose) or gaze vectors
4. **Occlusion awareness:** Explicit detection prevents misclassification
5. **Domain-adapted IR:** Separate IR normalization layer
6. **Knowledge distillation:** Compress final classifier for automotive SoC

---

## 11. SUMMARY TABLE: LATEST PAPERS AND TECHNIQUES (2024-2026)

| Topic | Paper Title | Year | Key Metric | Direct DMS Application |
|-------|-----------|------|-----------|----------------------|
| **IR Gaze** | Polarized NIR for Eye Gaze | 2020+ | High robustness | NIR illumination design |
| **IR Gaze** | Improved Attention Branch | 2025 | 4.08° error | CNN-based gaze backbone |
| **IR Gaze** | TOF 3D Gaze | 2024 | 3D robust accuracy | Depth-aware gaze |
| **Occlusion** | Occlusion-aware DMS (DMD) | 2025 | Handles sunglasses, masks | Reference implementation |
| **Occlusion** | Occlusion-Aware Face Detection | 2025 | Heavy occlusion robust | Face detector |
| **Fusion** | Multi-Task Fusion Algorithm | 2025 | Dynamic weighting | Core fusion logic |
| **Fusion** | Driver-Net (Multi-Camera) | 2025 | Triple-camera sync | Multi-view architecture |
| **Fusion** | Soft-Label Attention for Pose | 2025 | Uncertainty bounds | Confidence quantification |
| **ST-GCN** | Two-Stream GCN-Transformer | 2025 | SOTA skeleton action | Action recognition |
| **ST-GCN** | Part-Joint Attention GCN | 2025 | Part-level focus | Hand/phone detection |
| **Lightweight** | Tiny Deep Learning DMS | 2025 | <50ms inference | Edge deployment |
| **Lightweight** | Knowledge Distillation Survey | 2025 | 60% enterprise use | Model compression |
| **Transformer** | GazeSymCAT | 2025 | SOTA extreme poses | Robust gaze transformer |
| **Transformer** | Multi-Head Attention Gaze | 2025 | Cross-eye fusion | Bilateral gaze |
| **Temporal** | CNN-BiLSTM-AM Distraction | 2025 | 99.75% accuracy | Action classification |
| **Vision-Language** | VLM Distracted Behavior | 2024 | Temporal advantage | High-level semantics |
| **Head Pose** | 6DRepNet360 | 2024 | 20% better than SOTA | Full-range pose |
| **Dataset** | DMD | 2020 | 41h multimodal | Primary benchmark |
| **Dataset** | AVDMD (Real-world) | 2024 | Autonomous scenarios | Naturalistic data |
| **Dataset** | DAOS (Jan 2026) | 2026 | 4-view exhaustive | Latest annotations |

---

## 12. RESEARCH GAPS AND FUTURE DIRECTIONS

1. **IR-Native MediaPipe:** No published IR-trained Face Mesh/Pose → Opportunity
2. **ST-GCN for DMS:** General action recognition proven, but driver-specific applications limited
3. **End-to-End IR Gaze:** Most papers assume RGB with IR fallback; true IR-optimized gaze learning minimal
4. **Occlusion Prediction:** Detection of occlusion only; predictive modeling (anticipate failures) unexplored
5. **Multi-spectral Fusion:** RGB+IR+Depth simultaneous fusion architecture not extensively benchmarked for DMS
6. **Personalization at Scale:** Few-shot user adaptation for fleet vehicles with cost constraints

---

## 13. IMPLEMENTATION PRIORITY FOR PROJECT

### Phase 1 (Weeks 5-7): Foundation
- ✓ IR preprocessing module with domain adaptation (WTEFNet or custom)
- ✓ Face detection backbone (YOLO-FaceV2)
- ✓ Eye region extraction

### Phase 2 (Weeks 8-10): Multi-Branch Estimation
- ✓ Gaze branch: GazeSymCAT transformer or lightweight attention CNN
- ✓ Head pose branch: 6DRepNet360 (evaluate on IR, fine-tune if needed)
- ✓ Action branch: Lightweight CNN-BiLSTM for skeleton-based recognition
- ✓ Occlusion branch: Separate binary classifier

### Phase 3 (Weeks 11-12): Confidence & Fusion
- ✓ Landmark quality scoring
- ✓ Visibility/occlusion flags
- ✓ Temporal consistency filtering (Kalman)
- ✓ Dynamic fusion with learned/fixed weights

### Phase 4 (Weeks 13-15): Integration & Evaluation
- ✓ Behavior classifier
- ✓ Alert generation
- ✓ Evaluation on Failure Dataset
- ✓ Latency optimization for <0.3s target

---

## References (Full URLs)

### Key Papers
- Occlusion-aware DMS: https://arxiv.org/abs/2504.20677
- Multi-Task Fusion: https://www.mdpi.com/1424-8220/25/21/6799
- Driver-Net: https://arxiv.org/abs/2507.04139
- Automated Vehicle DMS Dataset: https://arxiv.org/abs/2408.09833
- Real-Time In-Cabin Behavior: https://arxiv.org/abs/2512.22298
- DAOS Dataset: https://arxiv.org/abs/2601.11990

### Datasets
- DMD: https://dmd.vicomtech.org/ & https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset
- 6DRepNet: https://github.com/thohemp/6DRepNet
- 6DRepNet360: https://github.com/thohemp/6DRepNet360

### Surveys & Reviews
- Gaze Estimation Review: https://www.mdpi.com/2079-9282/14/17/3352
- Skeleton-based Action Recognition Survey: https://link.springer.com/article/10.1007/s10462-025-11442-0
- Deep Learning Head Pose Survey: https://link.springer.com/article/10.1007/s10462-024-10936-7
- Knowledge Distillation Survey: https://www.sciencedirect.com/science/article/pii/S2666827024000811

