# Comprehensive DMS Research: Datasets, Evaluation Metrics, Data Augmentation, and Annotation Strategies

## 1. PUBLIC DMS DATASETS COMPARISON

### 1.1 DMD (Driver Monitoring Dataset)
**Source:** https://dmd.vicomtech.org/, https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset

**Size:**
- 41 hours of RGB, depth, and IR videos
- ~26 TB raw data (ROS bags), compressed to mp4 with H.264 (target bitrate 15000 kb/s)
- 37 drivers
- Multi-camera setup: face, body, hands (3 synchronized cameras)

**Modalities:** RGB, Infrared (IR), Depth

**Annotations:**
- VCD (Video Content Description) format
- Labels for distraction, fatigue, gaze-head pose
- Occlusion-aware annotations (occlusion detection under varying lighting)
- Gaze estimation by regions

**IR Availability:** Yes, full IR stream available

**Occlusion Coverage:** Explicit occlusion detection and handling

**Key Features:**
- Largest visual dataset for real driving actions
- Multi-stream synchronized recording
- Real car and driving simulator scenarios
- Most comprehensive IR coverage for DMS tasks

---

### 1.2 State Farm Distracted Driver Dataset
**Source:** https://www.kaggle.com/c/state-farm-distracted-driver-detection

**Size:**
- 22,424 training images (2D dashboard camera snapshots)
- 79,700+ unlabeled test images
- Equal distribution among classes

**Subjects/Hours:** Not explicitly specified (snapshot-based, not video)

**Annotation Types:**
- 10 classes: safe driving, texting (left/right), calling (left/right), radio, reaching behind, hair/makeup, drinking, talking to passenger

**Modalities:** RGB only

**IR Availability:** No

**Occlusion Coverage:** Limited (no explicit occlusion labels)

**Key Features:**
- Most widely used distraction dataset
- Simple class-based labeling
- No temporal information (frame-based)
- Good for classification tasks but limited for temporal analysis

---

### 1.3 NTHU-DDD (National Tsing Hua University Drowsy Driver Detection)
**Source:** http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/

**Size:**
- 36 subjects (various ethnicities)
- 9.5-10.5 hours of annotated footage
- Resolution: 640x480 (AVI format)

**Frame Rates:**
- Night scenarios: 15 fps
- Day scenarios: 30 fps

**Annotations:**
- Per-frame multi-modal annotations: stillness/drowsiness, head state, mouth state, eye state
- Drowsiness: 0 (still) / 1 (drowsy)
- Head: 0 (still) / 1 (nodding) / 2 (looking aside)
- Mouth: 0 (still) / 1 (yawning) / 2 (talking/laughing)
- Eyes: 0 (still) / 1 (sleepy)

**Modalities:** RGB only

**IR Availability:** No

**Occlusion Coverage:** Limited (glasses/sunglasses variations, no explicit occlusion labels)

**Key Features:**
- Day and night illumination conditions
- Dedicated drowsiness annotation
- Multiple behavioral indicators
- 4 annotated labels per frame

---

### 1.4 DAD (Driver Anomaly Detection Dataset)
**Source:** https://www.ce.cit.tum.de/mmk/dad/

**Size:**
- ~95 GB total (17% is test set)
- Multi-view: front and top views
- Two Infineon CamBoard pico flexx cameras

**Resolution:** 224 × 171 pixels, 45 fps

**Annotations:**
- Open/closed set annotations
- Manual multiclass annotations for test set
- Multi-modal structure for varying lighting

**Modalities:** 
- Infrared (IR) and Depth (synchronized)
- Multi-view (front + top)

**IR Availability:** Yes, primary modality with depth

**Occlusion Coverage:** Multi-view setup addresses occlusion from different angles

**Key Features:**
- Designed for operation in varying lighting conditions
- Multi-modal IR + depth combination
- Multi-view perspective (addresses occlusion via different viewpoints)
- Contrastive learning approach

---

### 1.5 DriveAHead (Driver Head Pose Dataset)
**Source:** https://cvhci.iar.kit.edu/1656.php

**Size:**
- 1,000,000 depth and infrared images
- 20 subjects
- Largest driver head pose dataset

**Annotations:**
- Frame-by-frame head position and orientation
- Yaw, pitch, roll angles
- Motion-capture system ground truth
- Occlusion annotations (face occlusion flags)

**Modalities:** Depth and Infrared (IR)

**IR Availability:** Yes, full IR stream with Kinect v2

**Occlusion Coverage:** Yes, frame-level occlusion labels

**Annotation Types:**
- Head pose (3D): yaw, pitch, roll
- Occlusion flags
- Motion-capture aligned at pixel level

**Key Features:**
- Largest IR-based head pose dataset
- Aligned RGB-IR-Depth at pixel level
- Motion-capture ground truth
- Divided into train/val/test sets

---

### 1.6 DGAZE (Driver Gaze Mapping Dataset)
**Source:** https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2020/DGAZE_Driver.pdf

**Size:**
- 20 drivers
- ~18-minute video per driver
- ~100,000 sample frames

**Participants:** 20 drivers (20-30 age group, male/female, some wearing glasses)

**Collection Setup:**
- Lab setting matching real driving conditions
- Mobile phone camera mounted at dashboard-like position
- Drivers asked to gaze at marked points
- No expensive eye-tracking hardware needed

**Annotations:**
- Gaze position targets (marked points)
- No complex landmark or head pose annotations
- Focus: gaze direction only

**Modalities:** RGB only (mobile camera)

**IR Availability:** No

**Key Features:**
- Hardware-free testing
- Controlled lab gaze collection
- 100K frames total dataset
- Simple gaze point annotation

---

### 1.7 3MDAD (Multiview Multimodal Multispectral Driver Action Dataset)
**Source:** https://www.researchgate.net/publication/343524896_A_novel_public_dataset_for_multimodal_multiview_and_multispectral_driver_distraction_analysis_3MDAD

**Size:**
- 69 participants total:
  - Daytime: 50 subjects (38M, 12F), aged 19-41
  - Nighttime: 19 subjects (11M, 8F), aged 19-53
- Two synchronized Kinect cameras (top + dashboard)

**Modalities:**
- Daytime: RGB + Depth (temporally synchronized)
- Nighttime: Infrared (IR) + Depth (temporally synchronized)

**IR Availability:** Yes (nighttime scenarios only)

**Annotations:**
- 14 activity classes: safe driving, hair/makeup, radio, GPS, texting, phone call, pictures, talking, singing/dancing, fatigue, somnolence, drinking, reaching, smoking

**Occlusion Coverage:** Multi-view design (addresses occlusion from two angles)

**Data Access:** Publicly available for non-commercial use

**Key Features:**
- Daytime/nighttime paired recordings
- Multi-view reduces single-view occlusion
- Comprehensive activity classification
- Large participant pool

---

### 1.8 MPIIGaze and MPIIFaceGaze
**Source:** https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/

**Size (MPIIGaze):**
- 213,659 full face images
- 15 subjects
- Collected over several months during everyday laptop use
- 37,667 images manually annotated

**Annotations:**
- Eye corners, mouth corners, pupil centers (manually annotated)
- 3D gaze direction
- 3D head pose
- (x,y) positions of 6 facial landmarks
- Pupil center positions

**MPIIFaceGaze:** Extended version with full facial landmark annotations

**Modalities:** RGB only

**IR Availability:** No

**Gaze Coverage:** Comprehensive (multiple gaze directions, varied head poses)

**Key Features:**
- In-the-wild collection (realistic variation)
- Experience sampling approach
- Extensive gaze direction coverage
- Benchmark for appearance-based gaze estimation

---

### 1.9 Columbia Gaze Dataset
**Source:** https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/

**Size:**
- 5,880 high-resolution images (5184 × 3456 pixels)
- 56 subjects (32M, 24F)
- Ages 18-36, 21 wearing glasses

**Configuration:**
- 5 horizontal head poses (0°, ±15°, ±30°)
- 21 gaze directions per head pose
- 7 horizontal gaze directions
- 3 vertical gaze angles

**Collection Setup:**
- Controlled: black background, grid dots at 5°/10° increments
- Camera distance: 2 meters
- Chin rest for head stability
- Camera at eye height

**Annotations:**
- Head pose (yaw, pitch, roll)
- Gaze direction (2D/3D)
- High accuracy ground truth

**Modalities:** RGB (high-resolution studio images)

**IR Availability:** No

**Key Features:**
- Controlled, high-quality acquisition
- Comprehensive pose-gaze combinations
- High resolution
- Excellent for gaze estimation benchmarking

---

## 2. EVALUATION METRICS FOR DMS

### 2.1 Standard Classification Metrics

**Accuracy:**
- Definition: Proportion of correct predictions among all predictions
- Formula: (TP + TN) / (TP + TN + FP + FN)
- Use: Overall model performance on balanced datasets

**Precision:**
- Definition: Proportion of true positive predictions among all positive predictions
- Formula: TP / (TP + FP)
- Use: Minimize false alarms in real distraction scenarios

**Recall (Sensitivity):**
- Definition: Proportion of true positives found among all actual positives
- Formula: TP / (TP + FN)
- Use: Ensure abnormal behaviors are not missed

**F1 Score:**
- Definition: Harmonic mean of Precision and Recall
- Formula: 2 × (Precision × Recall) / (Precision + Recall)
- Use: Balance precision and recall for imbalanced datasets (critical for DMS)

---

### 2.2 DMS-Specific Metrics

**PERCLOS (Percentage of Eyelid Closure):**
- Definition: Percentage of time in a minute that eyes are ≥80% closed
- Established benchmark: 1994 driving simulator study
- Finding: Most reliable measure of driver alertness level
- Typical threshold: PERCLOS > 15-20% indicates drowsiness

**Mean Angular Error (MAE) - Gaze Estimation:**
- Definition: Angular difference between predicted gaze vector and ground truth
- Units: Degrees
- Typical threshold for driver monitoring: <10° preferred
- Can be computed per axis (yaw, pitch, roll) or as single rotation angle

**Gaze Zone Classification Accuracy:**
- Definition: Accuracy of classifying driver gaze into defined zones (windshield, mirrors, dashboard, etc.)
- Typical zones: 9-12 regions in vehicle
- SOTA: YOLOv8 achieves near-perfect accuracy on well-labeled datasets

**Detection Latency:**
- Definition: Time from video frame capture to alert generation
- DMS requirement: <0.3 seconds (mentioned in capstone proposal)
- Critical for real-time warning systems

**False Positive Rate (FPR):**
- Definition: Proportion of non-abnormal situations incorrectly flagged as abnormal
- Critical in DMS: Normal behaviors (mirror check, dashboard viewing) misclassified as distraction
- Literature range: 0.2-0.4 (20-40%) in challenging conditions

---

### 2.3 SOTA Accuracy Benchmarks from Literature

**Driver Distraction Detection:**
- D-HCNN model: 95.59% (AUCD2 dataset), 99.87% (SFD3 dataset)
- Eye + mouth fusion: 94% accuracy with 0.86 kappa
- HOG features framework: 85.62% average accuracy
- Spatio-temporal representation learning: 76.2% accuracy

**Gaze Estimation (MAE in degrees):**
- GazeCapsNet: 4.06° MAE on MPIIFaceGaze
- SOTA models: ~3.96-4.08° on MPIIGaze/MPIIFaceGaze
- ETH-XGaze (challenging conditions): ~6-8° MAE

**Drowsiness Detection:**
- Various systems report 85-95% accuracy depending on conditions
- Performance degrades significantly in low-light scenarios

**Real-World False Positive Rates:**
- TPR: 0.609, FPR: 0.218, Precision: 0.325 (one framework on dashboard camera)
- Challenge: High false alarm rate due to environmental factors (lighting, vibration)

---

## 3. DATA AUGMENTATION FOR IR DOMAIN

### 3.1 Synthetic Occlusion Generation

**Random Erasing:**
- Randomly selects rectangular region and erases with random pixel values
- Training creates images with various occlusion levels
- Reduces overfitting and improves robustness
- More natural-looking occlusions than simple masking

**Cutout:**
- Randomly erases square regions in images
- Simulates partial object visibility
- Simple but effective for generalization

**Object Overlay:**
- Overlay synthetic objects (sunglasses, masks, hands) onto face regions
- Domain-specific approach for driver monitoring
- Can simulate realistic occlusion patterns

---

### 3.2 Brightness/Contrast Augmentation for Lighting Variations

**Target:** Simulate lighting condition changes (day/night transitions, tunnel entry/exit)

**Techniques:**
- Brightness adjustment (±20-40%)
- Contrast enhancement
- Gamma correction
- Adaptive histogram equalization

**IR-Specific Considerations:**
- IR images have different brightness distribution than RGB
- Focus on preserving texture details while adjusting overall intensity
- Avoid over-saturation that loses landmark visibility

---

### 3.3 IR-Specific Augmentation Techniques

**Thermal Image Colorization (GAN-based):**
- PearlGAN: Top-down guided attention + gradient alignment
- Colorizes nighttime thermal IR to daytime color
- Improves compatibility with RGB-trained models
- Facilitates transfer learning from RGB datasets

**Domain Adaptation via Style Transfer:**
- Multi-style transfer for feature-level adaptation
- Transfers curvatures and edges from source (RGB) to target (IR)
- Trains detection models on multi-style transferred images
- Improves robustness to IR-specific domain shift

**Contrast Enhancement Modules:**
- Denoising + edge restoration
- TE-GAN architecture: contrast enhancement + denoising
- Improves overall IR image quality

**Diffusion-Based Generation:**
- Emerging approach: diffusion models for synthetic IR image generation
- Can generate diverse IR variations from RGB inputs

---

### 3.4 GAN-Based IR Image Generation from RGB

**Key Approaches:**

1. **Pix2Pix (Conditional GAN):**
   - Maps RGB images to corresponding IR domain
   - Requires paired RGB-IR training data
   - Good for learning cross-modal translation

2. **CycleGAN (Unpaired):**
   - Enables translation without paired RGB-IR data
   - Useful when only separate RGB and IR datasets exist
   - Maintains semantic content across domains

3. **PearlGAN (Specialized for Driving):**
   - Specifically designed for nighttime thermal IR to daytime RGB
   - Uses top-down attention + gradient alignment
   - Reduces semantic encoding ambiguity

**Benefits:**
- Augments IR training data without additional capture
- Leverages larger RGB datasets
- Improves generalization to diverse lighting conditions

---

## 4. ANNOTATION TOOLS AND STRATEGIES FOR CUSTOM DMS DATASET

### 4.1 Recommended Annotation Tools

**CVAT (Computer Vision Annotation Tool)**
- **Type:** Open-source web-based platform
- **Video Support:** Full frame-by-frame annotation with keyframe interpolation
- **Best For:** Large-scale collaborative projects, high-volume labeling
- **Features:**
  - Team collaboration with role-based access control
  - Auto-interpolation between keyframes (time-saving)
  - Integration with pre-trained models for semi-automatic annotation
  - Keyboard shortcuts and efficient workflow
- **URL:** https://www.cvat.ai/, https://github.com/cvat-ai/cvat

**Label Studio**
- **Type:** Open-source image/video annotation tool
- **Video Support:** With plugins
- **Best For:** Flexibility, custom annotation schemas
- **Features:**
  - Multiple annotation task types
  - Model-assisted labeling (pre-labeling with predictions)
  - Reduces annotation time by ~60% vs manual
  - Active learning integration

**VIA (VGG Image Annotator)**
- **Type:** Lightweight, in-browser tool from Oxford's Visual Geometry Group
- **Video Support:** Limited (single images preferred)
- **Best For:** Quick, single-user annotation projects
- **Limitation:** No native video support, better for frame extraction

---

### 4.2 Annotation Schema for DMS Custom Dataset

**Gaze Zone Annotations:**
- Windshield (forward)
- Left side mirror
- Right side mirror
- Rear-view mirror
- Center console/navigation
- Speedometer/instrument cluster
- Handbrake/gear
- Left window
- Right window
- Out-of-frame/undefined

**Head Pose Annotations:**
- Yaw angle (±degrees)
- Pitch angle (±degrees)
- Roll angle (±degrees)
- Or directional classes: forward, left, right, down, up

**Occlusion Labels:**
- Visible: fully unobstructed
- Partially visible: <50% obscured
- Mostly visible: 50-80% obscured
- Mostly hidden: >80% obscured
- Object type: sunglasses, mask, hand, phone, hat, other

**Activity Classification:**
- Safe driving (normal forward viewing)
- Side mirror check
- Rear-view mirror check
- Speedometer check
- Phone use (handheld)
- Talking (passenger or phone)
- Drowsy (eyelid drooping)
- Other (specify)

**Temporal Consistency Flags:**
- Frame-level confidence score (0-1)
- Temporal smoothness flag (detects sudden jumps)

---

### 4.3 Semi-Automatic Annotation Using Pretrained Models

**Efficiency Gains:**
- 10× faster annotation with integrated AI models
- 60% reduction in annotation time (verification vs creation)
- Human-in-the-loop: AI pre-labels, human verifies and refines

**Workflow:**
1. Run pretrained face detector → bounding boxes
2. Run gaze estimator → gaze zones
3. Run pose estimator → head angles
4. Human annotators verify and correct
5. Feedback loop: retrain on corrected annotations

**Tools:**
- CVAT: Built-in model integration
- Label Studio: Model-assisted labeling
- Custom pipeline: Use off-the-shelf models (MediaPipe, OpenFace, etc.)

**Key Pretrained Models for DMS:**
- MediaPipe Face Mesh: Face detection + 468 landmarks
- Face Alignment Network (FAN): Facial landmarks
- 6D Head Pose Estimator: Head pose
- ETH-XGaze / MPIIGaze: Gaze direction

---

### 4.4 Dataset Size Requirements

**Recommended Video Annotation Scale:**
- **Minimum:** 6-12 hours of labeled footage for initial model training
- **Comprehensive:** 15-30 hours for robust system
- **Production-ready:** 50+ hours across diverse conditions

**Frame Considerations:**
- 60-second clip at 30 fps = 1,800 frames
- 10-minute video at 30 fps = 18,000 frames
- Smart frame sampling: extract 1-2 fps for general tasks (avoids data bloat)
- Tracking tasks: require higher fps

**Annotation Time Estimates (per 1 hour of video):**
- Manual full annotation: 40-60 hours (42x slowdown)
- Keyframe + interpolation: 5-10 hours (5-10x slowdown)
- Semi-automatic (AI pre-label + verify): 2-4 hours (2-4x slowdown)

**Practical Recommendation for Capstone:**
- Collect 15-20 hours custom failure data
- Annotate 5-10 hours in detail (critical failure scenarios)
- Use semi-automatic annotation to reduce burden
- Divide work: 5 team members × 2-3 hours each = feasible

---

## 5. BENCHMARK COMPARISON: SOTA DMS ACCURACY

### 5.1 Published SOTA Results

**Driver Distraction Classification:**
| Dataset | Model | Accuracy | Notes |
|---------|-------|----------|-------|
| AUCD2 | D-HCNN | 95.59% | Deep hybrid CNN |
| SFD3 | D-HCNN | 99.87% | Simplified setup |
| General fusion | Eye+Mouth | 94.00% | Multi-modal fusion |
| HOG-based | HOG+SVM | 85.62% | Traditional approach |
| Spatio-temporal | LSTM-CNN | 76.20% | Low-light conditions |

**Gaze Estimation (Mean Angular Error):**
| Dataset | Method | MAE (degrees) | Conditions |
|---------|--------|--------------|-----------|
| MPIIFaceGaze | GazeCapsNet | 4.06° | Controlled lighting |
| MPIIGaze | SOTA | 3.96-4.08° | In-the-wild |
| ETH-XGaze | Various | 6-8° | Extreme poses |
| Columbia | Various | 2-4° | Studio controlled |

**Drowsiness Detection:**
- Literature range: 85-95% accuracy (varies by conditions)
- Degrades significantly in low-light without IR

---

### 5.2 Real-World Performance Challenges

**False Positive Rate Issues:**
- Environmental factors: lighting changes, vehicle vibration
- High FPR typical: 20-40% (0.2-0.4)
- Example: TPR=0.609, FPR=0.218, Precision=0.325 on dashboard footage

**Capstone Project Target (from proposal):**
- Accuracy: ≥95% (integrated test set)
- MAE: ≤10° (gaze estimation)
- Latency: <0.3 seconds
- Recall: ≥90% (abnormal behavior detection)
- FP/NP ratio: Minimize false alarms in normal driving

---

### 5.3 Critical Research Gaps Your Project Can Address

1. **IR-domain performance:** Most SOTA papers use RGB; IR performance is under-explored
2. **Occlusion robustness:** Limited public datasets with explicit occlusion labels
3. **Degraded mode operation:** Few systems tested with partial information loss
4. **Temporal fusion:** Multi-branch confidence-aware fusion is emerging but not yet standard
5. **False positive reduction:** Most systems optimize accuracy, not real-world usability

---

## Summary: Key Takeaways for Your Capstone

### Best Datasets to Use
1. **DMD:** Most comprehensive IR data, best occlusion annotations → primary benchmark
2. **DAD:** Multi-modal IR+depth, multi-view design → validation/comparison
3. **DriveAHead:** Largest IR dataset, 1M images → pre-training
4. **3MDAD:** Day/night paired, multi-view → augment with failure scenarios
5. **Custom failure dataset:** Your key differentiator → addresses real-world gaps

### Annotation Strategy
- Use CVAT with semi-automatic pre-labeling (60% time savings)
- Define 10 gaze zones + occlusion states + head pose angles
- Collect 15-20 hours custom data, annotate 5-10 hours in depth
- Allocate 2-4 hours annotation per 1 hour video (with semi-automation)

### Augmentation Focus
- **Synthetic occlusions:** Random erasing + object overlays (sunglasses, masks, hands)
- **Lighting simulation:** Brightness/contrast for tunnel/transition scenarios
- **GAN-based enhancement:** Consider PearlGAN for IR→RGB translation if needed
- **Domain adaptation:** Style transfer for IR features

### Evaluation Strategy
- Primary: Accuracy ≥95% on integrated test set
- Secondary: MAE ≤10° (gaze), PERCLOS detection, F1 for rare events
- Differentiator: FPR measurement on normal driving (side mirror, dashboard checks)
- Temporal consistency: Measure frame-to-frame stability

### Realistic Benchmarks
- Expect 3-5% gap between controlled datasets and real driving
- IR performance typically 5-10% lower than RGB on same tasks
- Multi-branch fusion (your approach) can recover 2-4% accuracy
- Occlusion handling adds 3-5% robustness improvement

