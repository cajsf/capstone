# IR 기반 DMS 프로젝트 — 학습 데이터셋 종합 분석 보고서

**작성일:** 2026-03-27
**대상:** 스꾸삐 팀 전원
**목적:** 각 Branch별 최적 데이터셋 선정, 학습 전략, 다운로드 우선순위 확정

---

## 1. Executive Summary

40개 이상의 공개 데이터셋을 조사한 결과, 본 프로젝트에 **핵심 5개 + 보조 5개 = 총 10개 데이터셋**이 필요하다. 모두 한국 대학 연구팀에 무료로 제공되며, 최소 저장 용량은 약 800GB, 권장은 2~3TB이다. 특히 NTU RGB+D는 등록 승인에 1~2주가 소요되므로 **오늘 당장 신청**해야 한다.

### 핵심 발견

| 발견 | 내용 |
|------|------|
| **가장 중요한 데이터셋** | DMD — 유일하게 RGB+IR+Depth 3모달 + 운전 행동 라벨을 모두 포함 |
| **IR 스켈레톤 유일 소스** | NTU RGB+D — IR 영상 + 25관절 3D 스켈레톤이 있는 유일한 대규모 데이터셋 |
| **IR 머리 포즈 최적** | DriveAHead — 실제 운전 환경 IR+Depth, 1M 프레임, 모션캡처 GT |
| **시선 pretrain 최적** | ETH-XGaze — 110명 1M+ 이미지, 극단적 머리 포즈 커버 |
| **운전 중 행동 분류 최적** | Drive&Act — IR+RGB+Depth + 3D 스켈레톤 + 83개 행동 클래스 |
| **부족한 영역** | IR 환경 시선 추정 전용 데이터셋은 존재하지 않음 → 자체 구축 또는 도메인 적응 필수 |

---

## 2. 카테고리별 데이터셋 상세 분석

---

### A. 운전자 모니터링 / 주의분산 감지 데이터셋

---

#### A-1. DMD (Driver Monitoring Dataset) ⭐ 최우선

| 항목 | 내용 |
|------|------|
| **정식명칭** | DMD: A Large-Scale Multi-modal Driver Monitoring Dataset |
| **출시** | 2020, Vicomtech (스페인) |
| **다운로드** | https://dmd.vicomtech.org/ / GitHub 직접 다운로드 |
| **규모** | 41시간, 37명 운전자, ~5.7M 프레임 |
| **카메라** | 3대 동기화 (얼굴/몸통/손), **RGB + IR + Depth** |
| **해상도** | IR/Depth: 640×480, RGB: 1920×1080 |
| **어노테이션** | 시선 영역(9구역), 머리 포즈(yaw/pitch/roll), 얼굴 랜드마크, 주의분산 라벨(9행동), 졸음, 손-핸들 상호작용, 눈 감김 |
| **주행 시나리오** | 실제 주행 + 시뮬레이터, 주간/야간 |
| **차폐** | 선글라스, 안경, 손 가림, 핸드폰 |
| **라이선스** | 학술 연구 무료 |
| **용량** | ~500GB |
| **승인 소요** | 즉시 (GitHub 다운로드) |

**프로젝트 활용:**
- Branch A (시선): 9구역 시선 영역 라벨 → 시선 방향 분류 학습
- Branch B (포즈): 머리 포즈 GT → 6DRepNet IR fine-tuning 검증
- Branch C (스켈레톤): 몸통 카메라 영상 → RTMPose로 스켈레톤 추출 후 행동 인식 학습
- Branch D (차폐): 안경/선글라스 착용 장면 → 차폐 감지 학습
- 행동 분류: 9가지 주의분산 라벨 → TCN 분류기 학습의 주 데이터

**한계:** 스켈레톤 키포인트가 미리 추출되어 있지 않음 → 직접 추출 필요

---

#### A-2. Drive&Act ⭐ 행동/스켈레톤 핵심

| 항목 | 내용 |
|------|------|
| **정식명칭** | Drive&Act: A Multi-Modal Dataset for Fine-Grained Driver Behavior Recognition |
| **출시** | 2019, TU Darmstadt + DFKI (독일) |
| **다운로드** | https://www.driveandact.com/ (등록 필요) |
| **규모** | 12시간, 9.6M+ 프레임 |
| **카메라** | 6대 동기화 (다각도), **RGB + IR + Depth + 3D Skeleton** |
| **어노테이션** | 83개 행동 카테고리(계층적), 원자 행동, 객체 상호작용, 3D 스켈레톤, 세밀 행동 |
| **주행 시나리오** | 수동 주행 + 자율 주행 |
| **차폐** | 손-물체 상호작용, 부분 가시성 |
| **라이선스** | 연구 무료 |
| **용량** | ~1.5TB |
| **승인 소요** | 3~7일 (저자 연락) |

**프로젝트 활용:**
- **유일하게 IR + 3D 스켈레톤 + 행동 라벨을 모두 갖춘 운전 데이터셋**
- Branch C (스켈레톤): 3D 스켈레톤 → CTR-GCN 학습의 핵심 데이터
- 행동 분류: 83개 클래스 중 본 프로젝트 8개 클래스에 매핑 가능
- 6개 카메라 뷰 → cross-view 강건성 평가

**한계:** 1.5TB로 용량이 매우 큼 → RGB 뷰만 우선 다운로드 후 IR 추가

---

#### A-3. DriveAHead ⭐ IR 머리 포즈 최적

| 항목 | 내용 |
|------|------|
| **정식명칭** | DriveAHead: A Large-Scale Driver Head Pose Dataset |
| **출시** | 2017, Karlsruhe Institute of Technology (KIT) |
| **다운로드** | https://cvhci.anthropomatik.kit.edu/~aschwarz/driveahead/ (직접 다운로드) |
| **규모** | ~1M 프레임, 20명 운전자, ~50시간 |
| **카메라** | Kinect v2: **Depth + IR**, 512×424 |
| **어노테이션** | 머리 포즈(yaw/pitch/roll) 모션캡처 GT, 3D 머리 위치, 차폐 라벨 |
| **주행 시나리오** | 실제 차량 주행, 다양한 조명 |
| **차폐** | 손, 선글라스, 모자 |
| **라이선스** | 학술 무료 |
| **용량** | ~100GB |
| **승인 소요** | 즉시 |

**프로젝트 활용:**
- Branch B (포즈): **핵심 학습 데이터** — IR 환경 머리 포즈 + 모션캡처 GT
- Branch D (차폐): 차폐 라벨 → 차폐 감지 보조 학습
- IR Domain Adaptation: IR 영상으로 모델 fine-tuning의 직접적 소스

---

#### A-4. DAD (Driver Anomaly Detection)

| 항목 | 내용 |
|------|------|
| **정식명칭** | Driver Anomaly Detection Dataset |
| **출시** | 2021, TU Munich |
| **다운로드** | https://github.com/okankop/Driver-Anomaly-Detection |
| **규모** | ~650분 (학습 550분 정상 + 100분 이상), 31명 |
| **카메라** | 전면+상면 2뷰, **IR + Depth**, 224×171, 45fps |
| **어노테이션** | 정상 vs 이상 이진 분류, 8개 이상 행동(테스트: 16개 미학습 행동) |
| **라이선스** | 연구 무료 |
| **용량** | ~50GB |

**프로젝트 활용:**
- Open-set 이상 탐지 벤치마크 — 학습하지 않은 이상 행동도 탐지하는 능력 평가
- IR + Depth 모달리티로 본 프로젝트 IR 파이프라인과 직접 호환
- Contrastive learning 베이스라인 참고

---

#### A-5. 3MDAD (Multimodal Multi-View Driver Anomaly Detection)

| 항목 | 내용 |
|------|------|
| **출시** | 2020 |
| **카메라** | Kinect 2대 (전면+측면), **주간: RGB+Depth / 야간: IR+Depth** |
| **특징** | 주야간 모달리티 자동 전환 — 본 프로젝트의 IR 야간 시나리오에 직접 대응 |
| **용량** | ~30GB |

---

#### A-6. NTHU-DDD (Drowsy Driver Detection)

| 항목 | 내용 |
|------|------|
| **출시** | 2015, NTHU (대만) |
| **다운로드** | http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/ |
| **규모** | ~9.5시간, 36명 (다양한 인종) |
| **카메라** | RGB 단일, 640×480 |
| **어노테이션** | 졸음(alert/drowsy), 머리 상태(정지/고개끄덕/옆보기), 입 상태(정지/하품/말하기), 눈 상태(open/sleepy) |
| **차폐** | 안경, 선글라스 |
| **용량** | ~20GB |

**프로젝트 활용:**
- 졸음 감지 특화 — Branch A의 눈 감김/졸음 라벨 보조 학습
- RGB만이므로 IR 시뮬레이션(grayscale 변환) 필요

---

#### A-7. State Farm Distracted Driver Detection

| 항목 | 내용 |
|------|------|
| **다운로드** | https://www.kaggle.com/c/state-farm-distracted-driver-detection |
| **규모** | 22,424 이미지, 26명 운전자 |
| **클래스** | 10개 (안전운전, 문자, 전화(우), 전화(좌), 라디오, 음료, 뒤돌아봄, 머리손질, 대화, 기타) |
| **카메라** | RGB 단일, 고정 각도 |
| **용량** | ~5GB |
| **승인** | 즉시 (Kaggle) |

**프로젝트 활용:**
- 빠른 베이스라인 구축 — 5GB로 가볍고 즉시 다운로드 가능
- 10개 주의분산 클래스 → 본 프로젝트 8개 클래스와 매핑
- RGB only → IR 시뮬레이션 필요

---

#### A-8. Pandora ⭐ 차량 맥락 포즈

| 항목 | 내용 |
|------|------|
| **출시** | 2016, Modena University (이탈리아) |
| **다운로드** | https://aimagelab.ing.unimore.it/pandora/ (직접 다운로드) |
| **규모** | 250K+ 이미지, 22명 (남10, 여12) |
| **카메라** | Kinect v2: **RGB + Depth**, 1920×1080(RGB) / 512×424(Depth) |
| **어노테이션** | 머리 3D 위치, 머리 포즈(yaw ±125°, pitch ±100°, roll ±70°), 어깨 포즈, 상체 3D 스켈레톤, **차폐 어노테이션** |
| **차폐** | 안경, 선글라스, 스카프, 모자, 핸드폰 |
| **용량** | ~50GB |

**프로젝트 활용:**
- **차량 대시보드 시점** 시뮬레이션 — 실제 DMS 카메라 각도와 유사
- 머리 포즈 범위가 매우 넓음 (yaw ±125°) — 극단적 고개 돌림 커버
- 차폐 어노테이션 → Branch D 학습 보조
- 상체 스켈레톤 → Branch C 보조

---

### B. 시선 추정 데이터셋

---

#### B-1. ETH-XGaze ⭐ 시선 pretrain 최우선

| 항목 | 내용 |
|------|------|
| **출시** | 2020, ETH Zurich |
| **다운로드** | https://ait.ethz.ch/xgaze (등록 필요) |
| **규모** | 1M+ 이미지, 110명 |
| **카메라** | 18대 DSLR, RGB, 224×224 / 448×448 패치 |
| **어노테이션** | 3D 시선 방향(각도), 머리 포즈(yaw/pitch/roll), 눈 위치, 보정된 GT |
| **라이선스** | CC BY-NC-SA 4.0 |
| **용량** | 224×224: ~130GB / 448×448: ~497GB / Raw: ~7TB |
| **승인** | 2~5일 |

**프로젝트 활용:**
- Branch A (시선): **GazeCNN pretrain의 주 데이터셋**
- 극단적 머리 포즈 커버 (-90°~+90°) — 운전 중 고개 돌림 상황에 강건한 모델 학습
- RGB only → IR 적응을 위해 grayscale 변환 + domain adaptation 필요

**한계:** 실험실 환경, 운전 맥락 없음

---

#### B-2. Gaze360

| 항목 | 내용 |
|------|------|
| **출시** | 2019, MIT CSAIL |
| **다운로드** | https://gaze360.csail.mit.edu/ |
| **규모** | 238명, 실내+실외 |
| **어노테이션** | 3D 시선 방향 (360° 전방위), 불확실성 추정, 시계열 |
| **용량** | ~50GB |

**프로젝트 활용:** 360° 시선 커버리지 — 뒤돌아봄 등 극단적 시선 방향 학습에 보조적 활용

---

#### B-3. RT-GENE

| 항목 | 내용 |
|------|------|
| **출시** | 2018, Imperial College London |
| **다운로드** | https://github.com/Tobias-Fischer/rt_gene |
| **규모** | 277K 라벨 프레임, 15명 |
| **카메라** | Kinect (RGB-D), 다양한 거리 (0.5~2.9m) |
| **어노테이션** | 시선 방향(모션캡처 검증), 머리 포즈, Depth |
| **용량** | ~30GB |

**프로젝트 활용:** Depth 정보 포함 — 3D 시선 추정 검증에 보조 활용

---

#### B-4. MPIIGaze / MPIIFaceGaze

| 항목 | 내용 |
|------|------|
| **출시** | 2017, Max Planck Institute |
| **규모** | 213K 이미지, 15명, 3개월 이상 수집 |
| **어노테이션** | 시선 방향(2D 각도), 37K 이미지에 눈/입/동공 랜드마크, 머리 포즈 |
| **용량** | ~10GB |

**프로젝트 활용:** 일상 환경에서의 시선 데이터 — 조명 변화 강건성 학습 보조

---

#### B-5. Columbia Gaze

| 항목 | 내용 |
|------|------|
| **출시** | 2010, Columbia University |
| **다운로드** | https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/ |
| **규모** | 5,880 이미지, 56명 (다양한 인종: 아시아 21, 백인 19, 남아시아 8, 흑인 7, 히스패닉 4) |
| **어노테이션** | 5개 머리 포즈 × 21개 시선 방향, 고해상도 (5184×3456) |
| **특이사항** | **56명 중 21명이 안경 착용** — 차폐 시선 학습에 유용 |
| **용량** | ~3GB |

---

#### ⚠️ IR 시선 추정 전용 데이터셋 부재

2026년 3월 기준, **IR/NIR 환경 전용 시선 추정 데이터셋은 공개된 것이 없다.** 이것이 본 프로젝트의 데이터 측면 최대 도전과제이자 자체 Failure 데이터셋 구축의 학술적 가치가 되는 지점이다.

**대응 전략:**
1. ETH-XGaze RGB → grayscale 변환 + 히스토그램 매칭으로 IR 시뮬레이션
2. DMD의 IR 뷰 + 시선 영역 라벨로 fine-tuning
3. 자체 IR 카메라 촬영 데이터로 최종 fine-tuning

---

### C. 머리 포즈 추정 데이터셋

---

#### C-1. BIWI Kinect Head Pose

| 항목 | 내용 |
|------|------|
| **다운로드** | HuggingFace / Kaggle (즉시) |
| **규모** | 15K 이미지, 20명 |
| **카메라** | Kinect: RGB + Depth, 640×480 |
| **어노테이션** | yaw ±75°, pitch ±60°, 3D 머리 위치 |
| **용량** | ~3GB |

**활용:** Head Pose 베이스라인 벤치마크, 빠른 검증용

---

#### C-2. 300W-LP

| 항목 | 내용 |
|------|------|
| **규모** | 61,225 이미지 (3D 모델 기반 합성) |
| **어노테이션** | 68개 얼굴 랜드마크(2D+3D), 머리 포즈, 표정/조명 파라미터 |
| **포즈 범위** | yaw -90°~+90° |
| **용량** | ~8GB |

**활용:** 6DRepNet pretrain의 보조 데이터, 얼굴 랜드마크 학습

---

#### C-3. AFLW2000-3D

| 항목 | 내용 |
|------|------|
| **다운로드** | Kaggle 즉시 |
| **규모** | 2,000 이미지, 2,000명 |
| **어노테이션** | 68개 3D 얼굴 랜드마크, 머리 포즈 |
| **용량** | ~500MB |

**활용:** 6DRepNet 평가 벤치마크 (표준)

---

### D. 스켈레톤 / 포즈 / 행동 인식 데이터셋

---

#### D-1. NTU RGB+D 60 & 120 ⭐ 스켈레톤 학습 최우선

| 항목 | 내용 |
|------|------|
| **출시** | 2016(v60) / 2019(v120), NTU Singapore |
| **다운로드** | https://rose1.ntu.edu.sg/dataset/actionRecognition/ (**등록 필수, 1~2주 소요**) |
| **규모** | 114,480 샘플, 40명+, 120개 행동 클래스 |
| **카메라** | Kinect v2 × 3뷰: **RGB + Depth + IR**, 1920×1080 |
| **어노테이션** | **25관절 3D 스켈레톤** (x,y,z,confidence), 120개 행동, RGB 비디오, Depth 맵, **IR 비디오** |
| **라이선스** | 등록된 학술 사용자 무료 |
| **용량** | ~500GB |

**프로젝트 활용:**
- Branch C (스켈레톤): **CTR-GCN pretrain의 핵심 데이터셋**
- 25관절 → 본 프로젝트 13관절로 매핑 (상체만 추출)
- **IR 비디오 포함** — IR 환경 스켈레톤 추출기 fine-tuning에 직접 사용 가능
- 120개 행동 중 운전 관련 행동 선별: 전화하기(#23), 읽기(#11), 쓰기(#12), 먹기(#1) 등

**⚠️ 중요: 등록 승인에 1~2주 소요 — 오늘 당장 신청할 것**

---

#### D-2. COCO Keypoints

| 항목 | 내용 |
|------|------|
| **다운로드** | https://cocodataset.org/ (즉시) |
| **규모** | 200K+ 이미지, 17개 키포인트 |
| **용량** | ~25GB (키포인트만) |

**활용:** RTMPose pretrain의 기본 데이터셋 (이미 pretrained 모델에 포함)

---

#### D-3. Halpe Full-Body

| 항목 | 내용 |
|------|------|
| **다운로드** | https://github.com/Fang-Haoshu/Halpe-FullBody (즉시) |
| **규모** | 50K train + 5K test |
| **어노테이션** | **136개 키포인트** (몸 26 + 얼굴 68 + 좌손 21 + 우손 21) |
| **용량** | ~15GB |

**활용:** 얼굴+손+몸통 통합 키포인트 — Branch A(시선)와 Branch C(스켈레톤) 연결부 학습

---

### E. 차폐 / 얼굴 검출 데이터셋

---

#### E-1. WIDER FACE

| 항목 | 내용 |
|------|------|
| **다운로드** | http://shuoyang1213.me/WIDERFACE/ (즉시) |
| **규모** | 32,203 이미지, 393,703 얼굴 |
| **어노테이션** | 얼굴 bbox, 난이도(easy/medium/hard), 차폐/포즈/스케일 도전 |
| **용량** | ~50GB |

**활용:** YOLO-FaceV2 fine-tuning — 차폐 상황에서의 얼굴 검출 강건성 향상

---

#### E-2. MAFA (MAsked FAces)

| 항목 | 내용 |
|------|------|
| **다운로드** | Kaggle / https://imsg.ac.cn/research/maskedface.html |
| **규모** | 35,806 마스크 착용 얼굴 |
| **어노테이션** | 마스크 유형/위치/정도 세분류 |
| **용량** | ~10GB |

**활용:** Branch D (차폐 감지) 학습 — 마스크 유형 세분류 가능

---

### F. IR / NIR 전용 데이터셋

---

#### F-1. CASIA NIR-VIS 2.0

| 항목 | 내용 |
|------|------|
| **다운로드** | http://www.cbsr.ia.ac.cn/english/NIR-VIS-2.0/ (이메일 요청) |
| **규모** | 17,580 NIR+RGB 쌍, 725명 |
| **용량** | ~20GB |

**활용:** IR Domain Adaptation Layer 학습 — NIR↔RGB paired 데이터로 도메인 변환 학습

---

## 3. 모달리티 커버리지 매트릭스

```
데이터셋        | RGB | IR  | Depth | Video | 스켈레톤 | 시선 | 포즈 | 행동 | 차폐 | 운전맥락
──────────────┼─────┼─────┼───────┼───────┼──────────┼──────┼──────┼──────┼──────┼────────
DMD           |  ✓  |  ✓  |   ✓   |   ✓   |    ✗     |  ✓✓  |  ✓   |  ✓✓  |  ✓   |   ✓✓
Drive&Act     |  ✓  |  ✓  |   ✓   |   ✓   |    ✓✓    |  ✗   |  ✓   |  ✓✓  |  ✓   |   ✓✓
DriveAHead    |  ✗  |  ✓✓ |   ✓   |   ✓   |    ✗     |  ✗   |  ✓✓  |  ✗   |  ✓✓  |   ✓✓
NTU RGB+D     |  ✓  |  ✓  |   ✓   |   ✓   |    ✓✓    |  ✗   |  ✓   |  ✓✓  |  ✓   |   ✗
ETH-XGaze     |  ✓  |  ✗  |   ✗   |   ✗   |    ✗     |  ✓✓  |  ✓   |  ✗   |  ✗   |   ✗
Pandora       |  ✓  |  ✗  |   ✓   |   ✓   |    ✓     |  ✗   |  ✓✓  |  ✓   |  ✓✓  |   ✓
MAFA          |  ✓  |  ✗  |   ✗   |   ✗   |    ✗     |  ✗   |  ✗   |  ✗   |  ✓✓  |   ✗
WIDER FACE    |  ✓  |  ✗  |   ✗   |   ✗   |    ✗     |  ✗   |  ✗   |  ✗   |  ✓✓  |   ✗
CASIA NIR-VIS |  ✓  |  ✓✓ |   ✗   |   ✗   |    ✗     |  ✗   |  ✓   |  ✗   |  ✗   |   ✗
DAD           |  ✗  |  ✓  |   ✓   |   ✓   |    ✗     |  ✗   |  ✗   |  ✓   |  ✓   |   ✓
State Farm    |  ✓  |  ✗  |   ✗   |   ✗   |    ✗     |  ✗   |  ✗   |  ✓   |  ✓   |   ✓
──────────────┴─────┴─────┴───────┴───────┴──────────┴──────┴──────┴──────┴──────┴────────
✓✓ = 해당 영역의 최우수 데이터셋
```

---

## 4. Branch별 데이터셋 매핑 및 학습 전략

---

### 4.1 Branch A: 시선 추정 (GazeCNN)

```
학습 단계                    데이터셋                용도              비고
──────────────────────────────────────────────────────────────────────────
Stage 1: Pretrain          ETH-XGaze (1M)         시선 방향 학습       RGB, 224×224 패치 사용
Stage 2: IR 적응           ETH-XGaze → grayscale  도메인 적응          히스토그램 매칭으로 IR 시뮬
Stage 3: 운전 맥락 전이    DMD IR 뷰              시선 영역 fine-tune  9구역 → 본 프로젝트 8클래스 매핑
Stage 4: Failure 강화      자체 촬영 IR           최종 fine-tune       선글라스/야간/마스크 시나리오
평가                        DMD test + 자체 test   벤치마크             두 데이터 모두에서 평가
```

**시선 영역 매핑 (DMD 9구역 → 프로젝트 8클래스):**
```
DMD 영역         →  프로젝트 클래스
───────────────────────────────────
전방 구역        →  0: 전방 주시
좌측 미러 구역   →  1: 사이드미러 확인
후방 미러 구역   →  2: 백미러 확인
센터 콘솔 구역   →  3: 네비게이션 주시
하단 우측 구역   →  4: 핸드폰 사용 (행동과 결합)
눈감김/졸음      →  5: 졸음 (시선+눈상태 결합)
좌측 뒤쪽 구역   →  6: 뒤돌아봄
기타 구역        →  7: 기타 이상행동
```

**핵심 수치:**
- Pretrain 데이터: 1M+ 이미지 (ETH-XGaze)
- Fine-tune 데이터: ~50K 프레임 (DMD IR에서 시선 라벨 있는 부분)
- 자체 데이터: ~5K 프레임 (Failure 시나리오)
- 목표: MAE ≤ 10° (ETH-XGaze SOTA: 4.08°)

---

### 4.2 Branch B: 머리 포즈 추정 (6DRepNet360)

```
학습 단계                    데이터셋                     용도              비고
───────────────────────────────────────────────────────────────────────────────
Stage 1: Pretrain          300W-LP (61K)               얼굴 랜드마크+포즈   합성 데이터, 넓은 포즈
Stage 2: Fine-tune         DriveAHead IR (1M)          IR 운전 포즈         모션캡처 GT
Stage 3: 검증              AFLW2000-3D (2K)            벤치마크             표준 평가셋
Stage 4: 차량 맥락         Pandora (250K)              대시보드 시점        차폐 포함
Stage 5: Failure 강화      자체 촬영 IR                최종 fine-tune       극한 상황
평가                        BIWI + DriveAHead test      벤치마크
```

**핵심 수치:**
- Pretrain: 61K (300W-LP) + 1M (DriveAHead)
- IR 직접 학습: 1M 프레임 (DriveAHead) — **IR pretrain 가능!**
- 목표: MAE ≤ 5° (6DRepNet360 SOTA: 3~5°)

---

### 4.3 Branch C: 스켈레톤 행동 인식 (CTR-GCN + Temporal Transformer)

```
학습 단계                    데이터셋                     용도                 비고
────────────────────────────────────────────────────────────────────────────────
Stage 1: GCN Pretrain      NTU RGB+D (114K)            스켈레톤 행동 인식     25관절→13관절 매핑
Stage 2: 포즈 추출기       COCO Keypoints (200K)       RTMPose 기학습 확인   이미 pretrained
Stage 3: 운전 행동 전이    Drive&Act (9.6M)            운전 맥락 fine-tune   83→8 클래스 매핑
Stage 4: IR 적응           DMD IR 뷰 → RTMPose 추출   IR 스켈레톤 품질 확인
Stage 5: 통합 학습         DMD + Drive&Act + 자체      최종 fine-tune
평가                        DMD test + 자체 test
```

**NTU RGB+D 25관절 → 프로젝트 13관절 매핑:**
```
NTU 관절 ID    NTU 이름           →   프로젝트 관절
──────────────────────────────────────────────────
3              Head               →   0: 코 (근사)
4              Left eye (없음)    →   1: 좌눈 (Face Mesh에서)
5              Right eye (없음)   →   2: 우눈 (Face Mesh에서)
-              -                  →   3: 좌귀 (Face Mesh에서)
-              -                  →   4: 우귀 (Face Mesh에서)
4              Left shoulder      →   5: 좌어깨
8              Right shoulder     →   6: 우어깨
5              Left elbow         →   7: 좌팔꿈치
9              Right elbow        →   8: 우팔꿈치
6              Left wrist         →   9: 좌손목
10             Right wrist        →   10: 우손목
20             Spine (neck 근사)  →   11: 목
3              Head center        →   12: 머리중심
```

**Drive&Act 83클래스 → 프로젝트 8클래스 매핑 (예시):**
```
Drive&Act 클래스                              →  프로젝트 클래스
──────────────────────────────────────────────────────────────
driving, steering                             →  0: 전방 주시
checking_left_mirror, checking_right_mirror   →  1: 사이드미러
checking_rear_mirror                          →  2: 백미러
interacting_with_infotainment                 →  3: 네비게이션
using_phone, texting                          →  4: 핸드폰
yawning, fatigue_signs                        →  5: 졸음
looking_back                                  →  6: 뒤돌아봄
eating, drinking, reading, grooming           →  7: 기타
```

---

### 4.4 Branch D: 차폐 감지 (MobileNetV3-Small)

```
학습 단계                    데이터셋              용도                비고
──────────────────────────────────────────────────────────────────────
Stage 1: 얼굴 검출 기반    WIDER FACE (393K)    차폐 얼굴 검출       easy/medium/hard 구분
Stage 2: 마스크 특화       MAFA (35.8K)         마스크 유형 분류     세분류 어노테이션
Stage 3: 운전 맥락         Pandora (250K)       차량 내 차폐 학습    선글라스/모자/핸드폰
Stage 4: IR 적응           DriveAHead (1M)      IR 차폐 라벨         손/선글라스/모자
Stage 5: 통합              DMD + 자체           최종 fine-tune
```

**차폐 유형 5클래스 학습 데이터 소스:**
```
클래스          주 데이터셋              보조 데이터셋        자체 데이터
──────────────────────────────────────────────────────────────────────
none           DMD, Pandora             WIDER FACE (easy)   정상 운전
sunglasses     Pandora, DriveAHead      Columbia Gaze       선글라스 착용 촬영
mask           MAFA                     -                   마스크 착용 촬영
hand           DMD (손 카메라)          Pandora              손으로 얼굴 만짐
phone          DMD (행동 라벨)          State Farm           핸드폰 사용 촬영
```

---

## 5. 다운로드 우선순위 및 일정

### 5.1 Tier 1: 필수 (즉시 시작)

| 순위 | 데이터셋 | 용량 | 다운로드 | 긴급도 | 담당 |
|------|---------|------|---------|--------|------|
| **1** | **NTU RGB+D** | 500GB | 등록 신청 (1~2주 대기) | ⚠️ 오늘 당장 | PM |
| **2** | **DMD** | 500GB | GitHub 즉시 | 이번 주 | 포즈팀 |
| **3** | **ETH-XGaze** | 130GB | 등록 (2~5일) | 이번 주 | 시선팀 |
| **4** | **DriveAHead** | 100GB | 직접 다운로드 | 이번 주 | 포즈팀 |
| **5** | **WIDER FACE** | 50GB | 직접 다운로드 | 이번 주 | PM |

**Tier 1 소계: ~1,280GB (1.25TB)**

### 5.2 Tier 2: 강력 권장 (1~2주 내)

| 순위 | 데이터셋 | 용량 | 다운로드 | 담당 |
|------|---------|------|---------|------|
| 6 | Drive&Act | 1.5TB | 저자 연락 (3~7일) | PM |
| 7 | MAFA | 10GB | Kaggle 즉시 | 포즈팀 |
| 8 | Pandora | 50GB | 직접 다운로드 | 포즈팀 |
| 9 | CASIA NIR-VIS | 20GB | 이메일 요청 | 시선팀 |
| 10 | State Farm | 5GB | Kaggle 즉시 | PM |

### 5.3 Tier 3: 보조 (시간 여유 시)

| 데이터셋 | 용량 | 용도 |
|---------|------|------|
| BIWI | 3GB | Head Pose 빠른 벤치마크 |
| AFLW2000-3D | 500MB | Head Pose 평가 |
| 300W-LP | 8GB | 얼굴 랜드마크 보조 |
| Gaze360 | 50GB | 360° 시선 보조 |
| NTHU-DDD | 20GB | 졸음 특화 보조 |
| DAD | 50GB | 이상 탐지 보조 |
| Halpe | 15GB | 전신 키포인트 보조 |

---

## 6. 저장 공간 계획

```
최소 구성 (Tier 1만):           ~1.3TB
권장 구성 (Tier 1 + Tier 2):    ~2.9TB
전체 구성 (전부):               ~3.5TB

권장 환경:
├─ Colab Pro: Google Drive 저장 (기본 100GB, 추가 구매 고려)
├─ 외장 HDD/SSD: 2TB 이상 권장 (팀 공용)
└─ DGX Spark: 할당 시 대용량 학습용
```

---

## 7. 데이터 전처리 파이프라인

### 7.1 IR 시뮬레이션 (RGB → IR-like)

RGB only 데이터셋(ETH-XGaze, State Farm 등)을 IR 파이프라인에서 사용하기 위한 변환:

```python
import cv2
import numpy as np

def rgb_to_ir_simulation(rgb_image):
    """RGB → IR-like 이미지 시뮬레이션"""
    # 1) Grayscale 변환
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    # 2) IR 히스토그램 매칭 (DriveAHead IR 기준 분포)
    # IR은 피부 반사율이 높아 밝기 분포가 다름
    gray = cv2.normalize(gray, None, 20, 235, cv2.NORM_MINMAX)

    # 3) 노이즈 추가 (IR 센서 특성)
    noise = np.random.normal(0, 5, gray.shape).astype(np.float32)
    gray = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4) 약간의 블러 (IR 광학 특성)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.5)

    return gray
```

### 7.2 스켈레톤 포맷 통일

```python
# NTU RGB+D (25 joints) → 프로젝트 (13 joints) 변환
NTU_TO_PROJECT = {
    3: 0,   # head → 코
    # 1, 2 (좌눈, 우눈) → Face Mesh에서 별도 추출
    # 3, 4 (좌귀, 우귀) → Face Mesh에서 별도 추출
    4: 5,   # left_shoulder → 좌어깨
    8: 6,   # right_shoulder → 우어깨
    5: 7,   # left_elbow → 좌팔꿈치
    9: 8,   # right_elbow → 우팔꿈치
    6: 9,   # left_wrist → 좌손목
    10: 10, # right_wrist → 우손목
    20: 11, # spine → 목 (근사)
    3: 12,  # head → 머리중심
}

def convert_ntu_to_project(ntu_skeleton):
    """NTU 25관절 → 프로젝트 13관절 변환"""
    project_skeleton = np.zeros((13, 3))  # x, y, z
    for ntu_idx, proj_idx in NTU_TO_PROJECT.items():
        project_skeleton[proj_idx] = ntu_skeleton[ntu_idx]
    # 목(11) = 좌어깨(5) + 우어깨(6) 중점
    project_skeleton[11] = (project_skeleton[5] + project_skeleton[6]) / 2
    return project_skeleton
```

### 7.3 어노테이션 포맷 통일

```python
# 통일 어노테이션 포맷
unified_annotation = {
    'frame_id': int,
    'source_dataset': str,  # 'DMD', 'DriveAHead', 'NTU', ...
    'modality': str,        # 'IR', 'RGB', 'IR_simulated'

    # 얼굴 검출
    'face_bbox': [x1, y1, x2, y2],  # 정규화 좌표
    'face_landmarks_5pt': [[x,y], ...],

    # 시선
    'gaze_yaw': float,     # degrees
    'gaze_pitch': float,   # degrees
    'gaze_region': int,    # 0~7 클래스

    # 머리 포즈
    'head_yaw': float,
    'head_pitch': float,
    'head_roll': float,

    # 스켈레톤
    'skeleton_13': [[x,y,conf], ...],  # 13관절

    # 차폐
    'occlusion_type': str,  # 'none', 'sunglasses', 'mask', 'hand', 'phone'
    'occlusion_severity': float,  # 0.0~1.0

    # 행동
    'behavior_class': int,  # 0~7
    'behavior_name': str,
}
```

---

## 8. 자체 Failure 데이터셋 구축 계획

### 8.1 촬영 시나리오 (4종 + 보너스 2종)

| 시나리오 | 촬영 조건 | 목표 프레임 수 | 어노테이션 |
|---------|---------|-------------|----------|
| **S1: 야간 저조도** | 조명 OFF, IR 카메라만 | 3,000 | 행동 8클래스 + 시선 영역 |
| **S2: 선글라스** | 다양한 선글라스 3종+ | 2,000 | 행동 + 차폐=sunglasses |
| **S3: 마스크/모자** | KF94 + 야구모자 | 2,000 | 행동 + 차폐=mask |
| **S4: 핸드폰/손** | 통화, 문자, 얼굴 만짐 | 3,000 | 행동=phone/hand + 차폐 |
| S5: 졸음 연기 | 눈감김, 하품, 고개 숙임 | 2,000 | 행동=졸음 |
| S6: 정상 혼동 행동 | 미러 확인, 계기판, 뒤돌아봄 | 2,000 | 행동=정상(1,2) vs 이상(6) |

**총 목표: ~14,000 프레임 (60fps 기준 ~4분)**
**팀원 5명 × 피험자 역할 → 다양성 확보**

### 8.2 어노테이션 가이드라인

```
[라벨링 규칙]
1. 프레임 단위 라벨링 (1fps 샘플링 → 초당 1프레임 라벨)
2. 행동 클래스: 8개 중 하나 선택 (중복 불가)
3. 차폐 유형: 5개 중 하나 + 심각도(0~1)
4. 시선 영역: 8개 중 하나 (시선이 보이지 않으면 'unknown')
5. 신뢰도 플래그: 라벨러 확신도 (high/medium/low)

[도구]
- CVAT (Computer Vision Annotation Tool): 무료, 웹 기반
- 예상 작업 시간: 14,000 프레임 × 30초/프레임 ÷ 5명 = 약 23시간/인
```

---

## 9. 즉시 실행 체크리스트

### 오늘 (3/27)
- [ ] NTU RGB+D 등록 신청: https://rose1.ntu.edu.sg/dataset/actionRecognition/
- [ ] ETH-XGaze 접근 요청: https://ait.ethz.ch/xgaze
- [ ] CASIA NIR-VIS 이메일 요청
- [ ] State Farm 다운로드 시작 (Kaggle, 5GB, 즉시)
- [ ] BIWI 다운로드 시작 (HuggingFace, 3GB, 즉시)

### 이번 주 (3/27~4/3)
- [ ] DMD 다운로드 시작 (GitHub, ~500GB)
- [ ] DriveAHead 다운로드 시작 (직접, ~100GB)
- [ ] WIDER FACE 다운로드 (직접, ~50GB)
- [ ] MAFA 다운로드 (Kaggle, ~10GB)
- [ ] Drive&Act 저자 연락

### 다음 주 (4/3~4/10)
- [ ] ETH-XGaze 승인 확인 → 다운로드 시작
- [ ] NTU RGB+D 승인 대기 (미승인 시 리마인드)
- [ ] 다운로드 완료된 데이터셋 전처리 시작
- [ ] IR 시뮬레이션 파이프라인 구현

### 2주 후 (4/10~4/17)
- [ ] NTU RGB+D 다운로드 시작
- [ ] Drive&Act 접근 확인
- [ ] 전체 데이터 통일 포맷 변환 완료
- [ ] Branch별 학습 시작

---

## 10. 참고 — 데이터셋 논문 (필수 읽기)

| 순위 | 논문 | 데이터셋 | URL |
|------|------|---------|-----|
| 1 | Ortega et al., 2020 | DMD | https://arxiv.org/abs/2008.12085 |
| 2 | Martin et al., ICCV 2019 | Drive&Act | ICCV 2019 proceedings |
| 3 | Shahroudy et al., CVPR 2016 | NTU RGB+D | CVPR 2016 proceedings |
| 4 | Zhang et al., 2020 | ETH-XGaze | https://arxiv.org/abs/2007.15837 |
| 5 | Schwarz et al., CVPR-W 2017 | DriveAHead | CVPR 2017 Workshop |
| 6 | Yang et al., 2016 | WIDER FACE | https://arxiv.org/abs/1511.06523 |
| 7 | Ge et al., 2017 | MAFA | https://arxiv.org/abs/1709.05188 |
