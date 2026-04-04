# DMD: A Large-Scale Multi-Modal Driver Monitoring Dataset for Attention and Alertness Analysis

## 논문 정보

- **저자:** Juan Diego Ortega, Neslihan Kose, Paola Cañas, Min-An Chao, Alvaro Unzueta, Marcos Nieto, Oihana Otaegui, Luis Salgado
- **발표:** ECCV 2020 Workshops (Computer Vision – ECCV 2020 Workshops, Springer)
- **arXiv:** 2008.12085 (2020년 8월)
- **기관:** Vicomtech (스페인), Universidad Politécnica de Madrid

> ⚠️ **참고:** 본 요약은 논문 원문, GitHub 리포지토리, 공식 웹사이트 등 다수의 소스를 종합하여 작성되었습니다. 일부 세부 사항은 원문 PDF 직접 접근이 제한되어 공개된 정보를 기반으로 보완하였습니다.

---

## 1. Abstract (초록)

**핵심:** DMS(Driver Monitoring System) 개발의 병목은 충분히 크고 포괄적인 데이터셋의 부재이며, 이는 SAE Level-2에서 Level-3으로의 자동화 전환에 핵심적이다. DMD는 37명의 운전자를 대상으로 3대의 카메라(얼굴, 상체, 손)에서 RGB/Depth/IR 총 41시간의 영상을 수집한 대규모 다중 모달 데이터셋이다.

- 실제 차량 + 시뮬레이터 환경에서 수집
- 산만행동, 시선 배분, 졸음, 핸들-손 상호작용, 맥락 데이터 포함
- 기존 유사 데이터셋보다 규모, 다양성, 다목적성에서 우위
- 파생 데이터셋 dBehaviourMD (13개 산만행동)를 통한 딥러닝 학습 활용 예시 제시

---

## 2. Introduction (서론)

**핵심:** SAE Level-3 자율주행에서는 운전자가 시스템 요청 시 제어를 인수받아야 하므로, 운전자 상태(주의력, 졸음 등) 실시간 모니터링이 필수적이다.

- 기존 DMS 데이터셋은 규모가 작거나, 단일 모달리티이거나, 특정 행동만 다루는 한계
- Euro NCAP 2025 로드맵에서 DMS를 차량 안전 평가 항목에 포함할 계획
- DMD는 이러한 갭을 메우기 위해 설계된 포괄적 데이터셋
- 산만행동 인식, 시선 추적, 졸음 감지를 단일 데이터셋에서 연구 가능

---

## 3. Related Work (관련 연구)

**핵심:** 기존 운전자 모니터링 데이터셋과의 비교를 통해 DMD의 차별점을 명확히 한다.

### 기존 데이터셋의 한계

| 데이터셋 | 주요 한계 |
|----------|----------|
| StateFarm | 정지 이미지만 제공, 시간적 맥락 없음 |
| Drive&Act | 다양한 행동 포함하나 산만행동에 특화되지 않음 |
| DriveAHead | 머리 자세에만 초점 |
| DD-Pose | 소규모, 제한적 행동 |
| NTHU-DDD | 졸음에만 초점, 산만행동 미포함 |

### DMD의 차별성
- **다중 카메라** (3대 동시 촬영): 얼굴, 상체, 손
- **다중 모달리티** (RGB + Depth + IR): 조명 변화에 강건
- **다중 시나리오** (실차 + 시뮬레이터): 환경 다양성 확보
- **포괄적 어노테이션**: 산만행동, 졸음, 시선, 손-핸들 상호작용을 동시에 다룸

---

## 4. The DMD Dataset (데이터셋 구성)

### 4.1 Recording Setup (녹화 장비)

**핵심:** Intel RealSense 카메라 3대를 사용하여 운전자의 얼굴, 상체, 손을 동시에 다중 모달리티로 촬영한다.

- **카메라 모델:**
  - Intel RealSense D435 → 상체 촬영 (넓은 FOV)
  - Intel RealSense D415 → 얼굴 및 손 촬영 (저비용)
- **촬영 스트림 (카메라당 3개):**
  - RGB (가시광)
  - IR (근적외선)
  - Depth (깊이)
- **총 동시 스트림:** 3 카메라 × 3 모달리티 = 9개 스트림
- **녹화 환경:**
  - 실제 차량 (다양한 조명 조건)
  - 운전 시뮬레이터 (통제된 환경)

### 4.2 Recording Protocol (녹화 프로토콜)

**핵심:** 37명의 참가자가 사전 정의된 활동 프로토콜에 따라 산만행동, 시선 고정, 졸음 시뮬레이션을 수행한다.

- **참가자:** 37명 (다양한 연령, 성별)
- **총 녹화 시간:** 약 41시간
- **세션 구성 요소:**
  - 프로토콜 (수행할 활동)
  - 참가자
  - 환경 (실차/시뮬레이터)
  - 조명 조건

### 4.3 Annotation Scheme (어노테이션 체계)

**핵심:** 최대 7개 어노테이션 레벨로 구성된 계층적 라벨링 시스템을 사용하며, 각 레벨 내 라벨은 상호 배타적이다. VCD(Video Content Description) / OpenLABEL 포맷 사용.

#### 4.3.1 산만행동 (Distraction) 어노테이션

주의력 배분에 영향을 미치는 행동들:

| # | 라벨 | 설명 |
|---|------|------|
| 1 | Safe driving | 안전 운전 (정상 상태) |
| 2 | Texting (right) | 오른손 문자 |
| 3 | Phone call (right) | 오른손 통화 |
| 4 | Texting (left) | 왼손 문자 |
| 5 | Phone call (left) | 왼손 통화 |
| 6 | Operating the radio | 라디오/인포테인먼트 조작 |
| 7 | Drinking | 음료 마시기 |
| 8 | Reaching side | 옆으로 손 뻗기 |
| 9 | Hair and makeup | 머리 손질/화장 |
| 10 | Talking to passenger | 동승자와 대화 |
| 11 | Reaching backseat | 뒷좌석으로 손 뻗기 |
| 12 | Change gear | 기어 변경 |
| 13 | Standstill/Waiting | 정차/대기 상태 |

> **참고:** dBehaviourMD 파생 데이터셋은 위 13개 클래스를 사용

#### 4.3.2 시선 (Gaze) 어노테이션

**핵심:** 시선 어노테이션은 이진 분류(`gaze_on_road`)와 9개 세부 영역 분류의 두 가지 수준으로 제공된다.

**이진 분류 (gaze_on_road):**
- `looking_road` (도로 주시 중) — 약 84%
- `not_looking_road` (도로 미주시) — 약 16%

**9개 사전 정의 시선 고정 영역:**

| # | 영역 | 설명 |
|---|------|------|
| 1 | Forward (전방 도로) | 전방 도로를 바라봄 |
| 2 | Rearview mirror (룸미러) | 실내 후사경을 바라봄 |
| 3 | Left mirror (좌측 사이드미러) | 좌측 사이드미러를 바라봄 |
| 4 | Right mirror (우측 사이드미러) | 우측 사이드미러를 바라봄 |
| 5 | Left (좌측) | 좌측 창문/좌측 방향 |
| 6 | Right (우측) | 우측 창문/우측 방향 |
| 7 | Dashboard/Speedometer (계기판) | 속도계/계기판을 바라봄 |
| 8 | Center stack/Radio (센터페시아) | 라디오/인포테인먼트 영역 |
| 9 | Other (기타) | 위 영역에 해당하지 않는 곳 |

> ⚠️ **이전 질문에 대한 확인:** "전방 주시 중/아님"의 이진 분류(`gaze_on_road`)는 존재하지만, 이것이 DMD 시선 어노테이션의 전부는 아니다. 9개 세부 영역 분류가 별도로 존재한다.

#### 4.3.3 졸음 (Drowsiness) 어노테이션

피로와 가장 상관관계가 높은 징후들:

| # | 라벨 | 설명 |
|---|------|------|
| 1 | Normal/Alert | 정상/각성 상태 |
| 2 | Yawning | 하품 |
| 3 | Microsleep | 미세 수면 (짧은 눈 감음) |
| 4 | Nodding | 졸음에 의한 고개 끄덕임 |
| 5 | Eyelid closure | 눈꺼풀 내려감 (개안도 감소) |

#### 4.3.4 손-핸들 상호작용 (Hands-wheel Interaction)

- 양손의 핸들 위 위치 추적
- 핸들을 잡고 있는지 여부
- 손에 들고 있는 물체 (핸드폰, 빗, 병 등)

#### 4.3.5 어노테이션 포맷

- **포맷:** ASAM OpenLABEL (VCD - Video Content Description) JSON
- **도구:** TaTo (Temporal Annotation Tool)로 어노테이션 수행
- **탐색:** DEx (Dataset Explorer Tool)로 데이터 탐색
- **특성:**
  - 시공간적(spatio-temporal) 객체 어노테이션 지원
  - 행동(action), 이벤트(event), 관계(relation) 의미론 포함
  - 최대 7개 어노테이션 레벨, 각 레벨 내 상호 배타적

---

## 5. dBehaviourMD: 파생 데이터셋

**핵심:** DMD에서 산만행동 인식에 특화된 하위 데이터셋을 추출하여, 딥러닝 학습에 바로 사용할 수 있도록 정리한다.

- DMD 전체에서 13개 산만행동 클래스 추출
- 딥러닝 훈련 프로세스에 직접 사용 가능하도록 전처리
- 기하학적(geometrical) + 시간적(temporal) 기반 어노테이션 모두 포함
- 클래스 분포가 불균형 → safe driving이 가장 높은 비율 차지

---

## 6. Baseline Experiments (베이스라인 실험)

**핵심:** dBehaviourMD를 기반으로 실시간 운전자 행동 인식 시스템의 성능을 다양한 퓨전 전략으로 평가한다.

### 실험 설계
- **목표:** 비용 효율적인 CPU-only 플랫폼에서 실시간 운전자 행동 인식
- **입력:** 다중 카메라/모달리티 스트림
- **퓨전 전략 비교:**
  - 단일 스트림 (single stream)
  - 초기 퓨전 (early fusion)
  - 후기 퓨전 (late fusion)
  - 스코어 퓨전 (score fusion)

### 주요 결과
- 다중 카메라/모달리티 퓨전이 단일 스트림 대비 정확도 향상
- 모든 퓨전 전략에서 실시간 처리 가능
- IR 모달리티가 조명 변화에 강건한 성능 제공
- 얼굴 카메라 단독 대비 상체+손 카메라 추가 시 성능 개선

---

## 7. Conclusions (결론)

**핵심:** DMD는 현재 공개된 운전자 모니터링 데이터셋 중 가장 크고 포괄적이며, 다중 카메라·다중 모달리티·다중 시나리오 구성을 통해 DMS 연구의 새로운 기준을 제시한다.

- DMD는 산만행동, 졸음, 시선, 손-핸들 상호작용을 단일 데이터셋에서 연구 가능하게 함
- OpenLABEL/VCD 표준 포맷으로 재사용성 및 확장성 확보
- dBehaviourMD를 통해 실제 DMS 애플리케이션에의 적용 가능성 입증
- 향후 더 많은 참가자, 환경, 행동 추가를 통한 확장 계획

---

## 프로젝트 관련 핵심 시사점

### 데이터셋 활용 시 주의사항

1. **시선 레이블 선택:** 이진 분류(`gaze_on_road`)만 사용할지, 9개 영역 분류를 사용할지 프로젝트 목표에 맞게 결정 필요
2. **클래스 불균형:** safe driving 비율이 압도적 → 언더샘플링 또는 오버샘플링 전략 고려
3. **모달리티 선택:** IR은 조명 변화에 강건하나, 실제 배포 시 RGB만 사용 가능할 수 있음
4. **파생 데이터셋:** dBehaviourMD는 13개 산만행동에 바로 사용 가능하므로 빠른 프로토타이핑에 적합

### 베이스라인 모델 참고

- CPU-only 환경에서도 실시간 가능한 경량 모델 아키텍처
- 다중 스트림 퓨전이 성능을 향상시키지만 복잡도와의 트레이드오프 존재
- 프로젝트의 실시간 요구사항에 맞는 퓨전 전략 선택 필요

---

## 참고 자료

- **논문 arXiv:** https://arxiv.org/abs/2008.12085
- **공식 웹사이트:** https://dmd.vicomtech.org/
- **GitHub 리포지토리:** https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset
- **Springer (ECCV 2020):** https://link.springer.com/chapter/10.1007/978-3-030-66823-5_23
