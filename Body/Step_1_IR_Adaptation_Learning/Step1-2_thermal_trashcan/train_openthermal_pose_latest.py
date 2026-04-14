# train_openthermal_pose_latest.py

import os
import yaml
import torch
from ultralytics import YOLO


# =========================================================
# 1) 하드코딩 설정
# =========================================================
DATASET_ROOT = r"C:\Users\hyi8402\Desktop\Capstone\Body\Step_1_IR_Adaptation_Learning\1.Dataset\OpenThermalPose"

# 학습 결과 저장 폴더
RUNS_ROOT = r"C:\Users\hyi8402\Desktop\Capstone\Body\Step_1_IR_Adaptation_Learning\2.Code\2.train\runs_pose"
RUN_NAME = "yolo11n_pose_openthermal_partial_ft"

# 최신 계열 경량 pose 모델
MODEL_NAME = "yolo11n-pose.pt"

# 이어학습 여부
RESUME = True

# 전이학습: 앞쪽 일부 레이어 고정
FREEZE_LAYERS = 10

# 학습 하이퍼파라미터
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16
WORKERS = 4

OPTIMIZER = "AdamW"
LR0 = 5e-4
LRF = 1e-2
WEIGHT_DECAY = 5e-4
PATIENCE = 20


# =========================================================
# 2) 실제 데이터 구조에 맞는 경로
#    OpenThermalPose/
#      train/images, train/labels
#      val/images,   val/labels
#      test/images,  test/labels
# =========================================================
TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, "train", "images")
VAL_IMG_DIR   = os.path.join(DATASET_ROOT, "val", "images")
TEST_IMG_DIR  = os.path.join(DATASET_ROOT, "test", "images")

TRAIN_LBL_DIR = os.path.join(DATASET_ROOT, "train", "labels")
VAL_LBL_DIR   = os.path.join(DATASET_ROOT, "val", "labels")
TEST_LBL_DIR  = os.path.join(DATASET_ROOT, "test", "labels")

YAML_PATH = os.path.join(DATASET_ROOT, "openthermal_pose.yaml")


def make_dataset_yaml():
    """
    Ultralytics pose 학습용 dataset yaml 생성.
    OpenThermal 라벨이 COCO 17-keypoint 기준이라고 가정.
    """
    os.makedirs(DATASET_ROOT, exist_ok=True)

    data = {
        "path": DATASET_ROOT,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",

        # COCO human pose: 17 keypoints, (x, y, visible)
        "kpt_shape": [17, 3],

        # COCO 좌우 반전 대응 인덱스
        "flip_idx": [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],

        "names": {
            0: "person"
        }
    }

    with open(YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    print(f"[INFO] dataset yaml saved: {YAML_PATH}")


def count_images(folder):
    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ])


def count_labels(folder):
    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith(".txt")
    ])


def sanity_check():
    """
    폴더 존재 여부와 파일 개수 확인
    """
    required_dirs = [
        TRAIN_IMG_DIR, VAL_IMG_DIR, TEST_IMG_DIR,
        TRAIN_LBL_DIR, VAL_LBL_DIR, TEST_LBL_DIR
    ]

    for p in required_dirs:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"[ERROR] Missing directory: {p}")

    print(f"[INFO] train images: {count_images(TRAIN_IMG_DIR)}")
    print(f"[INFO] val images:   {count_images(VAL_IMG_DIR)}")
    print(f"[INFO] test images:  {count_images(TEST_IMG_DIR)}")

    print(f"[INFO] train labels: {count_labels(TRAIN_LBL_DIR)}")
    print(f"[INFO] val labels:   {count_labels(VAL_LBL_DIR)}")
    print(f"[INFO] test labels:  {count_labels(TEST_LBL_DIR)}")

    if count_images(TRAIN_IMG_DIR) == 0:
        raise RuntimeError("[ERROR] No training images found.")
    if count_images(VAL_IMG_DIR) == 0:
        raise RuntimeError("[ERROR] No validation images found.")

    if count_labels(TRAIN_LBL_DIR) == 0:
        raise RuntimeError("[ERROR] No training labels found.")
    if count_labels(VAL_LBL_DIR) == 0:
        raise RuntimeError("[ERROR] No validation labels found.")


def main():
    print("[INFO] running file:", __file__)
    print("[INFO] DATASET_ROOT:", DATASET_ROOT)
    print("[INFO] TRAIN_IMG_DIR:", TRAIN_IMG_DIR)
    print("[INFO] TRAIN_LBL_DIR:", TRAIN_LBL_DIR)

    make_dataset_yaml()
    sanity_check()

    os.makedirs(RUNS_ROOT, exist_ok=True)

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")

    # -----------------------------------------------------
    # 모델 로드
    # -----------------------------------------------------
    if RESUME:
        last_pt = os.path.join(RUNS_ROOT, RUN_NAME, "weights", "last.pt")
        if not os.path.isfile(last_pt):
            raise FileNotFoundError(f"[ERROR] Resume requested, but not found: {last_pt}")
        model = YOLO(last_pt)
        print(f"[INFO] Resuming from: {last_pt}")
    else:
        model = YOLO(MODEL_NAME)
        print(f"[INFO] Using pretrained model: {MODEL_NAME}")

    # -----------------------------------------------------
    # 학습
    # -----------------------------------------------------
    results = model.train(
        data=YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=device,

        # transfer learning
        pretrained=True,
        freeze=FREEZE_LAYERS,

        # optimizer / scheduler 관련
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        weight_decay=WEIGHT_DECAY,

        # thermal 데이터이므로 색상 증강은 비활성화
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        # pose에 크게 무리 없는 기하 증강만 약하게
        degrees=5.0,
        translate=0.05,
        scale=0.15,
        shear=0.0,
        perspective=0.0,

        fliplr=0.5,
        flipud=0.0,

        mosaic=0.2,
        mixup=0.0,

        # 기타
        amp=True,
        cache=False,
        plots=True,
        patience=PATIENCE,
        verbose=False,

        project=RUNS_ROOT,
        name=RUN_NAME,
        exist_ok=True,
        resume=RESUME
    )

    print("[INFO] Training finished.")
    print(results)

    best_pt = os.path.join(RUNS_ROOT, RUN_NAME, "weights", "best.pt")
    last_pt = os.path.join(RUNS_ROOT, RUN_NAME, "weights", "last.pt")

    print(f"[INFO] best.pt exists: {os.path.isfile(best_pt)}")
    print(f"[INFO] last.pt exists: {os.path.isfile(last_pt)}")

    if os.path.isfile(best_pt):
        print(f"[INFO] best.pt = {best_pt}")
    if os.path.isfile(last_pt):
        print(f"[INFO] last.pt = {last_pt}")


if __name__ == "__main__":
    main()