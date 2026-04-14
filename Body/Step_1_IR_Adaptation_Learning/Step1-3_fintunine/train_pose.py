import os
import yaml
import torch
from ultralytics import YOLO


# =========================================================
# 1) 하드코딩 설정
# =========================================================
DATASET_ROOT = r"C:\Users\hyi8402\Downloads\driveandact\yolopose_dataset"

RUNS_ROOT = os.path.join(DATASET_ROOT, "runs_pose")
RUN_NAME = "yolo11n_pose_driveandact_folder_split_17"

MODEL_NAME = "yolo11n-pose.pt"
RESUME = False
FREEZE_LAYERS = 10

EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16
WORKERS = 4

OPTIMIZER = "AdamW"
LR0 = 5e-4
LRF = 1e-2
WEIGHT_DECAY = 5e-4
PATIENCE = 15

# YAML 저장 위치
YAML_PATH = os.path.join(DATASET_ROOT, "dataset.yaml")

# =========================================================
# 2) 원본 데이터 구조
# =========================================================
SRC_TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, "images", "train")
SRC_VAL_IMG_DIR   = os.path.join(DATASET_ROOT, "images", "val")

SRC_TRAIN_LBL_DIR = os.path.join(DATASET_ROOT, "labels", "train")
SRC_VAL_LBL_DIR   = os.path.join(DATASET_ROOT, "labels", "val")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def count_valid_pairs(img_dir: str, lbl_dir: str):
    """
    이미지-라벨 쌍이 모두 있는 샘플 개수 확인
    """
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"[ERROR] Missing image dir: {img_dir}")
    if not os.path.isdir(lbl_dir):
        raise FileNotFoundError(f"[ERROR] Missing label dir: {lbl_dir}")

    image_files = [f for f in os.listdir(img_dir) if is_image_file(f)]
    valid_count = 0
    missing_labels = []

    for img_name in image_files:
        stem = os.path.splitext(img_name)[0]
        lbl_name = stem + ".txt"
        lbl_path = os.path.join(lbl_dir, lbl_name)

        if os.path.isfile(lbl_path):
            valid_count += 1
        else:
            missing_labels.append(lbl_name)

    print(f"[INFO] {img_dir} -> valid pairs: {valid_count}")
    if missing_labels:
        print(f"[WARN] missing labels: {len(missing_labels)} (first 10)")
        for x in missing_labels[:10]:
            print("   ", x)

    return valid_count


def make_dataset_yaml():
    """
    폴더 기반 split 구조용 YAML 생성
    """
    data = {
        "path": DATASET_ROOT,
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [17, 3],
        "flip_idx": [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
        "names": {
            0: "person"
        }
    }

    with open(YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    print(f"[INFO] dataset yaml saved: {YAML_PATH}")


def sanity_check():
    """
    폴더 구조 및 train/val 쌍 개수 점검
    """
    for p in [SRC_TRAIN_IMG_DIR, SRC_VAL_IMG_DIR, SRC_TRAIN_LBL_DIR, SRC_VAL_LBL_DIR]:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"[ERROR] Missing directory: {p}")

    train_count = count_valid_pairs(SRC_TRAIN_IMG_DIR, SRC_TRAIN_LBL_DIR)
    val_count = count_valid_pairs(SRC_VAL_IMG_DIR, SRC_VAL_LBL_DIR)

    if train_count == 0:
        raise RuntimeError("[ERROR] No valid train image-label pairs found.")
    if val_count == 0:
        raise RuntimeError("[ERROR] No valid val image-label pairs found.")

    print(f"[INFO] train valid pairs: {train_count}")
    print(f"[INFO] val valid pairs:   {val_count}")


def main():
    print("[INFO] running file:", __file__)
    print("[INFO] DATASET_ROOT:", DATASET_ROOT)
    print("[INFO] SRC_TRAIN_IMG_DIR:", SRC_TRAIN_IMG_DIR)
    print("[INFO] SRC_VAL_IMG_DIR:", SRC_VAL_IMG_DIR)
    print("[INFO] SRC_TRAIN_LBL_DIR:", SRC_TRAIN_LBL_DIR)
    print("[INFO] SRC_VAL_LBL_DIR:", SRC_VAL_LBL_DIR)

    ensure_dir(RUNS_ROOT)
    make_dataset_yaml()
    sanity_check()

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")

    if RESUME:
        last_pt = os.path.join(RUNS_ROOT, RUN_NAME, "weights", "last.pt")
        if not os.path.isfile(last_pt):
            raise FileNotFoundError(f"[ERROR] Resume requested, but not found: {last_pt}")
        model = YOLO(last_pt)
        print(f"[INFO] Resuming from: {last_pt}")
    else:
        model = YOLO(MODEL_NAME)
        print(f"[INFO] Using pretrained model: {MODEL_NAME}")

    results = model.train(
        data=YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=device,

        pretrained=True,
        freeze=FREEZE_LAYERS,

        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        weight_decay=WEIGHT_DECAY,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        degrees=5.0,
        translate=0.05,
        scale=0.15,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.2,
        mixup=0.0,

        amp=True,
        cache=True,
        plots=True,
        verbose=False,
        patience=PATIENCE,

        project=RUNS_ROOT,
        name=RUN_NAME,
        exist_ok=True,
        resume=RESUME,
        save=True,
        save_period=-1,
    )

    print("[INFO] Training finished.")
    print(results)

    best_pt = os.path.join(RUNS_ROOT, RUN_NAME, "weights", "best.pt")
    last_pt = os.path.join(RUNS_ROOT, RUN_NAME, "weights", "last.pt")

    print(f"[INFO] best.pt exists: {os.path.isfile(best_pt)}")
    print(f"[INFO] last.pt exists: {os.path.isfile(last_pt)}")

    if os.path.isfile(best_pt):
        print(f"[INFO] best.pt = {best_pt}")
        print("[INFO] Running final evaluation on val split...")
        best_model = YOLO(best_pt)
        metrics = best_model.val(
            data=YAML_PATH,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            workers=WORKERS,
            device=device,
            plots=True,
        )
        print("[INFO] Val metrics:")
        print(metrics)

    if os.path.isfile(last_pt):
        print(f"[INFO] last.pt = {last_pt}")


if __name__ == "__main__":
    main()