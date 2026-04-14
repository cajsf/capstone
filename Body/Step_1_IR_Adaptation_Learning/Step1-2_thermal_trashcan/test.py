# test_openthermal_pose_latest.py

import os
import random
import shutil
import yaml
from ultralytics import YOLO


# =========================================================
# 1) 하드코딩 설정
# =========================================================
DATASET_ROOT = r"C:\Users\hyi8402\Desktop\Capstone\Body\Dataset\OpenThermalPose"

# 학습 때 사용한 run 이름과 동일하게 맞춰야 함
RUNS_ROOT = r"C:\Users\hyi8402\Desktop\Capstone\Body\Step_1_IR_Adaptation_Learning\2.Code\2.train\runs_pose"
RUN_NAME = "yolo11n_pose_openthermal_partial_ft"

# best.pt 경로
BEST_PT = os.path.join(RUNS_ROOT, RUN_NAME, "weights", "best.pt")

# dataset yaml 경로
YAML_PATH = os.path.join(DATASET_ROOT, "openthermal_pose.yaml")

# test 이미지 폴더
TEST_IMG_DIR = os.path.join(DATASET_ROOT, "test", "images")

# 예측 시각화 저장 폴더
PRED_SAVE_DIR = os.path.join(RUNS_ROOT, RUN_NAME, "test_predictions")

# 예측할 샘플 수
NUM_SAMPLE_IMAGES = 20

# 평가 / 추론 설정
IMG_SIZE = 640
BATCH_SIZE = 8
DEVICE = 0  # GPU 있으면 0, 없으면 "cpu"로 바꿔도 됨
CONF_THRES = 0.25


def ensure_yaml_exists():
    """
    학습 코드와 동일한 dataset yaml이 없으면 생성
    """
    if os.path.isfile(YAML_PATH):
        print(f"[INFO] YAML exists: {YAML_PATH}")
        return

    data = {
        "path": DATASET_ROOT,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "kpt_shape": [17, 3],
        "flip_idx": [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
        "names": {0: "person"},
    }

    with open(YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    print(f"[INFO] YAML created: {YAML_PATH}")


def sanity_check():
    if not os.path.isfile(BEST_PT):
        raise FileNotFoundError(f"[ERROR] best.pt not found: {BEST_PT}")

    if not os.path.isdir(TEST_IMG_DIR):
        raise FileNotFoundError(f"[ERROR] test image dir not found: {TEST_IMG_DIR}")

    images = [
        f for f in os.listdir(TEST_IMG_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]
    if len(images) == 0:
        raise RuntimeError("[ERROR] No test images found.")

    print(f"[INFO] best.pt: {BEST_PT}")
    print(f"[INFO] test images: {len(images)}")


def run_test_eval(model):
    """
    test split 전체 정량 평가
    """
    print("\n[INFO] Running quantitative evaluation on test split...\n")

    metrics = model.val(
        data=YAML_PATH,
        split="test",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        conf=CONF_THRES,
        plots=True,
        save_json=False,
        project=RUNS_ROOT,
        name=f"{RUN_NAME}_test_eval",
        exist_ok=True,
        verbose=True
    )

    print("\n[INFO] Test evaluation finished.")

    # 버전에 따라 속성명이 조금 달라질 수 있어서 안전하게 처리
    for attr in ["box", "pose"]:
        obj = getattr(metrics, attr, None)
        if obj is not None:
            print(f"[INFO] {attr} metrics available")
            for m in ["map", "map50", "map75"]:
                if hasattr(obj, m):
                    print(f"  {attr}.{m}: {getattr(obj, m)}")

    # 요약 출력
    if hasattr(metrics, "summary"):
        try:
            summary = metrics.summary()
            print("[INFO] summary():")
            print(summary)
        except Exception as e:
            print(f"[WARN] summary() failed: {e}")

    if hasattr(metrics, "results_dict"):
        print("[INFO] results_dict:")
        print(metrics.results_dict)

    return metrics


def run_sample_predictions(model):
    """
    test 이미지 일부에 대해 예측 후 시각화 저장
    """
    print("\n[INFO] Running sample predictions...\n")

    os.makedirs(PRED_SAVE_DIR, exist_ok=True)

    all_images = [
        os.path.join(TEST_IMG_DIR, f)
        for f in os.listdir(TEST_IMG_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]
    all_images.sort()

    sample_images = all_images if len(all_images) <= NUM_SAMPLE_IMAGES else random.sample(all_images, NUM_SAMPLE_IMAGES)

    # 예측
    results = model.predict(
        source=sample_images,
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        device=DEVICE,
        save=False,       # 직접 저장할 거라 False
        verbose=True
    )

    saved_count = 0

    for img_path, result in zip(sample_images, results):
        # 예측 overlay 이미지 생성
        plotted = result.plot()

        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_img_path = os.path.join(PRED_SAVE_DIR, f"{stem}_pred.jpg")

        # cv2 저장
        import cv2
        cv2.imwrite(out_img_path, plotted)

        # 원본도 같이 복사해 두면 비교 편함
        raw_copy_path = os.path.join(PRED_SAVE_DIR, f"{stem}_raw{os.path.splitext(img_path)[1]}")
        if not os.path.exists(raw_copy_path):
            shutil.copy2(img_path, raw_copy_path)

        # keypoints / boxes 간단 출력
        num_boxes = 0
        if getattr(result, "boxes", None) is not None:
            try:
                num_boxes = len(result.boxes)
            except Exception:
                pass

        print(f"[INFO] saved: {out_img_path} | detections: {num_boxes}")
        saved_count += 1

    print(f"\n[INFO] sample prediction images saved: {saved_count}")
    print(f"[INFO] save dir: {PRED_SAVE_DIR}")


def main():
    print("[INFO] running file:", __file__)
    print("[INFO] DATASET_ROOT:", DATASET_ROOT)
    print("[INFO] BEST_PT:", BEST_PT)

    ensure_yaml_exists()
    sanity_check()

    model = YOLO(BEST_PT)
    print("[INFO] Model loaded.")

    # 1) test split 정량 평가
    run_test_eval(model)

    # 2) 샘플 예측 시각화 저장
    run_sample_predictions(model)

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()