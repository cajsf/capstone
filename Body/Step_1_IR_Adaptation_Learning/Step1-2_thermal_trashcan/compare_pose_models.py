# compare_pose_models.py

import os
import random
import shutil
import yaml
import cv2
from ultralytics import YOLO

# =========================================================
# 1) 경로 설정
# =========================================================
DATASET_ROOT = r"C:\Users\hyi8402\Desktop\Capstone\Body\Dataset\OpenThermalPose"

YAML_PATH = os.path.join(DATASET_ROOT, "openthermal_pose.yaml")
TEST_IMG_DIR = os.path.join(DATASET_ROOT, "test", "images")

# 모델 1: pretrained (fine-tuning 안한 모델)
MODEL_PRETRAINED = "yolo11n-pose.pt"

# 모델 2: fine-tuned 모델
MODEL_FINETUNED = os.path.join(
    r"C:\Users\hyi8402\Desktop\Capstone\Body\Step_1_IR_Adaptation_Learning\2.Code\2.train",
    "runs_pose",
    "yolo11n_pose_openthermal_partial_ft",
    "weights",
    "best.pt"
)

# 결과 저장 폴더
SAVE_ROOT = os.path.join(DATASET_ROOT, "compare_results")

NUM_SAMPLE = 20
IMG_SIZE = 640
DEVICE = 0


# =========================================================
# 2) YAML 생성 (없을 경우)
# =========================================================
def ensure_yaml():
    if os.path.isfile(YAML_PATH):
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

    print("[INFO] YAML created")


# =========================================================
# 3) 정량 평가
# =========================================================
def evaluate_model(model, name):
    print(f"\n========== {name} TEST EVAL ==========\n")

    metrics = model.val(
        data=YAML_PATH,
        split="test",
        imgsz=IMG_SIZE,
        device=DEVICE,
        verbose=False
    )

    res = metrics.results_dict

    print(f"[{name}]")
    print(f"Pose mAP50:      {res['metrics/mAP50(P)']:.4f}")
    print(f"Pose mAP50-95:   {res['metrics/mAP50-95(P)']:.4f}")
    print(f"Box  mAP50:      {res['metrics/mAP50(B)']:.4f}")
    print(f"Box  mAP50-95:   {res['metrics/mAP50-95(B)']:.4f}")

    return res


# =========================================================
# 4) 시각화 비교
# =========================================================
def visualize_comparison(model1, model2):
    os.makedirs(SAVE_ROOT, exist_ok=True)

    imgs = [
        os.path.join(TEST_IMG_DIR, f)
        for f in os.listdir(TEST_IMG_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    sample_imgs = random.sample(imgs, min(NUM_SAMPLE, len(imgs)))

    for img_path in sample_imgs:
        img = cv2.imread(img_path)

        res1 = model1.predict(img, imgsz=IMG_SIZE, verbose=False)[0]
        res2 = model2.predict(img, imgsz=IMG_SIZE, verbose=False)[0]

        plot1 = res1.plot()
        plot2 = res2.plot()

        # 좌우 비교 이미지 생성
        combined = cv2.hconcat([plot1, plot2])

        name = os.path.basename(img_path)
        save_path = os.path.join(SAVE_ROOT, f"compare_{name}")

        cv2.imwrite(save_path, combined)

    print(f"[INFO] comparison images saved to: {SAVE_ROOT}")


# =========================================================
# 5) main
# =========================================================
def main():
    ensure_yaml()

    print("[INFO] Loading models...")

    model_pre = YOLO(MODEL_PRETRAINED)
    model_ft = YOLO(MODEL_FINETUNED)

    print("[INFO] Models loaded.")

    # 1. 정량 비교
    res_pre = evaluate_model(model_pre, "PRETRAINED")
    res_ft = evaluate_model(model_ft, "FINE-TUNED")

    print("\n========== SUMMARY ==========\n")

    print("Pose mAP50-95 비교:")
    print(f"Pretrained : {res_pre['metrics/mAP50-95(P)']:.4f}")
    print(f"Finetuned  : {res_ft['metrics/mAP50-95(P)']:.4f}")

    diff = res_ft['metrics/mAP50-95(P)'] - res_pre['metrics/mAP50-95(P)']
    print(f"Difference : {diff:+.4f}")

    # 2. 시각화 비교
    visualize_comparison(model_pre, model_ft)


if __name__ == "__main__":
    main()