import os
import random
import cv2
import yaml
from ultralytics import YOLO

# =========================================================
# 1) OpenThermalPose 경로 하드코딩
# =========================================================
DATASET_ROOT = r"C:\Users\hyi8402\Desktop\Capstone\Body\Step_1_IR_Adaptation_Learning\1.Dataset\OpenThermalPose"

# 비교할 두 모델 경로
MODEL_PATH_1 = r"C:\Users\hyi8402\Desktop\Capstone\Body\Step_1_IR_Adaptation_Learning\2.Code\2.train\runs_pose_0411_0048_finetuning\yolo11n_pose_openthermal_partial_ft\weights\best.pt"
MODEL_PATH_2 = r"C:\Users\hyi8402\Desktop\Capstone\Body\Step_1_IR_Adaptation_Learning\2.Code\2.train\runs_pose_0410_2041_finetuning\yolo11n_pose_openthermal_partial_ft\weights\best.pt"

# OpenThermalPose test 이미지
TEST_IMG_DIR = os.path.join(DATASET_ROOT, "test", "images")

# YAML 경로
YAML_PATH = r"C:\Users\hyi8402\Desktop\Capstone\Body\Step_1_IR_Adaptation_Learning\2.Code\3.result\openthermal_pose.yaml"

# 비교 결과 저장 폴더
SAVE_ROOT = r"C:\Users\hyi8402\Desktop\Capstone\Body\Step_1_IR_Adaptation_Learning\3.Result\partialunfreezefine_vs_fullunfreezefine"

IMG_SIZE = 640
DEVICE = 0
NUM_SAMPLE = 20


# =========================================================
# 2) dataset yaml 보장
# =========================================================
def ensure_yaml():
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


# =========================================================
# 3) 정량 평가
# =========================================================
def evaluate_model(model, name):
    print(f"\n========== {name} : OpenThermalPose TEST EVAL ==========\n")

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
def visualize_comparison(model1, model2, name1="model1", name2="model2"):
    os.makedirs(SAVE_ROOT, exist_ok=True)

    imgs = [
        os.path.join(TEST_IMG_DIR, f)
        for f in os.listdir(TEST_IMG_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]

    if len(imgs) == 0:
        raise RuntimeError("[ERROR] No test images found.")

    sample_imgs = random.sample(imgs, min(NUM_SAMPLE, len(imgs)))

    for img_path in sample_imgs:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] failed to read image: {img_path}")
            continue

        res1 = model1.predict(img, imgsz=IMG_SIZE, device=DEVICE, verbose=False)[0]
        res2 = model2.predict(img, imgsz=IMG_SIZE, device=DEVICE, verbose=False)[0]

        plot1 = res1.plot()
        plot2 = res2.plot()

        # 위에 모델명 텍스트 붙이기
        h1, w1 = plot1.shape[:2]
        h2, w2 = plot2.shape[:2]

        top_pad = 40
        canvas1 = cv2.copyMakeBorder(plot1, top_pad, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        canvas2 = cv2.copyMakeBorder(plot2, top_pad, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        cv2.putText(canvas1, name1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(canvas2, name2, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        combined = cv2.hconcat([canvas1, canvas2])

        save_path = os.path.join(SAVE_ROOT, f"compare_{os.path.basename(img_path)}")
        cv2.imwrite(save_path, combined)

    print(f"[INFO] comparison images saved to: {SAVE_ROOT}")


# =========================================================
# 5) main
# =========================================================
def main():
    ensure_yaml()

    if not os.path.isfile(MODEL_PATH_1):
        raise FileNotFoundError(f"[ERROR] MODEL_PATH_1 not found: {MODEL_PATH_1}")
    if not os.path.isfile(MODEL_PATH_2):
        raise FileNotFoundError(f"[ERROR] MODEL_PATH_2 not found: {MODEL_PATH_2}")

    print("[INFO] Loading models...")
    model1 = YOLO(MODEL_PATH_1)
    model2 = YOLO(MODEL_PATH_2)
    print("[INFO] Models loaded.")

    # 1) OpenThermalPose test split으로 정량 비교
    res1 = evaluate_model(model1, "MODEL_1")
    res2 = evaluate_model(model2, "MODEL_2")

    print("\n========== SUMMARY ==========\n")
    print("Pose mAP50-95 비교:")
    print(f"MODEL_1 : {res1['metrics/mAP50-95(P)']:.4f}")
    print(f"MODEL_2 : {res2['metrics/mAP50-95(P)']:.4f}")
    print(f"Difference (MODEL_2 - MODEL_1): {res2['metrics/mAP50-95(P)'] - res1['metrics/mAP50-95(P)']:+.4f}")

    # 2) OpenThermalPose test 이미지 샘플 시각화 비교
    visualize_comparison(model1, model2, name1="MODEL_1", name2="MODEL_2")


if __name__ == "__main__":
    main()