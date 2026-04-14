import os
import random
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


# =========================================================
# 설정
# =========================================================
DATASET_ROOT = r"C:\Users\hyi8402\Downloads\driveandact\yolo_pose_inner_mirror_5fps\prepared_split"
IMG_DIR = os.path.join(DATASET_ROOT, "images", "train")
LBL_DIR = os.path.join(DATASET_ROOT, "labels", "train")

MODEL_PATH = r"C:\Users\hyi8402\Downloads\driveandact\yolo_pose_inner_mirror_5fps\runs_pose\yolo11n_pose_driveandact_partial_ft\weights\best.pt"

OUTPUT_DIR = r"C:\Users\hyi8402\Desktop\pose_debug_100"
NUM_SAMPLES = 100
CONF_THRES = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# skeleton 정의 (COCO 17)
# =========================================================
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),
    (5,7),(7,9),
    (6,8),(8,10),
    (5,11),(6,12),
    (11,12),
    (11,13),(13,15),
    (12,14),(14,16)
]


# =========================================================
# GT 그리기
# =========================================================
def draw_gt(img, label_path):
    h, w = img.shape[:2]

    if not os.path.isfile(label_path):
        return img

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        vals = list(map(float, line.strip().split()))
        if len(vals) < 8:
            continue

        # bbox
        cx, cy, bw, bh = vals[1:5]
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 남은 값 개수로 keypoint 수 자동 추론
        kpt_vals = vals[5:]
        if len(kpt_vals) % 3 != 0:
            print(f"[WARN] Invalid label format: {label_path} | len(kpt_vals)={len(kpt_vals)}")
            continue

        num_kpts = len(kpt_vals) // 3
        kpts = np.array(kpt_vals, dtype=np.float32).reshape(num_kpts, 3)

        # 일단 선 연결 없이 점만 그림
        for x, y, v in kpts:
            if v > 0:
                px = int(x * w)
                py = int(y * h)
                cv2.circle(img, (px, py), 3, (255, 0, 0), -1)

        cv2.putText(
            img,
            f"GT kpts={num_kpts}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )

    return img


# =========================================================
# Prediction 그리기
# =========================================================
def draw_pred(img, result):
    if result.keypoints is None or result.keypoints.xy is None:
        return img

    kpts_xy = result.keypoints.xy.cpu().numpy()

    for person in kpts_xy:
        for x, y in person:
            cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

    return img


# =========================================================
# 메인
# =========================================================
def main():
    model = YOLO(MODEL_PATH)

    img_files = [f for f in os.listdir(IMG_DIR)
                 if f.lower().endswith((".jpg",".png",".jpeg"))]

    samples = random.sample(img_files, min(NUM_SAMPLES, len(img_files)))

    print(f"[INFO] Sampling {len(samples)} images")

    for idx, fname in enumerate(tqdm(samples)):
        img_path = os.path.join(IMG_DIR, fname)
        lbl_path = os.path.join(LBL_DIR, os.path.splitext(fname)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue

        vis = img.copy()

        # -------------------------
        # GT (파란색)
        # -------------------------
        vis = draw_gt(vis, lbl_path)

        # -------------------------
        # Prediction (초록색)
        # -------------------------
        results = model.predict(
            source=img,
            conf=CONF_THRES,
            verbose=False
        )

        if len(results) > 0:
            vis = draw_pred(vis, results[0])

        # -------------------------
        # 저장
        # -------------------------
        out_path = os.path.join(OUTPUT_DIR, f"{idx:03d}.jpg")
        cv2.imwrite(out_path, vis)

    print(f"[INFO] Done. Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()