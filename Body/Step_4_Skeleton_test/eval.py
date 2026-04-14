import os
import random
import cv2
import numpy as np
from ultralytics import YOLO

# ===== 경로 설정 =====
model_path = r"C:\Users\hyi8402\Desktop\Capstone\Code\2.code\runs\pose\runs_pose\pose_debug5\weights\best.pt"
img_dir = r"C:\Users\hyi8402\Desktop\Capstone\Dataset\images\val2017"
label_dir = r"C:\Users\hyi8402\Desktop\Capstone\Dataset\labels\val2017"

# COCO 17-keypoint skeleton (0-index)
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6)
]

def read_image_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def pick_image_with_person(image_dir, label_dir):
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    random.shuffle(label_files)

    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)

        # 빈 라벨 제외
        if os.path.getsize(label_path) == 0:
            continue

        image_name = os.path.splitext(label_file)[0] + ".jpg"
        image_path = os.path.join(image_dir, image_name)

        if os.path.exists(image_path):
            return image_path, label_path

    return None, None

def draw_gt(img, label_path):
    h, w = img.shape[:2]

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        data = list(map(float, line.strip().split()))
        if len(data) < 5 + 17 * 3:
            continue

        # bbox
        cx, cy, bw, bh = data[1:5]
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # keypoints
        kpts = data[5:]
        pts = []
        for i in range(0, len(kpts), 3):
            x = int(kpts[i] * w)
            y = int(kpts[i + 1] * h)
            v = kpts[i + 2]
            if v > 0:
                pts.append((x, y))
                cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
            else:
                pts.append(None)

        for i, j in SKELETON:
            if i < len(pts) and j < len(pts):
                if pts[i] is not None and pts[j] is not None:
                    cv2.line(img, pts[i], pts[j], (0, 255, 0), 2)

    return img

def draw_pred(img, results):
    for r in results:
        if r.keypoints is None or r.keypoints.xy is None:
            continue

        persons = r.keypoints.xy.cpu().numpy()

        for person in persons:
            pts = []
            for x, y in person:
                x, y = int(x), int(y)
                pts.append((x, y))
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

            for i, j in SKELETON:
                if i < len(pts) and j < len(pts):
                    cv2.line(img, pts[i], pts[j], (0, 0, 255), 2)

    return img

def main():
    image_path, label_path = pick_image_with_person(img_dir, label_dir)

    if image_path is None:
        print("[ERROR] 사람 있는 val 이미지-라벨 쌍을 찾지 못함")
        return

    print("[IMAGE]", image_path)
    print("[LABEL]", label_path)

    model = YOLO(model_path)

    img = read_image_unicode(image_path)
    if img is None:
        print("[ERROR] 이미지 로드 실패")
        return

    vis = img.copy()
    vis = draw_gt(vis, label_path)

    results = model.predict(source=image_path, conf=0.25, verbose=False)
    vis = draw_pred(vis, results)

    cv2.imshow("GT=Green / Pred=Red", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()