import os
import random
import cv2
import numpy as np

# COCO skeleton
SKELETON = [
    (15,13),(13,11),(16,14),(14,12),(11,12),
    (5,11),(6,12),(5,6),(5,7),(6,8),
    (7,9),(8,10),(1,2),(0,1),(0,2),
    (1,3),(2,4),(3,5),(4,6)
]

def read_image_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def visualize(image_path, label_path):
    img = read_image_unicode(image_path)
    if img is None:
        print("[ERROR] 이미지 로드 실패")
        return

    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        data = list(map(float, line.strip().split()))

        cx, cy, bw, bh = data[1:5]

        # bbox
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)

        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

        # keypoints
        kpts = data[5:]
        pts = []

        for i in range(0, len(kpts), 3):
            x, y, v = kpts[i], kpts[i+1], kpts[i+2]

            if v > 0:
                px = int(x * w)
                py = int(y * h)
                pts.append((px, py))
                cv2.circle(img, (px, py), 3, (0,0,255), -1)
            else:
                pts.append(None)

        # skeleton
        for i, j in SKELETON:
            if i < len(pts) and j < len(pts):
                if pts[i] and pts[j]:
                    cv2.line(img, pts[i], pts[j], (255,0,0), 2)

    cv2.imshow("pose check", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pick_valid_sample(image_dir, label_dir):
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    random.shuffle(label_files)

    for f in label_files:
        img_name = f.replace(".txt", ".jpg")

        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, f)

        if os.path.exists(img_path):
            return img_path, label_path

    return None, None


if __name__ == "__main__":
    base_dir = r"C:\Users\hyi8402\Desktop\Capstone\Dataset"

    image_dir = os.path.join(base_dir, "train2017")
    label_dir = os.path.join(base_dir, "labels", "train2017")

    img_path, label_path = pick_valid_sample(image_dir, label_dir)

    if img_path is None:
        print("[ERROR] 이미지-라벨 매칭 없음 → 라벨/이미지 불일치")
    else:
        print("[OK] sample:", os.path.basename(img_path))
        visualize(img_path, label_path)