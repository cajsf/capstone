import os

# =========================================================
# 설정
# =========================================================
SRC_ROOT = r"C:\Users\hyi8402\Downloads\driveandact\yolo_pose_inner_mirror_5fps\labels"
DST_ROOT = r"C:\Users\hyi8402\Downloads\driveandact\yolo_pose_inner_mirror_5fps\labels_17"

# Body25 → COCO17 매핑
MAP_25_TO_17 = [
    0,   # nose
    16,  # left_eye
    15,  # right_eye
    18,  # left_ear
    17,  # right_ear
    5,   # left_shoulder
    2,   # right_shoulder
    6,   # left_elbow
    3,   # right_elbow
    7,   # left_wrist
    4,   # right_wrist
    12,  # left_hip
    9,   # right_hip
    13,  # left_knee
    10,  # right_knee
    14,  # left_ankle
    11   # right_ankle
]


# =========================================================
# 변환 함수
# =========================================================
def convert_file(src_path, dst_path):
    with open(src_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:
        vals = list(map(float, line.strip().split()))

        if len(vals) < 5:
            continue

        cls = vals[0]
        bbox = vals[1:5]
        kpts = vals[5:]

        # Body25 체크 (75개)
        if len(kpts) != 75:
            print(f"[SKIP] Not Body25: {src_path}")
            continue

        # (25,3) 형태
        kpts = [kpts[i:i+3] for i in range(0, 75, 3)]

        # 17개로 변환
        new_kpts = []
        for idx in MAP_25_TO_17:
            new_kpts.extend(kpts[idx])

        new_line = [cls] + bbox + new_kpts
        new_lines.append(" ".join(map(str, new_line)))

    if new_lines:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))


# =========================================================
# 전체 폴더 처리
# =========================================================
def process_split(split):
    src_dir = os.path.join(SRC_ROOT, split)
    dst_dir = os.path.join(DST_ROOT, split)

    if not os.path.isdir(src_dir):
        print(f"[WARN] Missing split: {split}")
        return

    files = [f for f in os.listdir(src_dir) if f.endswith(".txt")]

    print(f"[INFO] Processing {split}: {len(files)} files")

    for fname in files:
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        convert_file(src_path, dst_path)


def main():
    for split in ["train", "val"]:
        process_split(split)

    print("[INFO] Done converting to COCO17 format.")


if __name__ == "__main__":
    main()