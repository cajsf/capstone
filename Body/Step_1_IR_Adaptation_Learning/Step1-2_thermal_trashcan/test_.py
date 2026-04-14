import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


# =========================================================
# 1) 하드코딩 설정
# =========================================================
MODEL_PATH = r"C:\Users\hyi8402\Downloads\driveandact\yolo_pose_inner_mirror_5fps\runs_pose\yolo11n_pose_driveandact_partial_ft\weights\best.pt"
VIDEO_PATH = r"C:\Users\hyi8402\Downloads\dmd\gZ\34\s8\gZ_34_s4_2019-04-03T09;47;42+02;00_ir_body.mp4"

OUTPUT_DIR = r"C:\Users\hyi8402\Desktop\pose_test_result"
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "pose_overlay.mp4")
SAVE_TXT = True
TXT_DIR = os.path.join(OUTPUT_DIR, "keypoints_txt")

# 보기 옵션
SHOW_WINDOW = True          # 화면 띄울지
WINDOW_NAME = "Pose Test"
DISPLAY_WIDTH = 1280        # 화면 표시용 리사이즈 폭 (None이면 원본)
CONF_THRES = 0.25

# 저장 옵션
SAVE_VIDEO = True
VIDEO_CODEC = "mp4v"        # "mp4v" 또는 "XVID"

# keypoint 그리기 옵션
DRAW_KPT_RADIUS = 4
DRAW_LINE_THICKNESS = 2
DRAW_BOX = True

# COCO-17 skeleton 연결
SKELETON = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]


# =========================================================
# 2) 유틸
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def draw_pose(frame, xy, conf=None, box=None):
    """
    xy: (17, 2)
    conf: (17,) or None
    box: [x1, y1, x2, y2] or None
    """
    h, w = frame.shape[:2]

    valid = []
    for i in range(len(xy)):
        x, y = xy[i]
        c = conf[i] if conf is not None else 1.0
        ok = (c > 0.0) and (0 <= x < w) and (0 <= y < h)
        valid.append(ok)

    # skeleton lines
    for a, b in SKELETON:
        if a < len(xy) and b < len(xy) and valid[a] and valid[b]:
            x1, y1 = map(int, xy[a])
            x2, y2 = map(int, xy[b])
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), DRAW_LINE_THICKNESS)

    # keypoints
    for i in range(len(xy)):
        if valid[i]:
            x, y = map(int, xy[i])
            cv2.circle(frame, (x, y), DRAW_KPT_RADIUS, (0, 0, 255), -1)

    # bbox
    if DRAW_BOX and box is not None:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return frame


def save_keypoints_txt(txt_path, xy, conf=None, box=None):
    """
    사람이 여러 명일 수 있으므로 1명당 한 블록 저장.
    형식:
    x1 y1 c1 x2 y2 c2 ... x17 y17 c17 | box x1 y1 x2 y2
    """
    with open(txt_path, "a", encoding="utf-8") as f:
        vals = []
        for i in range(len(xy)):
            x, y = xy[i]
            c = conf[i] if conf is not None else 1.0
            vals.extend([f"{x:.4f}", f"{y:.4f}", f"{c:.4f}"])

        if box is not None:
            bx = " ".join([f"{v:.4f}" for v in box])
            f.write(" ".join(vals) + f" | box {bx}\n")
        else:
            f.write(" ".join(vals) + "\n")


def resize_for_display(frame, display_width=None):
    if display_width is None:
        return frame
    h, w = frame.shape[:2]
    if w <= display_width:
        return frame
    scale = display_width / w
    new_h = int(h * scale)
    return cv2.resize(frame, (display_width, new_h))


# =========================================================
# 3) 메인
# =========================================================
def main():
    ensure_dir(OUTPUT_DIR)
    if SAVE_TXT:
        ensure_dir(TXT_DIR)

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"[ERROR] MODEL_PATH not found: {MODEL_PATH}")
    if not os.path.isfile(VIDEO_PATH):
        raise FileNotFoundError(f"[ERROR] VIDEO_PATH not found: {VIDEO_PATH}")

    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] Failed to open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video: {VIDEO_PATH}")
    print(f"[INFO] FPS: {fps}")
    print(f"[INFO] Frames: {total_frames}")
    print(f"[INFO] Size: {width}x{height}")

    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"[ERROR] Failed to create writer: {OUTPUT_VIDEO_PATH}")

    pbar = tqdm(total=total_frames, desc="Pose inference", ncols=100)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # -------------------------------------------------
        # 추론
        # -------------------------------------------------
        results = model.predict(
            source=frame,
            conf=CONF_THRES,
            verbose=False,
            save=False
        )

        vis = frame.copy()

        if len(results) > 0:
            r = results[0]

            # boxes
            boxes = None
            if r.boxes is not None and r.boxes.xyxy is not None:
                boxes = r.boxes.xyxy.cpu().numpy()

            # keypoints
            if r.keypoints is not None and r.keypoints.xy is not None:
                kpts_xy = r.keypoints.xy.cpu().numpy()   # (N, 17, 2)

                kpts_conf = None
                if hasattr(r.keypoints, "conf") and r.keypoints.conf is not None:
                    kpts_conf = r.keypoints.conf.cpu().numpy()  # (N, 17)

                num_person = kpts_xy.shape[0]

                txt_path = None
                if SAVE_TXT:
                    txt_path = os.path.join(TXT_DIR, f"frame_{frame_idx:06d}.txt")
                    if os.path.isfile(txt_path):
                        os.remove(txt_path)

                for i in range(num_person):
                    xy = kpts_xy[i]
                    conf = kpts_conf[i] if kpts_conf is not None else None
                    box = boxes[i] if boxes is not None and i < len(boxes) else None

                    vis = draw_pose(vis, xy, conf=conf, box=box)

                    if SAVE_TXT:
                        save_keypoints_txt(txt_path, xy, conf=conf, box=box)

                cv2.putText(
                    vis,
                    f"Persons: {num_person}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        # -------------------------------------------------
        # 저장 / 표시
        # -------------------------------------------------
        if SAVE_VIDEO and writer is not None:
            writer.write(vis)

        if SHOW_WINDOW:
            disp = resize_for_display(vis, DISPLAY_WIDTH)
            cv2.imshow(WINDOW_NAME, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("[INFO] Interrupted by user.")
                break

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print("[INFO] Done.")
    if SAVE_VIDEO:
        print(f"[INFO] Saved video: {OUTPUT_VIDEO_PATH}")
    if SAVE_TXT:
        print(f"[INFO] Saved keypoints txt dir: {TXT_DIR}")


if __name__ == "__main__":
    main()