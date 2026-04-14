# test_yolopose_eval.py
import os
from pathlib import Path
from multiprocessing import freeze_support
from ultralytics import YOLO

# =========================
# 1. 사용자 설정
# =========================
TEST_ROOT = r"C:\Users\hyi8402\Desktop\dmd_labeling\dmd_split\test"
MODEL_PATH = r"C:\Users\hyi8402\Desktop\Capstone\Body\Step_1_IR_Adaptation_Learning\Step1-1_COCO_2_YOLOPOSE\COCO_POSE_result\weights\best.pt"
OUT_DIR = r"C:\Users\hyi8402\Desktop\Capstone\Body\Step_1_IR_Adaptation_Learning\Step1-3_fintunine\test_results"

KPT_SHAPE = [17, 3]
CLASS_NAMES = ["person"]

IMG_SIZE = 640
BATCH = 16
CONF = 0.001
IOU = 0.6
DEVICE = 0
WORKERS = 0   # Windows에서는 우선 0으로 두는 게 가장 안전함

def main():
    test_root = Path(TEST_ROOT)
    images_dir = test_root / "images"
    labels_dir = test_root / "labels"
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not test_root.exists():
        raise FileNotFoundError(f"[ERROR] TEST_ROOT 없음: {test_root}")
    if not images_dir.exists():
        raise FileNotFoundError(f"[ERROR] images 폴더 없음: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"[ERROR] labels 폴더 없음: {labels_dir}")
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"[ERROR] 모델 파일 없음: {MODEL_PATH}")

    yaml_path = out_dir / "dmd_pose_test.yaml"

    yaml_text = f"""path: {test_root.parent.as_posix()}
train: {test_root.as_posix()}/images
val: {test_root.as_posix()}/images
test: {test_root.as_posix()}/images

kpt_shape: {KPT_SHAPE}
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

names:
"""
    for i, name in enumerate(CLASS_NAMES):
        yaml_text += f"  {i}: {name}\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print(f"[INFO] data.yaml 저장 완료: {yaml_path}")

    print(f"[INFO] 모델 로드 중: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print("[INFO] test evaluation 시작")
    results = model.val(
        data=str(yaml_path),
        split="test",
        imgsz=IMG_SIZE,
        batch=BATCH,
        conf=CONF,
        iou=IOU,
        device=DEVICE,
        workers=WORKERS,
        project=str(out_dir),
        name="yolopose_test_eval",
        save_json=False,
        plots=True,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("[TEST EVALUATION RESULT]")
    print("=" * 60)

    try:
        print(f"Box  mAP50      : {results.box.map50:.4f}")
        print(f"Box  mAP50-95   : {results.box.map:.4f}")
        print(f"Pose mAP50      : {results.pose.map50:.4f}")
        print(f"Pose mAP50-95   : {results.pose.map:.4f}")
    except Exception as e:
        print("[WARN] metric 출력 실패")
        print(e)
        print(results)

if __name__ == "__main__":
    freeze_support()
    main()