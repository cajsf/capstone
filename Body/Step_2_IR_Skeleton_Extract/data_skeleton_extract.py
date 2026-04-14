import json
import shutil
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO


# =========================
# 설정
# =========================
MODEL_PATH = r"C:\Users\hyi8402\Downloads\best.pt"
INPUT_ROOT = r"C:\Users\hyi8402\Downloads\dmd"
OUTPUT_ROOT = r"C:\Users\hyi8402\Downloads\dmd_finetune"

CONF_THRES = 0.25
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_G_FOLDERS = {"gA","gB","gC","gE","gF","gZ"}


# =========================
# 유틸
# =========================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def is_number_folder(path: Path) -> bool:
    return path.is_dir() and path.name.isdigit()


def is_image_folder(path: Path) -> bool:
    return path.is_dir() and path.name.endswith("_image")


def list_image_files(folder: Path):
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: p.name)
    return files


def format_seconds(sec: float) -> str:
    sec = int(round(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h}시간 {m}분 {s}초"
    if m > 0:
        return f"{m}분 {s}초"
    return f"{s}초"


def copy_annotation_if_exists(src_clip_dir: Path, dst_clip_dir: Path):
    src_ann = src_clip_dir / "annotation.json"
    if src_ann.exists():
        shutil.copy2(src_ann, dst_clip_dir / "annotation.json")
        with open(src_ann, "r", encoding="utf-8") as f:
            ann = json.load(f)
        return ann
    return None


def choose_best_person(result):
    if result.keypoints is None:
        return None

    xy = result.keypoints.xy
    conf = result.keypoints.conf

    if xy is None or len(xy) == 0:
        return None

    xy = xy.cpu().numpy()
    conf = conf.cpu().numpy() if conf is not None else None

    if conf is None:
        return {
            "person_index": 0,
            "xy": xy[0],
            "conf": None
        }

    mean_conf = np.nanmean(conf, axis=1)
    best_idx = int(np.argmax(mean_conf))

    return {
        "person_index": best_idx,
        "xy": xy[best_idx],
        "conf": conf[best_idx]
    }


def run_pose_on_image(model, image_path: Path):
    results = model.predict(
        source=str(image_path),
        conf=CONF_THRES,
        verbose=False
    )

    if not results:
        return {
            "detected": False,
            "num_persons": 0,
            "keypoints": None,
            "conf": None
        }

    result = results[0]

    num_persons = 0
    if result.keypoints is not None and result.keypoints.xy is not None:
        num_persons = int(len(result.keypoints.xy))

    picked = choose_best_person(result)
    if picked is None:
        return {
            "detected": False,
            "num_persons": num_persons,
            "keypoints": None,
            "conf": None
        }

    return {
        "detected": True,
        "num_persons": num_persons,
        "keypoints": picked["xy"],
        "conf": picked["conf"]
    }


def save_skeleton_outputs(dst_clip_dir: Path, ann: dict, frame_results: list):
    json_out = {
        "clip_id": ann.get("clip_id") if ann else dst_clip_dir.name,
        "primary_label": ann.get("primary_label") if ann else None,
        "labels": ann.get("labels") if ann else None,
        "original_frame_count": ann.get("original_frame_count") if ann else None,
        "saved_frame_count": ann.get("saved_frame_count") if ann else len(frame_results),
        "frame_results": frame_results
    }

    with open(dst_clip_dir / "skeleton_ir.json", "w", encoding="utf-8") as f:
        json.dump(json_out, f, ensure_ascii=False, indent=2)

    num_kpts = None
    for fr in frame_results:
        if fr["keypoints"] is not None:
            num_kpts = len(fr["keypoints"])
            break

    if num_kpts is None:
        np.savez_compressed(
            dst_clip_dir / "skeleton_ir.npz",
            image_names=np.array([fr["image_name"] for fr in frame_results], dtype=object),
            detected=np.array([fr["detected"] for fr in frame_results], dtype=bool),
            num_persons=np.array([fr["num_persons"] for fr in frame_results], dtype=np.int32),
            keypoints=np.empty((0,), dtype=np.float32),
            conf=np.empty((0,), dtype=np.float32),
        )
        return

    T = len(frame_results)
    keypoints_arr = np.full((T, num_kpts, 2), np.nan, dtype=np.float32)
    conf_arr = np.full((T, num_kpts), np.nan, dtype=np.float32)
    detected_arr = np.zeros((T,), dtype=bool)
    num_persons_arr = np.zeros((T,), dtype=np.int32)
    image_names = []

    for i, fr in enumerate(frame_results):
        image_names.append(fr["image_name"])
        detected_arr[i] = fr["detected"]
        num_persons_arr[i] = fr["num_persons"]

        if fr["keypoints"] is not None:
            keypoints_arr[i] = np.asarray(fr["keypoints"], dtype=np.float32)

        if fr["conf"] is not None:
            conf_arr[i] = np.asarray(fr["conf"], dtype=np.float32)

    np.savez_compressed(
        dst_clip_dir / "skeleton_ir.npz",
        image_names=np.array(image_names, dtype=object),
        detected=detected_arr,
        num_persons=num_persons_arr,
        keypoints=keypoints_arr,
        conf=conf_arr
    )


def process_clip(model, clip_dir: Path, out_clip_dir: Path):
    ensure_dir(out_clip_dir)

    ann = copy_annotation_if_exists(clip_dir, out_clip_dir)

    ir_dir = clip_dir / "ir"
    if not ir_dir.exists() or not ir_dir.is_dir():
        return False

    image_files = list_image_files(ir_dir)
    if not image_files:
        return False

    frame_results = []

    for img_path in image_files:
        try:
            pose_out = run_pose_on_image(model, img_path)

            frame_results.append({
                "image_name": img_path.name,
                "detected": pose_out["detected"],
                "num_persons": pose_out["num_persons"],
                "keypoints": pose_out["keypoints"].tolist() if pose_out["keypoints"] is not None else None,
                "conf": pose_out["conf"].tolist() if pose_out["conf"] is not None else None
            })

        except Exception:
            frame_results.append({
                "image_name": img_path.name,
                "detected": False,
                "num_persons": 0,
                "keypoints": None,
                "conf": None
            })

    save_skeleton_outputs(out_clip_dir, ann, frame_results)
    return True


def main():
    input_root = Path(INPUT_ROOT)
    output_root = Path(OUTPUT_ROOT)

    ensure_dir(output_root)

    print("[모델 로드 중]")
    model = YOLO(MODEL_PATH)
    print("[모델 로드 완료]")

    g_dirs = [p for p in input_root.iterdir() if p.is_dir() and p.name in VALID_G_FOLDERS]
    g_dirs.sort(key=lambda p: p.name)

    total_clips = 0
    total_success = 0

    for g_dir in g_dirs:
        num_dirs = [p for p in g_dir.iterdir() if is_number_folder(p)]
        num_dirs.sort(key=lambda p: int(p.name))

        for num_dir in num_dirs:
            image_dirs = [p for p in num_dir.iterdir() if is_image_folder(p)]
            image_dirs.sort(key=lambda p: p.name)

            for image_dir in image_dirs:
                clip_dirs = [p for p in image_dir.iterdir() if p.is_dir()]
                clip_dirs.sort(key=lambda p: p.name)

                if not clip_dirs:
                    continue

                block_name = f"{g_dir.name}/{num_dir.name}/{image_dir.name}"
                block_total = len(clip_dirs)
                block_start = time.time()
                block_success = 0

                print(f"\n[처리 시작] {block_name}")
                print(f"총 clip 수: {block_total}")

                for clip_dir in clip_dirs:
                    rel_path = clip_dir.relative_to(input_root)
                    out_clip_dir = output_root / rel_path

                    ok = process_clip(model, clip_dir, out_clip_dir)
                    total_clips += 1
                    if ok:
                        total_success += 1
                        block_success += 1

                block_elapsed = time.time() - block_start
                avg_time = block_elapsed / block_total if block_total > 0 else 0.0

                print(f"[저장 완료] {block_name}")
                print(f"성공 clip 수: {block_success}/{block_total}")
                print(f"총 소요 시간: {format_seconds(block_elapsed)}")
                print(f"clip당 평균 시간: {avg_time:.2f}초")

    print("\n[전체 처리 완료]")
    print(f"전체 clip 수: {total_clips}")
    print(f"성공 clip 수: {total_success}")


if __name__ == "__main__":
    main()