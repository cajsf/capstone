import json
import random
import cv2
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple


# =========================================================
# 설정
# =========================================================

ROOT_DIR = Path(r"C:\Users\hyi8402\Downloads\dmd\gZ")
LABEL_REGISTRY_PATH = ROOT_DIR / "label_registry.json"

CLIP_LEN = 50   # 정상 클립에만 사용
ABNORMAL_CONTEXT_BEFORE = 0
ABNORMAL_CONTEXT_AFTER = 0
ABNORMAL_STRIDE = 25
NORMAL_STRIDE = 25

JPG_QUALITY = 75
SAVE_IR = True
SAVE_RGB = False
USE_BODY_FRAME_SHIFT = True

OUTPUT_SUFFIX = "_image"

FRAME_STEP = 1

NORMAL_RATIO = 0.3

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# =========================================================
# 유틸
# =========================================================

def load_json(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(json_path: Path, data: dict):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jpg(img_path: Path, image, quality: int = 95):
    img_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def read_sampled_frames(cap: cv2.VideoCapture, start_frame: int, end_frame: int, frame_step: int):
    """
    start_frame ~ end_frame 구간 전체를 frame_step 간격으로 읽는다.
    예: start=100, end=109, step=2 -> 100,102,104,106,108
    """
    frames = []
    sampled_indices = list(range(start_frame, end_frame + 1, frame_step))

    for frame_idx in sampled_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return None, sampled_indices
        frames.append(frame)

    return frames, sampled_indices


def find_subject_dirs(root_dir: Path) -> List[Path]:
    """
    root_dir 아래 모든 하위 폴더를 재귀적으로 탐색해서
    s1, s2, s3 ... 형태의 폴더를 전부 찾는다.
    """
    subject_dirs = []

    for p in root_dir.rglob("*"):
        if p.is_dir() and p.name.startswith("s"):
            subject_dirs.append(p)

    return sorted(subject_dirs)


def find_files_in_subject_dir(subject_dir: Path) -> Tuple[Path, Path, Path]:
    ann_files = list(subject_dir.glob("*_ann_distraction.json"))
    ir_files = list(subject_dir.glob("*_ir_body.mp4"))
    rgb_files = list(subject_dir.glob("*_rgb_body.mp4"))

    if not ann_files:
        raise FileNotFoundError(f"[{subject_dir.name}] *_ann_distraction.json 없음")
    if not ir_files:
        raise FileNotFoundError(f"[{subject_dir.name}] *_ir_body.mp4 없음")
    if not rgb_files:
        raise FileNotFoundError(f"[{subject_dir.name}] *_rgb_body.mp4 없음")

    return ann_files[0], ir_files[0], rgb_files[0]


def get_body_stream_info(openlabel: dict) -> Tuple[int, int]:
    streams = openlabel.get("streams", {})
    body_info = streams.get("body_camera", {})
    stream_props = body_info.get("stream_properties", {})
    sync = stream_props.get("sync", {})

    frame_shift = int(sync.get("frame_shift", 0))
    total_frames = stream_props.get("total_frames", None)
    if total_frames is not None:
        total_frames = int(total_frames)

    return frame_shift, total_frames


def to_body_frame(global_frame: int, body_frame_shift: int) -> int:
    if USE_BODY_FRAME_SHIFT:
        return global_frame - body_frame_shift
    return global_frame


# =========================================================
# 라벨 정책
# =========================================================

def classify_action_type(action_type: str) -> str:
    if not action_type:
        return "ignore"

    if action_type == "driver_actions/safe_drive":
        return "normal"

    if action_type == "driver_actions/unclassified":
        return "ignore"

    if action_type.startswith("hands_using_wheel/"):
        return "ignore"

    if action_type.startswith("driver_actions/"):
        return "abnormal"

    if action_type.startswith("talking/"):
        return "abnormal"

    if action_type == "gaze_on_road/not_looking_road":
        return "abnormal"

    if action_type == "gaze_on_road/looking_road":
        return "ignore"

    return "ignore"


# =========================================================
# 라벨 레지스트리
# =========================================================

def init_or_load_label_registry(registry_path: Path) -> dict:
    if registry_path.exists():
        registry = load_json(registry_path)
    else:
        registry = {
            "all_action_types": [],
            "normal_types": [],
            "abnormal_types": [],
            "ignored_types": [],
            "type_to_policy": {},
            "seen_in_files": {},
        }
    return registry


def update_label_registry(registry: dict, action_types: Set[str], source_name: str) -> dict:
    changed = False

    for action_type in sorted(action_types):
        policy = classify_action_type(action_type)

        if action_type not in registry["all_action_types"]:
            registry["all_action_types"].append(action_type)
            changed = True

        if action_type not in registry["type_to_policy"]:
            registry["type_to_policy"][action_type] = policy
            changed = True

        if action_type not in registry["seen_in_files"]:
            registry["seen_in_files"][action_type] = []
            changed = True
        if source_name not in registry["seen_in_files"][action_type]:
            registry["seen_in_files"][action_type].append(source_name)
            changed = True

    normal_types = []
    abnormal_types = []
    ignored_types = []

    for action_type in registry["all_action_types"]:
        policy = registry["type_to_policy"].get(action_type, classify_action_type(action_type))

        if policy == "normal":
            normal_types.append(action_type)
        elif policy == "abnormal":
            abnormal_types.append(action_type)
        else:
            ignored_types.append(action_type)

    if registry["normal_types"] != normal_types:
        registry["normal_types"] = normal_types
        changed = True
    if registry["abnormal_types"] != abnormal_types:
        registry["abnormal_types"] = abnormal_types
        changed = True
    if registry["ignored_types"] != ignored_types:
        registry["ignored_types"] = ignored_types
        changed = True

    if changed:
        registry["all_action_types"] = sorted(registry["all_action_types"])
        registry["normal_types"] = sorted(registry["normal_types"])
        registry["abnormal_types"] = sorted(registry["abnormal_types"])
        registry["ignored_types"] = sorted(registry["ignored_types"])
        for k in registry["seen_in_files"]:
            registry["seen_in_files"][k] = sorted(registry["seen_in_files"][k])

    return registry


# =========================================================
# Interval / Action 처리
# =========================================================

def get_actions(openlabel: dict) -> Dict[str, dict]:
    return openlabel.get("actions", {})


def get_action_types(openlabel: dict) -> Set[str]:
    action_types = set()
    actions = get_actions(openlabel)
    for _, action_info in actions.items():
        action_type = action_info.get("type", "").strip()
        if action_type:
            action_types.add(action_type)
    return action_types


def collect_intervals_by_type(openlabel: dict, body_frame_shift: int) -> Dict[str, List[Tuple[int, int]]]:
    actions = get_actions(openlabel)
    intervals_by_type = defaultdict(list)

    for _, action_info in actions.items():
        action_type = action_info.get("type", "").strip()
        if not action_type:
            continue

        for interval in action_info.get("frame_intervals", []):
            gs = int(interval["frame_start"])
            ge = int(interval["frame_end"])

            bs = to_body_frame(gs, body_frame_shift)
            be = to_body_frame(ge, body_frame_shift)

            if be < 0:
                continue
            if bs < 0:
                bs = 0

            intervals_by_type[action_type].append((bs, be))

    return intervals_by_type


def split_intervals_by_policy(intervals_by_type: Dict[str, List[Tuple[int, int]]]):
    normal_intervals = defaultdict(list)
    abnormal_intervals = defaultdict(list)
    ignored_intervals = defaultdict(list)

    for action_type, intervals in intervals_by_type.items():
        policy = classify_action_type(action_type)
        if policy == "normal":
            normal_intervals[action_type].extend(intervals)
        elif policy == "abnormal":
            abnormal_intervals[action_type].extend(intervals)
        else:
            ignored_intervals[action_type].extend(intervals)

    return normal_intervals, abnormal_intervals, ignored_intervals


def build_frame_to_abnormal_labels(
    abnormal_intervals: Dict[str, List[Tuple[int, int]]],
    max_frame: int,
) -> Dict[int, Set[str]]:
    frame_to_labels = defaultdict(set)

    for action_type, intervals in abnormal_intervals.items():
        for s, e in intervals:
            s = max(0, s)
            e = min(max_frame - 1, e)
            for f in range(s, e + 1):
                frame_to_labels[f].add(action_type)

    return frame_to_labels


def interval_to_clip_starts(s: int, e: int, clip_len: int, stride: int) -> List[int]:
    if e < s:
        return []

    region_len = e - s + 1

    if region_len <= clip_len:
        center = (s + e) // 2
        start = center - clip_len // 2
        if start < 0:
            start = 0
        return [start]

    starts = []
    cur = s
    while cur + clip_len - 1 <= e:
        starts.append(cur)
        cur += stride

    last_start = e - clip_len + 1
    if len(starts) == 0 or starts[-1] != last_start:
        starts.append(last_start)

    return starts


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    겹치거나 붙어 있는 interval을 하나로 합친다.
    예: (10,20), (18,30), (31,40) -> (10,40)
    """
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [list(intervals[0])]

    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e + 1:
            merged[-1][1] = max(last_e, e)
        else:
            merged.append([s, e])

    return [(s, e) for s, e in merged]


def build_abnormal_clips(
    abnormal_intervals: Dict[str, List[Tuple[int, int]]],
    frame_to_abnormal_labels: Dict[int, Set[str]],
    total_frames: int,
    context_before: int,
    context_after: int,
) -> List[dict]:
    """
    비정상 행동 하나의 연속 구간 전체를 클립 하나로 저장.
    같은 라벨에서 붙어 있거나 겹치는 interval은 merge 후 저장.
    """
    clips = []

    for action_type, intervals in abnormal_intervals.items():
        merged_intervals = merge_intervals(intervals)

        for core_s, core_e in merged_intervals:
            start = max(0, core_s - context_before)
            end = min(total_frames - 1, core_e + context_after)

            if start > end:
                continue

            labels = set()
            for f in range(start, end + 1):
                labels.update(frame_to_abnormal_labels.get(f, set()))

            if not labels:
                labels = {action_type}

            clips.append({
                "clip_type": "abnormal",
                "start_frame_body": start,
                "end_frame_body": end,
                "labels": sorted(labels),
                "primary_label": action_type,
            })

    clips = sorted(clips, key=lambda x: (x["start_frame_body"], x["end_frame_body"], x["primary_label"]))
    return clips


def build_normal_clips(
    normal_intervals: Dict[str, List[Tuple[int, int]]],
    abnormal_frame_set: Set[int],
    total_frames: int,
    clip_len: int,
    stride: int,
    target_count: int,
) -> List[dict]:
    safe_intervals = normal_intervals.get("driver_actions/safe_drive", [])
    candidates = []
    seen = set()

    for s, e in safe_intervals:
        starts = interval_to_clip_starts(s, e, clip_len, stride)
        for start in starts:
            end = start + clip_len - 1
            if end >= total_frames:
                continue

            overlap = False
            for f in range(start, end + 1):
                if f in abnormal_frame_set:
                    overlap = True
                    break
            if overlap:
                continue

            key = (start, end)
            if key in seen:
                continue
            seen.add(key)

            candidates.append({
                "clip_type": "normal",
                "start_frame_body": start,
                "end_frame_body": end,
                "labels": ["driver_actions/safe_drive"],
                "primary_label": "driver_actions/safe_drive",
            })

    random.shuffle(candidates)
    return candidates[:min(target_count, len(candidates))]


# =========================================================
# 저장
# =========================================================

def save_clip(
    clip_idx: int,
    clip_info: dict,
    out_dir: Path,
    ir_cap: cv2.VideoCapture,
    rgb_cap: cv2.VideoCapture,
    ann_name: str,
    ir_video_name: str,
    rgb_video_name: str,
    body_frame_shift: int,
    ir_fps: float,
    rgb_fps: float,
):
    start = clip_info["start_frame_body"]
    end = clip_info["end_frame_body"]

    ir_total = int(ir_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rgb_total = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start < 0 or end >= ir_total or end >= rgb_total:
        print(f"[SKIP] clip_{clip_idx:06d}: start={start}, end={end}, ir_total={ir_total}, rgb_total={rgb_total}")
        return None

    clip_dir = out_dir / f"clip_{clip_idx:06d}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    saved_ir = []
    saved_rgb = []

    sampled_frame_indices = list(range(start, end + 1, FRAME_STEP))
    saved_frame_count = len(sampled_frame_indices)
    original_frame_count = end - start + 1

    if SAVE_IR:
        ir_frames, _ = read_sampled_frames(ir_cap, start, end, FRAME_STEP)
        if ir_frames is None:
            raise RuntimeError(f"IR clip 읽기 실패: {start}~{end}")
        for i, frame in enumerate(ir_frames):
            img_path = clip_dir / "ir" / f"frame_{i:03d}.jpg"
            save_jpg(img_path, frame, JPG_QUALITY)
            saved_ir.append(str(img_path.relative_to(out_dir)))

    if SAVE_RGB:
        rgb_frames, _ = read_sampled_frames(rgb_cap, start, end, FRAME_STEP)
        if rgb_frames is None:
            raise RuntimeError(f"RGB clip 읽기 실패: {start}~{end}")
        for i, frame in enumerate(rgb_frames):
            img_path = clip_dir / "rgb" / f"frame_{i:03d}.jpg"
            save_jpg(img_path, frame, JPG_QUALITY)
            saved_rgb.append(str(img_path.relative_to(out_dir)))

    clip_ann = {
        "clip_id": f"clip_{clip_idx:06d}",
        "clip_type": clip_info["clip_type"],
        "primary_label": clip_info["primary_label"],
        "labels": clip_info["labels"],
        "start_frame_body": start,
        "end_frame_body": end,
        "original_frame_count": original_frame_count,
        "frame_step": FRAME_STEP,
        "saved_frame_count": saved_frame_count,
        "sampled_frame_indices_body": sampled_frame_indices,
        "estimated_global_start": start + body_frame_shift if USE_BODY_FRAME_SHIFT else start,
        "estimated_global_end": end + body_frame_shift if USE_BODY_FRAME_SHIFT else end,
        "timestamp_start_ir_sec": start / ir_fps if ir_fps > 0 else None,
        "timestamp_end_ir_sec": end / ir_fps if ir_fps > 0 else None,
        "timestamp_start_rgb_sec": start / rgb_fps if rgb_fps > 0 else None,
        "timestamp_end_rgb_sec": end / rgb_fps if rgb_fps > 0 else None,
        "source_annotation": ann_name,
        "source_ir_video": ir_video_name,
        "source_rgb_video": rgb_video_name,
        "saved_ir_frames": saved_ir,
        "saved_rgb_frames": saved_rgb,
    }

    save_json(clip_dir / "annotation.json", clip_ann)
    return clip_ann


# =========================================================
# 메인
# =========================================================

def process_subject_dir(subject_dir: Path, registry: dict) -> dict:
    print(f"\n===== 처리 시작: {subject_dir.name} =====")

    ann_path, ir_video_path, rgb_video_path = find_files_in_subject_dir(subject_dir)
    data = load_json(ann_path)
    openlabel = data.get("openlabel", {})

    body_frame_shift, body_total_frames_ann = get_body_stream_info(openlabel)

    action_types = get_action_types(openlabel)
    registry = update_label_registry(registry, action_types, ann_path.name)

    print(f"annotation: {ann_path.name}")
    print(f"발견 action types:")
    for t in sorted(action_types):
        print(f"  - {t} [{classify_action_type(t)}]")

    ir_cap = cv2.VideoCapture(str(ir_video_path))
    rgb_cap = cv2.VideoCapture(str(rgb_video_path))

    if not ir_cap.isOpened():
        raise RuntimeError(f"IR 영상 열기 실패: {ir_video_path}")
    if not rgb_cap.isOpened():
        raise RuntimeError(f"RGB 영상 열기 실패: {rgb_video_path}")

    ir_total = int(ir_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rgb_total = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ir_fps = ir_cap.get(cv2.CAP_PROP_FPS)
    rgb_fps = rgb_cap.get(cv2.CAP_PROP_FPS)

    total_frames = min(ir_total, rgb_total)
    if body_total_frames_ann is not None:
        total_frames = min(total_frames, body_total_frames_ann)

    intervals_by_type = collect_intervals_by_type(openlabel, body_frame_shift)
    normal_intervals, abnormal_intervals, ignored_intervals = split_intervals_by_policy(intervals_by_type)

    frame_to_abnormal_labels = build_frame_to_abnormal_labels(abnormal_intervals, total_frames)
    abnormal_frame_set = set(frame_to_abnormal_labels.keys())

    abnormal_clips = build_abnormal_clips(
        abnormal_intervals=abnormal_intervals,
        frame_to_abnormal_labels=frame_to_abnormal_labels,
        total_frames=total_frames,
        context_before=ABNORMAL_CONTEXT_BEFORE,
        context_after=ABNORMAL_CONTEXT_AFTER,
    )

    normal_clips = build_normal_clips(
        normal_intervals=normal_intervals,
        abnormal_frame_set=abnormal_frame_set,
        total_frames=total_frames,
        clip_len=CLIP_LEN,
        stride=NORMAL_STRIDE,
        target_count=max(1, int(len(abnormal_clips) * NORMAL_RATIO)),
    )

    print(f"abnormal clips: {len(abnormal_clips)}")
    print(f"normal clips  : {len(normal_clips)}")

    all_clips = abnormal_clips + normal_clips
    all_clips = sorted(all_clips, key=lambda x: (x["start_frame_body"], x["end_frame_body"], x["clip_type"]))

    out_dir = subject_dir.parent / f"{subject_dir.name}{OUTPUT_SUFFIX}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    class_counter = defaultdict(int)

    for clip_idx, clip_info in enumerate(all_clips):
        clip_ann = save_clip(
            clip_idx=clip_idx,
            clip_info=clip_info,
            out_dir=out_dir,
            ir_cap=ir_cap,
            rgb_cap=rgb_cap,
            ann_name=ann_path.name,
            ir_video_name=ir_video_path.name,
            rgb_video_name=rgb_video_path.name,
            body_frame_shift=body_frame_shift,
            ir_fps=ir_fps,
            rgb_fps=rgb_fps,
        )
        if clip_ann is None:
            continue

        manifest.append(clip_ann)
        class_counter[clip_ann["primary_label"]] += 1

        if (clip_idx + 1) % 50 == 0:
            print(f"저장 중... {clip_idx + 1}/{len(all_clips)} clips")

    ir_cap.release()
    rgb_cap.release()

    with open(out_dir / "annotations.jsonl", "w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "subject_folder": subject_dir.name,
        "source_annotation": str(ann_path),
        "source_ir_video": str(ir_video_path),
        "source_rgb_video": str(rgb_video_path),
        "clip_len_for_normal_only": CLIP_LEN,
        "abnormal_context_before": ABNORMAL_CONTEXT_BEFORE,
        "abnormal_context_after": ABNORMAL_CONTEXT_AFTER,
        "normal_stride": NORMAL_STRIDE,
        "frame_step": FRAME_STEP,
        "use_body_frame_shift": USE_BODY_FRAME_SHIFT,
        "body_frame_shift": body_frame_shift,
        "normal_types_in_this_file": sorted(list(normal_intervals.keys())),
        "abnormal_types_in_this_file": sorted(list(abnormal_intervals.keys())),
        "ignored_types_in_this_file": sorted(list(ignored_intervals.keys())),
        "total_saved_clips": len(manifest),
        "saved_clip_counts_by_primary_label": dict(class_counter),
    }
    save_json(out_dir / "summary.json", summary)

    return registry


def main():
    registry = init_or_load_label_registry(LABEL_REGISTRY_PATH)

    subject_dirs = find_subject_dirs(ROOT_DIR)
    if not subject_dirs:
        print("s1, s2, s3 ... 폴더를 찾지 못함")
        return

    for subject_dir in subject_dirs:
        try:
            registry = process_subject_dir(subject_dir, registry)
            save_json(LABEL_REGISTRY_PATH, registry)
        except Exception as e:
            print(f"[오류] {subject_dir.name}: {e}")

    print("\n===== 최종 누적 라벨 레지스트리 =====")
    print(f"저장 위치: {LABEL_REGISTRY_PATH}")
    print("all_action_types:")
    for t in registry["all_action_types"]:
        print(f"  - {t} [{registry['type_to_policy'].get(t, 'unknown')}]")

    save_json(LABEL_REGISTRY_PATH, registry)


if __name__ == "__main__":
    main()