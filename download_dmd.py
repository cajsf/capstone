#!/usr/bin/env python3
"""DMD Dataset Downloader for /data/shared/DMD.

Parses signed URLs from dmd_dataset_links.md, downloads tar.gz files in parallel,
extracts them while the next file is being downloaded, verifies integrity with
ffprobe, then builds a manifest.json and per-video keyframe index for fast seek.

Usage:
    python3 download_dmd.py                              # full pipeline
    python3 download_dmd.py --dry-run                    # preflight only
    python3 download_dmd.py --skip-download              # verify + manifest only
    python3 download_dmd.py --workers 3                  # 3 parallel downloads
    python3 download_dmd.py --no-keyframe-index          # skip Phase 3 keyframes

State is persisted to _meta/state.json so reruns skip completed files.
Run under tmux; Ctrl+B D to detach, session survives disconnects.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: tqdm not installed. Run: pip install --user tqdm", file=sys.stderr)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────
# Config & constants
# ─────────────────────────────────────────────────────────────

DEFAULT_LINKS = Path("/data/shared/DMD/dmd_dataset_links.md")
DEFAULT_DEST = Path("/data/shared/DMD")
URL_PATTERN = re.compile(
    r"\[(?P<name>[^\]]+\.tar\.gz)\]\((?P<url>https://datasets\.vicomtech\.org[^\)]+)\)"
)
# e.g. dmd-dataset-rgb_ir-gA-1.tar.gz  -> dataset=distraction (rgb_ir), group=gA, subject=1
# e.g. dmd-dataset-gaze-gB-10.tar.gz   -> dataset=gaze, group=gB, subject=10
FILENAME_PATTERN = re.compile(r"dmd-dataset-(?P<kind>rgb_ir|gaze)-(?P<group>g[A-Z])-(?P<sid>\d+)\.tar\.gz")
KIND_TO_DATASET = {"rgb_ir": "distraction", "gaze": "gaze"}


@dataclass
class Download:
    filename: str
    url: str
    dataset: str          # "distraction" | "gaze"
    group: str            # "gA".."gZ"
    subject: int          # 1..37
    size_bytes: int = 0
    download_sec: float = 0.0
    extract_sec: float = 0.0
    status: str = "pending"   # pending | downloaded | extracted | verified | failed
    error: str = ""


# ─────────────────────────────────────────────────────────────
# Link parsing
# ─────────────────────────────────────────────────────────────

def parse_links(md_path: Path) -> list[Download]:
    text = md_path.read_text()
    out: list[Download] = []
    seen = set()
    for m in URL_PATTERN.finditer(text):
        name = m["name"]
        if name in seen:
            continue
        seen.add(name)
        fm = FILENAME_PATTERN.match(name)
        if not fm:
            # READMEs and other non-data files are skipped
            continue
        out.append(Download(
            filename=name,
            url=m["url"],
            dataset=KIND_TO_DATASET[fm["kind"]],
            group=fm["group"],
            subject=int(fm["sid"]),
        ))
    out.sort(key=lambda d: (d.dataset, d.group, d.subject))
    return out


# ─────────────────────────────────────────────────────────────
# State persistence (resume support)
# ─────────────────────────────────────────────────────────────

class State:
    def __init__(self, path: Path):
        self.path = path
        self.lock = Lock()
        self.data: dict[str, dict] = {}
        if path.exists():
            self.data = json.loads(path.read_text())

    def update(self, dl: Download) -> None:
        with self.lock:
            self.data[dl.filename] = asdict(dl)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.data, indent=2))
            tmp.replace(self.path)

    def get_status(self, filename: str) -> str:
        return self.data.get(filename, {}).get("status", "pending")


# ─────────────────────────────────────────────────────────────
# Phase 1: Download + Extract pipeline
# ─────────────────────────────────────────────────────────────

def wget_download(url: str, dest: Path) -> tuple[bool, str]:
    """Download via wget with resume. Returns (ok, error_msg)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    # -c: continue partial, -q: quiet, --tries=3, --timeout=60
    cmd = [
        "wget", "-c", "-q", "--tries=3", "--timeout=60",
        "--show-progress", "--progress=dot:giga",
        "-O", str(dest), url,
    ]
    try:
        # 6h overall timeout; large files under shared bandwidth can take >1h.
        # wget's own --timeout=60 still aborts stalled transfers quickly.
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=21600)
        if r.returncode != 0:
            return False, f"wget rc={r.returncode}: {r.stderr[-200:]}"
        if not dest.exists() or dest.stat().st_size < 1024:
            return False, f"downloaded size too small ({dest.stat().st_size if dest.exists() else 0} bytes) — URL may be expired"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "wget timeout (6h)"
    except Exception as e:
        return False, f"wget exception: {e}"


def extract_tar(tar_path: Path, dest_root: Path) -> tuple[bool, str]:
    """Extract tar.gz to dest_root. Returns (ok, err)."""
    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            # Safe extract: reject absolute paths or .. traversal
            members = tf.getmembers()
            for m in members:
                mp = Path(m.name)
                if mp.is_absolute() or ".." in mp.parts:
                    return False, f"unsafe member: {m.name}"
            tf.extractall(dest_root)
        return True, ""
    except tarfile.ReadError as e:
        # Try plain tar in case of bad gzip wrapper
        try:
            with tarfile.open(tar_path, "r:") as tf:
                tf.extractall(dest_root)
            return True, ""
        except Exception as e2:
            return False, f"tar ReadError: {e}; fallback failed: {e2}"
    except Exception as e:
        return False, f"extract exception: {e}"


def download_worker(dl: Download, tmp_dir: Path, state: State, pbar: tqdm) -> Download:
    if state.get_status(dl.filename) in ("extracted", "verified"):
        pbar.update(1)
        return dl
    tar_path = tmp_dir / dl.filename
    t0 = time.time()
    ok, err = wget_download(dl.url, tar_path)
    dl.download_sec = time.time() - t0
    if not ok:
        dl.status = "failed"
        dl.error = err
        state.update(dl)
        pbar.update(1)
        return dl
    dl.size_bytes = tar_path.stat().st_size
    dl.status = "downloaded"
    state.update(dl)
    pbar.update(1)
    return dl


def extractor_loop(q: Queue, dest_root: Path, state: State, pbar_ex: tqdm) -> None:
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            return
        dl, tar_path = item
        if dl.status == "failed":
            q.task_done()
            pbar_ex.update(1)
            continue
        t0 = time.time()
        # Separate extraction by dataset: distraction/ and gaze/ each get their own root.
        dataset_root = dest_root / dl.dataset
        dataset_root.mkdir(parents=True, exist_ok=True)
        ok, err = extract_tar(tar_path, dataset_root)
        dl.extract_sec = time.time() - t0
        if ok:
            dl.status = "extracted"
            try:
                tar_path.unlink()
            except OSError:
                pass
        else:
            dl.status = "failed"
            dl.error = err
        state.update(dl)
        pbar_ex.update(1)
        q.task_done()


def phase1_download_and_extract(
    downloads: list[Download], dest: Path, tmp_dir: Path, state: State, workers: int
) -> None:
    pending = [d for d in downloads if state.get_status(d.filename) not in ("extracted", "verified")]
    if not pending:
        print("Phase 1: all files already extracted. Skipping.")
        return
    total_gb = sum(d.size_bytes for d in pending) / (1024**3) or None
    print(f"Phase 1: downloading {len(pending)}/{len(downloads)} files "
          f"({'~' + f'{total_gb:.1f} GB' if total_gb else 'size TBD'}), {workers} parallel workers")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    q: Queue = Queue(maxsize=workers * 2)

    pbar_dl = tqdm(total=len(pending), desc="download ", unit="file", position=0)
    pbar_ex = tqdm(total=len(pending), desc="extract  ", unit="file", position=1)

    extractor = Thread(target=extractor_loop, args=(q, dest, state, pbar_ex), daemon=True)
    extractor.start()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_worker, d, tmp_dir, state, pbar_dl): d for d in pending}
        for fut in as_completed(futures):
            dl = fut.result()
            if dl.status == "downloaded":
                q.put((dl, tmp_dir / dl.filename))
            else:  # failed — still advance extractor pbar via sentinel
                q.put((dl, None))
    q.put(None)
    q.join()
    extractor.join()
    pbar_dl.close()
    pbar_ex.close()

    ok = sum(1 for d in downloads if d.status in ("extracted", "verified"))
    fail = sum(1 for d in downloads if d.status == "failed")
    print(f"Phase 1 done: {ok} extracted, {fail} failed")
    for d in downloads:
        if d.status == "failed":
            print(f"  FAIL {d.filename}: {d.error}")


# ─────────────────────────────────────────────────────────────
# Phase 2: Integrity verification (ffprobe)
# ─────────────────────────────────────────────────────────────

def ffprobe_video(path: Path) -> Optional[dict]:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_packets",  # packets are much faster than counting decoded frames
        "-show_entries", "stream=width,height,nb_read_packets,r_frame_rate,duration",
        "-of", "json", str(path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return None
        info = json.loads(r.stdout)
        streams = info.get("streams", [])
        if not streams:
            return None
        s = streams[0]
        # r_frame_rate like "2976/100"
        num, den = s.get("r_frame_rate", "0/1").split("/")
        fps = float(num) / float(den) if float(den) else 0.0
        return {
            "width": int(s.get("width", 0)),
            "height": int(s.get("height", 0)),
            "fps": round(fps, 2),
            "duration_sec": float(s.get("duration", 0) or 0),
            "packets": int(s.get("nb_read_packets", 0) or 0),
        }
    except Exception:
        return None


def phase2_verify(dest: Path, meta_dir: Path) -> list[dict]:
    videos = sorted(list(dest.rglob("*.mp4")) + list(dest.rglob("*.avi")))
    if not videos:
        print("Phase 2: no videos found to verify.")
        return []
    print(f"Phase 2: verifying {len(videos)} videos with ffprobe...")
    failures = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(ffprobe_video, v): v for v in videos}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="verify  ", unit="vid"):
            v = futures[fut]
            info = fut.result()
            if info is None or info["width"] == 0 or info["packets"] < 10:
                failures.append({"path": str(v), "reason": "ffprobe failed or too few packets"})
    if failures:
        (meta_dir / "verify_fail.json").write_text(json.dumps(failures, indent=2))
        print(f"  {len(failures)} failures logged to _meta/verify_fail.json")
    else:
        print("  all videos OK")
    return failures


# ─────────────────────────────────────────────────────────────
# Phase 3: Manifest + keyframe index
# ─────────────────────────────────────────────────────────────

# e.g. gA_1_s6_2019-03-22T11;59;56+01;00_ir_face.mp4
VIDEO_NAME_RE = re.compile(
    r"(?P<group>g[A-Z])_(?P<sid>\d+)_(?P<session>s\d+)_(?P<ts>[^_]+)_(?P<stream>rgb|ir|depth)_(?P<camera>face|body|hands)\.(?P<ext>mp4|avi)$"
)
ANN_NAME_RE = re.compile(
    r"(?P<group>g[A-Z])_(?P<sid>\d+)_(?P<session>s\d+)_(?P<ts>[^_]+)_rgb_ann_(?P<dim>gaze|hands|distraction)\.json$"
)


def keyframe_index(video: Path) -> Optional[list[float]]:
    """Return list of keyframe PTS (in seconds) for fast random seek."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,flags",
        "-of", "csv=print_section=0", str(video),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return None
        keys: list[float] = []
        for line in r.stdout.splitlines():
            parts = line.split(",")
            if len(parts) >= 2 and "K" in parts[1]:
                try:
                    keys.append(float(parts[0]))
                except ValueError:
                    pass
        return keys
    except Exception:
        return None


def phase3_build_manifest(dest: Path, meta_dir: Path, with_keyframes: bool) -> None:
    print("Phase 3: building manifest.json...")
    videos = sorted(list(dest.rglob("*.mp4")) + list(dest.rglob("*.avi")))
    jsons = sorted(dest.rglob("*_ann_*.json"))

    # Index annotations by (group, sid, session, ts) → {dim: path}
    ann_index: dict[tuple, dict[str, str]] = {}
    for j in jsons:
        m = ANN_NAME_RE.match(j.name)
        if not m:
            continue
        key = (m["group"], int(m["sid"]), m["session"], m["ts"])
        ann_index.setdefault(key, {})[m["dim"]] = str(j)

    manifest: dict[str, dict] = {}
    idx_dir = meta_dir / "keyframes"
    if with_keyframes:
        idx_dir.mkdir(exist_ok=True)

    def process(v: Path) -> tuple[str, dict] | None:
        m = VIDEO_NAME_RE.match(v.name)
        if not m:
            return None
        key = (m["group"], int(m["sid"]), m["session"], m["ts"])
        info = ffprobe_video(v) or {}
        entry = {
            "path": str(v),
            "rel_path": str(v.relative_to(dest)),
            "group": m["group"],
            "subject": int(m["sid"]),
            "session": m["session"],
            "timestamp": m["ts"],
            "stream": m["stream"],
            "camera": m["camera"],
            "size_mb": round(v.stat().st_size / (1024 * 1024), 1),
            "width": info.get("width"),
            "height": info.get("height"),
            "fps": info.get("fps"),
            "duration_sec": info.get("duration_sec"),
            "packets": info.get("packets"),
            "annotations": ann_index.get(key, {}),
        }
        if with_keyframes:
            kfs = keyframe_index(v)
            if kfs is not None:
                idx_path = idx_dir / (v.stem + ".idx.json")
                idx_path.write_text(json.dumps({"keyframes_sec": kfs}))
                entry["keyframe_index"] = str(idx_path.relative_to(meta_dir))
                entry["num_keyframes"] = len(kfs)
        return str(v.relative_to(dest)), entry

    with ThreadPoolExecutor(max_workers=8) as pool:
        for res in tqdm(pool.map(process, videos), total=len(videos), desc="manifest", unit="vid"):
            if res:
                manifest[res[0]] = res[1]

    out = meta_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest: {len(manifest)} entries → {out}")
    print(f"  annotations indexed: {sum(len(v) for v in ann_index.values())} across {len(ann_index)} sessions")


# ─────────────────────────────────────────────────────────────
# Preflight & main
# ─────────────────────────────────────────────────────────────

def preflight(links: Path, dest: Path) -> list[Download]:
    print("─── preflight ───")
    for tool in ("wget", "tar", "ffprobe"):
        if not shutil.which(tool):
            print(f"  MISSING: {tool}")
            sys.exit(2)
        print(f"  {tool}: {shutil.which(tool)}")
    if not links.exists():
        print(f"  links file not found: {links}")
        sys.exit(2)
    dest.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(dest).free / (1024**3)
    print(f"  dest: {dest} (free {free_gb:.1f} GB)")
    if free_gb < 500:
        print(f"  WARN: <500 GB free; DMD extracted is ~300~500 GB")

    downloads = parse_links(links)
    if not downloads:
        print(f"  no tar.gz URLs parsed from {links}")
        sys.exit(2)
    by_ds: dict[str, int] = {}
    for d in downloads:
        by_ds[d.dataset] = by_ds.get(d.dataset, 0) + 1
    print(f"  parsed {len(downloads)} files: {by_ds}")
    print(f"  first: {downloads[0].filename}")
    print(f"  last:  {downloads[-1].filename}")
    return downloads


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--links", type=Path, default=DEFAULT_LINKS)
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    p.add_argument("--workers", type=int, default=2, help="parallel download workers (default 2)")
    p.add_argument("--dry-run", action="store_true", help="preflight only, do nothing")
    p.add_argument("--skip-download", action="store_true", help="skip Phase 1, verify + manifest existing files")
    p.add_argument("--skip-verify", action="store_true", help="skip Phase 2 ffprobe verification")
    p.add_argument("--no-keyframe-index", action="store_true", help="skip keyframe .idx.json generation in Phase 3")
    args = p.parse_args()

    downloads = preflight(args.links, args.dest)
    if args.dry_run:
        print("dry-run: exiting before any work")
        return

    meta_dir = args.dest / "_meta"
    meta_dir.mkdir(exist_ok=True)
    tmp_dir = args.dest / "_tmp"
    state = State(meta_dir / "state.json")

    t_start = time.time()
    if not args.skip_download:
        phase1_download_and_extract(downloads, args.dest, tmp_dir, state, args.workers)
        # Best-effort cleanup of _tmp
        if tmp_dir.exists() and not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()

    if not args.skip_verify:
        phase2_verify(args.dest, meta_dir)

    phase3_build_manifest(args.dest, meta_dir, with_keyframes=not args.no_keyframe_index)

    elapsed = time.time() - t_start
    print(f"\n all phases done in {elapsed/60:.1f} min")
    print(f" next steps:")
    print(f"   - read /data/shared/DMD/CLAUDE.md for usage")
    print(f"   - manifest: {meta_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
