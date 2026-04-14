#!/usr/bin/env bash
# DMD download status reporter for Telegram.
# Usage: ./dmd_status.sh [--send]
#   no args  → print report to stdout
#   --send   → also push to Telegram (reads ~/.config/dmd-bot.env)
set -euo pipefail

DEST=/data/shared/DMD
STATE="$DEST/_meta/state.json"
TMP="$DEST/_tmp"

report=$(python3 - <<'PY'
import json, os, time
from pathlib import Path
from collections import Counter

DEST = Path("/data/shared/DMD")
STATE = DEST / "_meta" / "state.json"
TMP = DEST / "_tmp"

lines = []
now = time.strftime("%H:%M:%S")
lines.append(f"📦 DMD {now}")

# state.json
if STATE.exists():
    d = json.loads(STATE.read_text())
    c = Counter(v["status"] for v in d.values())
    total = 46  # 31 distraction + 15 gaze
    done = c.get("extracted", 0) + c.get("verified", 0)
    pct = done * 100 / total if total else 0
    lines.append(f"진행: {done}/{total} ({pct:.0f}%)")
    status_parts = [f"{k}={v}" for k, v in sorted(c.items())]
    lines.append("  " + " ".join(status_parts))

    # ETA
    ext = [v for v in d.values() if v["status"] in ("extracted", "verified") and v.get("download_sec", 0) > 0]
    if ext and done > 0:
        avg_dl = sum(v["download_sec"] for v in ext) / len(ext)
        remaining = total - done
        eta_sec = remaining * avg_dl / 3  # 3 workers
        lines.append(f"  평균 DL: {avg_dl:.0f}s, ETA: ~{eta_sec/60:.0f}분")

    # failures
    fails = [(k, v.get("error", "?")) for k, v in d.items() if v["status"] == "failed"]
    if fails:
        lines.append(f"⚠️ 실패 {len(fails)}:")
        for name, err in fails[:3]:
            lines.append(f"  {name}: {err[:60]}")
else:
    lines.append("state.json 아직 없음")

# _tmp contents
if TMP.exists():
    tmp_files = list(TMP.glob("*.tar.gz"))
    if tmp_files:
        total_mb = sum(f.stat().st_size for f in tmp_files) / (1024*1024)
        lines.append(f"🔄 다운로드 중: {len(tmp_files)}개 ({total_mb:.0f} MB)")
        for f in sorted(tmp_files)[:4]:
            mb = f.stat().st_size / (1024*1024)
            lines.append(f"  {f.name[:40]} {mb:.0f}MB")

# Disk usage
try:
    import shutil
    used = 0
    for sub in ("distraction", "gaze"):
        p = DEST / sub
        if p.exists():
            # du-like: walk
            for dirpath, _, files in os.walk(p):
                for f in files:
                    try:
                        used += (Path(dirpath) / f).stat().st_size
                    except OSError:
                        pass
    lines.append(f"💾 추출 용량: {used/(1024**3):.1f} GB")
    free = shutil.disk_usage(DEST).free / (1024**3)
    lines.append(f"   여유: {free:.0f} GB")
except Exception as e:
    pass

# Process check
try:
    import subprocess
    r = subprocess.run(["pgrep", "-f", "download_dmd.py"], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        lines.append(f"✅ 스크립트 실행 중 (pid {r.stdout.strip().replace(chr(10), ',')})")
    else:
        lines.append(f"🛑 스크립트 미실행")
except Exception:
    pass

print("\n".join(lines))
PY
)

if [[ "${1:-}" == "--send" ]]; then
    source ~/.config/dmd-bot.env
    curl -sS -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
        -d "chat_id=${TG_CHAT_ID}" --data-urlencode "text=${report}" > /dev/null
fi

echo "$report"
