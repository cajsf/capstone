#!/usr/bin/env python3
"""DMD Telegram command poller — one polling iteration.

Fetches new Telegram messages addressed to this bot, executes commands,
replies. Designed to be called repeatedly from a Claude /loop or cron.

Security: only messages from the configured chat_id are processed.

Commands:
  /status    — send current download status
  /fail      — list failed files with reasons
  /log [N]   — last N log lines (default 15)
  /tmp       — current _tmp/ contents
  /disk      — disk usage
  /pause     — kill running download_dmd.py
  /resume    — restart download_dmd.py inside tmux session 'dmd'
  /help      — this list
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ENV_PATH = Path.home() / ".config" / "dmd-bot.env"
OFFSET_PATH = Path("/data/shared/DMD/_meta/tg_offset.json")
STATE_PATH = Path("/data/shared/DMD/_meta/state.json")
LOG_PATH = Path("/data/shared/DMD/_meta/download.log")
TMP_PATH = Path("/data/shared/DMD/_tmp")
DEST = Path("/data/shared/DMD")
TOTAL = 46  # 31 distraction + 15 gaze


def load_env() -> dict[str, str]:
    d = {}
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    return d


ENV = load_env()
TOKEN = ENV["TG_TOKEN"]
CHAT_ID = int(ENV["TG_CHAT_ID"])
API = f"https://api.telegram.org/bot{TOKEN}"


def tg_call(method: str, **params) -> dict:
    data = urllib.parse.urlencode(params).encode()
    with urllib.request.urlopen(f"{API}/{method}", data=data, timeout=30) as r:
        return json.loads(r.read())


def send(text: str) -> None:
    # Telegram message limit ≈ 4096 chars.
    for chunk in [text[i : i + 3900] for i in range(0, max(len(text), 1), 3900)] or [text]:
        try:
            tg_call("sendMessage", chat_id=CHAT_ID, text=chunk)
        except Exception as e:
            print(f"send failed: {e}", file=sys.stderr)


def get_updates() -> list[dict]:
    offset = 0
    if OFFSET_PATH.exists():
        try:
            offset = json.loads(OFFSET_PATH.read_text()).get("offset", 0)
        except Exception:
            pass
    try:
        r = tg_call("getUpdates", offset=offset + 1, timeout=0, allowed_updates='["message"]')
    except Exception as e:
        print(f"getUpdates failed: {e}", file=sys.stderr)
        return []
    return r.get("result", [])


def save_offset(update_id: int) -> None:
    OFFSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    OFFSET_PATH.write_text(json.dumps({"offset": update_id}))


# ─── command handlers ───

def cmd_status() -> str:
    return run_script(["/home/yg/capstone/dmd_status.sh"])


def cmd_fail() -> str:
    if not STATE_PATH.exists():
        return "state.json 없음"
    d = json.loads(STATE_PATH.read_text())
    fails = [(k, v.get("error", "?")) for k, v in d.items() if v.get("status") == "failed"]
    if not fails:
        return "✅ 실패 없음"
    out = [f"❌ 실패 {len(fails)}개:"]
    for name, err in fails:
        out.append(f"• {name}\n  {err[:150]}")
    return "\n".join(out)


def cmd_log(args: list[str]) -> str:
    n = 15
    if args and args[0].isdigit():
        n = min(int(args[0]), 80)
    if not LOG_PATH.exists():
        return "로그 없음"
    lines = LOG_PATH.read_text(errors="replace").splitlines()[-n:]
    return "— log tail —\n" + "\n".join(lines)


def cmd_tmp() -> str:
    if not TMP_PATH.exists():
        return "_tmp 비어있음"
    files = sorted(TMP_PATH.glob("*.tar.gz"))
    if not files:
        return "_tmp 비어있음 (추출 끝)"
    out = [f"🔄 다운로드 중 {len(files)}개:"]
    total = 0
    for f in files:
        mb = f.stat().st_size / (1024 * 1024)
        total += mb
        out.append(f"  {f.name[:44]} {mb:.0f}MB")
    out.append(f"합계: {total:.0f} MB")
    return "\n".join(out)


def cmd_disk() -> str:
    import shutil
    used = 0
    for sub in ("distraction", "gaze"):
        p = DEST / sub
        if p.exists():
            for dirpath, _, files in os.walk(p):
                for f in files:
                    try:
                        used += (Path(dirpath) / f).stat().st_size
                    except OSError:
                        pass
    free = shutil.disk_usage(DEST).free / (1024**3)
    return f"💾 추출: {used/(1024**3):.1f} GB\n여유: {free:.0f} GB"


def cmd_pause() -> str:
    r = subprocess.run(["pkill", "-f", "download_dmd.py"], capture_output=True)
    if r.returncode == 0:
        return "⏸ download_dmd.py 종료"
    return "실행 중인 스크립트 없음"


def cmd_resume() -> str:
    # Check if already running
    r = subprocess.run(["pgrep", "-f", "download_dmd.py"], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return "이미 실행 중"
    # Try tmux session 'dmd'
    r = subprocess.run(["tmux", "has-session", "-t", "dmd"], capture_output=True)
    if r.returncode != 0:
        subprocess.run(["tmux", "new-session", "-d", "-s", "dmd"], check=True)
    cmd = "python3 /home/yg/capstone/download_dmd.py --workers 3 2>&1 | tee -a /data/shared/DMD/_meta/download.log"
    subprocess.run(["tmux", "send-keys", "-t", "dmd", cmd, "Enter"], check=True)
    return "▶️ tmux 'dmd' 세션에서 재개"


def cmd_help() -> str:
    return (
        "명령어:\n"
        "/status — 진행 요약\n"
        "/tmp — 현재 받는 중\n"
        "/fail — 실패 목록\n"
        "/log [N] — 로그 끝 N줄\n"
        "/disk — 용량\n"
        "/pause — 중지\n"
        "/resume — 재개\n"
        "/help — 이 메시지"
    )


def run_script(cmd: list) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (r.stdout or r.stderr or "(no output)").strip()
    except subprocess.TimeoutExpired:
        return "스크립트 타임아웃"
    except Exception as e:
        return f"error: {e}"


def handle(text: str) -> str:
    parts = text.strip().split()
    if not parts:
        return ""
    cmd = parts[0].lower().lstrip("/")
    args = parts[1:]
    table = {
        "status": cmd_status, "s": cmd_status,
        "fail": cmd_fail, "failures": cmd_fail,
        "tmp": cmd_tmp,
        "disk": cmd_disk,
        "pause": cmd_pause, "stop": cmd_pause,
        "resume": cmd_resume, "start": cmd_resume,
        "help": cmd_help, "h": cmd_help,
    }
    if cmd == "log":
        return cmd_log(args)
    fn = table.get(cmd)
    if not fn:
        return f"모르는 명령: {text}\n/help 로 목록 확인"
    return fn()


def main() -> int:
    updates = get_updates()
    n_handled = 0
    for u in updates:
        save_offset(u["update_id"])
        msg = u.get("message") or u.get("edited_message")
        if not msg:
            continue
        if msg.get("chat", {}).get("id") != CHAT_ID:
            continue  # silently ignore foreign chats
        text = msg.get("text") or ""
        if not text:
            continue
        reply = handle(text)
        if reply:
            send(reply)
            n_handled += 1
    print(f"polled, handled={n_handled}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
