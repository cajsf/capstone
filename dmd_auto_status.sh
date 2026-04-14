#!/usr/bin/env bash
# DMD auto status reporter — sends /status every hour until download completes.
# Run independently in a detached tmux session:
#   tmux new -d -s dmd-auto '/home/yg/capstone/dmd_auto_status.sh'

set -uo pipefail
INTERVAL=3600   # 1 hour
STATE=/data/shared/DMD/_meta/state.json
TOTAL=46

send_final() {
    source ~/.config/dmd-bot.env
    curl -sS -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
        -d "chat_id=${TG_CHAT_ID}" \
        --data-urlencode "text=$1" > /dev/null
}

echo "[$(date '+%F %T')] auto-status daemon start (interval ${INTERVAL}s)"
while true; do
    /home/yg/capstone/dmd_status.sh --send
    # completion check
    if [[ -f "$STATE" ]]; then
        done=$(python3 -c "import json; d=json.load(open('$STATE')); print(sum(1 for v in d.values() if v['status'] in ('extracted','verified')))" 2>/dev/null || echo 0)
        failed=$(python3 -c "import json; d=json.load(open('$STATE')); print(sum(1 for v in d.values() if v['status']=='failed'))" 2>/dev/null || echo 0)
        if [[ "$done" -ge "$TOTAL" ]]; then
            send_final "🎉 DMD 전체 다운로드 완료! ($done/$TOTAL)"
            echo "[$(date '+%F %T')] all done, exiting"
            exit 0
        fi
        if [[ "$((done+failed))" -ge "$TOTAL" ]]; then
            send_final "⚠️ 완료=$done, 실패=$failed — /fail 로 확인"
            # 실패 있어도 계속 감시 (resume 가능)
        fi
    fi
    sleep "$INTERVAL"
done
