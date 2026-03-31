#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# run_online_demo.sh — Automated online RL demo with zeroclaw
#
# Procedure:
#   1. Launch online_rollout.py and wait for the gateway address.
#   2. Patch ~/.zeroclaw/config.toml: point the "localhost" provider at the
#      gateway and set the api key.
#   3. Ask the model "how many r's are in the word strawberry?" via zeroclaw.
#      If wrong, correct it and ask again.  If still wrong, set reward = 0.
#   4. Repeat step 3 three more times (4 total trajectories = batch_size).
#   5. Check whether online_rollout.py printed a databatch summary.
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── Paths ──────────────────────────────────────────────────────────────────
ONLINE_ROLLOUT="$REPO_ROOT/examples/experimental/inference_service/online_rollout.py"
CONFIG_YAML="$REPO_ROOT/examples/experimental/inference_service/online_rollout.yaml"
SET_REWARD="$REPO_ROOT/examples/experimental/inference_service/set_reward.py"
ZEROCLAW_CONFIG="$HOME/.zeroclaw/config.toml"

# ── Tunables ───────────────────────────────────────────────────────────────
ACTOR_PATH="/storage/openpsi/models/Qwen__Qwen3-0.6B"
REQUEST_TIMEOUT=3600
ADMIN_KEY="sk-test123456"
QUESTION="how many r's are in the word strawberry?"
CORRECT_ANSWER_RE='\b3\b|three'
GATEWAY_WAIT_SECS=600

ROLLOUT_LOG="$(mktemp /tmp/online_rollout_XXXXXX.log)"
ROLLOUT_PID=""

# ── Cleanup ────────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Cleanup"
    echo "════════════════════════════════════════════════════════════════"
    if [[ -n "$ROLLOUT_PID" ]] && kill -0 "$ROLLOUT_PID" 2>/dev/null; then
        echo "  Stopping online_rollout.py (PID $ROLLOUT_PID) ..."
        kill "$ROLLOUT_PID" 2>/dev/null || true
        wait "$ROLLOUT_PID" 2>/dev/null || true
    fi
    if [[ -f "${ZEROCLAW_CONFIG}.demo_bak" ]]; then
        cp "${ZEROCLAW_CONFIG}.demo_bak" "$ZEROCLAW_CONFIG"
        rm -f "${ZEROCLAW_CONFIG}.demo_bak"
        echo "  Restored original zeroclaw config."
    fi
    echo "  Rollout log preserved at: $ROLLOUT_LOG"
}
trap cleanup EXIT

# ── Helpers ────────────────────────────────────────────────────────────────
strip_ansi() {
    sed 's/\x1b\[[0-9;]*m//g'
}

update_zeroclaw_key() {
    local key="$1"
    if grep -q '^api_key = ' "$ZEROCLAW_CONFIG"; then
        sed -i "s|^api_key = \".*\"|api_key = \"${key}\"|" "$ZEROCLAW_CONFIG"
    else
        sed -i "1 a api_key = \"${key}\"" "$ZEROCLAW_CONFIG"
    fi
    echo "    api_key  → $key"
}

update_zeroclaw_base_url() {
    local url="$1"
    sed -i "s|^default_provider = \".*\"|default_provider = \"custom:${url}\"|" "$ZEROCLAW_CONFIG"
    echo "    default_provider → custom:$url"
}

answer_is_correct() {
    echo "$1" | grep -qiE "$CORRECT_ANSWER_RE"
}

run_set_reward() {
    local api_key="$1"
    local reward="$2"
    echo "    Setting reward=${reward} for key=${api_key}"
    PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}" \
        python3 "$SET_REWARD" "$GATEWAY_ADDR" \
            --api-key "$api_key" \
            --reward "$reward" || true
}

# ────────────────────────────────────────────────────────────────────────────
# Step 1: Launch online_rollout.py
# ────────────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════"
echo "  Step 1: Launch online_rollout.py"
echo "════════════════════════════════════════════════════════════════"

cd "$REPO_ROOT"
python3 "$ONLINE_ROLLOUT" \
    --config "$CONFIG_YAML" \
    actor.path="$ACTOR_PATH" \
    rollout.request_timeout="$REQUEST_TIMEOUT" \
    >"$ROLLOUT_LOG" 2>&1 &
ROLLOUT_PID=$!
echo "  PID : $ROLLOUT_PID"
echo "  Log : $ROLLOUT_LOG"
echo "  Waiting for gateway address (up to ${GATEWAY_WAIT_SECS}s) ..."

GATEWAY_ADDR=""
for (( i = 1; i <= GATEWAY_WAIT_SECS; i++ )); do
    if ! kill -0 "$ROLLOUT_PID" 2>/dev/null; then
        echo "  ERROR: online_rollout.py exited prematurely.  Log tail:"
        tail -40 "$ROLLOUT_LOG"
        exit 1
    fi
    if line=$(sed 's/\x1b\[[0-9;]*m//g' "$ROLLOUT_LOG" | grep -oP "Proxy gateway available at \Khttp://\S+" 2>/dev/null | head -1); then
        GATEWAY_ADDR="$line"
        break
    fi
    sleep 1
done

if [[ -z "$GATEWAY_ADDR" ]]; then
    echo "  ERROR: Timed out waiting for gateway.  Log tail:"
    tail -40 "$ROLLOUT_LOG"
    exit 1
fi
echo "  Gateway: $GATEWAY_ADDR"

# ────────────────────────────────────────────────────────────────────────────
# Step 2: Patch zeroclaw config
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Step 2: Update ~/.zeroclaw/config.toml"
echo "════════════════════════════════════════════════════════════════"

cp "$ZEROCLAW_CONFIG" "${ZEROCLAW_CONFIG}.demo_bak"
echo "  Backed up → ${ZEROCLAW_CONFIG}.demo_bak"
update_zeroclaw_base_url "$GATEWAY_ADDR"
update_zeroclaw_key "$ADMIN_KEY"
echo "  Done."

# ────────────────────────────────────────────────────────────────────────────
# Chat-and-reward round
#
# Usage:  do_round <api_key> <label>
#
#   1. Ask the strawberry question.
#   2. If wrong → correct the model and ask again.
#   3. If still wrong → set reward 0.  Otherwise → set reward 1.
# ────────────────────────────────────────────────────────────────────────────
do_round() {
    local api_key="$1"
    local label="$2"
    local session_file
    session_file="$(mktemp /tmp/zeroclaw_session_XXXXXX.json)"

    echo ""
    echo "  ── $label ──"

    # ---------- First attempt ----------
    echo "  Q: $QUESTION"
    local resp
    resp=$(ZEROCLAW_API_KEY="$api_key" zeroclaw agent -m "$QUESTION" \
        --session-state-file "$session_file" 2>&1 | strip_ansi) || true
    echo "  A: $resp"

    if answer_is_correct "$resp"; then
        echo "  ✔ Correct on first try."
        run_set_reward "$api_key" 1.0
        rm -f "$session_file"
        return
    fi

    # ---------- Second attempt ----------
    echo "  ✘ Wrong — giving corrective feedback and asking again ..."
    local correction="That's wrong. The word 'strawberry' contains 3 r's. Let me ask once more: $QUESTION"
    resp=$(ZEROCLAW_API_KEY="$api_key" zeroclaw agent -m "$correction" \
        --session-state-file "$session_file" 2>&1 | strip_ansi) || true
    echo "  A: $resp"

    if answer_is_correct "$resp"; then
        echo "  ✔ Correct on second try."
        run_set_reward "$api_key" 1.0
        rm -f "$session_file"
        return
    fi

    # ---------- Still wrong ----------
    echo "  ✘ Still wrong after two attempts — setting reward to 0."
    run_set_reward "$api_key" 0.0
    rm -f "$session_file"
}

# ────────────────────────────────────────────────────────────────────────────
# Steps 3–4: Four HITL rounds (4 trajectories = batch_size)
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Steps 3–4  (4 HITL rounds)"
echo "════════════════════════════════════════════════════════════════"
do_round "$ADMIN_KEY" "Trajectory 0"
do_round "$ADMIN_KEY" "Trajectory 1"
do_round "$ADMIN_KEY" "Trajectory 2"
do_round "$ADMIN_KEY" "Trajectory 3"

# ────────────────────────────────────────────────────────────────────────────
# Step 5: Verify the online rollout printed databatch info
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Step 5: Check online_rollout output for databatch"
echo "════════════════════════════════════════════════════════════════"

echo "  Waiting a few seconds for the rollout to process ..."
for (( i = 1; i <= 30; i++ )); do
    if grep -q "Rollout complete" "$ROLLOUT_LOG" 2>/dev/null; then
        break
    fi
    sleep 2
done

echo ""
echo "  ── Rollout log (last 40 lines) ──"
tail -40 "$ROLLOUT_LOG"
echo ""

if grep -q "Rollout complete" "$ROLLOUT_LOG"; then
    echo "  ✔ Databatch detected in online_rollout output:"
    grep "Rollout complete" "$ROLLOUT_LOG"
else
    echo "  ✘ No 'Rollout complete' message found yet."
    echo "    The rollout may still be collecting trajectories"
    echo "    (batch_size=4 — need 4 completed trajectories with rewards)."
fi

echo ""
echo "  Demo finished."
