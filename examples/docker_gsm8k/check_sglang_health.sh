#!/bin/bash
# SGLang Server Health Check Script
# Checks if SGLang server is running and responding

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "SGLang Server Health Check"
echo "=========================================="
echo ""

# Check if SGLang server process is running
echo "1. Checking SGLang server processes..."
SGLANG_PROCS=$(ps aux | grep -i sglang | grep -v grep | wc -l)
if [ "$SGLANG_PROCS" -gt 0 ]; then
    echo "   ✅ Found $SGLANG_PROCS SGLang process(es)"
    ps aux | grep -i sglang | grep -v grep
else
    echo "   ❌ No SGLang processes found"
fi
echo ""

# Check for server logs
echo "2. Checking SGLang server logs..."
LOG_DIRS=$(find outputs/grpo/logs -type d -name "*" 2>/dev/null | head -5)
if [ -n "$LOG_DIRS" ]; then
    echo "   ✅ Found log directories:"
    for dir in $LOG_DIRS; do
        echo "      - $dir"
        if [ -f "$dir/llm_server.log" ]; then
            echo "        Last 5 lines:"
            tail -5 "$dir/llm_server.log" | sed 's/^/          /'
        fi
    done
else
    echo "   ⚠️  No log directories found"
fi
echo ""

# Check for connection errors in logs
echo "3. Checking for connection errors..."
ERROR_COUNT=0
for log_file in $(find outputs/grpo/logs -name "*.log" 2>/dev/null); do
    ERRORS=$(grep -i "connection.*refused\|connection.*failed" "$log_file" 2>/dev/null | wc -l)
    if [ "$ERRORS" -gt 0 ]; then
        ERROR_COUNT=$((ERROR_COUNT + ERRORS))
        echo "   ⚠️  Found $ERRORS connection errors in: $log_file"
        echo "      Last error:"
        grep -i "connection.*refused\|connection.*failed" "$log_file" 2>/dev/null | tail -1 | sed 's/^/         /'
    fi
done
if [ "$ERROR_COUNT" -eq 0 ]; then
    echo "   ✅ No connection errors found in logs"
else
    echo "   ❌ Total connection errors: $ERROR_COUNT"
fi
echo ""

# Check GPU memory
echo "4. Checking GPU memory..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    echo ""
    MEM_USAGE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    MEM_PERCENT=$((MEM_USAGE * 100 / MEM_TOTAL))
    if [ "$MEM_PERCENT" -gt 95 ]; then
        echo "   ⚠️  GPU memory usage is very high: ${MEM_PERCENT}%"
    elif [ "$MEM_PERCENT" -gt 85 ]; then
        echo "   ⚠️  GPU memory usage is high: ${MEM_PERCENT}%"
    else
        echo "   ✅ GPU memory usage: ${MEM_PERCENT}%"
    fi
else
    echo "   ⚠️  nvidia-smi not available"
fi
echo ""

# Check system memory
echo "5. Checking system memory..."
if command -v free &> /dev/null; then
    free -h
    MEM_AVAIL=$(free -m | awk 'NR==2{printf "%.0f", $7*100/$2}')
    if [ "$MEM_AVAIL" -lt 10 ]; then
        echo "   ⚠️  System memory is very low: ${MEM_AVAIL}% available"
    else
        echo "   ✅ System memory: ${MEM_AVAIL}% available"
    fi
else
    echo "   ⚠️  free command not available"
fi
echo ""

# Check for OOM kills
echo "6. Checking for OOM kills..."
if command -v dmesg &> /dev/null; then
    OOM_COUNT=$(dmesg | grep -i "out of memory\|killed.*python\|killed.*sglang" | wc -l)
    if [ "$OOM_COUNT" -gt 0 ]; then
        echo "   ❌ Found $OOM_COUNT OOM kill(s):"
        dmesg | grep -i "out of memory\|killed.*python\|killed.*sglang" | tail -3 | sed 's/^/      /'
    else
        echo "   ✅ No OOM kills found"
    fi
else
    echo "   ⚠️  dmesg not available"
fi
echo ""

# Check checkpoints
echo "7. Checking checkpoints..."
EXPERIMENT_NAME="${EXPERIMENT_NAME:-gsm8k-grpo-full-local}"
TRIAL_NAME="${TRIAL_NAME:-trial0}"
# Checkpoints are saved under {fileroot}/checkpoints/{username}/{experiment_name}/{trial_name}
# Try both with and without "root" username
CHECKPOINT_DIR1="outputs/grpo/checkpoints/${EXPERIMENT_NAME}/${TRIAL_NAME}"
CHECKPOINT_DIR2="outputs/grpo/checkpoints/root/${EXPERIMENT_NAME}/${TRIAL_NAME}"
if [ -d "$CHECKPOINT_DIR2" ]; then
    CHECKPOINT_DIR="$CHECKPOINT_DIR2"
elif [ -d "$CHECKPOINT_DIR1" ]; then
    CHECKPOINT_DIR="$CHECKPOINT_DIR1"
else
    CHECKPOINT_DIR=""
fi

if [ -n "$CHECKPOINT_DIR" ] && [ -d "$CHECKPOINT_DIR" ]; then
    # Count checkpoints in default subdirectory
    DEFAULT_DIR="$CHECKPOINT_DIR/default"
    if [ -d "$DEFAULT_DIR" ]; then
        CHECKPOINT_COUNT=$(ls -1d "$DEFAULT_DIR"/epoch*globalstep* 2>/dev/null | wc -l)
        echo "   ✅ Found $CHECKPOINT_COUNT checkpoint(s) in $CHECKPOINT_DIR/default"
        echo "      Latest checkpoints:"
        ls -1t "$DEFAULT_DIR"/epoch*globalstep* 2>/dev/null | head -5 | sed 's|.*/|        |'
    else
        CHECKPOINT_COUNT=$(ls -1 "$CHECKPOINT_DIR" 2>/dev/null | wc -l)
        echo "   ✅ Found $CHECKPOINT_COUNT checkpoint(s) in $CHECKPOINT_DIR"
    fi
else
    echo "   ⚠️  No checkpoints found at: $CHECKPOINT_DIR1 or $CHECKPOINT_DIR2"
fi
echo ""

# Check recovery info
echo "8. Checking recovery info..."
# Recovery info is saved under {fileroot}/recover/{username}/{experiment_name}/{trial_name}
RECOVER_DIR1="outputs/grpo/recover/${EXPERIMENT_NAME}/${TRIAL_NAME}"
RECOVER_DIR2="outputs/grpo/recover/root/${EXPERIMENT_NAME}/${TRIAL_NAME}"
RECOVER_INFO_DIR1="outputs/grpo/checkpoints/${EXPERIMENT_NAME}/${TRIAL_NAME}/recover_info"
RECOVER_INFO_DIR2="outputs/grpo/checkpoints/root/${EXPERIMENT_NAME}/${TRIAL_NAME}/recover_info"

if [ -d "$RECOVER_INFO_DIR2" ]; then
    RECOVER_DIR="$RECOVER_INFO_DIR2"
elif [ -d "$RECOVER_INFO_DIR1" ]; then
    RECOVER_DIR="$RECOVER_INFO_DIR1"
elif [ -d "$RECOVER_DIR2" ]; then
    RECOVER_DIR="$RECOVER_DIR2"
elif [ -d "$RECOVER_DIR1" ]; then
    RECOVER_DIR="$RECOVER_DIR1"
else
    RECOVER_DIR=""
fi

if [ -n "$RECOVER_DIR" ] && [ -d "$RECOVER_DIR" ]; then
    RECOVER_FILES=$(ls -1 "$RECOVER_DIR" 2>/dev/null | wc -l)
    echo "   ✅ Found $RECOVER_FILES recovery file(s) in $RECOVER_DIR"
    if [ -f "$RECOVER_DIR/step_info.json" ]; then
        echo "      Recovery info (step_info.json):"
        cat "$RECOVER_DIR/step_info.json" 2>/dev/null | sed 's/^/        /' || echo "        (could not read)"
    fi
else
    echo "   ⚠️  No recovery info found"
fi
echo ""

echo "=========================================="
echo "Health Check Complete"
echo "=========================================="
echo ""
echo "If SGLang server is down:"
echo "  1. Check logs for crash reason"
echo "  2. Apply fixes (memory reduction, etc.)"
echo "  3. Resume training: bash examples/docker_gsm8k/run_full_training.sh"
echo ""

