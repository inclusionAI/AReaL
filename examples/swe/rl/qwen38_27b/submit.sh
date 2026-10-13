#!/usr/bin/env bash
# Usage: QWEN_ENV_FILE=/absolute/path/to/private/site.env bash submit.sh
set -euo pipefail
umask 077
recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$recipe_dir/runtime.sh"
: "${QWEN_CONTROLLER_NODE:?Set the controller node in the site profile}"
export QWEN_SOURCE_REVISION
QWEN_SOURCE_REVISION=$(git -C "$QWEN_REPO" rev-parse HEAD)
if [[ -n $(git -C "$QWEN_REPO" status --porcelain) ]]; then
  echo 'Commit the reviewed source before submitting' >&2
  exit 1
fi
checkout_root=$(realpath "$QWEN_REPO")
for output_dir in "$EXPERIMENT_ROOT" "$QWEN_OUTPUT_ROOT"; do
  case "$(realpath -m "$output_dir")/" in
    "$checkout_root/"*) echo 'Keep run files and output outside the Git checkout' >&2; exit 1 ;;
  esac
done
mkdir -p "$EXPERIMENT_ROOT" "$QWEN_OUTPUT_ROOT/logs"
if [[ -f $QWEN_CONFIG ]]; then
  cmp "$recipe_dir/train.yaml" "$QWEN_CONFIG"
  test "$(cat "$EXPERIMENT_ROOT/source-revision")" = "$QWEN_SOURCE_REVISION"
else
  cp "$recipe_dir/train.yaml" "$QWEN_CONFIG"
  printf '%s\n' "$QWEN_SOURCE_REVISION" > "$EXPERIMENT_ROOT/source-revision"
fi
export QWEN_SUBMITTED_RESERVATION=$QWEN_RESERVATION QWEN_SUBMITTED_NODELIST=$QWEN_NODELIST
args=(--parsable --partition="$QWEN_PARTITION" --nodelist="$QWEN_CONTROLLER_NODE"
  --chdir="$QWEN_REPO" --job-name="gray19-$QWEN_ARENA_TRIAL"
  --output="$QWEN_OUTPUT_ROOT/logs/gray19-$QWEN_ARENA_TRIAL-ctl-%j.out"
  --error="$QWEN_OUTPUT_ROOT/logs/gray19-$QWEN_ARENA_TRIAL-ctl-%j.err"
  --export=ALL)
if [[ -n $QWEN_RESERVATION ]]; then
  args+=(--reservation="$QWEN_RESERVATION")
fi
exec sbatch "${args[@]}" "$recipe_dir/train.sbatch"
