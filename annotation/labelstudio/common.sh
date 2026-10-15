# Shared settings of launch.sh and stop.sh. cycle.sh exports DATA_ROOT, OUTPUT_ROOT and LS_DATA_DIR
# for the run it drives; set them yourself to call a script directly.
# Every other value can be overridden from the environment before the call, for example LS_PORT=8800.

LS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANNOTATION_DIR="$(dirname "$LS_DIR")"

# the dataset of the run (the frames and annotations/clip.json) and the output of the loop
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the dataset directory of the run}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to the output directory of the loop}"
RUN_NAME="${RUN_NAME:-tracked}"
VIDEO_ID="${VIDEO_ID:-clip}"
# the directory with round_XX_queries.json and round_XX_answers.json of the video; the loop names
# the strategy directory after the "name" inside the strategy config thc_wpu_duw, not after the file
STRATEGY_DIR="THC+WPU_Coresetfilter"
ORACLE_DIR="$OUTPUT_ROOT/$RUN_NAME/simplebaseline/$STRATEGY_DIR/$VIDEO_ID/oracle"

# the environment of Label Studio (its own, isolated; the loop uses annotation/.venv)
LS_VENV="$ANNOTATION_DIR/.venv-labelstudio"
LS_DATA_DIR="${LS_DATA_DIR:?set LS_DATA_DIR to the directory of the Label Studio database}"
LS_HOST="127.0.0.1"
LS_PORT="${LS_PORT:-8765}"
LS_URL="http://localhost:$LS_PORT"
LS_PID_FILE="$LS_DATA_DIR/label-studio.pid"
LS_LOG="$LS_DATA_DIR/label-studio.log"

# a throwaway local account; it exists only in the local database of the run
LS_USER="demo@example.com"
LS_PASSWORD="demo-password-2026"
LS_TOKEN="vatl4pose-demo-token-0000000000000000"
