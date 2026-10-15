#!/usr/bin/env bash
# Install the annotation tool in its own environment (annotation/.venv); leave other environments unchanged.
set -euo pipefail
ANNOTATION_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "$ANNOTATION_ROOT")
mkdir -p "$ANNOTATION_ROOT/.deps"
if [[ ! -x "$ANNOTATION_ROOT/.venv/bin/python" ]]; then
  uv venv --python 3.12 "$ANNOTATION_ROOT/.venv"
fi
for repo in SAMIDARE sam2; do
  revision=$("$ANNOTATION_ROOT/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$ANNOTATION_ROOT/upstream.json" "$repo")
  case "$repo" in
    SAMIDARE) url=https://github.com/ZabuZabuZabu/SAMIDARE.git ;;
    sam2) url=https://github.com/facebookresearch/sam2.git ;;
  esac
  if [[ ! -d "$ANNOTATION_ROOT/.deps/$repo/.git" ]] || \
      ! git -C "$ANNOTATION_ROOT/.deps/$repo" rev-parse --verify HEAD >/dev/null 2>&1; then
    echo "Fetching pinned $repo commit $revision"
    git init "$ANNOTATION_ROOT/.deps/$repo"
    GIT_TERMINAL_PROMPT=0 git -C "$ANNOTATION_ROOT/.deps/$repo" fetch --depth 1 "$url" "$revision"
    git -C "$ANNOTATION_ROOT/.deps/$repo" checkout --detach FETCH_HEAD
  fi
  installed=$(git -C "$ANNOTATION_ROOT/.deps/$repo" rev-parse HEAD)
  if [[ "$installed" != "$revision" ]]; then
    echo "$repo is at $installed; expected $revision. Move that checkout away and run setup again." >&2
    exit 1
  fi
  if [[ -n $(git -C "$ANNOTATION_ROOT/.deps/$repo" status --porcelain --untracked-files=no) ]]; then
    echo "$repo has local source changes; restore or preserve them before setup." >&2
    exit 1
  fi
done
uv pip sync --python "$ANNOTATION_ROOT/.venv/bin/python" --exclude-newer 2030-01-01 \
  --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu124 \
  "$ANNOTATION_ROOT/requirements.lock"
uv pip install --python "$ANNOTATION_ROOT/.venv/bin/python" --no-deps \
  -e "$REPO_ROOT[cu124]"
# Explicitly disable the optional connected-components extension and its postprocessing.
SAM2_BUILD_CUDA=0 uv pip install --python "$ANNOTATION_ROOT/.venv/bin/python" \
  --no-build-isolation --no-deps -e "$ANNOTATION_ROOT/.deps/sam2"
mkdir -p "$ANNOTATION_ROOT/.deps/checkpoints"
"$ANNOTATION_ROOT/.venv/bin/python" - "$ANNOTATION_ROOT" <<'PY'
import hashlib, json, os, pathlib, subprocess, sys, urllib.request
root = pathlib.Path(sys.argv[1])
checkpoint = root / '.deps/checkpoints/sam2.1_hiera_large.pt'
if not checkpoint.exists():
    partial = checkpoint.with_suffix('.partial')
    urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt', partial)
    partial.replace(checkpoint)
expected = json.loads((root/'models.json').read_text())[checkpoint.name]
with checkpoint.open('rb') as handle:
    actual = hashlib.file_digest(handle, 'sha256').hexdigest()
if actual != expected:
    raise RuntimeError(f'{checkpoint}: SHA256 {actual}, expected {expected}; preserve the file and download a clean checkpoint')
commits = {name: subprocess.check_output(['git', '-C', str(root/'.deps'/name), 'rev-parse', 'HEAD'], text=True).strip()
           for name in ('SAMIDARE', 'sam2')}
(root/'.deps/upstream_commits.json').write_text(json.dumps(commits, indent=2)+'\n')
import sam2, torch, torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
os.environ['TORCH_HOME'] = str(root/'.deps/torch-models')
FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True,
                                                   map_location='cpu', weights_only=True)
print('Tracking dependencies installed:', torch.__version__, torchvision.__version__)
print('CUDA visible:', torch.cuda.is_available())
PY
uv pip check --python "$ANNOTATION_ROOT/.venv/bin/python"
"$ANNOTATION_ROOT/.venv/bin/python" - "$ANNOTATION_ROOT" <<'PY'
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
(root / '.deps/requirements.lock.sha256').write_text(hashlib.sha256((root / 'requirements.lock').read_bytes()).hexdigest() + '\n')
PY
