#!/bin/bash
set -euo pipefail
STUDIO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$STUDIO_ROOT"
export PYTHONNOUSERSITE=1
mkdir -p .state outputs
if [[ ! -x .venv/bin/python ]] || [[ ! -f .state/venv_location ]] || [[ "$(cat .state/venv_location)" != "$STUDIO_ROOT" ]]; then
  echo '首次启动：创建当前路径专用环境，安装随包离线依赖…'
  if [[ -d .venv ]]; then mv .venv ".state/old-venv-$(date +%s)"; fi
  /usr/bin/python3 -m venv --system-site-packages .venv
  .venv/bin/python -m pip install --ignore-installed --no-index --find-links environment/wheels -r environment/requirements.txt
  echo "$STUDIO_ROOT" > .state/venv_location
fi
PORT="${STUDIO_PORT:-8765}"
set +e
.venv/bin/python - "$PORT" "$STUDIO_ROOT" <<'PY'
import sys,urllib.request
try:
    import json
    data=json.load(urllib.request.urlopen('http://127.0.0.1:'+sys.argv[1]+'/api/bootstrap',timeout=2))
    if data.get('package_root')!=sys.argv[2]:
        print('此端口已被另一份工作台占用，请关闭另一份，或设置 STUDIO_PORT=8766。',flush=True)
        sys.exit(2)
except Exception:sys.exit(1)
PY
STUDIO_EXISTING=$?
set -e
if [[ "$STUDIO_EXISTING" == 2 ]]; then exit 2; fi
if [[ "$STUDIO_EXISTING" == 0 ]]; then
  .venv/bin/python - "$PORT" <<'PYOPEN'
import sys
sys.path.insert(0,'app')
from server import open_browser
open_browser('http://127.0.0.1:'+sys.argv[1])
PYOPEN
  exit 0
fi
.venv/bin/python tools/check_environment.py 2>&1 | tee .state/environment.log
chmod +x browser/chrome-linux-arm64/chrome browser/chrome-linux-arm64/chrome_crashpad_handler
exec .venv/bin/python app/server.py --port "$PORT" "$@"
