#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[AReaL] Building English version..."
jupyter-book build "$SCRIPT_DIR/en" --all --path-output "$SCRIPT_DIR/_build/en"

echo "[AReaL] Building Chinese version..."
jupyter-book build "$SCRIPT_DIR/zh" --all --path-output "$SCRIPT_DIR/_build/zh"

# Create root index page with auto-redirect based on localStorage
ROOT_INDEX="$SCRIPT_DIR/_build/index.html"
mkdir -p "$SCRIPT_DIR/_build"

cat > "$ROOT_INDEX" <<'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>AReaL Documentation</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <style>
    body{font:14px/1.4 system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;padding:40px;max-width:720px;margin:auto;color:#222}
    a{color:#0969da;text-decoration:none}a:hover{text-decoration:underline}
    .lang-links{margin-top:1.2rem;display:flex;gap:1rem}
    .note{margin-top:2rem;font-size:12px;color:#666}
  </style>
  <script>
    (function(){
      var stored = null;
      try{stored = localStorage.getItem('areal-doc-lang');}catch(e){}
      var path = (stored === 'zh') ? 'zh/' : (stored === 'en') ? 'en/' : null;
      if(path){ window.location.replace(path); }
    })();
  </script>
</head>
<body>
  <h1>AReaL Documentation</h1>
  <p>Select language:</p>
  <p class="lang-links"><a href="en/">English</a> <a href="zh/">中文</a></p>
  <p class="note">Auto-redirect uses your last choice if stored; else pick above.</p>
</body>
</html>
EOF

echo "Build complete!"
echo "  English: _build/en/index.html"
echo "  Chinese: _build/zh/index.html"
echo "  Root:    _build/index.html"
