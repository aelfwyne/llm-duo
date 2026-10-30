#!/usr/bin/env bash
# Build LLM Duo in release mode and install it for the current user (~/.local).
# Usage: scripts/install-local.sh            (build + install)
#        scripts/install-local.sh --uninstall
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="$HOME/.local/bin/llm-duo"
DESKTOP="$HOME/.local/share/applications/io.github.llmduo.desktop"
ICON="$HOME/.local/share/icons/hicolor/scalable/apps/io.github.llmduo.svg"

if [[ "${1:-}" == "--uninstall" ]]; then
    rm -f "$BIN" "$DESKTOP" "$ICON"
    echo "Removed LLM Duo (settings in ~/.config/llm-duo and logs in ~/.local/share/llm-duo were kept)."
    exit 0
fi

export QMAKE="${QMAKE:-$(command -v qmake6 || command -v qmake)}"
echo "Using qmake: $QMAKE ($("$QMAKE" -query QT_VERSION))"
# a fresh stamp every install: forces a rebuild even when unpacked files carry old timestamps,
# and shows up in the window title so you can see the new build is the one running
export LLM_DUO_BUILD="$(date '+%Y-%m-%d %H:%M')"
touch src/main.rs src/bridge.rs src/duo/*.rs qml/*.qml
cargo build --release

install -Dm755 target/release/llm-duo "$BIN"
install -Dm644 packaging/io.github.llmduo.desktop "$DESKTOP"
# absolute path: ~/.local/bin is often not on the desktop session's PATH until you log in again
sed -i "s|^Exec=.*|Exec=$BIN|" "$DESKTOP"
install -Dm644 packaging/io.github.llmduo.svg "$ICON"
command -v update-desktop-database >/dev/null && update-desktop-database "$HOME/.local/share/applications" || true
command -v kbuildsycoca6 >/dev/null && kbuildsycoca6 >/dev/null 2>&1 || true

echo "Installed: $BIN (build $LLM_DUO_BUILD)"
echo "Start it from the application launcher (LLM Duo) or run: llm-duo"
