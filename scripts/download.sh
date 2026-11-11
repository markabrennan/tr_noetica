#!/usr/bin/env bash
# Download Qwen2.5-Instruct weights + tokenizer from Hugging Face with curl.
# Usage: scripts/download.sh [size]   (size: 0.5B | 1.5B | 3B, default 1.5B)
set -euo pipefail
SIZE="${1:-1.5B}"
REPO="Qwen/Qwen2.5-${SIZE}-Instruct"
DEST="models/qwen2.5-${SIZE,,}-instruct"
mkdir -p "$DEST"
for f in config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors; do
  if [[ -s "$DEST/$f" ]]; then echo "have $f"; continue; fi
  echo "fetching $f"
  curl -fL --retry 3 -o "$DEST/$f.part" "https://huggingface.co/$REPO/resolve/main/$f"
  mv "$DEST/$f.part" "$DEST/$f"
done
echo "done: $DEST"
