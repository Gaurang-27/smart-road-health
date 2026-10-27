#!/bin/sh
# Run on the target Nano, with JetPack's trtexec, not on a desktop GPU.
set -eu
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: sh tools/build_engine.sh models/pothole.onnx [workspace-MiB]" >&2
    exit 2
fi
model=$1
workspace=${2:-256}
case "$model" in *.onnx) ;; *) echo "Expected an .onnx file" >&2; exit 2 ;; esac
case "$workspace" in ''|*[!0-9]*|0) echo "Workspace must be a positive integer" >&2; exit 2 ;; esac
test -f "$model"
engine=${model%.onnx}.engine
if [ -e "$engine" ]; then
    echo "Engine already exists: $engine. Move it aside before rebuilding." >&2
    exit 1
fi
temporary=$engine.building
trap 'rm -f "$temporary"' EXIT HUP INT TERM
/usr/src/tensorrt/bin/trtexec --onnx="$model" --saveEngine="$temporary" \
    --explicitBatch --fp16 --workspace="$workspace" --buildOnly
mv "$temporary" "$engine"
echo "Built $engine on this device"
