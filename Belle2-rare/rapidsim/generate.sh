#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
src="$repo/vendor/RapidSim"
out=${2:-"$repo/runs/rapidsim_signal"}
n=${1:-100000}
mkdir -p "$(dirname "$out")"
mkdir "$out"
cp "$repo"/rapidsim/BToKnunu.{decay,config} "$out/"
cd "$out"
RAPIDSIM_EVTGEN_BUILTIN_ONLY=1 pixi run --manifest-path "$src/pixi.toml" --frozen -- "$src/.build/src/RapidSim.exe" BToKnunu "$n" 1
