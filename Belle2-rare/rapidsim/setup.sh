#!/usr/bin/env bash
# Create a pinned, project-local checkout. Existing checkouts are left to their owner.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
src="$repo/vendor/RapidSim"
if [[ -e "$src" ]]; then
    echo "Checkout already exists at $src. See rapidsim/README.md for rebuild instructions." >&2
    exit 1
fi
mkdir -p "$repo/vendor"
git clone https://github.com/lhcb/RapidSim.git "$src"
git -C "$src" checkout --detach 5fcfc328f0924f8ecb7a9d0129aed7c415a265a6
git -C "$src" apply "$repo/rapidsim/evtgen_builtin_only.patch"
git -C "$src" apply "$repo/rapidsim/register_knunu.patch"
cp "$repo"/simulation/models/EvtKnunu.{cc,h} "$src/src/"
cp "$repo/rapidsim/pixi.lock" "$src/pixi.lock"
(cd "$src" && pixi run --frozen build)
