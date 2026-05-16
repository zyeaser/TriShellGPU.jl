#!/usr/bin/env bash
set -euo pipefail

JULIA_VERSION="release"
PROJECT_DIR="${PROJECT_DIR:-/TriShellGPU.jl}"

echo "Installing Julia using juliaup..."
curl -fsSL https://install.julialang.org | sh -s -- --yes

export PATH="$HOME/.juliaup/bin:$PATH"

echo "Setting Julia default version..."
juliaup default "$JULIA_VERSION"

JULIA="$HOME/.juliaup/bin/julia"

echo "Julia version:"
"$JULIA" --version

cd "$PROJECT_DIR"
"$JULIA" --project=. -e 'import Pkg; Pkg.instantiate()'
