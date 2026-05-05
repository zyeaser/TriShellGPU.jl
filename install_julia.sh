#!/usr/bin/env bash
# =============================================================================
# install_julia.sh
#
# Installs Julia on the Euler login node (no root required).
# Run once from the FinalProject root:
#   bash install_julia.sh
#
# What this does:
#   1. Installs juliaup (the Julia version manager) into ~/.juliaup
#   2. Adds Julia 1.12.6 (the version pinned in Manifest.toml)
#   3. Sets 1.12.6 as the default
#   4. Adds juliaup to your PATH in ~/.bashrc (if not already there)
#   5. Instantiates this project's Julia packages
# =============================================================================

set -euo pipefail

JULIA_VERSION="1.12.6"
PROJECT_DIR="${PROJECT_DIR:-$HOME/repo759/FinalProject}"

echo "============================================================"
echo "Julia installer for Euler login node"
echo "Target version : Julia $JULIA_VERSION"
echo "Project dir    : $PROJECT_DIR"
echo "Home           : $HOME"
echo "============================================================"
echo ""

# ── Step 1: Install juliaup if not already present ───────────────────────────
if command -v juliaup &>/dev/null; then
    echo "[1/5] juliaup already installed: $(juliaup --version 2>&1 || true)"
else
    echo "[1/5] Installing juliaup..."
    curl -fsSL https://install.julialang.org | sh -s -- --yes
    echo "      juliaup installed."
fi

# ── Step 2: Make juliaup available in this shell session ──────────────────────
export PATH="$HOME/.juliaup/bin:$PATH"

# ── Step 3: Install Julia 1.12.6 ─────────────────────────────────────────────
echo ""
echo "[2/5] Installing Julia $JULIA_VERSION via juliaup..."
if juliaup status 2>&1 | grep -q "$JULIA_VERSION"; then
    echo "      Julia $JULIA_VERSION already installed."
else
    juliaup add "$JULIA_VERSION"
fi

# ── Step 4: Set Julia 1.12.6 as the default ──────────────────────────────────
echo ""
echo "[3/5] Setting Julia $JULIA_VERSION as default..."
juliaup default "$JULIA_VERSION"

# ── Step 5: Add ~/.juliaup/bin to ~/.bashrc (idempotent) ─────────────────────
echo ""
echo "[4/5] Ensuring ~/.juliaup/bin is on PATH in ~/.bashrc..."
BASHRC="$HOME/.bashrc"
MARKER='# juliaup'
if grep -qF "$MARKER" "$BASHRC" 2>/dev/null; then
    echo "      PATH entry already present in ~/.bashrc."
else
    {
        echo ""
        echo "$MARKER"
        echo 'export PATH="$HOME/.juliaup/bin:$PATH"'
    } >> "$BASHRC"
    echo "      Added to ~/.bashrc."
fi

# ── Step 6: Confirm install and instantiate the project ──────────────────────
JULIA="$HOME/.juliaup/bin/julia"

echo ""
echo "[5/5] Julia version check and project instantiation..."
"$JULIA" --version

cd "$PROJECT_DIR"

# Write LocalPreferences.toml before Pkg.instantiate() so CUDA.jl precompiles
# with the local toolkit setting and avoids "no CUDA runtime" warnings.
echo ""
echo "Writing CUDA local-toolkit preference..."
if ! grep -q 'CUDA_Runtime_jll' LocalPreferences.toml 2>/dev/null; then
    cat >> LocalPreferences.toml << 'TOML'

[CUDA_Runtime_jll]
local = true
TOML
    echo "      Written."
else
    echo "      Already set."
fi

echo ""
echo "Instantiating Julia packages for this project (may take a few minutes)..."
"$JULIA" --project=. -e 'import Pkg; Pkg.instantiate()'

echo ""
echo "============================================================"
echo "Done! Julia $JULIA_VERSION is ready."
echo ""
echo "To use Julia in new shells, run:"
echo "  source ~/.bashrc"
echo ""
echo "Then launch Julia with:"
echo "  julia --project=."
echo "============================================================"
