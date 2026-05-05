# GPU-Accelerated Triangular Shell Finite Element Solver

A fully GPU-accelerated finite element solver for triangular shell elements, implemented in Julia using CUDA.jl. The solver assembles and solves the global stiffness system entirely on the GPU — element stiffness computation, sparse matrix assembly, boundary condition enforcement, and iterative solve — with no CPU fallback for large problems.

Developed for CS 759 (High Performance Computing), UW–Madison.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup on Euler](#setup-on-euler)
- [Running the Examples](#running-the-examples)
- [Running on the Euler Cluster (SLURM)](#running-on-the-euler-cluster-slurm)
- [Experiments and Benchmarks](#experiments-and-benchmarks)
- [CPU Reference Implementations](#cpu-reference-implementations)
- [GPU Pipeline Architecture](#gpu-pipeline-architecture)
- [Results and Reports](#results-and-reports)

---

## Overview

The solver targets the **triangular shell element** with 6 DOFs per node (ux, uy, w, θx, θy, θz). It reproduces the CPU reference formulation exactly while offloading every compute-heavy stage to the GPU:

- **One CUDA thread per element** for the 18×18 local stiffness matrix (ke)
- **Atomic scatter** into a pre-built CSR sparse matrix
- **Jacobi-preconditioned Conjugate Gradient** (PCG) via cuBLAS / CUSOLVER
- Optional **geometric stiffness** kernel for linear buckling analysis

Three problem types are supported: membrane, bending, and buckling (eigenvalue).

---

## Project Structure

```
FinalProject/
├── install_julia.sh                  # One-time Julia installer for Euler login node
├── run_line_pressure_gpu.sh          # SLURM job script for the line-pressure example
├── Project.toml                      # Julia package dependencies
│
├── src/
│   ├── gpu/                          # GPU kernel suite
│   │   ├── gpu_solve.jl              # Top-level solver orchestrator (entry point)
│   │   ├── device_functions.jl       # Inline math helpers (shape functions, Jacobian, etc.)
│   │   ├── element_stiffness_kernel.jl  # CUDA kernel: 18×18 ke per element
│   │   ├── assembly_kernel.jl        # GPU CSR sparse matrix assembly (atomic adds)
│   │   ├── dirichlet_kernel.jl       # GPU Dirichlet boundary condition enforcement
│   │   ├── pcg_solver.jl             # PCG solver (custom CUDA, cuBLAS, CUSOLVER)
│   │   └── geometric_stiffness_kernel.jl  # GPU geometric stiffness for buckling
│   ├── TriShellFiniteElement_Membrane_and_Bending.jl  # Standalone CPU example script
│   └── TriShellFiniteElement.jl/     # Julia package: custom IP3/IP6 interpolations
│       ├── Project.toml
│       ├── src/TriShellFiniteElement.jl
│       └── test/                     # 16 unit tests (ke, Kg, Jacobian, assembly, etc.)
│
├── example/
│   ├── gpu_plate_membrane_line_pressure.jl    # Line load example (in-plane & out-of-plane)
│   └── gpu_plate_membrane_surface_pressure.jl # Surface pressure example
│
│
├── cpu_reference/                    # CPU-only reference implementations (ground truth)
│   ├── base_plate_r0.jl
│   ├── Plate_membrane_t100/
│   ├── Plate_membrane_t2/
│   ├── Plate_bending_t100/
│   ├── Plate_bending_t2/
│   ├── Plate_membrane_t100_Pressure/
│   ├── Plate_membrane_t2_Pressure/
│   ├── Plate_bending_t100_Pressure/
│   └── Plate_bending_t2_Pressure/
│
├── misc/
│   ├── benchmarks/
│   │   ├── run_all.sh                # Master SLURM script: runs all benchmarks + plots
│   │   ├── run_membrane.jl           # Membrane benchmark sweep (CPU + 3 GPU solvers)
│   │   ├── run_bending.jl            # Bending benchmark sweep
│   │   ├── run_buckling.jl           # Buckling eigenvalue benchmark
│   │   └── speedup_plots.jl          # Generates speedup/timing plots from CSVs
│   └── plots/                        # Generated benchmark plots
│
└── resources/                        # Reference papers, result files, comparison plots
    ├── GPU-accelerated-Finite-Element-Method-using-Python-and-CUDA/
    ├── performance_comparison_cpu_h100_rtx4000ada.csv
    ├── result_comparison.xlsx
    └── *.png, *.txt                  # Benchmark result files and speedup charts
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Julia | 1.12.6 (via juliaup) |
| CUDA Toolkit | 13.0.0 (available as `nvidia/cuda/13.0.0` on Euler) |
| GPU | Any NVIDIA GPU with CUDA compute capability ≥ 6.0 |

Julia packages (installed automatically via `Pkg.instantiate()`):

- `CUDA` — GPU support via CUDA.jl
- `Ferrite` — FEM framework (mesh, DOF handler, grid generation)
- `FerriteGmsh` — Gmsh mesh import
- `Tensors`, `StaticArrays` — performant math types
- `BenchmarkTools` — timing utilities
- `TriShellFiniteElement` — custom package in `src/TriShellFiniteElement.jl/`

---

## Setup on Euler

**Run once from the FinalProject directory on the Euler login node:**

```bash
bash install_julia.sh
source ~/.bashrc
```

This script:
1. Installs `juliaup` (Julia version manager) into `~/.juliaup`
2. Adds Julia 1.12.6 and sets it as the default
3. Appends `~/.juliaup/bin` to `PATH` in `~/.bashrc`
4. Runs `Pkg.instantiate()` to download and precompile all dependencies

After that, run the one-time package registration (registers the local `TriShellFiniteElement` package and configures CUDA.jl to use the system CUDA toolkit):

```bash
julia --project=. experiments/setup_euler.jl
```

---

## Running the Examples

Both examples can be run locally (with a GPU) or via SLURM on Euler.

### Line pressure example

Solves a 100 mm × 1000 mm plate (t = 2 mm or 100 mm) under a line load of 1000 N/mm applied at mid-span. Runs both in-plane (ux) and out-of-plane (w) load cases across a sweep of mesh sizes and writes results to `example/results_line_pressure_mesh_thin.txt`.

```bash
julia --project=. example/gpu_plate_membrane_line_pressure.jl
```

### Surface pressure example

Solves the same plate geometry under a uniform distributed surface pressure, for both membrane and bending formulations.

```bash
julia --project=. example/gpu_plate_membrane_surface_pressure.jl
```

---

## Running on the Euler Cluster (SLURM)

### Line pressure example (recommended starting point)

```bash
sbatch run_line_pressure_gpu.sh
```

This script is at the project root with correct paths. By default it targets the **RTX 4000 Ada** on the `instruction` partition. To use a different GPU, open the script and uncomment the appropriate block:

```bash
# RTX 4000 Ada — instruction partition (default)
#SBATCH -p instruction
#SBATCH --gres=gpu:rtx4000ada:1

# A100 — research partition
##SBATCH -p research
##SBATCH --gres=gpu:a100:1

# H100 — research partition
##SBATCH -p research
##SBATCH --gres=gpu:h100:1
```

Output is written to `line_pressure_<jobid>.out` and `line_pressure_<jobid>.err`.

### Submit to all GPU types at once

```bash
bash experiments/submit_line_pressure_all_gpus.sh
```

This submits parallel jobs to all five GPU types (RTX 4000 Ada, RTX A4500, RTX 2080 Ti, A100, H100) and labels each job accordingly.

### Other SLURM scripts

| Script | Purpose |
|--------|---------|
| `experiments/run_fem_solver.sh` | Element stiffness kernel benchmark |
| `experiments/submit_fem_all_gpus.sh` | FEM benchmark on all GPU types |
| `experiments/run_cuda_test.sh` | Quick GPU connectivity check |
| `misc/benchmarks/run_all.sh` | Full benchmark suite + plot generation |

---

## Experiments and Benchmarks

### Validate the GPU solver

```bash
# Check GPU ke == CPU ke for a single element
julia --project=. experiments/test_single_element.jl

# End-to-end solve on a small mesh
julia --project=. experiments/test_small_mesh.jl

# PCG solver convergence
julia --project=. experiments/test_pcg.jl

# Dirichlet BC enforcement
julia --project=. experiments/test_dirichlet.jl

# Geometric stiffness matrix
julia --project=. experiments/test_geometric_stiffness.jl
```

### Element stiffness kernel benchmark (GPU vs CPU)

Sweeps mesh sizes from 100 to 50,000 elements and compares GPU kernel, serial CPU loop, and Ferrite reference. Results written to `experiments/benchmark_results_fem.csv`.

```bash
julia --project=. experiments/benchmark_fem_solver.jl
# or on Euler:
sbatch experiments/run_fem_solver.sh
```

### Full problem benchmarks

These scripts live in `misc/benchmarks/` and perform comprehensive sweeps (mesh sizes, solver variants, problem types) with accuracy validation against Abaqus reference values:

```bash
julia --project=. misc/benchmarks/run_membrane.jl
julia --project=. misc/benchmarks/run_bending.jl
julia --project=. misc/benchmarks/run_buckling.jl
```

### Generate plots

After benchmarks complete, generate speedup and timing breakdown plots:

```bash
julia --project=. misc/benchmarks/speedup_plots.jl
```

---

## CPU Reference Implementations

The `cpu_reference/` directory contains pure CPU Julia/Ferrite solvers for all problem variants. These use Ferrite's built-in sparse direct solver (`K \ F`) and serve as ground truth for validation and as the serial baseline for speedup measurements.

| Directory | Problem |
|-----------|---------|
| `Plate_membrane_t100/` | Membrane, thick plate (t = 100 mm), line load |
| `Plate_membrane_t2/` | Membrane, thin plate (t = 2 mm), line load |
| `Plate_bending_t100/` | Bending, thick plate, line load |
| `Plate_bending_t2/` | Bending, thin plate, line load |
| `Plate_membrane_t100_Pressure/` | Membrane, thick plate, uniform pressure |
| `Plate_membrane_t2_Pressure/` | Membrane, thin plate, uniform pressure |
| `Plate_bending_t100_Pressure/` | Bending, thick plate, uniform pressure |
| `Plate_bending_t2_Pressure/` | Bending, thin plate, uniform pressure |

Run any reference solver directly:

```bash
julia --project=. cpu_reference/Plate_membrane_t100/TriShellFiniteElement_Membrane_and_Bending.jl
```

---

## GPU Pipeline Architecture

The main entry point is `gpu_solve()` in `src/gpu/gpu_solve.jl`. It accepts a mesh, material parameters, fixed DOFs, and a force vector, and returns the displacement vector and timing statistics.

```
CPU: mesh + DOF setup → CSR pattern construction → upload to GPU
         ↓
GPU: element_stiffness_kernel  (one thread per element → 18×18 ke)
         ↓
GPU: assembly_kernel           (atomic scatter ke → global CSR matrix K)
         ↓
GPU: dirichlet_kernel          (zero fixed rows/cols, set diagonal to 1)
         ↓
GPU: pcg_solver                (Jacobi-preconditioned CG → displacement u)
         ↓
CPU: download u → extract results
```

**Solver backends** (selectable via `solver=` keyword):

| Backend | Keyword | Best for |
|---------|---------|---------|
| Custom CUDA PCG | `:custom_pcg` (default) | General use |
| cuBLAS PCG | `:cublas_pcg` | Fewer host-device transfers |
| CUSOLVER direct | `:cusolver` | Small meshes |

**Element formulation:**
- 18 DOFs per element (6 DOFs/node: ux, uy, w, θx, θy, θz)
- IP3 shape functions for membrane and bending
- IP6 shape functions for shear
- Dunavant 3-point quadrature rule
- Drilling DOF (θz) stabilized via penalty and pinned to zero

---

## Results and Reports

- `Final_Project_Report_Zubayer.docx` — Full project report with methodology, results, and analysis
- `FinalProjectProposal_Yeaser.pdf` — Original project proposal
- `resources/performance_comparison_cpu_h100_rtx4000ada.csv` — CPU vs H100 vs RTX 4000 Ada comparison
- `resources/result_comparison.xlsx` — Displacement results vs Abaqus reference
- `resources/*.png` — Speedup charts and timing breakdown plots
