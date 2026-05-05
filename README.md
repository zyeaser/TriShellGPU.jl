# GPU-Accelerated Triangular Shell Finite Element Solver

A fully GPU-accelerated finite element solver for triangular shell elements, implemented in Julia using CUDA.jl. The solver assembles and solves the global stiffness system entirely on the GPU — element stiffness computation, sparse matrix assembly, boundary condition enforcement, and iterative solve.

Developed for CS 759 (High Performance Computing), UW–Madison.

---

## Repository Contents

```
FinalProject/
├── src/                              # Source code
├── example/                          # Runnable example problems
├── Project.toml                      # Julia package dependencies
├── Manifest.toml                     # Locked dependency versions (Julia 1.12.6)
├── install_julia.sh                  # One-time Julia installer for Euler login node
├── run_line_pressure_gpu.sh          # SLURM job script (line-pressure example)
├── FinalProjectProposal_Yeaser.pdf   # Original project proposal
└── Final_Project_Report_Zubayer.docx # Final project report
```

---

## src/

Contains all solver source code, split into two parts.

### `src/gpu/` — GPU kernel suite

| File | Description |
|------|-------------|
| `gpu_solve.jl` | Entry point. Orchestrates the full solve pipeline: CPU preprocessing → GPU upload → element stiffness → sparse assembly → boundary conditions → PCG solve → result download. Call `gpu_solve(mesh_params, material, fixed_dofs, F)` from any script. |
| `device_functions.jl` | Inline CUDA math helpers — shape functions (IP3/IP6), Jacobian, rotation matrices, quadrature rules. Zero heap allocation; called inside GPU kernels. |
| `element_stiffness_kernel.jl` | CUDA kernel. One thread per element computes the full 18×18 local stiffness matrix (ke) covering membrane, bending, and shear contributions. |
| `assembly_kernel.jl` | GPU sparse matrix assembly. Builds the CSR sparsity pattern on CPU once, then uses GPU atomic-add scatter to fill the global stiffness matrix K. |
| `dirichlet_kernel.jl` | GPU boundary condition enforcement. Zeroes fixed DOF rows and columns and sets the diagonal to 1, preserving matrix symmetry. |
| `pcg_solver.jl` | Jacobi-preconditioned Conjugate Gradient solver with three backends: custom CUDA (default), cuBLAS-optimized, and CUSOLVER direct. |
| `geometric_stiffness_kernel.jl` | GPU geometric stiffness kernel for linear buckling analysis. Computes 18×18 Kg from pre-buckling element stresses. |

### `src/TriShellFiniteElement.jl/` — Julia package

Defines custom finite element interpolations `IP3` (3-node linear) and `IP6` (6-node quadratic) that extend Ferrite's interpolation interface. Registered as a local package so it can be loaded with `using TriShellFiniteElement`.

### `src/TriShellFiniteElement_Membrane_and_Bending.jl`

Standalone CPU solver script used during development. Uses Ferrite's built-in direct sparse solver (`K \ F`) without any GPU code.

---

## example/

Two ready-to-run GPU example problems. Both use `gpu_solve()` from `src/gpu/gpu_solve.jl` and write results to a tab-separated `.txt` file in the same directory.

### `gpu_plate_membrane_line_pressure.jl`

**Problem:** 100 mm × 1000 mm plate under a line load of 1000 N/mm at mid-span (y = 500 mm).  
**Variants:** in-plane (ux) and out-of-plane (w) loading, thin (t = 2 mm) and thick (t = 100 mm) plates.  
**Output:** `example/results_line_pressure_mesh_thin.txt` — displacements and timing for each mesh size.

```bash
julia --project=. example/gpu_plate_membrane_line_pressure.jl
```

### `gpu_plate_membrane_surface_pressure.jl`

**Problem:** Same plate geometry under uniform distributed surface pressure (p = 1 N/mm²).  
**Variants:** membrane and bending formulations at both plate thicknesses.

```bash
julia --project=. example/gpu_plate_membrane_surface_pressure.jl
```

---

## Setup

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Julia | 1.12.6 (via juliaup) |
| CUDA Toolkit | 13.0.0 (`nvidia/cuda/13.0.0` on Euler) |
| GPU | NVIDIA GPU, CUDA compute capability ≥ 6.0 |

### Julia packages (`Project.toml` / `Manifest.toml`)

Installed automatically by `Pkg.instantiate()`. Key dependencies:

| Package | Purpose |
|---------|---------|
| `CUDA` | GPU kernels via CUDA.jl |
| `Ferrite` | FEM mesh, DOF handler, grid generation |
| `Tensors`, `StaticArrays` | Performant math types |
| `BenchmarkTools` | Timing utilities |
| `TriShellFiniteElement` | Local package — custom IP3/IP6 interpolations |

### `install_julia.sh` — first-time setup on Euler

Run once from the project root on the Euler **login node**:

```bash
bash install_julia.sh
source ~/.bashrc
```

This script:
1. Installs `juliaup` into `~/.juliaup` (skips if already present)
2. Adds Julia 1.12.6 and sets it as the default
3. Appends `~/.juliaup/bin` to `PATH` in `~/.bashrc`
4. Writes `LocalPreferences.toml` to configure CUDA.jl to use the system CUDA toolkit
5. Runs `Pkg.instantiate()` to download and precompile all packages

---

## Running on Euler (SLURM)

### `run_line_pressure_gpu.sh`

Submits the line-pressure example as a SLURM job. Default target: **RTX 4000 Ada** on the `instruction` partition (15 min, 4 CPUs, 64 GB RAM).

```bash
sbatch run_line_pressure_gpu.sh
```

To switch GPU, open the script and uncomment one of the other blocks:

```bash
# RTX 4000 Ada — instruction partition (default, active)
#SBATCH -p instruction
#SBATCH --gres=gpu:rtx4000ada:1

# A100 — research partition
##SBATCH -p research
##SBATCH --gres=gpu:a100:1

# H100 — research partition
##SBATCH -p research
##SBATCH --gres=gpu:h100:1
```

Job output: `line_pressure_<jobid>.out` / `.err`

---

## GPU Pipeline

```
CPU  →  mesh setup, CSR pattern construction, force vector assembly
GPU  →  element_stiffness_kernel   (18×18 ke, one thread per element)
GPU  →  assembly_kernel            (atomic scatter → global K in CSR)
GPU  →  dirichlet_kernel           (apply boundary conditions)
GPU  →  pcg_solver                 (Jacobi-PCG → displacement vector u)
CPU  →  download u, extract results
```

**Element formulation:** 18 DOFs/element (6 DOFs/node: ux, uy, w, θx, θy, θz). Drilling DOF (θz) is stabilized by a penalty term and pinned to zero.

---

## Reports

| File | Contents |
|------|----------|
| `Final_Project_Report_Zubayer.docx` | Full report: formulation, GPU implementation, benchmarks, speedup results |
| `FinalProjectProposal_Yeaser.pdf` | Original proposal: objectives, scope, methodology |
