# Kinetically-Constrained-Model Project

This repository is a computational archive for the work:

**Thermodynamic phase transitions in lattice spin systems with severe kinetic constraints: Numerical simulation results**

It studies the most severely constrained Fredrickson-Andersen (FA) model with kinetic hyperparameter `K = 1` on finite-dimensional lattices.

This repository contains research code, processed datasets, and post-processing notebooks. It is not organized as a general-purpose software package; parameters are mainly controlled directly in the C++ sources and notebook cells.

## Scientific Summary

In the manuscript, each lattice site has a binary packing state:

- `c_i = 0`: empty,
- `c_i = 1`: occupied.

The model is initialized from a completely random configuration with initial occupied-site density `rho`. For the `K = 1` FA kinetic rule, a site is blocked from changing state whenever it has one or more occupied nearest neighbors. This severe local rule partitions the lattice into:

- **frozen sites**, which remain fixed to their initial state,
- **unfrozen sites**, which can still flip at least occasionally.

The unfrozen subsystem is the central object studied in this repository.

The manuscript reports two main classes of thermodynamic phenomena:

1. **Connectivity collapse of the unfrozen subsystem**
   
2. **Ground-state ordering transition**

## Repository Layout

- `ComputeLargeComponent3.cpp`
  computes the size density of the largest connected component of the unfrozen subsystem as a function of `rho`.
- `MIS-multi-L-v11-6.cpp`
  computes ground-state observables of the unfrozen subsystem through the maximum independent set / minimum vertex cover formulation by non-equilibrium sample.
- `MVC-equ-MCMC-LSB-AD-ACF-mul-L-rho-0322.cpp`
  implements the same FA `K = 1` unfrozen-subsystem pipeline, but goes one step further by explicitly sampling the minimum-vertex-cover / ground-state ensemble through conventional Markov-chain Monte Carlo on the unfrozen bipartite graph. It also outputs the coarse-grained site counts `N_1`, `N_0`, and `N_*` on the two sublattices.
- `percolation-NSA.ipynb`
  post-processing and scaling analysis for the largest-component data.
- `K-core-figures - final/data/MIS-data-2d/summary_stats-d-2.txt`
  processed ground-state summary table for two-dimensional systems.
- `data/max-cluster-size-0708/`
  processed data for the collapse transition of the giant connected component.

- `data/MVS-MCMC/`
  main-line disorder-averaged and disorder-resolved outputs from the explicit MVC/MCMC sampling workflow.

- `mvs-mcmc-absR-NSA-bootstrap-collapse-0401.ipynb`
  bootstrap finite  scaling analysis for `|R|`.
- `mvs-mcmc-chi-dis-NSA-bootstrap-collapse-0401.ipynb`
  bootstrap finite scaling analysis for disorder susceptibility.
- `CLT2.cpp`
  auxiliary lattice-analysis code for checkboard configurations.
- `MCPottsModel2.cpp`
  auxiliary disordered Potts-model simulation code.

## Data Guide

Unless otherwise stated:

- one **disorder sample** means one independent random initial configuration at fixed `(L, rho)`;
- disorder averages are written as $[\cdots]$;
- thermal or ground-state averages at fixed disorder are written as $\langle \cdots \rangle$;
- `N` = total number of lattice sites;
- $N_{\mathrm{uf}}$ = number of unfrozen sites in the active graph used by the MIS/MVC analysis;
- in the explicit-MCMC data below, the sampled imbalance order parameter is written as $m$, although the C++ code stores it in a variable named `R`.

The most important observables are:

- $s_{\max} = N_{\mathrm{LCC}} / N$:
  density of the largest connected component of the unfrozen subsystem.
- $E = |\mathrm{MIS}| / N_{\mathrm{uf}}$:
  MIS density of the unfrozen active graph.
- $R_{\mathrm{MIS}} = |\mathrm{MIS} \cap U| / |U| - |\mathrm{MIS} \cap V| / |V|$:
  MIS imbalance between the two checkerboard partitions $U$ and $V$.
- $m_I = \frac{(N_{1A} - N_{0A}) - (N_{1B} - N_{0B})}{N_{1A} + N_{0A} + N_{1B} + N_{0B}}$:
  manuscript imbalance order parameter built from type-1 and type-0 site counts.
- $m = \frac{N_{\mathrm{MVC}}^{(U)} - N_{\mathrm{MVC}}^{(V)}}{N_{\mathrm{MVC}}^{(U)} + N_{\mathrm{MVC}}^{(V)}}$:
  explicit-MCMC imbalance observable used in `R_mcmc_stat-d-*`.

### `Zhou/data/max-cluster-size-0708/`

This directory contains processed data for the collapse transition of the unfrozen subsystem.

`max-cluster-density-stat-d=2.txt`

- each row = one parameter point `(L, rho)`.
- `L` = system linear size.
- `rho` = initial occupied-site density in the manuscript convention.
- `mean_density` = $[s_{\max}]$.
- `second_moment_uncentered` = $[s_{\max}^2]$.
- `forth_moment_uncentered` = $[s_{\max}^4]$.

the parameter note file in this directory stores run notes for the largest-component scan.

### `Zhou/data/MVS-MCMC/`

This directory contains the explicit MVC/MCMC data products for the same FA `K = 1` problem.

`absR_thermal-d-*.txt`

- each row = one parameter point `(d, L, W, rho)`.
- `d` = spatial dimension.
- `L` = first lattice length.
- `W` = transverse lattice length.
- `rho` = initial occupied-site density.
- `sample_1`, `sample_2`, ..., `sample_{N_{\mathrm{samples}}}` = $\langle |m| \rangle_1$, $\langle |m| \rangle_2$, ..., $\langle |m| \rangle_{N_{\mathrm{samples}}}$.

This is the most direct file for reconstructing $[\langle |m| \rangle]$.

`Frozen_stat-d-*.txt`

- each row = one disorder sample at fixed `(d, L, W, rho)`.
- `sample_idx` = disorder-sample index.
- `N_1A` = $N_{1A}$.
- `N_1B` = $N_{1B}$.
- `N_0A` = $N_{0A}$.
- `N_0B` = $N_{0B}$.
- `N_starA` = $N_{*A}$.
- `N_starB` = $N_{*B}$.

From these raw counts one may reconstruct

- $m_I = \frac{(N_{1A} - N_{0A}) - (N_{1B} - N_{0B})}{N_{1A} + N_{0A} + N_{1B} + N_{0B}}$.
- $|m_I| = \left| \frac{(N_{1A} - N_{0A}) - (N_{1B} - N_{0B})}{N_{1A} + N_{0A} + N_{1B} + N_{0B}} \right|$.

`WCC_sizes-d-*.txt`

- each row = one disorder sample at fixed `(d, L, W, rho)`.
- `sample_idx` = disorder-sample index.
- `wcc_size_1`, `wcc_size_2`, ... = $|C_1|$, $|C_2|$, ..., the weakly connected component sizes of the reduced state-flexible core used in the MVC/MCMC workflow.

`R_mcmc_stat-d-*.txt`

- each row = one parameter point `(d, L, W, rho)`.
- `d` = spatial dimension.
- `L` = first lattice length.
- `W` = transverse lattice length.
- `rho` = initial occupied-site density.
- `N_samples` = $N_{\mathrm{dis}}$, the number of disorder samples.
- `N_thermal` = $N_{\mathrm{th}}$, the number of thermal measurements per disorder sample.
- `mean_sweeps` = mean number of MCMC sweeps used per disorder sample.
- `sweeps_std` = standard deviation of the sweep count across disorder samples.
- `dis_mR` = $[\langle m \rangle]$.
- `dis_mAbsR` = $[\langle |m| \rangle]$.
- `dis_mR2` = $[\langle m^2 \rangle]$.
- `dis_mR3` = $[\langle m^3 \rangle]$.
- `dis_mAbsR3` = $[\langle |m|^3 \rangle]$.
- `dis_mR4` = $[\langle m^4 \rangle]$.
- `q_EA` = $[\langle m \rangle^2]$.
- `dis_mAbsR_sq` = $[\langle |m| \rangle^2]$.
- `dis_mR_p4` = $[\langle m \rangle^4]$.
- `dis_mAbsR_p4` = $[\langle |m| \rangle^4]$.
- `dis_mR2_sq` = $[\langle m^2 \rangle^2]$.
- `chi_th_abs` = $N\bigl([\langle m^2 \rangle] - [\langle |m| \rangle^2]\bigr)$.
- `chi_dis_abs` = $N\bigl([\langle |m| \rangle^2] - [\langle |m| \rangle]^2\bigr)$.
- `chi_tot_abs` = $N\bigl([\langle m^2 \rangle] - [\langle |m| \rangle]^2\bigr)$.
- `chi_tot_abs_sem_jk` = jackknife standard error of $\chi_{\mathrm{tot}}^{|m|}$.
- `chi_th_R` = $N\bigl([\langle m^2 \rangle] - [\langle m \rangle^2]\bigr)$.
- `chi_dis_R` = $N\bigl([\langle m \rangle^2] - [\langle m \rangle]^2\bigr)$.
- `chi_tot_R` = $N\bigl([\langle m^2 \rangle] - [\langle m \rangle]^2\bigr)$.
- `U4` = $1 - [\langle m^4 \rangle] / \bigl(3[\langle m^2 \rangle]^2\bigr)$.
- `U_EA` = $1 - [\langle m \rangle^4] / \bigl(3[\langle m \rangle^2]^2\bigr)$.
- `U_EA_abs` = $1 - [\langle |m| \rangle^4] / \bigl(3[\langle |m| \rangle^2]^2\bigr)$.
- `U22` = $1 - [\langle m^2 \rangle^2] / \bigl(3[\langle m^2 \rangle]^2\bigr)$.
- `acf_probe_warn_count` = $n_{\mathrm{warn}}$, the number of disorder samples whose short ACF probe was flagged as too short.
- `acf_probe_warn_frac` = $n_{\mathrm{warn}} / N_{\mathrm{dis}}$.

## Build and Run

The code is currently compiled file-by-file. From the repository root:

```bash
cd Zhou
g++ -O3 -fopenmp -march=native ComputeLargeComponent3.cpp -o ComputeLargeComponent
g++ -O3 -fopenmp -march=native MIS-multi-L-v11-6.cpp -o MIS
g++ -O3 -march=native -fopenmp -std=c++17 MVC-equ-MCMC-LSB-AD-ACF-mul-L-rho-0322.cpp -o MVC_MCMC
```

Then run, for example:

```bash
./ComputeLargeComponent
./MIS
./MVC_MCMC
```

Most production parameters are hard-coded near `main()` or in compile-time constants. 

## Citation

If you use this repository in academic work, please cite the corresponding article:

**Thermodynamic phase transitions in lattice spin systems with severe kinetic constraints: Numerical simulation results**



## License

This repository is distributed under the terms given in [`LICENSE`](LICENSE).
