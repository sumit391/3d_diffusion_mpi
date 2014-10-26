3d_diffusion_mpi
================
3D - Diffusion code for project using MPI, automatic vecotarization 
and OpenMp.

Assumptions and Initial Conditions
Domain: [-1, 1] 3 divided as 1024 x 1024 x 1024 cells
Diffusion coefficient (to compute the time step) for stability = 1
Diffusion PDE Solver: FTCS (Forward-Time Central-Space) Finite Difference Method
Number of Time steps = 20
Initial condition = All cells within (-0.5 , 0.5) 3 would have density values of 1 and others 0


Observations and Summary
• The speedup compared to serial code increased with increasing number of threads.
• For MPI code on 4 and 8 Nodes without OpenMP, the time of execution exceeded the
time for serial code. This could be because of increased overhead due to increased
communication between nodes through MPI and suboptimal number of nodes for the
data size.
• The speedup compared to serial code increased drastically with increasing number of
nodes per run of the diffusion equation. The solver code scaled well and the speedup
rate was steadily increased for both MPI only and MPI-OpenMP Hybrid codes.
• For the Tődi architecture the Solver performance speedup was highest for 16 Thread
on 256 Nodes, compared to the serial code for the constant domain grid size of 1024 x
1024 x 1024.
