#!/bin/bash
#
# run_mono1.sh – LAMMPS mono1 production run on 2x 128-core node
#

#SBATCH --job-name=mono1
#SBATCH --account=25fa-mse4851a-eng
#SBATCH --partition=eng-instruction
#SBATCH --nodes=2
#SBATCH --ntasks=256
#SBATCH --time=12:00:00
#SBATCH --output=mono1_%j.out
#SBATCH --error=mono1_%j.err

# Optional: make sure we start in the submit directory
cd "$SLURM_SUBMIT_DIR"

# Clean environment so conda/micromamba MPI doesn't clash with system modules
module purge

# Activate your micromamba base env (where lmp_mpi is installed)
eval "$(micromamba shell hook -s bash)"
micromamba activate base

# Fix OpenMPI PML issues we saw earlier
unset OMPI_MCA_pml
unset OMPI_MCA_btl
unset UCX_TLS
unset UCX_NET_DEVICES
export OMPI_MCA_pml=ob1

# Just for sanity in the job output
echo "Running on nodes: $SLURM_NODELIST"
echo "NTASKS = $SLURM_NTASKS, NODES = $SLURM_JOB_NUM_NODES"
which lmp_mpi

# Launch LAMMPS with 128 MPI ranks on this node
mpirun -np 256 lmp_mpi -in mono1.in > mono1.out
