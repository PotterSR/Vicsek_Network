#!/bin/bash
#PBS -N vicsek_sim
#PBS -l nodes=1:ppn=25
#PBS -l mem=32gb
#PBS -l walltime=72:00:00
#PBS -j oe

cd "$PBS_O_WORKDIR"

# module load julia
export JULIA_NUM_THREADS=25

echo "started on $(hostname) at $(date)"
julia --project=. vicsek.jl
echo "finished at $(date)"
