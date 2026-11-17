#!/bin/bash
#PBS -N vicsek_network
#PBS -l nodes=1:ppn=25
#PBS -l mem=32gb
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -t 1-10

# job array: each of the 10 jobs runs vicsek.jl with one value of η
# (on PBS Pro replace the line above with "#PBS -J 1-10")

cd "$PBS_O_WORKDIR"

# module load julia
export JULIA_NUM_THREADS=25

# noise strength η: angle noise is η ξ with ξ uniform in [-π, π]
ETAS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

ID=${PBS_ARRAYID:-$PBS_ARRAY_INDEX}         # Torque sets the first, PBS Pro the second
eta=${ETAS[$((ID - 1))]}                    # bash arrays start at 0, job indices at 1

echo "job $ID: η = $eta, started on $(hostname) at $(date)"
julia --project=. vicsek.jl $eta
echo "job $ID: finished at $(date)"
