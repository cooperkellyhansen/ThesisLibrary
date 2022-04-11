#!/bin/csh

#SBATCH --time=12:00:00 # walltime, abbreviated by -t
#SBATCH -o FIPgrainEV_stack100-%j.out # name of the stdout
#SBATCH -e error-%j.err # name of the stderr
#SBATCH --nodes=1 # number of nodes you're reserving
#SBATCH --ntasks=1 # number of MPI tasks, abbreviated by -n # additional information for allocated clusters 
#SBATCH --account=hochhalter-np # account - abbreviated by -A 
#SBATCH --partition=hochhalter-shared-np # partition, abbreviated by -p # # set data and working directories
#SBATCH --mail-type=END
#SBATCH --mail-user=cooperkellyhansen@gmail.com
ml gcc mpich
setenv PYTHONPATH /uufs/chpc.utah.edu/common/home/u1164746/bingo

source /uufs/chpc.utah.edu/common/home/u6019587/.venv/bingovenv_v2/bin/activate.csh
mpiexec -n $SLURM_NTASKS python ~/Thesis/ThesisLibrary/main.py 
