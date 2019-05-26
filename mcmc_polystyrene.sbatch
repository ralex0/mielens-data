#!/bin/bash
#SBATCH -J mcmc_polystyrene
#SBATCH -o mcmc_polystyrene.%j.out
#SBATCH -e mcmc_polystyrene.%j.err
#SBATCH -p test
#SBATCH -t 15
#SBATCH -n 2
#SBATCH --mem-per-cpu=1000

ML_PREFIX="/n/manoharan/alexander/mielens-data"
DATA_DIR="data/Polystyrene2-4um-60xWater-042919/processed-256"
FITS_DIR="fits/sedimentation/newdata"

## SETUP STORAGE ##
STORAGE_DIR="/scratch/$USER/${SLURM_JOB_ID}"
mkdir -pv $STORAGE_DIR
mkdir -pv $STORAGE_DIR/$DATA_DIR
mkdir -pv $STORAGE_DIR/$FITS_DIR

## COPY INPUT FILES AND MOVE TO WORKING DIR##
cp -R $ML_PREFIX/$DATA_DIR $STORAGE_DIR/$DATA_DIR
cp $ML_PREFIX/$FITS_DIR/*.pkl $STORAGE_DIR/$FITS_DIR
cd $STORAGE_DIR

# Load required modules
source new-modules.sh
module purge
module load Anaconda/5.0.1-fasrc01
module load gcc/7.1.0-fasrc01 openmpi/2.1.0-fasrc01
source activate holopy-dev

## RUN PROGRAM ##
srun -n 2 --mpi=pmi2 python $SLURM_SUBMIT_DIR/mcmc_polystyrene.py

## COPY OUTPUT BACK ##
cp *.pkl $FITS_DIR/