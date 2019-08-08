#!/bin/bash
#SBATCH -J mcmc_mieonly_polystyrene
#SBATCH -o mcmc_mo_polystyrene_%j.out
#SBATCH -e mcmc_mo_polystyrene_%j.err
#SBATCH -p shared
#SBATCH -t 03:00:00
#SBATCH -n 32
#SBATCH -N 1
#SBATCH --mem=8000

ML_PREFIX="/n/manoharan/alexander/mielens-data"
DATA_DIR="data/Polystyrene2-4um-60xWater-042919/processed-256"
FITS_DIR="fits/Polystyrene2-4um-60xWater-042919/"
PARTICLE="polystyrene"
THEORY='mieonly'

## SETUP STORAGE ##
STORAGE_DIR="/scratch/$USER/${SLURM_JOB_ID}"
mkdir -pv $STORAGE_DIR
mkdir -pv $STORAGE_DIR/$DATA_DIR
mkdir -pv $STORAGE_DIR/$FITS_DIR

## COPY INPUT FILES AND MOVE TO WORKING DIR##
cp -R $ML_PREFIX/$DATA_DIR $STORAGE_DIR/$DATA_DIR
cp $ML_PREFIX/$DATA_DIR/fits_mo4.json $STORAGE_DIR/$DATA_DIR/best_guesses.json
cd $STORAGE_DIR

# Load required modules
source new-modules.sh
module purge
module load Anaconda3/5.0.1-fasrc02
source activate holopy-dev

## RUN PROGRAM ##
python $SLURM_SUBMIT_DIR/fit_sedimentation_data.py $PARTICLE $THEORY
