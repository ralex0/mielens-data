#!/bin/bash
#SBATCH -J fit_Si_mieonly
#SBATCH -o fit_Si_mieonly%j.out
#SBATCH -e fit_Si_mieonly%j.err
#SBATCH -p shared
#SBATCH -t 03:00:00
#SBATCH -n 32
#SBATCH -N 1
#SBATCH --mem=3000

ML_PREFIX="/n/manoharan/alexander/mielens-data"
DATA_DIR="data/Silica1um-60xWater-080619/processed0-256"
FITS_DIR="fits/Silica1um-60xWater-080619/"
PARTICLE="silica"
THEORY='mieonly'

## SETUP STORAGE ##
STORAGE_DIR="/scratch/$USER/${SLURM_JOB_ID}"
mkdir -pv $STORAGE_DIR
mkdir -pv $STORAGE_DIR/$DATA_DIR
mkdir -pv $STORAGE_DIR/$FITS_DIR

## COPY INPUT FILES AND MOVE TO WORKING DIR##
cp -R $ML_PREFIX/$DATA_DIR $STORAGE_DIR/$DATA_DIR
cp $ML_PREFIX/$DATA_DIR/best_guesses.json $STORAGE_DIR/$DATA_DIR/best_guesses.json
cd $STORAGE_DIR

# Load required modules
source new-modules.sh
module purge
module load Anaconda3/5.0.1-fasrc02
source activate holopy-dev

## RUN PROGRAM ##
python $SLURM_SUBMIT_DIR/fit_sedimentation_data.py $PARTICLE $THEORY
