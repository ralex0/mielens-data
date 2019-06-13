#!/bin/bash
#SBATCH -J mcmc_polystyrene
#SBATCH -o mcmc_polystyrene.out
#SBATCH -e mcmc_polystyrene.err
#SBATCH -p shared
#SBATCH -t 8:00:00
#SBATCH -n 128
#SBATCH --mem=8182

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
cp $ML_PREFIX/$FITS_DIR/fits_mo4.json $STORAGE_DIR/$FITS_DIR/fits_mo4.json
cp $ML_PREFIX/$FITS_DIR/fits_ml3.json $STORAGE_DIR/$FITS_DIR/fits_ml3.json
cd $STORAGE_DIR

# Load required modules
source new-modules.sh
module purge
module load Anaconda/5.0.1-fasrc01
source activate holopy-dev

## RUN PROGRAM ##
python $SLURM_SUBMIT_DIR/mcmc_polystyrene.py
