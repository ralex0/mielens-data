for IM_NUMBER in $(seq 3 4); do
#
export IM_NUMBER
#
sbatch mcmc_polystyrene.sbatch
#
sleep 1
done
