



SEEDS=(1 2 3 4 5 6 7 8 9 10)


for SEED in ${SEEDS[@]}; do
    python 'run_experiment.py' $SEED &
done