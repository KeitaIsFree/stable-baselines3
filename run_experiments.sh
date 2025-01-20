



SEEDS=(11 12 13 14 15)

# for SEED in ${SEEDS[@]}; do
#     python 'run_experiment.py' $SEED "SAC" "cuda:0" $ENV_NAME &
# done

# for SEED in ${SEEDS[@]}; do
#     python 'run_experiment.py' $SEED "A2C" "cuda:1" $ENV_NAME &
# done

for SEED in ${SEEDS[@]}; do
    # python 'run_experiment.py' $SEED "A2C" "cuda:0" $ENV_NAME &
    python 'run_experiment.py' seed=$SEED &
done

# wait 

# for SEED in ${SEEDS[@]}; do
#     python 'run_experiment.py' $SEED "PPO" "cuda:0" $ENV_NAME &
# done

# for SEED in ${SEEDS[@]}; do
#     python 'run_experiment.py' $SEED "TD3" "cuda:1" $ENV_NAME &
# done

# for SEED in ${SEEDS[@]}; do
#     python 'run_experiment.py' $SEED "OURS" "cuda:0" &
# done