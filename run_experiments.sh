

ENV_NAME="BipedalWalker-v3"

SEEDS=(1 2 3 4 5 6 7 8 9 10)

for SEED in ${SEEDS[@]}; do
    python 'run_experiment.py' $SEED "SAC" "cuda:0" $ENV_NAME &
done

for SEED in ${SEEDS[@]}; do
    python 'run_experiment.py' $SEED "A2C" "cuda:1" $ENV_NAME &
done

for SEED in ${SEEDS[@]}; do
    python 'run_experiment.py' $SEED "OURS" "cuda:2" $ENV_NAME &
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