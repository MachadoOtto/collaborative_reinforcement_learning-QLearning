# collaborative_reinforcement_learning-QLearning

This repository presents a collaborative reinforcement learning approach leveraging High-Performance Computing (HPC) for training an artificial intelligence model using Q-Learning and OpenAI Gym. The system consists of a master node that manages a global model and multiple slave nodes running simulations on that model.

## Examples

## Train a model

```bash
 python reinforcement_learner/slaveV2.py --env_name="LunarLander-v2" --n_games=1000 --save_checkpoint=True --save_suffix="-base1"
```

This will train a model for the LunarLander-v2 environment for 1000 episodes and save the model checkpoint
(.tar file) inside outputs/models folder with the name LunarLander-v2-1000-base1.tar

It will also save a csv file with the columns

```markdown
episode,cumulative_reward,episode_length,time,epsilon
```

inside the outputs/stats folder with the name LunarLander-v2-base1.csv

## Plot

```bash
python reinforcement_learner/plot.py --csv="reinforcement_learner/outputs/stats/LunarLander-v2-base1.csv"
```

## Evaluate and see the model in action

```bash
 python reinforcement_learner/play.py --env_name="LunarLander-v2" --model="LunarLander-v2-1000-base1.tar" --evaluate=500 --watch=5
```

## Merge models using the FedAvg algorithm

```bash
python reinforcement_learner/master.py --env_name="LunarLander-v2" --suffixes="-1000-base1,-1000-base2,-2000-base3"
```

This will look for the models LunarLander-v2-1000-base1.tar, LunarLander-v2-1000-base2.tar, and LunarLander-v2-2000-base3.tar inside the outputs/models folder and merge them.
The resulting model will be saved inside the outputs/models folder with the name

```markdown
outputs/models/LunarLander-v2-1000-base1-1000-base2-2000-base3_merged_model.pt
```

> Note: You can specify the out folder and model name by passing the --out_path argument.

## Notes:

-   Output file name formats:
    -   plots/<environment_name>-<model_suffix>-<episode>.png : The plot of the rewards per episode.
    -   models/<environment_name>-<total_trained_episodes>-<model_suffix>.pt : The model state dict checkpoint.
-   If you get the error "error: command 'swig' failed: No such file or directory" while installing the gymnasium[box2d] package you may want to install the swig package in your OS first.
