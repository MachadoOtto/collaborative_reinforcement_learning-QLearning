# MPI Distributed Reinforcement Learning for Gym Environments

This repository contains the implementation of a distributed reinforcement learning system using MPI (Message Passing Interface) for Gym environments. The system includes both master and slave processes that work together to train reinforcement learning models in parallel.

## Authors
- Guido Dinello (5.031.022-5)
- Jorge Machado (4.876.616-9)

HPC - High Performance Computing - 2024  
Facultad de Ingeniería - Universidad de la República, Uruguay

## Description

This code facilitates distributed reinforcement learning by leveraging MPI to parallelize the training process across multiple nodes. The master node coordinates the workflow, sending models to slave nodes, collecting results, and merging models. Slave nodes execute reinforcement learning tasks and send results back to the master.

## Environment and Dependencies

- **MPI**: Ensure that MPI is installed and properly configured on your system.
- **Python**: A Python environment is required to run the reinforcement learning scripts (`master.py` and `slave.py`) located in the `./reinforcement_learner/` directory.
- **Gym**: The Gym environment used in this implementation is `CartPole-v1`, but it can be modified as needed.

### Python Dependencies

Install the Python dependencies by navigating to the `./reinforcement_learner/` directory and running:

```bash
pip install -r requirements.txt
```

## File Structure

- **master.c**: Contains the logic for the master node, including model distribution and collection.
- **slave.c**: Contains the logic for the slave nodes, including model receipt, training execution, and model sending.
- **master.py**: Python script for merging models on the master node.
- **slave.py**: Python script for training reinforcement learning models on the slave nodes.

## Compilation and Execution

To compile the MPI code, use the following command:

```bash
mpicc -o mpi_reinforcement_learning master.c slave.c
```

To run the MPI code, use the following command:

```bash
mpirun -np <num_processes> ./mpi_reinforcement_learning
```

Replace `<number_of_processes>` with the total number of processes you want to run (1 master + n slaves).

## Configuration
- ENV_NAME: The Gym environment to use (e.g., "CartPole-v1").
- MAX_ORDERS: Maximum number of orders to be sent from the master to the slaves.
- TIMEOUT_TIME: Time (in seconds) for each reinforcement learning task to run.
- N_GAMES: Number of games to play during training.
- BASE_MODEL_PATH: Path where models are saved and loaded from.

Modify these settings in the source code as needed.

## Reinforcement Learning Python Scripts

## Train a model

```bash
 python reinforcement_learner/slave.py --env_name="LunarLander-v2" --n_games=1000 --save_checkpoint=True --save_suffix="-base1"
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