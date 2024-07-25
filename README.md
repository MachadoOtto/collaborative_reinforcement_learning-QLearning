# collaborative_reinforcement_learning-QLearning

This repository presents a collaborative reinforcement learning approach leveraging High-Performance Computing (HPC) for training an artificial intelligence model using Q-Learning and OpenAI Gym. The system consists of a master node that manages a global model and multiple slave nodes running simulations on that model.

## Notes:

-   Output file name formats:
    -   plots/<environment_name>-<episode>.png : The plot of the rewards per episode.
    -   models/<environment_name>-<episode>-<score>.pt : The model state dict checkpoint.
-   If you get the error "error: command 'swig' failed: No such file or directory" while installing the gymnasium[box2d] package you may want to install the swig package in your OS first.
