import os

import torch

LUNAR_LANDER = {
    "env": {
        "name": "LunarLander-v2",
        "kwargs": {},
    },
    "agent": {
        "hyperparams": {
            "gamma": 0.99,
            "epsilon": 1.0,
            "eps_end": 0.01,
            "lr": 0.0001,
        },
    },
}

FLAPPY_BIRD = {
    "env": {
        "name": "FlappyBird-v0",
        "kwargs": {"use_lidar": False},
        "import": "flappy_bird_gymnasium",
    },
    "agent": {
        "hyperparams": {
            "gamma": 0.99,
            "epsilon": 1.0,
            "eps_end": 0.01,
            "lr": 0.001,
        },
    },
}

CART_POLE = {
    "env": {
        "name": "CartPole-v1",
        "kwargs": {},
    },
    "agent": {
        "hyperparams": {
            "gamma": 0.99,
            "epsilon": 1.0,
            "eps_end": 0.01,
            "lr": 0.001,
        },
    },
}

ACROBOT = {
    "env": {
        "name": "Acrobot-v1",
        "kwargs": {},
    },
    "agent": {
        "hyperparams": {
            "gamma": 0.99,
            "epsilon": 1.0,
            "eps_end": 0.01,
            "lr": 0.001,
        },
    },
}
# Me tira. AttributeError: `np.float_` was removed in the NumPy 2.0 release. Use `np.float64` instead.. Did you mean: 'float16'?


CONFIGS = {
    "LunarLander-v2": LUNAR_LANDER,
    "FlappyBird-v0": FLAPPY_BIRD,
    "CartPole-v1": CART_POLE,
    "Acrobot-v1": ACROBOT,
}

DQN = {
    "layer1_size": 256,  # el tutorial usa 128,
    "layer2_size": 256,  # el tutorial usa 128,
}
BATCH_SIZE = 64

TUTORIAL = {
    "batch_size": 128,
    "gamma": 0.99,
    "eps_start": 0.9,
    "eps_end": 0.05,
    "eps_decay": 1000,
    "tau": 0.005,
    "lr": 1e-4,
}

OUT_DIR = f"{os.path.dirname(os.path.realpath(__file__))}/outputs"
DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
