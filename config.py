from reinforcement_learner.new_dqn.main import BATCH_SIZE

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


CONFIGS = {
    "LunarLander": LUNAR_LANDER,
    "FlappyBird": FLAPPY_BIRD,
    "CartPole": CART_POLE,
    "Acrobot": ACROBOT,
}

DQN = {
    "fc1_dims": 256,  # el tutorial usa 128,
    "fc2_dims": 256,  # el tutorial usa 128,
}
BATCH_SIZE = 64
