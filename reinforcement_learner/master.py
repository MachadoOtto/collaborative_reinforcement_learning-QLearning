import copy
import importlib
import logging
from argparse import ArgumentParser

import config
import gymnasium as gym
import torch
from reinforcement_learner.discrete_agent.Agent import Agent

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(message)s")


def fedavg(global_model, agents, out_path: str) -> None:
    """
    Implement Federated Averaging (FedAvg) for DQN agents.
    """
    global_weights = global_model.policy_net.state_dict()
    local_weights = agents

    # Get the number of samples (experiences) from each agent
    num_samples = [10_000 for _ in agents]  # [len(agent.memory) for agent in agents]
    total_samples = sum(num_samples)

    # Compute weighted average
    fed_state_dict = copy.deepcopy(global_weights)
    for key in global_weights.keys():
        fed_state_dict[key] = torch.zeros_like(global_weights[key])
        for i in range(len(agents)):
            weight = num_samples[i] / total_samples
            fed_state_dict[key] += weight * local_weights[i][key]

    # Update global model
    global_model.policy_net.load_state_dict(fed_state_dict)

    torch.save(global_model.policy_net.state_dict(), out_path)

    logging.info("Master model updated at %s", out_path)


def main(env_name: str, suffixes: list[str], **kwargs) -> None:
    if module := config.CONFIGS[env_name]["env"].get("import"):
        importlib.import_module(module)

    env = gym.make(
        config.CONFIGS[env_name]["env"]["name"],
        **config.CONFIGS[env_name]["env"]["kwargs"],
    )
    agent = Agent(
        **config.TUTORIAL,
        env=env,
    )

    model_states = [
        torch.load(f"{config.OUT_DIR}/models/{env_name}{suff}.pt") for suff in suffixes
    ]

    out_path = kwargs.get("out_path", None)
    if out_path is None:
        out_path = (
            f"{config.OUT_DIR}/models/{env_name}{"".join(suffixes)}_merged_model.pt"
        )
    logging.debug("Using out_path: %s", out_path)

    fedavg(agent, model_states, out_path)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    # required
    arg_parser.add_argument(
        "--env_name", type=str, required=True, help="Environment to use"
    )
    arg_parser.add_argument(
        "--suffixes",
        type=lambda x: x.split(","),
        required=True,
        help="Comma-separated list of suffixes of the same environment models to merge. Ex. '-1000-base1,-1000-base2'",
    )
    # optional
    arg_parser.add_argument(
        "--out_path",
        type=str,
        help="Path to save the merged model. Default: '{OUT_DIR}/models/{env_name}-{suffixes}_merged_model.pt' ",
    )

    args = arg_parser.parse_args()
    logging.debug("Script called with args: %s", args)

    main(**vars(args))
