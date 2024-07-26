import logging
import os

import config
import gymnasium as gym
import torch
from discrete_agent.AgentV2 import Agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

OUT_DIR = f"{os.path.dirname(os.path.realpath(__file__))}/outputs"


def merge_models(agent: Agent, model_states: list[dict], out_path: str) -> None:
    # https://stackoverflow.com/questions/48560227/how-to-take-the-average-of-the-weights-of-two-networks

    # capaz los model_states deberian ser paths a los model_states
    # the agent should be initialized with the same parameters as the models. the slaves agents

    for i, st in enumerate(model_states):
        for k in st.keys():
            logging.debug(
                "Model %s, Key: %s, Type: %s, Shape: %s", i, k, type(st[k]), st[k].shape
            )

    # Simple averaging strategy - can be replaced with more sophisticated merging
    # Merge the corresponding weights of the models
    beta = 0.5  # The interpolation parameter
    merged_state = model_states[1]

    for k in model_states[0].keys():
        merged_state[k].data.copy_(
            (beta * model_states[0][k].data + (1 - beta) * model_states[1][k].data) / 2
        )

    agent.policy_net.load_state_dict(merged_state)

    torch.save(agent.policy_net.state_dict(), out_path)
    logging.info("Master model updated at %s", out_path)


def main():
    env_name = "FlappyBird-v0"
    if env_name == "FlappyBird-v0":
        import flappy_bird_gymnasium  # noqa: F401

    env = gym.make(
        config.CONFIGS[env_name]["env"]["name"],
        **config.CONFIGS[env_name]["env"]["kwargs"],
    )
    agent = Agent(
        **config.TUTORIAL,
        env=env,
    )

    suffixes = ["-base1", "-base2"]
    model_states = [
        torch.load(
            f"{OUT_DIR}/models/{env_name}-1000{suff}.tar",
            map_location=torch.device("cpu"),
        )["model_state_dict"]
        for suff in suffixes
    ]

    MASTER_MODEL_PATH = (
        f"{OUT_DIR}/models/{env_name}-{"".join(suffixes)}_merged_model.pt"
    )

    merge_models(agent, model_states, MASTER_MODEL_PATH)


if __name__ == "__main__":
    main()
