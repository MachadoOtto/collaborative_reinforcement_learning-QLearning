import logging
import os

import torch
from discrete_agent.Agent import Agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

OUT_DIR = f"{os.path.dirname(os.path.realpath(__file__))}/outputs"
MASTER_MODEL_PATH = f"{OUT_DIR}/models/merged_model.pt"


def merge_models(agent: Agent, model_states: list[dict]) -> None:
    # capaz los model_states deberian ser paths a los model_states
    # the agent should be initialized with the same parameters as the models. the slaves agents

    for i, st in enumerate(model_states):
        for k in st.keys():
            logging.debug(
                "Model %s, Key: %s, Type: %s, Shape: %s", i, k, type(st[k]), st[k].shape
            )

    # Simple averaging strategy - can be replaced with more sophisticated merging
    # Merge the corresponding weights of the models
    merged_state = {
        k: sum(st[k] for st in model_states) / len(model_states)
        for k in model_states[0].keys()
    }

    agent.Q_eval.load_state_dict(merged_state)

    torch.save(agent.Q_eval.state_dict(), MASTER_MODEL_PATH)
    logging.info("Master model updated at %s", MASTER_MODEL_PATH)


def main():
    agent = Agent(
        gamma=0.99,
        epsilon=1.0,
        eps_dec=0.99941,
        eps_end=0.01,
        batch_size=64,
        n_actions=4,
        input_dims=[8],
        lr=0.0001,
    )

    model_states = [
        torch.load(
            f"{OUT_DIR}/models/LunarLander-v2-1000{suff}.tar",
            map_location=torch.device("cpu"),
        )["model_state_dict"]
        for suff in ["-base1", "-base2"]
    ]

    merge_models(agent, model_states)


if __name__ == "__main__":
    main()
